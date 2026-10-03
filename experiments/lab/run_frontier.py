"""Component ladder for native frontier computation on exact-search Countdown.

Every model sees start numbers + target and answers "reachable?". The frontier
models run the search inside their forward pass; their success is also checked
exactly: did the states they kept really reach the target?

Ladder (each adds one operator):
  vanilla        plain transformer
  vanilla_aux    plain transformer + the same move-physics labels the frontier gets
  frontier       learned expansion, random pruning (no judge, no interference, no merge)
  +survival      learned survival score picks the K states that advance
  +interference  candidates attend to each other before survival scores them
  full           + merge: latent-equivalent candidates collapse and pool support

Usage: python run_frontier.py [steps] [seed ...]
"""
import json
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import frontier_env as E
from frontier_model import FrontierNet, VanillaReach, first_move_labels

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K, BS, LR = 4, 128, 1e-3
LADDER = {
    "vanilla": None,
    "vanilla_aux": None,
    "vanilla_matched": None,       # d=512 L=6: 567 MFLOPs/example, matches full (573)
    "frontier": dict(survival=False, interference=False, merge=False),
    "+survival": dict(survival=True, interference=False, merge=False),
    "+interference": dict(survival=True, interference=True, merge=False),
    "full": dict(survival=True, interference=True, merge=True),
}


def build(name):
    if name == "vanilla_matched":
        return VanillaReach(d=512, layers=6).to(DEVICE)
    if name.startswith("vanilla"):
        return VanillaReach(aux=name == "vanilla_aux").to(DEVICE)
    return FrontierNet(k=K, **LADDER[name]).to(DEVICE)


def tensors(batch):
    s = torch.tensor([b[0] for b in batch], device=DEVICE)
    t = torch.tensor([b[1] for b in batch], device=DEVICE)
    y = torch.tensor([b[2] for b in batch], device=DEVICE, dtype=torch.float)
    return s, t, y


def loss_of(model, name, batch, gen):
    s, t, y = tensors(batch)
    if name.startswith("vanilla"):
        p, aux = model(s, t)
        loss = F.binary_cross_entropy(p.clamp(1e-6, 1 - 1e-6), y)
        if aux is not None:
            loss = loss + F.cross_entropy(aux.reshape(-1, E.V), first_move_labels(s).reshape(-1))
        return loss
    _, hit_mass, aux, _ = model(s, t, gen=gen)
    loss = model.physics_loss(aux)
    if model.survival and bool(y.sum() > 0):
        loss = loss - torch.log(hit_mass[y > 0] + 1e-6).mean()
    return loss


@torch.no_grad()
def evaluate(model, name, examples):
    """Task metrics for every model; mechanism metrics for frontier models,
    each checked against exact state identity from the environment."""
    model.eval()
    scores, true_hits, pos_mask = [], [], []
    surv_auc, same_cos, diff_cos, kept_distinct, kept_dead, dec_acc, leg_acc = [], [], [], [], [], [], []
    merge_pairs = merge_correct = 0
    gen = torch.Generator(device=DEVICE).manual_seed(1234)
    for i in range(0, len(examples), 256):
        batch = examples[i:i + 256]
        s, t, y = tensors(batch)
        if name.startswith("vanilla"):
            scores += model(s, t)[0].tolist()
            continue
        p, _, aux, rec = model(s, t, gen=gen, record=True)
        scores += p.tolist()
        true_hits += ((rec["final_val"] == t.unsqueeze(1)) & rec["final_ok"]).any(1).tolist()
        pos_mask += (y > 0).tolist()
        for new, leg, r, ok in aux:
            legal = ok & (r > 0)
            if legal.any():
                dec_acc.append((model.decode(new[legal]).argmax(-1) == r[legal]).float().mean().item())
            leg_acc.append(((leg[ok] > 0) == (r[ok] > 0)).float().mean().item())
        for step in (0, 1):
            st = rec[f"step{step}"]
            evs, valid = st["ev"].tolist(), st["valid"]
            for b in range(len(batch)):
                cand = [(j, tuple(sorted(v for v in evs[b][j] if v))) for j in range(len(evs[b])) if valid[b, j]]
                cand = [(j, c) for j, c in cand if c]       # drop dead (illegal-move) states
                if not cand:
                    continue
                if step == 0 and model.survival:
                    a = E.auc([st["score"][b, j].item() for j, _ in cand],
                              [int(batch[b][1] in E.finals(c)) for _, c in cand])
                    if a is not None:
                        surv_auc.append(a)
                if step == 1:
                    c = F.normalize(st["c"][b, [j for j, _ in cand]], dim=-1)
                    cos = (c @ c.T).tolist()
                    for x in range(len(cand)):
                        for z in range(x + 1, len(cand)):
                            same = cand[x][1] == cand[z][1]
                            (same_cos if same else diff_cos).append(cos[x][z])
                            if cos[x][z] > model.merge_tau:
                                merge_pairs += 1
                                merge_correct += same
                    kept_distinct.append(len({tuple(sorted(v for v in evs[b][j] if v))
                                              for j, ok in zip(st["kept"][b].tolist(), st["kept_ok"][b].tolist())
                                              if ok and any(evs[b][j])}))
                    kept_dead.append(sum(1 for j, ok in zip(st["kept"][b].tolist(), st["kept_ok"][b].tolist())
                                         if ok and not any(evs[b][j])))
    labels = [y for *_, y in examples]
    out = {"reach_auc": E.auc(scores, labels),
           "reach_acc": sum((sc > 0.5) == bool(y) for sc, y in zip(scores, labels)) / len(labels)}
    if name.startswith("vanilla"):
        return out

    def mean(v):
        return sum(v) / len(v) if v else None

    n_same, n_diff = len(same_cos), len(diff_cos)
    out.update({
        "frontier_solve_rate": mean([h for h, pm in zip(true_hits, pos_mask) if pm]),
        "physics_decode_acc": mean(dec_acc),
        "physics_legal_acc": mean(leg_acc),
        "survival_auc_step1": mean(surv_auc),
        "path_invariance_auc": E.auc(same_cos + diff_cos, [1] * n_same + [0] * n_diff) if n_same and n_diff else None,
        "same_state_cos": mean(same_cos),
        "diff_state_cos": mean(diff_cos),
        "merge_precision": merge_correct / merge_pairs if merge_pairs else None,
        "merge_recall": merge_correct / n_same if n_same else None,
        "kept_distinct_states_step2": mean(kept_distinct),
        "kept_dead_states_step2": mean(kept_dead),
    })
    return out


def reference(test_eval, seed):
    rng = random.Random(seed)
    pos = [(s, t) for s, t, y in test_eval if y]
    return {f"exact_search_K{K}_{'oracle' if j else 'random'}_judge": sum(E.exact_search(s, t, K, j, True, rng) for s, t in pos) / len(pos)
            for j in (False, True)}


def run(name, steps, seed):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    train_starts, test_starts = E.split_starts(seed)
    index = E.by_target(train_starts)
    test_eval = E.eval_set(test_starts, seed)
    model = build(name)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=steps, pct_start=0.05)
    t0, log = time.time(), []
    for step in range(1, steps + 1):
        model.train()
        loss = loss_of(model, name, E.sample(index, BS, rng), gen)
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        opt.step()
        sched.step()
        if step % max(steps // 5, 1) == 0:
            log.append({"step": step, "loss": round(loss.item(), 4), "grad_norm": round(gn, 3)})
    train_sec = time.time() - t0
    res = {"run": name, "seed": seed, "steps": steps, "params": sum(p.numel() for p in model.parameters()),
           "train_seconds": round(train_sec, 1), "log": log, "device": DEVICE}
    res.update(evaluate(model, name, test_eval))
    res["train_split_reach_auc"] = evaluate(model, name, E.eval_set(train_starts[:400], seed))["reach_auc"]
    return res


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    seeds = [int(s) for s in sys.argv[2:]] or [0]
    only = os.environ.get("LADDER")
    names = only.split(",") if only else list(LADDER)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    all_res = []
    for seed in seeds:
        _, test_starts = E.split_starts(seed)
        test_eval = E.eval_set(test_starts, seed)
        train_starts, _ = E.split_starts(seed)
        print("REFERENCE", seed, json.dumps(reference(test_eval, seed)),
              "SHORTCUTS", json.dumps(E.shortcut_scores(train_starts, test_eval)), flush=True)
        for name in names:
            r = run(name, steps, seed)
            all_res.append(r)
            json.dump(r, open(os.path.join(out_dir, f"frontier_{name}_s{seed}.json"), "w"), indent=1)
            print("RESULT", json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                        for k, v in r.items() if k != "log"}), flush=True)
    print("\n=== mean over seeds", seeds, "===")
    keys = ["reach_auc", "frontier_solve_rate", "survival_auc_step1", "path_invariance_auc",
            "merge_precision", "kept_distinct_states_step2", "kept_dead_states_step2",
            "physics_decode_acc", "train_seconds"]
    for name in names:
        rs = [r for r in all_res if r["run"] == name]
        row = {}
        for k in keys:
            v = [r[k] for r in rs if r.get(k) is not None]
            row[k] = round(sum(v) / len(v), 3) if v else None
        print(f"{name:14s}", json.dumps(row))


if __name__ == "__main__":
    main()
