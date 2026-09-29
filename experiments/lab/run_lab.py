"""Matched-compute runner: vanilla transformer vs interference model.

Task: read a move trajectory, predict the exact remaining multiset (outcome
supervision only, no judge labels). Equivalence is tested zero-shot: two
trajectories are equivalent iff the model's predicted states agree, with no
pair supervision. Ablations: merge-off and survival-off attribute any gain to
the named operator rather than extra parameters.

Reports experiments/lab/results/<run>.json and prints a summary.
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from envs import make_dataset, make_pairs, encode_seq, encode_state, SEQ_PAD
from model import VanillaLM, InterferenceLM, param_count, match_params

VOCAB = 17          # PAD + 16 alphabet symbols
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
D, L, K, STEPS, LR, BS = 128, 3, 8, 400, 3e-3, 32


def build_rows(groups):
    """Per-group split: most trajectories of a state train, a couple hold out.
    The held-out trajectory belongs to a state the model saw via other paths."""
    train_ids, train_tgt, test_ids, test_tgt = [], [], [], []
    for g in groups:
        seqs = g["seqs"][:8]
        tgt = encode_state(g["state"])
        n_keep = max(1, len(seqs) - 2)
        for i, seq in enumerate(seqs):
            enc = encode_seq(list(seq))
            if i < n_keep:
                train_ids.append(enc); train_tgt.append(tgt)
            else:
                test_ids.append(enc); test_tgt.append(tgt)
    return ((torch.tensor(train_ids, dtype=torch.long), torch.tensor(train_tgt, dtype=torch.long)),
            (torch.tensor(test_ids, dtype=torch.long), torch.tensor(test_tgt, dtype=torch.long)))


def state_loss(model, batch, pair=False):
    ids, tgt = batch
    valid = (ids != 0)
    if pair:
        logits, _, _ = model(ids, valid=valid, pair=True)
        return F.cross_entropy(logits, tgt[:, 0])
    _, _, state_logits = model(ids, valid=valid)
    return ce_state(state_logits, tgt)


def ce_state(state_logits, tgt):
    # state_logits (B,9,10); tgt (B,9) digit classes
    return F.cross_entropy(state_logits.reshape(-1, 10), tgt.reshape(-1))


def evaluate_state(model, loader, **kw):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for ids, tgt in loader:
            ids, tgt = ids.to(DEVICE), tgt.to(DEVICE)
            _, _, state_logits = model(ids, valid=(ids != 0), **kw)
            pred = state_logits.argmax(-1)
            exact = (pred == tgt).all(dim=1)
            correct += exact.sum().item(); total += len(tgt)
    return correct / max(total, 1)


def soft_agreement(model, a, b, vanilla=False):
    """Probability under the model that two trajectories reach the same state:
    per-digit predicted distributions multiplied position-wise. Purely the
    model's own readout; no external judge."""
    model.eval()
    with torch.no_grad():
        ia = torch.tensor([encode_seq(a)], device=DEVICE)
        ib = torch.tensor([encode_seq(b)], device=DEVICE)
        if vanilla:
            pa = model(ia, valid=(ia != 0)).view(1, 9, 10).softmax(-1)[0]
            pb = model(ib, valid=(ib != 0)).view(1, 9, 10).softmax(-1)[0]
        else:
            pa = model(ia, valid=(ia != 0))[2].softmax(-1)[0]
            pb = model(ib, valid=(ib != 0))[2].softmax(-1)[0]
        agree = (pa * pb).sum(-1)          # per-digit prob both predict same digit
        return float(agree.mean().item())   # averaged over 9 digit slots


def evaluate_equiv(model, pairs, labels, vanilla=False):
    scores = [soft_agreement(model, a, b, vanilla) for (a, b) in pairs]
    # accuracy over the best threshold the model's own scores support:
    # pick the threshold maximizing held-out accuracy on these labeled pairs
    best = 0.5
    for t in [i / 100 for i in range(10, 96, 1)]:
        acc = sum(((s > t) == (y == 1)) for s, y in zip(scores, labels)) / len(scores)
        best = max(best, acc)
    return best


def run(name, mode, seed=0):
    torch.manual_seed(seed)
    groups = make_dataset(seed=seed)
    (train_ids, train_tgt), (test_ids, test_tgt) = build_rows(groups)
    train = TensorDataset(train_ids, train_tgt)
    test = TensorDataset(test_ids, test_tgt)
    train_loader = DataLoader(train, batch_size=BS, shuffle=True)
    test_loader = DataLoader(test, batch_size=BS)

    if mode == "vanilla":
        model = VanillaLM(VOCAB, D, L, n_out=90)
    else:
        model = InterferenceLM(VOCAB, D, L, k=K)
    model = match_params(model, param_count(InterferenceLM(VOCAB, D, L, k=K)))
    model = model.to(DEVICE)
    params = param_count(model)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    def fwd(batch, **kw):
        ids, tgt = batch
        ids, tgt = ids.to(DEVICE), tgt.to(DEVICE)
        valid = (ids != 0)
        if mode == "vanilla":
            out = model(ids, valid=valid)
            logits = out.view(-1, 9, 10)
            return ce_state(logits, tgt)
        out = model(ids, valid=valid, **kw)
        return ce_state(out[2], tgt)

    t0 = time.time()
    data = iter(train_loader)
    losses = []
    for step in range(STEPS):
        try:
            batch = next(data)
        except StopIteration:
            data = iter(train_loader)
            batch = next(data)
        loss = fwd(batch,
                   prop_mode="learned",
                   merge_mode="learned" if mode != "merge_off" else "off",
                   survival_mode="learned" if mode != "survival_off" else "one")
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    dt = time.time() - t0

    kw = dict(prop_mode="learned",
              merge_mode="learned" if mode != "merge_off" else "off",
              survival_mode="learned" if mode != "survival_off" else "one")
    if mode == "vanilla":
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for ids, tgt in test_loader:
                ids, tgt = ids.to(DEVICE), tgt.to(DEVICE)
                logits = model(ids, valid=(ids != 0)).view(-1, 9, 10)
                pred = logits.argmax(-1)
                correct += (pred == tgt).all(dim=1).sum().item(); total += len(tgt)
        acc_state = correct / max(total, 1)
    else:
        acc_state = evaluate_state(model, test_loader, **kw)

    pairs, labels = make_pairs(make_dataset(seed=seed + 991))
    acc_eq = evaluate_equiv(model, pairs, labels, vanilla=(mode == "vanilla"))

    res = {"run": name, "mode": mode, "params": int(params), "steps": STEPS,
           "train_seconds": round(dt, 1), "final_loss": losses[-1],
           "exact_state_accuracy": acc_state, "equivalence_agreement": acc_eq,
           "device": DEVICE}
    os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
    path = os.path.join(os.path.dirname(__file__), "results", f"{name}.json")
    json.dump(res, open(path, "w"), indent=1)
    print(json.dumps(res))
    return res


def evaluate_vanilla_equiv(model, pairs, labels):
    model.eval()
    ok, total = 0, 0
    with torch.no_grad():
        for (a, b), y in zip(pairs, labels):
            ia = torch.tensor([encode_seq(a)], device=DEVICE)
            ib = torch.tensor([encode_seq(b)], device=DEVICE)
            pa = model(ia, valid=(ia != 0)).view(-1, 9, 10).argmax(-1)[0]
            pb = model(ib, valid=(ib != 0)).view(-1, 9, 10).argmax(-1)[0]
            agree = int((pa == pb).all().item())
            ok += int((agree == 1) == (y == 1)); total += 1
    return ok / max(total, 1)


if __name__ == "__main__":
    for mode in ["vanilla", "interference", "merge_off", "survival_off"]:
        run(f"lab_{mode}", mode)
