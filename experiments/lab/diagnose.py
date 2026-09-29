"""Mechanism diagnostics for the interference lab. Measures, does not assume.

1. Dead parameters: which weights receive no gradient from the training loss.
2. Frontier trace at init: slot norm and slot diversity per token step, for
   each mode. Diversity = mean pairwise cosine between slots (1.0 = all slots
   identical, i.e. no branching).
3. Merge operator at init, isolated: does it contract slots toward their mean,
   or reflect them through it?
4. Equal-step training (same data, same steps for every model), logging loss,
   grad norm, test accuracy, frontier norm, slot cosine, survival, equiv.

Usage: python diagnose.py [steps] [seed]
"""
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from envs import make_dataset
from model import InterferenceLM, VanillaLM, match_params, param_count
from run_lab import BS, D, DEVICE, K, L, LR, VOCAB, build_rows, ce_state, evaluate_state

MODES = {
    "interference": dict(prop_mode="learned", merge_mode="learned", survival_mode="learned"),
    "merge_off": dict(prop_mode="learned", merge_mode="off", survival_mode="learned"),
    "survival_off": dict(prop_mode="learned", merge_mode="learned", survival_mode="one"),
}


def slot_stats(f):
    fn = F.normalize(f, dim=-1)
    cos = fn @ fn.transpose(1, 2)
    k = f.size(1)
    off = ~torch.eye(k, dtype=torch.bool, device=f.device)
    spread = (f - f.mean(1, keepdim=True)).norm(dim=-1).mean() / f.norm(dim=-1).mean().clamp_min(1e-9)
    return {"norm": f.norm(dim=-1).mean().item(),
            "slot_cos": cos[:, off].mean().item(),
            "rel_spread": spread.item()}


@torch.no_grad()
def trace(model, ids, valid, kw):
    B = ids.size(0)
    x = model.tok(ids)
    for b in model.blocks:
        x = b(x)
    f = model.slot_init.unsqueeze(0).expand(B, model.k, model.d).contiguous()
    last = valid.long().sum(1) - 1
    rows = [dict(t=-1, **slot_stats(f))]
    for t in range(int(last.max()) + 1):
        f_new, _, st = model.unit(f, x[:, t], **kw)
        f = torch.where(valid[:, t].view(B, 1, 1), f_new, f)
        rows.append(dict(t=t, **slot_stats(f), **st))
    return rows


@torch.no_grad()
def merge_isolated(model):
    """Feed the merge step a frontier with known spread; report spread after."""
    unit = model.unit
    B, k, d = 64, model.k, model.d
    f = torch.randn(B, k, d, device=DEVICE)
    fi = f.unsqueeze(2).expand(B, k, k, d)
    fj = f.unsqueeze(1).expand(B, k, k, d)
    rel = unit.rel(torch.cat([fi, fj], -1))
    equiv = torch.sigmoid(rel[..., 2])
    a = torch.sigmoid(unit.alpha(torch.cat([fi, fj], -1)).squeeze(-1))
    f2 = f + ((equiv * a).unsqueeze(-1) * (fj - fi)).sum(2)
    dev1 = f - f.mean(1, keepdim=True)
    dev2 = f2 - f2.mean(1, keepdim=True)
    ratio = (dev2.norm(dim=-1).mean() / dev1.norm(dim=-1).mean()).item()
    corr = F.cosine_similarity(dev1.flatten(1), dev2.flatten(1), dim=-1).mean().item()
    return {"equiv_mean": equiv.mean().item(), "alpha_mean": a.mean().item(),
            "sum_j_equiv_x_alpha": (equiv * a).sum(2).mean().item(),
            "spread_ratio_after_merge": ratio,
            "cos(deviation_before, deviation_after)": corr}


def build_model(mode):
    target = param_count(InterferenceLM(VOCAB, D, L, k=K))
    if mode == "vanilla":
        return match_params(VanillaLM(VOCAB, D, L, n_out=90), target).to(DEVICE)
    return InterferenceLM(VOCAB, D, L, k=K).to(DEVICE)


def loss_of(model, mode, ids, tgt):
    valid = ids != 0
    if mode == "vanilla":
        return ce_state(model(ids, valid=valid).view(-1, 9, 10), tgt)
    return ce_state(model(ids, valid=valid, **MODES[mode])[2], tgt)


@torch.no_grad()
def acc_of(model, mode, loader):
    if mode != "vanilla":
        return evaluate_state(model, loader, **MODES[mode])
    model.eval()
    c = n = 0
    for ids, tgt in loader:
        ids, tgt = ids.to(DEVICE), tgt.to(DEVICE)
        pred = model(ids, valid=ids != 0).view(-1, 9, 10).argmax(-1)
        c += (pred == tgt).all(1).sum().item(); n += len(tgt)
    return c / max(n, 1)


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    torch.manual_seed(seed)
    (tr_ids, tr_tgt), (te_ids, te_tgt) = build_rows(make_dataset(seed=seed))
    train_loader = DataLoader(TensorDataset(tr_ids, tr_tgt), batch_size=BS, shuffle=True)
    test_loader = DataLoader(TensorDataset(te_ids, te_tgt), batch_size=BS)
    probe_ids = tr_ids[:64].to(DEVICE)
    probe_valid = probe_ids != 0
    report = {"device": DEVICE, "steps": steps, "seed": seed}

    # 0. compute and numerics: FLOPs per forward, and GPU agrees with CPU
    from torch.utils.flop_counter import FlopCounterMode
    flops = {}
    for mode in ["vanilla", "interference"]:
        torch.manual_seed(seed)
        m = build_model(mode).eval()
        with FlopCounterMode(display=False) as fc, torch.no_grad():
            if mode == "vanilla":
                m(probe_ids, valid=probe_valid)
            else:
                m(probe_ids, valid=probe_valid, **MODES["interference"])
        flops[mode] = fc.get_total_flops()
    report["flops_per_64_rows"] = flops
    report["flops_ratio_interference_over_vanilla"] = flops["interference"] / flops["vanilla"]
    torch.manual_seed(seed)
    m = build_model("interference").eval()
    with torch.no_grad():
        g_out = m(probe_ids, valid=probe_valid, **MODES["interference"])[2].float().cpu()
        m_cpu = m.to("cpu")
        c_out = m_cpu(probe_ids.cpu(), valid=probe_valid.cpu(), **MODES["interference"])[2]
    report["gpu_vs_cpu_max_abs_diff"] = (g_out - c_out).abs().max().item()

    # 1. dead parameters
    torch.manual_seed(seed)
    m = build_model("interference")
    loss = loss_of(m, "interference", tr_ids[:BS].to(DEVICE), tr_tgt[:BS].to(DEVICE))
    m.zero_grad(); loss.backward()
    dead = [(n, p.numel()) for n, p in m.named_parameters()
            if p.grad is None or p.grad.abs().sum().item() == 0]
    report["dead_params"] = {"tensors": [n for n, _ in dead],
                             "count": sum(c for _, c in dead),
                             "total": param_count(m)}
    report["param_counts"] = {"interference": param_count(m),
                              "vanilla": param_count(build_model("vanilla"))}

    # 2-3. frontier trace and isolated merge at init
    report["init_trace"] = {}
    for mode, kw in MODES.items():
        torch.manual_seed(seed)
        m = build_model(mode)
        m.eval()
        report["init_trace"][mode] = trace(m, probe_ids, probe_valid, kw)
    torch.manual_seed(seed)
    report["merge_isolated_init"] = merge_isolated(build_model("interference"))

    # 4. equal-step training
    report["train"] = {}
    for mode in ["vanilla", "interference", "merge_off", "survival_off"]:
        torch.manual_seed(seed)
        m = build_model(mode)
        opt = torch.optim.AdamW(m.parameters(), lr=LR)
        log, it, t0, gmax = [], iter(train_loader), time.time(), 0.0
        for step in range(1, steps + 1):
            m.train()
            try:
                ids, tgt = next(it)
            except StopIteration:
                it = iter(train_loader); ids, tgt = next(it)
            loss = loss_of(m, mode, ids.to(DEVICE), tgt.to(DEVICE))
            opt.zero_grad(); loss.backward()
            g = torch.nn.utils.clip_grad_norm_(m.parameters(), float("inf")).item()
            gmax = max(gmax, g if g == g else float("inf"))
            opt.step()
            if step % max(steps // 8, 1) == 0 or step == steps:
                row = {"step": step, "loss": loss.item(), "grad_norm": g, "grad_norm_max_so_far": gmax,
                       "test_acc": acc_of(m, mode, test_loader), "sec": round(time.time() - t0, 1)}
                if mode != "vanilla":
                    m.eval()
                    tr = trace(m, probe_ids, probe_valid, MODES[mode])[-1]
                    row.update({k2: tr[k2] for k2 in ("norm", "slot_cos", "rel_spread",
                                                      "survival_mean", "equiv", "support", "conflict")})
                log.append(row)
                print(mode, json.dumps(row), flush=True)
        report["train"][mode] = log

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", f"diagnose_s{seed}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(report, open(out, "w"), indent=1)
    print("WROTE", out)
    print("DEAD", json.dumps(report["dead_params"]))
    print("PARAMS", json.dumps(report["param_counts"]))
    print("FLOPS", json.dumps(report["flops_per_64_rows"]),
          "ratio", round(report["flops_ratio_interference_over_vanilla"], 3))
    print("GPU_VS_CPU", report["gpu_vs_cpu_max_abs_diff"])
    print("MERGE_INIT", json.dumps(report["merge_isolated_init"]))
    for mode, rows in report["init_trace"].items():
        pick = [rows[0], rows[1], rows[len(rows) // 2], rows[-1]]
        print("INIT", mode, json.dumps([{k2: round(v, 4) if isinstance(v, float) else v
                                          for k2, v in r.items()} for r in pick]))


if __name__ == "__main__":
    main()
