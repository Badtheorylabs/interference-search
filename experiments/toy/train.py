"""REINFORCE with grouped baselines (GRPO-style) for the multi-stream policy."""
import argparse
import json
import time

import torch

from env import TreeSearch
from baselines import Local
from model import StreamPolicy

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True)
ap.add_argument("--k", type=int, default=4)
ap.add_argument("--iters", type=int, default=1500)
ap.add_argument("--inst", type=int, default=128, help="instances per batch")
ap.add_argument("--group", type=int, default=4, help="rollouts per instance")
ap.add_argument("--noise", type=float, default=1.0)
ap.add_argument("--refute-p", type=float, default=0.25)
ap.add_argument("--steps", type=int, default=24)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--ent", type=float, default=0.01)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--eval-every", type=int, default=100)
ap.add_argument("--bc-iters", type=int, default=0, help="imitation of independent greedy before RL")
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default=None)
args = ap.parse_args()
torch.manual_seed(args.seed)
torch.set_num_threads(4)
DEV = torch.device(args.device)


def rollout(model, n, k, group=1, gen=None, sample=True, expert=False):
    env = TreeSearch(n * group, k, noise=args.noise, steps=args.steps, gen=gen, refute_p=args.refute_p, device=DEV)
    env.reset()
    if group > 1:  # same instance (goal and hints) for every rollout in a group
        env.goal = env.goal[::group].repeat_interleave(group)
        env.hint = env.hint[::group].repeat_interleave(group, 0)
    N, T = n * group, args.steps
    sids = torch.rand(N, 16, generator=gen).argsort(1)[:, :k].to(DEV)
    obs = env.observe(torch.zeros(N, k, dtype=torch.bool, device=DEV))
    dead_node = torch.full((N, k), -1, dtype=torch.long, device=DEV)
    buf = {x: [] for x in ("pos_code", "ch_hint", "is_leaf", "dead", "dead_code", "child_code")}
    acts, alive = [], []
    for t in range(T):
        buf["pos_code"].append(env.code[obs["pos"]])
        buf["ch_hint"].append(obs["ch_hint"])
        buf["is_leaf"].append(obs["is_leaf"])
        buf["dead"].append(obs["dead"])
        buf["dead_code"].append(env.code[dead_node.clamp(min=0)] * (dead_node >= 0).unsqueeze(-1) - (dead_node < 0).unsqueeze(-1).long())
        buf["child_code"].append(env.code[env.children(obs["pos"]).clamp(max=env.n_nodes - 1)])
        if expert:
            if t == 0:
                ex = Local(env, "greedy_self")
            a = ex.act()
        else:
            inp = {x: torch.stack(v, 2) for x, v in buf.items()}
            with torch.no_grad():
                logits = model(**inp, sids=sids)[:, :, -1]
            a = torch.distributions.Categorical(logits=logits).sample() if sample else logits.argmax(-1)
        alive.append(~env.solved.clone())
        acts.append(a)
        obs, dead_node = env.step(a)
        if expert:
            ex.update(dead_node)
    inp = {x: torch.stack(v, 2) for x, v in buf.items()}
    reward = env.solved.float() * (1 + (T - env.solve_t) / T)
    return inp, sids, torch.stack(acts, 2), torch.stack(alive, 1), reward, env


def evaluate(model, k, n=1000, seed=123):
    g = torch.Generator().manual_seed(seed)
    model.eval()
    _, _, _, _, _, env = rollout(model, n, k, gen=g)
    model.train()
    return env.solved.float().mean().item()


model = StreamPolicy(args.arm, steps=args.steps).to(DEV)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
g = torch.Generator().manual_seed(args.seed)
log = []
t0 = time.time()
# behaviour cloning of the independent greedy expert (no coordination in the teacher)
for it in range(args.bc_iters):
    inp, sids, acts, alive, _, _ = rollout(model, args.inst * args.group, args.k, gen=g, expert=True)
    logits = model(**inp, sids=sids)
    ce = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), acts.reshape(-1), reduction="none")
    m = alive.unsqueeze(1).expand_as(acts).reshape(-1).float()   # ignore steps after the goal was reached
    loss = (ce * m).sum() / m.sum().clamp(min=1)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if it % 100 == 0:
        print(json.dumps({"bc_iter": it, "loss": round(loss.item(), 4), "sec": round(time.time() - t0)}), flush=True)
# RL gets a fresh optimizer: Adam moments from imitation would otherwise shrink RL steps for ~1/(1-beta2) iterations
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
for it in range(args.iters + 1):
    if it % args.eval_every == 0:
        ks = (2, 4, 8) if it in (0, args.iters) else (args.k,)
        ev = {f"K{k}": evaluate(model, k) for k in ks}
        rec = {"iter": it, "sec": round(time.time() - t0), **ev}
        if args.arm in ("signed", "unsigned"):
            s = model.raw.item()
            rec["strength"] = -torch.nn.functional.softplus(model.raw).item() if args.arm == "signed" else s
        log.append(rec)
        print(json.dumps(rec), flush=True)
    if it == args.iters:
        break
    inp, sids, acts, alive, reward, _ = rollout(model, args.inst, args.k, group=args.group, gen=g)
    r = reward.view(args.inst, args.group)
    adv = ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-6)).view(-1)
    logits = model(**inp, sids=sids)
    dist = torch.distributions.Categorical(logits=logits)
    mask = alive.unsqueeze(1).float()                        # (N,1,T): steps before the group solved
    logp = (dist.log_prob(acts) * mask).sum((1, 2))
    ent = (dist.entropy() * mask).sum((1, 2)) / (mask.sum((1, 2)) * args.k).clamp(min=1)  # per stream-step
    loss = -(adv * logp).mean() / (args.k * args.steps) - args.ent * ent.mean()
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if it % 10 == 0:
        print(json.dumps({"rl_iter": it, "sec": round(time.time() - t0, 1), "train_reward": round(reward.mean().item(), 4),
                          "train_solved": round((reward > 0).float().mean().item(), 4),
                          "entropy": round(ent.mean().item(), 3)}), flush=True)

if args.out:
    json.dump({"args": vars(args), "log": log}, open(args.out, "w"))
    torch.save(model.state_dict(), args.out.replace(".json", ".pt"))
