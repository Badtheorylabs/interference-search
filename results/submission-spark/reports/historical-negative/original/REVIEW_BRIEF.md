# Interference Search: review brief

I'd like a critical review of this research direction, its experiments, and the architecture being tested. Please look for design flaws, confounds, bugs in reasoning, and anything that would make a positive result meaningless. Everything below ran locally on an Apple M2 with 16 GB. Some results are still pending; those are marked.

## 1. Goal

LLMs spend inference compute in two ways today. They think longer (one sequential chain), or they think wider (N independent samples, subagents, best-of-N, tree search). Wide sampling has a known weakness: samples from the same model share the same biases, so they tend to make the same mistakes. Sixty-four samples can fail in the same way sixty-four times.

The idea comes from quantum computing. A quantum computer's advantage comes from interference, not from "trying everything at once": amplitudes can be negative, so wrong paths cancel before measurement. The LLM analogue I want to test is **destructive coupling**: K parallel reasoning streams inside one model, where a stream's verified dead end actively pushes its siblings away from the same region of the search space.

The concrete question: does a dedicated, sign-constrained suppression pathway between parallel streams buy more search success per unit of compute than (a) independent streams or (b) streams that can simply attend to each other?

## 2. Prior art (searched 2026-09-22)

- **Streams that see each other:** Hogwild! Inference (2504.06261, shared KV cache, training-free), Group Think (2505.11107, token-level visibility between threads), Multi-Stream LLMs (2605.12460, trained cross-stream causal attention on Qwen up to 27B). All give positive visibility. None has an explicit mechanism for one stream to suppress another.
- **Native parallel branching with RL:** Parallel-R1 (2509.07980), ParaThinker (2509.04475), Native Parallel Reasoner (2512.07461), TSLM (2601.22688). Branches are isolated during generation and meet only at a merge.
- **Central pruning:** Parallel-Probe (2602.03845), DeepPrune (2510.08483), HyPER (2602.06527). An external controller kills branches using answer agreement or reward scores. Branches never learn why a sibling failed.
- **Latent superposition:** "Reasoning by Superposition" (2505.12514) shows continuous thoughts can hold several BFS frontiers. "The Illusion of Superposition?" (2604.06374) finds this does not happen in pretrained models retrofitted with latent CoT; only models trained from scratch superpose.
- **Supporting evidence:** Parason (2608.24658) finds 55 to 74% of parallelizable reasoning steps are competing trials where most fail. Reject, Resample, Repeat (2603.07887) analyzes parallel sampling as Sequential Monte Carlo, the closest classical cousin.

The gap I believe exists: learned, content-bearing, sign-constrained suppression between parallel streams, trained end to end. Please check whether I missed prior work.

## 3. What was run on a real LLM, and what it showed

Model: Qwen3-1.7B, 4-bit, MLX, batched generation. Task: Countdown (reach a target from given numbers with + - * /). Every attempt is an arithmetic expression, so I extract attempts from reasoning traces, evaluate them exactly, and canonicalize them (commutative operations sorted) with no LLM judge. K = 8 streams per problem, lockstep.

### Phase 0: is there redundancy to remove?

Six hard 4-number problems (filtered so no single pair of numbers lands within 25 of the target), 2,048 tokens per stream.

- 381 failed attempts. **29.7%** repeated a failure that a sibling stream had already produced earlier in token time. **45.1%** repeated the stream's own earlier failure.
- Share of all tokens spent on sibling-redundant failures: **16.5%** on the pre-registered primary metric (between the 10% kill and 25% go thresholds, so the pre-registered rule was to run the cheap Phase 1 test before any training).
- 46% of tokens came after some stream had already found the answer. That is the "tell siblings you're done" signal, which Parallel-Probe already exploits, so it does not count for this idea.

### Phase 1: shared textual ledger (prompting, no training)

24 problems, 256-token chunks, 8 rounds. Three arms with identical seeds, so streams are identical until the first note:

- **A:** independent.
- **P:** each stream is periodically reminded of its own verified wrong expressions.
- **C:** each stream is also told its siblings' verified wrong expressions ("already checked, WRONG, do not retry").

| Arm | Found (of 24) | Mean tokens to find | Sibling-redundant failures | Self-repeats |
|---|---|---|---|---|
| A | 22 | 5,803 | 23.0% | 55.1% |
| P | 21 | 6,485 | 21.7% | 55.6% |
| C | 22 | 5,376 | 27.3% | 54.3% |

No significant difference (sign test p = 1.0). Near ceiling: 16 of 24 problems were decided before notes could act. **The mechanism check failed in an informative way:** in the 80 tokens after a shared note, 33.6% of C's attempts repeated a sibling refutation, against about 21% in A and P. The model goes back and checks what the note lists. "Do not retry X" primes X. Private reminders did not reduce self-repeats either.

### Phase 1b: rewind and resample (decode-time suppression)

When a stream wrote an already-refuted full attempt, it was rewound to the start of that line and resampled, with discarded tokens charged to its budget. Stopped after 5 problems: on 5-number problems the model wrote almost no complete attempts, so there was nothing to suppress. One observation mattered. A stream was rewound 16 times and regenerated the same wrong expression every time. The cause of the mistake lives upstream in the context, so deleting it at decode time changes nothing.

**Conclusion from the LLM phases:** a pretrained sequential model cannot be prompted or decoded into destructive coupling. If the behaviour exists, it has to be in the weights. That motivated the from-scratch test below.

## 4. The from-scratch architecture test (running now)

### Environment: hidden-goal tree search

- Complete tree, branching 3, depth 4 (81 leaves), one goal leaf chosen at random per episode.
- Every node has a hint: mean 1 on the root-to-goal path, 0 elsewhere, plus Gaussian noise with sigma 1.0. Hints are fixed per episode and **shared by all streams**, so streams are drawn to the same wrong subtrees. This is the property that makes the toy a model of correlated LLM samples.
- K streams move in lockstep for 24 steps. Actions: go to child 0, 1, or 2, or go back to the parent. Reaching a non-goal leaf is a dead end: the event is recorded and the stream teleports to the root. The group succeeds when any stream reaches the goal.
- A stream observes its current node, the hints of that node's children, and a dead-end flag plus the identity of the leaf it just found dead.

### Hand-coded bounds (4,000 episodes; each uses only what a stream can observe)

| Policy | K=1 | K=2 | K=4 | K=8 |
|---|---|---|---|---|
| random | 0.07 | 0.13 | 0.27 | 0.45 |
| greedy, own dead-end memory | 0.37 | 0.37 | 0.37 | 0.37 |
| greedy, shared dead-end memory | 0.37 | 0.37 | 0.37 | 0.37 |
| coordinator (shared memory + avoid subtrees a sibling is in or just chose) | 0.37 | 0.50 | 0.71 | 0.85 |

Greedy is flat in K because every stream makes the same choices. The coordinator shows the prize for real teamwork: +34 points at K=4.

### Model

One transformer shared by all K streams. Token (k, t) is stream k's observation at step t; its output gives stream k's action at step t. The sequence is the K×T grid flattened. Width d = 64, 2 layers, 4 heads, pre-norm, no dropout. About 112k parameters (120k for the dead-end arms).

Token embedding = node embedding + linear(child hints) + linear(is_leaf, dead flag) + linear(node embedding of the leaf just found dead, if any) + time embedding + stream-ID embedding (IDs drawn at random from 16 per episode, so width can change at test time).

Node embedding = sum over depth levels of a learned embedding for the child index taken at that level. A leaf therefore shares directions with its ancestors, so "near a dead leaf" is expressible as similarity.

Action logits: for child c, `logit_c = (W h) · E(c) / sqrt(d) + a · hint_c + b`, and a separate linear head for "back".

### The four arms

1. **indep:** attention mask lets token (k, t) see only (k, s ≤ t).
2. **visible:** token (k, t) sees every stream's tokens at s ≤ t.
3. **unsigned:** visible, plus a dead-end term added to child logits: `logit_c += λ · D_c`.
4. **signed:** visible, plus `logit_c -= softplus(ρ) · D_c`.

The dead-end term:

```
D_c(k,t) = Σ_e  sigmoid( q(h_kt) · k(E(e)) / sqrt(d) ) · relu( cos(E(c), E(e)) )
```

The sum runs over all dead-end events e (own and siblings) at steps ≤ t. D_c is always non-negative, so in the signed arm the pathway can only lower the logits of children that resemble known dead ends. It is a sum rather than a softmax, so more dead leaves under a subtree give more inhibition. In the unsigned arm λ is a free scalar. Both start as the same function: softplus(ρ₀) = 0.13 and λ₀ = −0.13, both inhibitory. **The sign constraint is the only difference between arms 3 and 4.**

An earlier draft subtracted in hidden space. I replaced it because learned projections downstream could flip the sign and make the constraint meaningless.

### Training

- **Imitation warm start, 300 iterations:** every arm first imitates the "greedy, own memory" policy, which has no coordination in it. Cross-entropy is masked after the goal is reached.
- **RL, 1,000 iterations:** REINFORCE with GRPO-style grouped baselines. 128 instances × 4 rollouts each. Advantage = (R − group mean) / group std. Reward = solved × (1 + (T − solve_step) / T). Entropy bonus 0.01 per stream-step, learning rate 3e-4, Adam, gradient clip 1.0, trained at K = 4.
- **Tuning:** hyperparameters were tuned on the indep arm only. The first attempts failed from policy collapse and an entropy term that was mis-scaled by K; both were fixed before the main runs.
- **Evaluation:** 4,000 fresh episodes at K = 2, 4, 8 with sampled actions. I also measure redundancy: the share of dead-end hits that land on a leaf any stream had already found dead.
- **Seeds:** 3 per arm (0 and 1 first, seed 2 queued).

### Results so far (end-of-training evals on 1,000 episodes; final 4,000-episode eval pending)

| Arm | Seed | K=2 | K=4 | K=8 |
|---|---|---|---|---|
| indep | 0 | 0.332 | 0.513 | 0.727 |
| indep | 1 | 0.365 | 0.542 | 0.757 |
| visible | 0 | 0.321 | 0.525 | 0.718 |
| visible | 1 | 0.375 | 0.542 | 0.754 |
| unsigned | 0, 1 | pending | pending | pending |
| signed | 0, 1 | pending | pending | pending |

The 4,000-episode eval of indep (2 seeds) gives 0.343, 0.542, 0.728 at K = 2, 4, 8, with 16.9% redundancy at K = 4.

**Interim reading:** plain cross-stream attention gave no gain over independent streams. Both learned to beat greedy, apparently by sampling diversity, and both sit about 18 points under the coordinator at K = 4. The information needed to coordinate is available to the visible arm, and it did not learn to use it in 1,000 iterations.

## 5. Known issues and open questions

1. **The strength scalar barely moves.** In the unsigned arm, λ went from −0.1144 to −0.1144 over the first 100 RL iterations. It drifted from −0.13 to −0.114 during imitation. Its gradient may be tiny or noisy, so the "free sign" control might not really be free. Does the unsigned arm test what I claim it tests?
2. **Compute cost.** The dead-end term is O((KT)² · B · d) per forward pass, which makes those arms about 2.5× slower per iteration. Fine for a toy; it would need a sparse or cached form for real use.
3. **Is the hand-designed similarity doing the work?** relu(cos(E(c), E(e))) builds tree structure into the pathway. A signed-arm win could come from that inductive bias, not from the sign constraint. The unsigned arm shares the same term, which is why it is the key control, but I'd like a view on whether that is enough.
4. **Imitation teacher.** Warm-starting from an uncoordinated greedy policy may create a basin that RL can't escape at 1,000 iterations. It affects all arms equally, but it could hide real differences.
5. **Width generalization.** Training is at K = 4 only. K = 2 and K = 8 are out-of-distribution tests.
6. **Statistics.** Two or three seeds per arm is thin. With end-of-training noise around ±0.02, a real effect needs to be several points to be credible.
7. **Transfer.** Even a clean toy win says nothing yet about language models. The planned next step is retrofitting the signed pathway into a pretrained model (Multi-Stream-style cross-stream attention plus the dead-end term) and fine-tuning with RL on Countdown, where refutations are exactly checkable.

## 6. What I'd like from the review

- Is the signed-vs-unsigned comparison a fair isolation of "negative coupling"? What would you change?
- Is there a simpler baseline that would explain away a signed-arm win?
- Are the Phase 0 and Phase 1 conclusions justified by those numbers?
- What prior work did I miss?
- If the signed arm wins, what is the most convincing next experiment on a real LLM that fits a small budget?

---

## Appendix A: technical specification

### A.1 Notation and shapes

- N = batch of episodes, K = streams, T = steps (24), B = branching (3), D = depth (4), d = 64.
- Nodes are indexed in level order: root 0, children of n are nB+1 … nB+B. 121 nodes, leaves are 40…120.
- `code[n]` ∈ {−1, 0, 1, 2}^D is the child index taken at each level on the root-to-n path, padded with −1.
- Inputs per token (N, K, T, ·): `pos_code` (D), `ch_hint` (B), `is_leaf` (1), `dead` (1), `dead_code` (D, all −1 when no event), `child_code` (B × D). Stream IDs `sids` (N, K).

### A.2 Embeddings

```
E(n)     = Σ_{l=1..D, code[n][l] ≥ 0}  L[l, code[n][l]]            L ∈ R^{D×B×d}, init N(0, 0.3²)
x_{k,t}  = E(pos) + W_h·ch_hint + W_f·[is_leaf, dead]
           + dead · W_dn·E(dead_leaf) + Emb_time[t] + Emb_sid[sid_k]
```

### A.3 Attention mask (flattened index i = k·T + t, stream-major)

```
allowed(i → j) = (t_j ≤ t_i)                          visible, unsigned, signed
allowed(i → j) = (t_j ≤ t_i) and (k_j = k_i)          indep
```

Siblings' tokens at the same step t are visible. All streams observe simultaneously before acting, so this doesn't leak future information. Encoder: `nn.TransformerEncoder`, 2 layers, 4 heads, FFN 4d, pre-norm, dropout 0, then a final LayerNorm, giving h_{k,t} ∈ R^d.

I checked causal consistency numerically: logits from step-by-step rollout (a growing prefix) match a single full-sequence forward to 3e-7.

### A.4 Action head

```
logit_c    = (W_o h_{k,t}) · E(child_c) / √d  +  a·hint_c  +  b            c = 0..B−1
logit_back = w_b · h_{k,t} + b_b
children masked to −1e9 when the stream is on a leaf (only happens after the goal is reached)
```

### A.5 Dead-end term (unsigned and signed arms only)

```
events  e = (k', s) with dead_{k',s} = 1 and s ≤ t          own and siblings
w_{kt,e} = sigmoid( (W_q h_{k,t}) · (W_k E(e)) / √d )       gate, not a softmax
sim_{c,e} = relu( cos( E(child_c), E(e) ) )
D_c       = Σ_e w_{kt,e} · sim_{c,e}                         ≥ 0

unsigned:  logit_c ← logit_c + λ·D_c                          λ ∈ R, init −0.13
signed:    logit_c ← logit_c − softplus(ρ)·D_c                ρ init −1.97 → softplus = 0.13
```

Cost is O((KT)²·B·d) per forward pass. D_c is added directly to the logits, so no learned map downstream can flip its sign.

### A.6 Parameter counts

| Component | visible | signed / unsigned |
|---|---|---|
| node embedding L | 768 | 768 |
| hint, flags, dead_node, time, sid | 7,232 | 7,232 |
| transformer (2 layers) + final norm | 100,096 | 100,096 |
| action head (W_o, a, b, back) | 4,227 | 4,227 |
| dead-end term (W_q, W_k, ρ or λ) | 0 | 8,321 |
| **total** | **112,323** | **120,644** |

indep has the same count as visible; only the mask differs. The dead-end arms have 7% more parameters. A parameter-matched visible control (for example d widened, or an unused W_q/W_k) has not been run.

### A.7 Training procedure

1. **Imitation (300 iterations).** Roll out the teacher ("greedy, own memory": at each node pick the highest-hint child whose subtree still has a leaf this stream hasn't found dead; go back if none) on 512 episodes at K = 4. Cross-entropy on the teacher's actions, masked after the group reaches the goal.
2. **RL (1,000 iterations).** Each iteration: 128 instances, each replicated 4 times with the same goal and hints, sampled actions, then one full-sequence forward for log-probs.

```
R      = solved · (1 + (T − t_solve)/T)
A_i    = (R_i − mean_group) / (std_group + 1e-6)
loss   = − mean_i[ A_i · Σ_{k,t < t_solve} log π(a_{k,t}) ] / (K·T)  −  0.01 · H̄
H̄      = mean entropy per stream-step, over steps before the goal was reached
Adam, lr 3e-4, grad-norm clip 1.0
```

3. **Evaluation.** 4,000 fresh episodes per width (K = 2, 4, 8), actions sampled from the policy, fixed eval seed. Success = any stream reached the goal within 24 steps. Redundancy = dead-end hits on a leaf already in the shared dead set, divided by all dead-end hits, counted before the goal is reached.

### A.8 Hyperparameters

| | value |
|---|---|
| tree | B = 3, D = 4, 81 leaves |
| hint noise σ | 1.0 |
| steps T | 24 |
| training width K | 4 |
| d, layers, heads | 64, 2, 4 |
| imitation iterations | 300 |
| RL iterations | 1,000 |
| instances × group | 128 × 4 |
| learning rate | 3e-4 (tuned on indep only) |
| entropy coefficient | 0.01 per stream-step |
| seeds | 0, 1, 2 |

### A.9 Bugs found and fixed before the main runs

- The entropy term was averaged per step instead of per stream-step (K times too large). It drove policies toward uniform and made RL look like it was degrading.
- The first imitation teacher ranked full root-to-leaf paths using hints the learner never observes. It couldn't be imitated, and it inflated the "ceiling" numbers. Both the teacher and the bounds were replaced with policies that use only observable information.
- The imitation loss wasn't masked after the goal was reached. A stream sitting on the goal leaf had its child actions masked to −1e9, so the loss hit 2×10⁸.
- The first coordinator baseline didn't break ties within a step, so all streams moved identically. It now avoids children that earlier streams chose in the same step.

## Appendix B: source code

Four files, 437 lines. PyTorch, CPU.

### toy/env.py

```python
"""Hidden-goal tree search, vectorized over a batch of episodes and K streams.

A complete tree with branching B and depth D has one goal leaf. Every node carries
a noisy hint: mean 1 on the goal path, 0 elsewhere, plus Gaussian noise. The hint
is fixed per node within an episode, so every stream sees the same misleading
hints and tends to make the same mistakes, like samples from one LLM.

Streams move in lockstep. Actions: 0..B-1 go to that child, B goes back to the
parent. Reaching a non-goal leaf is a dead end and sends the stream to the root.
The group succeeds when any stream reaches the goal.
"""
import torch


class TreeSearch:
    def __init__(self, batch, k, branching=3, depth=4, noise=1.0, steps=24, device="cpu", gen=None):
        self.N, self.K, self.B, self.D = batch, k, branching, depth
        self.noise, self.T, self.dev = noise, steps, device
        self.g = gen
        # node ids: level-order, root = 0; children of n are n*B+1 .. n*B+B
        self.n_nodes = sum(branching ** d for d in range(depth + 1))
        self.first_leaf = self.n_nodes - branching ** depth
        par = torch.zeros(self.n_nodes, dtype=torch.long)
        for n in range(1, self.n_nodes):
            par[n] = (n - 1) // branching
        self.parent = par.to(device)
        dep = torch.zeros(self.n_nodes, dtype=torch.long)
        for n in range(1, self.n_nodes):
            dep[n] = dep[par[n]] + 1
        self.depth_of = dep.to(device)
        # path code: child index taken at each level, -1 padded (for embeddings)
        code = torch.full((self.n_nodes, depth), -1, dtype=torch.long)
        for n in range(1, self.n_nodes):
            code[n] = code[par[n]]
            code[n, dep[n] - 1] = (n - 1) % branching
        self.code = code.to(device)

    def reset(self):
        N, dev = self.N, self.dev
        leaves = self.n_nodes - self.first_leaf
        self.goal = self.first_leaf + torch.randint(leaves, (N,), generator=self.g, device=dev)
        on_path = torch.zeros(N, self.n_nodes, device=dev)
        node = self.goal.clone()
        for _ in range(self.D + 1):
            on_path[torch.arange(N), node] = 1.0
            node = self.parent[node]
        self.hint = on_path + self.noise * torch.randn(N, self.n_nodes, generator=self.g, device=dev)
        self.pos = torch.zeros(N, self.K, dtype=torch.long, device=dev)
        self.t = 0
        self.solved = torch.zeros(N, dtype=torch.bool, device=dev)
        self.solve_t = torch.full((N,), self.T, device=dev)
        self.dead = torch.zeros(N, self.n_nodes, dtype=torch.bool, device=dev)  # leaves found dead
        return self.observe(torch.zeros(N, self.K, dtype=torch.bool, device=dev))

    def children(self, pos):
        base = pos * self.B + 1
        return base.unsqueeze(-1) + torch.arange(self.B, device=self.dev)

    def observe(self, dead_event):
        """Per stream: current node, hints of its children, whether it is a leaf, dead-end flag."""
        is_leaf = self.pos >= self.first_leaf
        ch = self.children(self.pos).clamp(max=self.n_nodes - 1)
        ch_hint = torch.gather(self.hint.unsqueeze(1).expand(-1, self.K, -1), 2, ch)
        ch_hint = torch.where(is_leaf.unsqueeze(-1), torch.zeros_like(ch_hint), ch_hint)
        return {"pos": self.pos.clone(), "ch_hint": ch_hint, "is_leaf": is_leaf, "dead": dead_event}

    def step(self, action, last_node=None):
        """action: (N, K) in 0..B. Returns obs, the node each stream just left dead (or -1)."""
        is_leaf = self.pos >= self.first_leaf
        down = self.children(self.pos).clamp(max=self.n_nodes - 1)
        go_child = torch.gather(down, 2, action.clamp(max=self.B - 1).unsqueeze(-1)).squeeze(-1)
        up = self.parent[self.pos]
        new = torch.where(action == self.B, up, torch.where(is_leaf, self.pos, go_child))
        self.pos = new
        self.t += 1
        at_leaf = self.pos >= self.first_leaf
        hit_goal = self.pos == self.goal.unsqueeze(1)
        newly = hit_goal.any(1) & ~self.solved
        self.solve_t = torch.where(newly, torch.full_like(self.solve_t, self.t), self.solve_t)
        self.solved |= hit_goal.any(1)
        dead_event = at_leaf & ~hit_goal
        dead_node = torch.where(dead_event, self.pos, torch.full_like(self.pos, -1))
        # mark dead leaves and send those streams back to the root
        idx = dead_node.clamp(min=0)
        self.dead.scatter_(1, idx, self.dead.gather(1, idx) | dead_event)
        self.pos = torch.where(dead_event, torch.zeros_like(self.pos), self.pos)
        return self.observe(dead_event), dead_node

    def done(self):
        return self.t >= self.T
```

### toy/model.py

```python
"""Multi-stream policy: one transformer shared by K streams over lockstep tokens.

Token (k, t) is stream k's observation at step t; its output picks stream k's action
at step t. Arms differ only in what a token may read and how dead-end information
enters the action head:

  indep     own stream only
  visible   every stream's tokens at steps <= t
  unsigned  visible + a dead-end head that shifts child logits by a free-sign amount
  signed    visible + the same head constrained to subtract (strength >= 0):
            known dead ends can only push the policy away, never toward
Both dead-end arms start as the same function (strength 0.13, inhibitory), so the
sign constraint is the only difference between them.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

ARMS = ("indep", "visible", "unsigned", "signed")


class NodeEmbed(nn.Module):
    """Node = sum of per-level child-index embeddings, so a leaf shares direction with its ancestors."""

    def __init__(self, branching, depth, d):
        super().__init__()
        self.lvl = nn.Parameter(torch.randn(depth, branching, d) * 0.3)
        self.depth = depth

    def forward(self, code):  # code: (..., depth) with -1 padding
        mask = (code >= 0).unsqueeze(-1).float()
        idx = code.clamp(min=0)
        emb = self.lvl[torch.arange(self.depth, device=code.device), idx]  # (..., depth, d)
        return (emb * mask).sum(-2)


class StreamPolicy(nn.Module):
    def __init__(self, arm, branching=3, depth=4, d=64, layers=2, heads=4, steps=24, n_ids=16):
        super().__init__()
        assert arm in ARMS
        self.arm, self.B, self.d = arm, branching, d
        self.node = NodeEmbed(branching, depth, d)
        self.hint = nn.Linear(branching, d)
        self.flags = nn.Linear(2, d)
        self.dead_node = nn.Linear(d, d)
        self.time = nn.Embedding(steps + 1, d)
        self.sid = nn.Embedding(n_ids, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d)
        if arm in ("unsigned", "signed"):
            self.q = nn.Linear(d, d)
            self.k = nn.Linear(d, d)
            # signed: strength = softplus(raw) >= 0, applied as a subtraction.
            # unsigned: same pathway, strength is a free real number (starts at 0).
            # softplus(-1.97) = 0.13, so both arms start as the same inhibitory function
            self.raw = nn.Parameter(torch.tensor(-1.97 if arm == "signed" else -0.13))
        self.head_node = nn.Linear(d, d)
        self.head_hint = nn.Linear(1, 1)
        self.back = nn.Linear(d, 1)

    def tokens(self, pos_code, ch_hint, is_leaf, dead, dead_code, t_idx, sids):
        # pos_code (N,K,T,D) ch_hint (N,K,T,B) is_leaf/dead (N,K,T) dead_code (N,K,T,D)
        x = self.node(pos_code) + self.hint(ch_hint)
        x = x + self.flags(torch.stack([is_leaf.float(), dead.float()], -1))
        x = x + self.dead_node(self.node(dead_code)) * dead.unsqueeze(-1).float()
        x = x + self.time(t_idx) + self.sid(sids).unsqueeze(2)
        return x

    def dead_term(self, h, dead, dead_code, ch, N, K, T):
        """Per-child logit shift from every dead-end event (own and siblings) at steps <= t.

        term_c = sum_e sigmoid(q_t . k_e) * relu(cos(child_c, dead_e)).  The relu and the
        sigmoid keep the term >= 0, so its only possible effect in the signed arm is to
        lower the logits of children that resemble known dead ends.
        """
        ev = self.node(dead_code).reshape(N, K * T, self.d)
        is_ev = dead.reshape(N, K * T)
        tq = torch.arange(T, device=h.device).repeat(K)
        allowed = (tq.unsqueeze(1) >= tq.unsqueeze(0)).unsqueeze(0) & is_ev.unsqueeze(1)
        w = torch.sigmoid(self.q(h) @ self.k(ev).transpose(1, 2) / self.d ** 0.5) * allowed  # (N,KT,KT)
        chn = F.normalize(ch.reshape(N, K * T, self.B, self.d), dim=-1)
        evn = F.normalize(ev, dim=-1)
        sim = torch.relu(torch.einsum("nqbd,ned->nqbe", chn, evn))                           # (N,KT,B,KT)
        term = (sim * w.unsqueeze(2)).sum(-1).reshape(N, K, T, self.B)
        if self.arm == "signed":
            return -F.softplus(self.raw) * term
        return self.raw * term

    def attn_mask(self, K, T, device):
        t = torch.arange(T, device=device)
        s = torch.arange(K, device=device)
        tq = t.repeat(K)            # flattened (k, t) order: k major
        sq = s.repeat_interleave(T)
        causal = tq.unsqueeze(1) >= tq.unsqueeze(0)
        if self.arm == "indep":
            allowed = causal & (sq.unsqueeze(1) == sq.unsqueeze(0))
        else:
            allowed = causal
        return ~allowed  # True = blocked

    def forward(self, pos_code, ch_hint, is_leaf, dead, dead_code, child_code, sids):
        N, K, T = is_leaf.shape
        t_idx = torch.arange(T, device=is_leaf.device).view(1, 1, T).expand(N, K, T)
        x = self.tokens(pos_code, ch_hint, is_leaf, dead, dead_code, t_idx, sids)
        x = x.reshape(N, K * T, self.d)
        h = self.norm(self.tf(x, mask=self.attn_mask(K, T, x.device)))
        h = h.reshape(N, K, T, self.d)
        # action logits: children scored by similarity to the state, plus their raw hint
        ch = self.node(child_code)                              # (N,K,T,B,d)
        logit_c = (self.head_node(h).unsqueeze(-2) * ch).sum(-1) / self.d ** 0.5
        logit_c = logit_c + self.head_hint(ch_hint.unsqueeze(-1)).squeeze(-1)
        logit_c = logit_c.masked_fill(is_leaf.unsqueeze(-1), -1e9)
        if self.arm in ("unsigned", "signed"):
            logit_c = logit_c + self.dead_term(h.reshape(N, K * T, self.d), dead, dead_code, ch, N, K, T)
        logit_b = self.back(h)
        return torch.cat([logit_c, logit_b], -1)                # (N,K,T,B+1)
```

### toy/train.py

```python
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
ap.add_argument("--steps", type=int, default=24)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--ent", type=float, default=0.01)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--eval-every", type=int, default=100)
ap.add_argument("--bc-iters", type=int, default=300, help="imitation of independent greedy before RL")
ap.add_argument("--out", default=None)
args = ap.parse_args()
torch.manual_seed(args.seed)
torch.set_num_threads(4)


def rollout(model, n, k, group=1, gen=None, sample=True, expert=False):
    env = TreeSearch(n * group, k, noise=args.noise, steps=args.steps, gen=gen)
    env.reset()
    if group > 1:  # same instance (goal and hints) for every rollout in a group
        env.goal = env.goal[::group].repeat_interleave(group)
        env.hint = env.hint[::group].repeat_interleave(group, 0)
    N, T = n * group, args.steps
    sids = torch.stack([torch.randperm(16, generator=gen)[:k] for _ in range(N)])
    obs = env.observe(torch.zeros(N, k, dtype=torch.bool))
    dead_node = torch.full((N, k), -1, dtype=torch.long)
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
    redundant = None
    return inp, sids, torch.stack(acts, 2), torch.stack(alive, 1), reward, env


def evaluate(model, k, n=1000, seed=123):
    g = torch.Generator().manual_seed(seed)
    model.eval()
    _, _, _, _, _, env = rollout(model, n, k, gen=g)
    model.train()
    return env.solved.float().mean().item()


model = StreamPolicy(args.arm, steps=args.steps)
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
        print(json.dumps({"rl_iter": it, "train_reward": round(reward.mean().item(), 4),
                          "train_solved": round((reward > 0).float().mean().item(), 4),
                          "entropy": round(ent.mean().item(), 3)}), flush=True)

if args.out:
    json.dump({"args": vars(args), "log": log}, open(args.out, "w"))
    torch.save(model.state_dict(), args.out.replace(".json", ".pt"))
```

### toy/baselines.py

```python
"""Hand-coded local policies that bound what learned streams can achieve.

Every policy here uses only what a learned stream can observe: the hints of the
current node's children, plus memory of which leaves were found dead.

  random         uniform over children
  greedy_self    go to the child with the highest hint whose subtree is not fully dead
                 for this stream (own memory only; the imitation teacher)
  greedy_shared  same, with the dead set shared by all streams
  coordinator    shared dead set, and prefer children whose subtree no sibling is
                 currently inside (the ceiling for coordination from local views)
"""
import torch

from env import TreeSearch


def leaf_ranges(env):
    """First and last leaf index (0-based among leaves) under every node."""
    lo = torch.zeros(env.n_nodes, dtype=torch.long)
    hi = torch.zeros(env.n_nodes, dtype=torch.long)
    for n in range(env.n_nodes - 1, -1, -1):
        if n >= env.first_leaf:
            lo[n] = hi[n] = n - env.first_leaf
        else:
            c0, c1 = n * env.B + 1, n * env.B + env.B
            lo[n], hi[n] = lo[c0], hi[c1]
    return lo, hi


class Local:
    def __init__(self, env, kind, gen=None):
        self.env, self.kind, self.g = env, kind, gen
        self.lo, self.hi = leaf_ranges(env)
        n_leaves = env.n_nodes - env.first_leaf
        self.own = torch.zeros(env.N, env.K, n_leaves, dtype=torch.bool)
        # prefix sums let us test "subtree fully dead" in O(1)
        self.leaf_idx = torch.arange(n_leaves)

    def exhausted(self, dead, nodes):
        """dead: (N, L) bool, nodes: (N, B) -> (N, B) whether every leaf under node is dead."""
        cs = torch.cat([torch.zeros(dead.shape[0], 1), dead.float().cumsum(1)], 1)
        lo, hi = self.lo[nodes], self.hi[nodes]
        cnt = cs.gather(1, hi + 1) - cs.gather(1, lo)
        return cnt >= (hi - lo + 1)

    def act(self):
        env = self.env
        if self.kind == "random":
            return torch.randint(env.B, (env.N, env.K), generator=self.g)
        acts, nxt = [], []
        shared = env.dead[:, env.first_leaf:]
        for s in range(env.K):
            ch = env.children(env.pos[:, s]).clamp(max=env.n_nodes - 1)       # (N, B)
            h = env.hint.gather(1, ch)
            dead = self.own[:, s] if self.kind == "greedy_self" else shared
            score = h.masked_fill(self.exhausted(dead, ch), -1e9)
            if self.kind == "coordinator":
                # penalise children whose subtree holds a sibling right now
                occ = torch.zeros_like(score, dtype=torch.bool)
                for j in range(env.K):
                    if j == s:
                        continue
                    # a sibling at node p is inside child c's subtree iff p's leaf range nests in c's
                    lo_j, hi_j = self.lo[env.pos[:, j]], self.hi[env.pos[:, j]]
                    inside = (lo_j.unsqueeze(1) >= self.lo[ch]) & (hi_j.unsqueeze(1) <= self.hi[ch]) \
                        & (env.pos[:, j] != 0).unsqueeze(1)
                    occ |= inside
                for nj in nxt:  # children already chosen this step by earlier streams
                    occ |= (self.lo[nj].unsqueeze(1) >= self.lo[ch]) & (self.hi[nj].unsqueeze(1) <= self.hi[ch]) \
                        & (nj != 0).unsqueeze(1)
                score = score - 50.0 * occ.float()
            a = score.argmax(1)
            a = torch.where(score.max(1).values < -1e8, torch.full_like(a, env.B), a)
            acts.append(a)
            nxt.append(torch.where(a < env.B, ch.gather(1, a.clamp(max=env.B - 1).unsqueeze(1)).squeeze(1),
                                   env.parent[env.pos[:, s]]))
        return torch.stack(acts, 1)

    def update(self, dead_node):
        hit = dead_node >= 0
        for s in range(self.env.K):
            h = hit[:, s]
            self.own[h, s, dead_node[h, s] - self.env.first_leaf] = True


def run(kind, batch=4000, k=4, steps=24, noise=1.5, seed=0):
    g = torch.Generator().manual_seed(seed)
    env = TreeSearch(batch, k, noise=noise, steps=steps, gen=g)
    env.reset()
    pol = Local(env, kind, gen=g)
    while not env.done():
        _, dead_node = env.step(pol.act())
        pol.update(dead_node)
    return env.solved.float().mean().item()


if __name__ == "__main__":
    for noise in (1.0, 1.5):
        print(f"noise {noise}")
        for k in (1, 2, 4, 8):
            print(f"  K={k}: " + "   ".join(f"{p} {run(p, k=k, noise=noise):.2f}"
                                            for p in ("random", "greedy_self", "greedy_shared", "coordinator")))
```
