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
    def __init__(self, batch, k, branching=3, depth=4, noise=1.0, steps=24, device="cpu", gen=None,
                 refute_p=0.0):
        """refute_p: chance that entering an off-goal-path internal node (depth >= 1) reveals
        "this whole subtree is dead", like a bad intermediate value in Countdown killing every
        expression built on it. 0 reproduces the original leaf-only environment."""
        self.N, self.K, self.B, self.D = batch, k, branching, depth
        self.refute_p = refute_p
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
        lo = torch.zeros(self.n_nodes, dtype=torch.long)
        hi = torch.zeros(self.n_nodes, dtype=torch.long)
        for n in range(self.n_nodes - 1, -1, -1):
            if n >= self.first_leaf:
                lo[n] = hi[n] = n - self.first_leaf
            else:
                lo[n], hi[n] = lo[n * branching + 1], hi[n * branching + branching]
        self.leaf_lo, self.leaf_hi = lo.to(device), hi.to(device)

    def reset(self):
        N, dev = self.N, self.dev
        leaves = self.n_nodes - self.first_leaf
        self.goal = self.first_leaf + torch.randint(leaves, (N,), generator=self.g).to(dev)
        on_path = torch.zeros(N, self.n_nodes, device=dev)
        node = self.goal.clone()
        for _ in range(self.D + 1):
            on_path[torch.arange(N, device=dev), node] = 1.0
            node = self.parent[node]
        self.hint = on_path + self.noise * torch.randn(N, self.n_nodes, generator=self.g).to(dev)
        self.on_path = on_path.bool()
        # which off-path internal nodes reveal themselves as dead when entered (fixed per episode)
        internal = (torch.arange(self.n_nodes, device=dev) > 0) & (torch.arange(self.n_nodes, device=dev) < self.first_leaf)
        self.refutes = (~self.on_path) & internal & (torch.rand(N, self.n_nodes, generator=self.g).to(dev) < self.refute_p)
        self.pos = torch.zeros(N, self.K, dtype=torch.long, device=dev)
        self.t = 0
        self.solved = torch.zeros(N, dtype=torch.bool, device=dev)
        self.solve_t = torch.full((N,), self.T, device=dev)
        self.dead = torch.zeros(N, self.n_nodes, dtype=torch.bool, device=dev)  # nodes known dead
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
        refuted = self.refutes.gather(1, self.pos)
        dead_event = (at_leaf & ~hit_goal) | refuted
        dead_node = torch.where(dead_event, self.pos, torch.full_like(self.pos, -1))
        # mark every node in each dead subtree (a dead leaf is a subtree of one) and send those streams home
        n_idx = torch.arange(self.n_nodes, device=self.dev)
        for s in range(self.K):
            dn = dead_node[:, s]
            has = dn >= 0
            if has.any():
                lo, hi = self.leaf_lo[dn.clamp(min=0)], self.leaf_hi[dn.clamp(min=0)]
                nl, nh = self.leaf_lo[n_idx], self.leaf_hi[n_idx]
                inside = (nl.unsqueeze(0) >= lo.unsqueeze(1)) & (nh.unsqueeze(0) <= hi.unsqueeze(1)) & has.unsqueeze(1)
                self.dead |= inside
        self.pos = torch.where(dead_event, torch.zeros_like(self.pos), self.pos)
        return self.observe(dead_event), dead_node

    def done(self):
        return self.t >= self.T
