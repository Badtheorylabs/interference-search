"""Hand-coded local policies that bound what learned streams can achieve.

Every policy here uses only what a learned stream can observe: the hints of the
current node's children, plus memory of which leaves were found dead.

  random         uniform over children
  greedy_self    go to the child with the highest hint whose subtree is not fully dead
                 for this stream (own memory only; the imitation teacher)
  greedy_shared  same, with the dead set shared by all streams
  soft_self      softmax over child hints (temperature 0.5), own dead memory
  soft_shared    same, shared dead memory
  coordinator    shared dead set, and prefer children whose subtree no sibling is
                 currently inside or just chose (a heuristic, like virtual loss in
                 parallel MCTS; a learned policy could beat it)
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
            dead = self.own[:, s] if self.kind in ("greedy_self", "soft_self") else shared
            score = h.masked_fill(self.exhausted(dead, ch), -1e9)
            if self.kind.startswith("soft"):
                gumbel = -torch.log(-torch.log(torch.rand(score.shape, generator=self.g).clamp(1e-9, 1)))
                score = torch.where(score > -1e8, score / 0.5 + gumbel, score)
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
        """Record each dead subtree (a dead leaf is a subtree of one) in the stream's own memory."""
        leaf = torch.arange(self.own.shape[-1])
        for s in range(self.env.K):
            dn = dead_node[:, s]
            has = dn >= 0
            lo, hi = self.lo[dn.clamp(min=0)], self.hi[dn.clamp(min=0)]
            self.own[:, s] |= (leaf.unsqueeze(0) >= lo.unsqueeze(1)) & (leaf.unsqueeze(0) <= hi.unsqueeze(1)) & has.unsqueeze(1)


def run(kind, batch=4000, k=4, steps=24, noise=1.0, seed=0, refute_p=0.0):
    g = torch.Generator().manual_seed(seed)
    env = TreeSearch(batch, k, noise=noise, steps=steps, gen=g, refute_p=refute_p)
    env.reset()
    pol = Local(env, kind, gen=g)
    while not env.done():
        _, dead_node = env.step(pol.act())
        pol.update(dead_node)
    return env.solved.float().mean().item()


if __name__ == "__main__":
    pols = ("random", "greedy_self", "greedy_shared", "soft_self", "soft_shared", "coordinator")
    for rp in (0.0, 0.5, 1.0):
        print(f"noise 1.0, refute_p {rp}")
        for k in (1, 2, 4, 8):
            print(f"  K={k}: " + "  ".join(f"{p} {run(p, k=k, refute_p=rp):.2f}" for p in pols))
