"""Native frontier network: Interference Search inside one forward pass.

The model gets the start numbers and a target. Inside the forward pass it
keeps a frontier of K latent states and advances it move by move:

  expand      a learned arithmetic layer turns each state into its children
  interfere   candidate states attend to each other and to the target
  survive     a learned score picks the K states that advance (top-K, like MoE routing)
  merge       candidates whose latent states coincide collapse into one,
              pooling their support; the frontier width is not spent on copies

A state is a set of up to 4 latent number vectors. Numbers are never snapped
back to symbols: a child's new number is whatever the arithmetic layer
computes, so two paths to the same multiset share a representation only if
the network learned to put them there.

Training signals:
  physics   the value and legality of every move the network expands (the
            environment's rules, what the paper's environment executes)
  outcome   for reachable examples, the likelihood that the frontier's own
            search path hits the target. The final hit is decoded with
            detached heads, so this loss cannot make the network claim a hit;
            it reaches the arithmetic layer only through survival scores.

No label says which state should survive or which states are equal.

`exact_apply` and the `ev` tensors carry true values for labels and metrics
only; the forward computation never reads them.
"""
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from frontier_env import CAP, N_NUMS, OPS, V

PAIRS = list(itertools.combinations(range(N_NUMS), 2))
M = len(PAIRS) * len(OPS)                      # 24 moves per state
_I = torch.tensor([p[0] for p in PAIRS for _ in OPS])
_J = torch.tensor([p[1] for p in PAIRS for _ in OPS])
_O = torch.tensor([[k for k in range(N_NUMS) if k not in p] for p in PAIRS for _ in OPS])
_OP = torch.tensor([o for _ in PAIRS for o in range(len(OPS))])


def exact_apply(x, y, op_idx):
    """True move results on integer tensors; 0 marks illegal (or a padded slot)."""
    a, b = torch.maximum(x, y), torch.minimum(x, y)
    bs = b.clamp_min(1)
    res = torch.stack([a + b, a * b, a - b, torch.where((b > 1) & (a % bs == 0), a // bs, 0)])
    r = res.gather(0, op_idx.expand_as(x).unsqueeze(0)).squeeze(0)
    return torch.where((x > 0) & (y > 0) & (r > 0) & (r <= CAP), r, 0)


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


class FrontierNet(nn.Module):
    def __init__(self, d=128, k=4, survival=True, interference=True, merge=True):
        super().__init__()
        self.d, self.k = d, k
        self.survival, self.interference, self.merge = survival, interference, merge
        self.num = nn.Embedding(V, d)
        self.op = nn.Embedding(len(OPS), d)
        self.arith = nn.Sequential(mlp(4 * d, 2 * d, d), nn.LayerNorm(d))
        self.legal = mlp(4 * d, d, 1)
        self.dec_bias = nn.Parameter(torch.zeros(V))   # decoder tied to the number embedding
        self.log_scale = nn.Parameter(torch.tensor(2.5))
        self.phi = mlp(d, d, d)
        self.pool = nn.Sequential(mlp(d, d, d), nn.LayerNorm(d))
        self.tgt = nn.Embedding(V, d)
        if interference:
            self.inter = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
        if survival:
            self.score = mlp(3 * d, d, 1)
        # merge: two candidates are one state when their latent states point the
        # same way. The threshold is fixed, not trained: an outcome loss would
        # always reward merging more (more pooled mass), so a learned threshold
        # would drift toward merging everything. What is learned is the state
        # space itself; merge_tau is audited against exact identity.
        self.merge_tau = 0.98
        self.norm = nn.LayerNorm(d)

    # ----- operators -----
    def codes(self):
        """Number code book: the same normalized vectors the model reads its
        input numbers with. Tying the decoder to it means a computed 12 is
        pushed toward the vector of an input 12, the only way two paths can
        share a state. It does not say which states are equal."""
        return self.norm(self.num.weight)

    def decode(self, h, detach=False):
        w, b, s = self.codes(), self.dec_bias, self.log_scale.exp()
        if detach:
            w, b, s = w.detach(), b.detach(), s.detach()
        return F.normalize(h, dim=-1) @ F.normalize(w, dim=-1).T * s + b

    def state_rep(self, n, m):
        """Permutation-invariant latent state from its number vectors."""
        return self.pool((self.phi(n) * m.unsqueeze(-1)).sum(-2))

    def expand(self, n, m, ev):
        """n (B,S,4,d), m (B,S,4) -> children (B,S,M,4,d), masks, legality logits, new-number vectors."""
        dev = n.device
        I, J, O, OP = _I.to(dev), _J.to(dev), _O.to(dev), _OP.to(dev)
        ni, nj = n[:, :, I], n[:, :, J]
        feat = torch.cat([ni + nj, ni * nj, (ni - nj).abs(), self.op(OP).expand_as(ni)], -1)
        new = self.arith(feat)
        leg = self.legal(feat).squeeze(-1)
        pair_ok = m[:, :, I] & m[:, :, J]
        cn = torch.cat([n[:, :, O], new.unsqueeze(-2), torch.zeros_like(new).unsqueeze(-2)], -2)
        cm = torch.cat([m[:, :, O], pair_ok.unsqueeze(-1), torch.zeros_like(pair_ok).unsqueeze(-1)], -1)
        r = exact_apply(ev[:, :, I], ev[:, :, J], OP.expand_as(ev[:, :, I]))
        cev = torch.cat([ev[:, :, O], r.unsqueeze(-1), torch.zeros_like(r).unsqueeze(-1)], -1)
        # A child reached by an illegal move is not a real state: zero its true
        # numbers so nothing downstream can count it (no false hits from states
        # that skipped a number). The network may still keep it if its learned
        # legality is wrong; that shows up as wasted frontier width.
        cev = cev * (r > 0).unsqueeze(-1)
        return cn, cm, leg, new, pair_ok, r, cev

    def select(self, c_raw, c_int, t, valid, gen):
        """Pick K states. Returns kept indices, their support mass, alive flags,
        plus scores and equivalences for metrics."""
        B, C, _ = c_raw.shape
        if self.survival:
            tt = t.unsqueeze(1).expand_as(c_int)
            score = self.score(torch.cat([c_int, tt, c_int * tt], -1)).squeeze(-1)
        else:                                   # no judge: random order
            score = torch.rand(B, C, device=c_raw.device, generator=gen)
        score = score.masked_fill(~valid, -1e9)
        q = torch.softmax(score, -1) * valid
        pick = score
        if self.training and self.survival:
            u = torch.rand(B, C, device=c_raw.device, generator=gen).clamp(1e-6, 1 - 1e-6)
            pick = score - torch.log(-torch.log(u))         # Gumbel exploration
        alive = valid.clone()
        cn = F.normalize(c_raw.detach(), dim=-1)
        idx, mass, ok = [], [], []
        ar = torch.arange(B, device=c_raw.device)
        for _ in range(self.k):
            a = pick.masked_fill(~alive, -1e9).argmax(-1)
            live = alive[ar, a]
            m_a = q[ar, a]
            alive[ar, a] = False
            if self.merge:
                cos = (cn * cn[ar, a].unsqueeze(1)).sum(-1)
                sup = (cos > self.merge_tau) & alive & live.unsqueeze(1)
                m_a = m_a + (q * sup).sum(-1)
                alive = alive & ~sup
            idx.append(a); mass.append(m_a * live); ok.append(live)
        return torch.stack(idx, 1), torch.stack(mass, 1), torch.stack(ok, 1), score

    def interfere(self, c, t, valid):
        if not self.interference:
            return c
        x = torch.cat([t.unsqueeze(1), c], 1)
        pad = torch.cat([torch.zeros_like(valid[:, :1]), ~valid], 1)
        return self.inter(x, src_key_padding_mask=pad)[:, 1:]

    # ----- forward -----
    def forward(self, starts, targets, gen=None, record=False):
        """starts (B,4) ints, targets (B,) ints."""
        B = starts.size(0)
        dev = starts.device
        n = self.norm(self.num(starts)).unsqueeze(1)          # (B,1,4,d)
        m = (starts > 0).unsqueeze(1)
        ev = starts.unsqueeze(1)
        alive = torch.ones(B, 1, dtype=torch.bool, device=dev)
        path_mass = torch.ones(B, 1, device=dev)
        t = self.tgt(targets)
        aux, rec = [], {}
        ar = torch.arange(B, device=dev)
        for step in range(N_NUMS - 1):
            cn, cm, leg, new, pair_ok, r, cev = self.expand(n, m, ev)
            S = n.size(1)
            ok_lab = pair_ok & alive.unsqueeze(-1)
            real = (ev > 0).any(-1, keepdim=True)          # label-side only: parent is a true state
            aux.append((new, leg, r, ok_lab & real))
            valid = (ok_lab & (leg > 0)).reshape(B, S * M)
            if step == N_NUMS - 2:                             # final move: every child is a single number
                # Outcome path: decoder and legality heads are detached, so the
                # outcome loss cannot make the network claim a hit that the
                # arithmetic did not produce.
                logits = self.decode(new.detach(), detach=True)
                match = torch.softmax(logits, -1).gather(-1, targets.view(B, 1, 1, 1).expand(B, S, M, 1)).squeeze(-1)
                hit = torch.sigmoid(leg.detach()) * match * ok_lab
                p = 1 - torch.prod((1 - hit).reshape(B, -1).clamp_min(1e-6), -1)
                hit_mass = (hit * path_mass.unsqueeze(-1)).reshape(B, -1).sum(-1)
                if record:
                    rec["final_val"] = r.reshape(B, -1)
                    rec["final_ok"] = ok_lab.reshape(B, -1)
                return p, hit_mass, aux, rec
            c_raw = self.state_rep(cn, cm).reshape(B, S * M, -1)
            c_int = self.interfere(c_raw, t, valid)
            kidx, kmass, kok, score = self.select(c_raw, c_int, t, valid, gen)
            parent = kidx // M
            path_mass = path_mass[ar.unsqueeze(1), parent] * kmass
            n = cn.reshape(B, S * M, N_NUMS, -1)[ar.unsqueeze(1), kidx]
            m = cm.reshape(B, S * M, N_NUMS)[ar.unsqueeze(1), kidx]
            ev = cev.reshape(B, S * M, N_NUMS)[ar.unsqueeze(1), kidx]
            alive = kok
            if record:
                rec[f"step{step}"] = dict(c=c_raw.detach(), score=score.detach(), valid=valid,
                                           ev=cev.reshape(B, S * M, N_NUMS), kept=kidx, kept_ok=kok)
        raise AssertionError("unreachable: the final move returns")

    def physics_loss(self, aux):
        ce, bce = [], []
        for new, leg, r, ok in aux:
            legal = ok & (r > 0)
            if legal.any():
                ce.append(F.cross_entropy(self.decode(new[legal]), r[legal]))
            bce.append(F.binary_cross_entropy_with_logits(leg[ok], (r[ok] > 0).float()))
        return sum(ce) / max(len(ce), 1) + sum(bce) / len(bce)


class VanillaReach(nn.Module):
    """Plain transformer: start numbers + target tokens -> reachable?"""

    def __init__(self, d=256, layers=4, aux=False):
        super().__init__()
        self.num = nn.Embedding(V, d)
        self.role = nn.Embedding(2, d)
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, 4, 4 * d, dropout=0.0, batch_first=True, norm_first=True), layers,
            enable_nested_tensor=False)
        self.out = nn.Linear(d, 1)
        self.aux_head = nn.Linear(d, M * V) if aux else None

    def forward(self, starts, targets):
        x = torch.cat([self.num(starts) + self.role.weight[0], (self.num(targets) + self.role.weight[1]).unsqueeze(1)], 1)
        h = self.enc(x).mean(1)
        aux = self.aux_head(h).view(-1, M, V) if self.aux_head is not None else None
        return torch.sigmoid(self.out(h).squeeze(-1)), aux


def first_move_labels(starts):
    """True result of each of the M first moves (0 = illegal)."""
    dev = starts.device
    I, J, OP = _I.to(dev), _J.to(dev), _OP.to(dev)
    return exact_apply(starts[:, I], starts[:, J], OP.expand(starts.size(0), -1))
