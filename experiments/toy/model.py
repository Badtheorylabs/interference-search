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
            # unsigned: same pathway, strength is a free real number.
            # softplus(-1.97) = 0.13, so both arms start as the same inhibitory function
            self.raw = nn.Parameter(torch.tensor(-1.97 if arm == "signed" else -0.13))
        self.head_node = nn.Linear(d, d)
        self.head_hint = nn.Linear(1, 1)
        self.back = nn.Linear(d, 1)
        with torch.no_grad():  # untrained policy is near-uniform over children and rarely goes back
            self.back.bias.fill_(-2.0)
            self.head_node.weight.zero_()
            self.head_node.bias.zero_()
            self.head_hint.weight.zero_()
            self.head_hint.bias.zero_()
            self.back.weight.zero_()

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
