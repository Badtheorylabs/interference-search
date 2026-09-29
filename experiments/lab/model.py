"""Interference Transformer lab models.

Mechanism name: native latent interference (precise term; not "native reasoning").

VanillaLM      standard causal-attention trunk + state readout.
InterferenceLM same trunk, plus a recurrent interference unit advancing a
               learned frontier F across the token positions. Four operators:
               proposal, relation (support/conflict/equivalence), merge,
               survival. Supervision is the outcome label only: no external
               judge, no search loop supervises merge or survival.

Sequences are causal; readout happens at the last valid token. The frontier
skips padded positions.
"""
import torch
import torch.nn as nn


def causal_mask(L, device):
    return torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), 1)


def last_valid_index(valid):
    """Per-row index of the last valid token (valid rows are prefix-only)."""
    return valid.long().sum(dim=1) - 1


def gather_rows(x, idx):
    """x: (B, L, D); idx: (B,) -> (B, D)"""
    return x.gather(1, idx.view(-1, 1, 1).expand(-1, 1, x.size(-1))).squeeze(1)


class Block(nn.Module):
    def __init__(self, d, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.n1 = nn.LayerNorm(d)
        self.n2 = nn.LayerNorm(d)
        self.comp = None  # optional compensation MLP set by the trainer for param matching

    def forward(self, x):
        L = x.size(1)
        m = causal_mask(L, x.device)
        h = self.n1(x)
        x = x + self.attn(h, h, h, attn_mask=m)[0]
        x = x + self.mlp(self.n2(x))
        if self.comp is not None:
            x = x + self.comp(x)
        return x


class VanillaLM(nn.Module):
    def __init__(self, vocab, d, layers, heads=4, n_out=2):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.head = nn.Linear(d, n_out)

    def forward(self, ids, valid=None):
        x = self.tok(ids)
        for b in self.blocks:
            x = b(x)
        if valid is None:
            valid = torch.ones(ids.shape[0], ids.shape[1], dtype=torch.bool, device=ids.device)
        h = gather_rows(x, last_valid_index(valid))
        return self.head(h)


class InterfereUnit(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.d = d
        self.k = k
        self.prop = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d))
        self.rel = nn.Linear(2 * d, 3)          # support / conflict / equivalence
        self.alpha = nn.Linear(2 * d, 1)         # learned merge mixing
        self.surv = nn.Linear(2 * d + 3, 1)
        # survival starts near open: a random sigmoid(~0) gate multiplied per token
        # collapses the frontier (0.5^10) before relations are learned, which the
        # first run measured as survival hurting state accuracy. Start at ~0.95.
        nn.init.constant_(self.surv.bias, 3.0)
        nn.init.zeros_(self.surv.weight)
        self.up = nn.Linear(d, d)

    def forward(self, f, x, prop_mode="learned", merge_mode="learned", survival_mode="learned"):
        B, K, d = f.shape
        xc = x.unsqueeze(1).expand(B, K, d)
        # proposal: new hypotheses from each slot conditioned on the token state
        if prop_mode == "learned":
            f = f + self.prop(torch.cat([f, xc], -1))
        # relation: pairwise semantic scores between slots
        fi = f.unsqueeze(2).expand(B, K, K, d)
        fj = f.unsqueeze(1).expand(B, K, K, d)
        rel = self.rel(torch.cat([fi, fj], -1))                    # (B,K,K,3)
        support = torch.sigmoid(rel[..., 0])
        conflict = torch.sigmoid(rel[..., 1])
        equiv = torch.sigmoid(rel[..., 2])
        if merge_mode == "off":
            equiv = torch.zeros_like(equiv)
        # merge: move each slot toward partners judged equivalent, learned mixing
        a = torch.sigmoid(self.alpha(torch.cat([fi, fj], -1)).squeeze(-1))
        if merge_mode == "off":
            a = torch.zeros_like(a)
        delta = (equiv * a).unsqueeze(-1) * (fj - fi)
        f = f + delta.sum(dim=2)
        # survival: learned cancellation using slot, token state, pooled relations
        pooled = torch.stack([support.mean(2), conflict.mean(2), equiv.mean(2)], -1)
        if survival_mode == "one":
            s = torch.ones(B, K, 1, device=f.device)
        else:
            s = torch.sigmoid(self.surv(torch.cat([f, xc, pooled], -1)))
        f = f * s
        # interference output injected into the token representation
        x = x + self.up(f.mean(dim=1))
        stats = {"support": support.mean().item(), "conflict": conflict.mean().item(),
                 "equiv": equiv.mean().item(), "survival_mean": s.mean().item()}
        return f, x, stats


class InterferenceLM(nn.Module):
    def __init__(self, vocab, d, layers, k=8, heads=4):
        super().__init__()
        self.d = d
        self.tok = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.k = k
        self.slot_init = nn.Parameter(torch.randn(k, d) * 0.02)
        self.unit = InterfereUnit(d, k)
        self.head = nn.Linear(2 * d, 2)
        self.state_head = nn.Linear(k * d, 9 * 10)   # remaining multiset, 3 numbers x digits
        self.pair_head = nn.Linear(2 * d, 2)          # equivalence judge, from frontier

    def forward(self, ids, valid=None, prop_mode="learned", merge_mode="learned",
                survival_mode="learned", pair=False):
        B, L = ids.shape[0], ids.shape[1]
        if valid is None:
            valid = torch.ones(B, L, dtype=torch.bool, device=ids.device)
        x = self.tok(ids)
        for b in self.blocks:
            x = b(x)                            # standard attention over tokens
        f = self.slot_init.unsqueeze(0).expand(B, self.k, self.d).contiguous()
        last = last_valid_index(valid)
        stats_last = None
        max_len = int(last.max().item()) + 1
        for t in range(max_len):
            step_valid = valid[:, t]
            f_new, xt, stats_last = self.unit(f, x[:, t], prop_mode, merge_mode, survival_mode)
            f = torch.where(step_valid.view(B, 1, 1), f_new, f)
        h = torch.cat([gather_rows(x, last), f.mean(dim=1)], -1)
        if pair:
            logits = self.pair_head(h)
            return logits, stats_last if stats_last else {}, None
        logits = self.head(h)
        state_logits = self.state_head(f.flatten(1)).view(B, 9, 10)
        return logits, stats_last if stats_last else {}, state_logits


def param_count(m):
    return sum(p.numel() for p in m.parameters())


def match_params(vanilla, target):
    """Add compensation MLPs to the vanilla trunk until it reaches target params."""
    while param_count(vanilla) < target:
        d = vanilla.tok.weight.size(1)
        gap = target - param_count(vanilla)
        units = max(1, gap // (2 * d * len(vanilla.blocks)))
        for b in vanilla.blocks:
            if b.comp is None:
                b.comp = nn.Sequential(nn.Linear(d, d + units), nn.GELU(), nn.Linear(d + units, d))
            else:
                old = b.comp
                b.comp = nn.Sequential(nn.Linear(d, d + units), nn.GELU(),
                                       nn.Linear(d + units, d + units), nn.GELU(),
                                       nn.Linear(d + units, d))
    return vanilla
