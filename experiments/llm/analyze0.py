"""Phase 0 analysis: shared-failure rate among independent samples.

Streams are treated as running in lockstep, one token per step. A failed attempt
in stream k at token t is *sibling-redundant* if another stream produced the same
canonical expression at a token < t, meaning a shared refutation ledger would
already have contained it.

Attempt cost = tokens since the previous attempt in the same stream (the first
attempt's cost runs from the start of generation; the "strict" variant sets it to 0).
"""
import argparse
import json
from collections import defaultdict

from transformers import AutoTokenizer

from interference_search.countdown import extract_attempts

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--tokenizer", default="mlx-community/Qwen3-1.7B-4bit")
ap.add_argument("--show", type=int, default=0, help="print N example redundant attempts")
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.tokenizer)


def char_to_tok(text):
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    offs = [s for s, _ in enc["offset_mapping"]]
    import bisect
    return lambda c: bisect.bisect_right(offs, c), len(offs)


T = defaultdict(float)
examples = []
per_problem = []
for path in args.runs:
    for line in open(path):
        rec = json.loads(line)
        nums, target = rec["numbers"], rec["target"]
        streams = []
        for s in rec["samples"]:
            c2t, n = char_to_tok(s["text"])
            atts = extract_attempts(s["text"], nums, target)
            # drop everything after the stream's own first correct attempt:
            # later "attempts" are verification or overthinking, not search
            cut = next((i for i, a in enumerate(atts) if a["correct"]), None)
            if cut is not None:
                atts = atts[:cut + 1]
            prev = 0
            for a in atts:
                a["tok"] = c2t(a["offset"])
                a["cost"] = a["tok"] - prev
                a["first"] = prev == 0
                prev = a["tok"]
            # primary cost: first attempt priced at the stream's median later-attempt cost
            later = sorted(a["cost"] for a in atts[1:])
            med = later[len(later) // 2] if later else 0
            for a in atts:
                a["cost_mid"] = med if a["first"] else a["cost"]
            streams.append((atts, n, s["solved"]))
        # earliest token each canonical failure appears, per stream
        first_seen = defaultdict(dict)
        for k, (atts, _, _) in enumerate(streams):
            for a in atts:
                if not a["correct"]:
                    first_seen[a["canon"]].setdefault(k, a["tok"])
        solve_tok = min((a["tok"] for atts, _, _ in streams for a in atts if a["correct"]),
                        default=None)
        P = defaultdict(float)
        for k, (atts, n, solved) in enumerate(streams):
            P["tokens"] += n
            P["solved_streams"] += solved
            if solve_tok is not None:
                P["post_solve_tokens"] += max(0, n - solve_tok)
            seen_self = set()
            for a in atts:
                if a["correct"]:
                    continue
                P["fail"] += 1
                P["fail_tok"] += a["cost"]
                sib = any(t < a["tok"] for j, t in first_seen[a["canon"]].items() if j != k)
                if a["canon"] in seen_self:
                    P["self_repeat"] += 1
                    P["self_repeat_tok"] += a["cost"]
                elif sib:
                    P["sib"] += 1
                    P["sib_tok"] += a["cost"]
                    P["sib_tok_strict"] += 0 if a["first"] else a["cost"]
                    P["sib_tok_mid"] += a["cost_mid"]
                    if len(examples) < args.show:
                        examples.append((rec["idx"], k, a["tok"], a["canon"], a["value"]))
                seen_self.add(a["canon"])
        P["distinct_fail"] = len(first_seen)
        P["any_solved"] = float(P["solved_streams"] > 0)
        per_problem.append((rec["idx"], dict(P)))
        for key, v in P.items():
            T[key] += v

n = len(per_problem)
print(f"problems {n}   streams/problem {len(rec['samples'])}   model {rec['model']}   budget {rec['max_tokens']}")
print(f"stream solve rate          {T['solved_streams'] / (n * len(rec['samples'])):.1%}")
print(f"problem solved by any      {T['any_solved'] / n:.1%}")
print(f"failed attempts            {int(T['fail'])}  ({T['fail'] / n:.1f}/problem, {int(T['distinct_fail'])} distinct)")
print(f"  sibling-redundant        {int(T['sib'])}  = {T['sib'] / max(T['fail'], 1):.1%} of failures")
print(f"  self-repeats             {int(T['self_repeat'])}  = {T['self_repeat'] / max(T['fail'], 1):.1%} of failures")
print(f"tokens total               {int(T['tokens'])}")
print(f"  in sibling-redundant     PRIMARY {T['sib_tok_mid'] / T['tokens']:.1%}   (strict {T['sib_tok_strict'] / T['tokens']:.1%}, loose {T['sib_tok'] / T['tokens']:.1%})")
print(f"  in self-repeats          {T['self_repeat_tok'] / T['tokens']:.1%}")
print(f"  after first group solve  {T['post_solve_tokens'] / T['tokens']:.1%}")
print()
print("idx  solved  fails  sib  self  sib_tok%")
for idx, P in per_problem:
    print(f"{idx:3d}  {int(P['solved_streams']):3d}    {int(P.get('fail', 0)):4d}  {int(P.get('sib', 0)):4d} {int(P.get('self_repeat', 0)):4d}   {P.get('sib_tok', 0) / P['tokens']:.1%}")
for e in examples:
    print("example", e)
