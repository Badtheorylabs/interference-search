"""Phase 1 analysis: does a shared refutation ledger help, beyond a private one?

Per arm: group find rate within budget, anytime curve (found by round r),
generated tokens to first find (unfound problems counted at full budget), and
the share of failed attempts that repeat a refutation already known elsewhere.
Paired over problems: C vs P and C vs A, exact two-sided sign test on discordant pairs.
"""
import argparse
import json
from collections import defaultdict
from math import comb

from transformers import AutoTokenizer

from interference_search.countdown import extract_attempts

tok = AutoTokenizer.from_pretrained("mlx-community/Qwen3-1.7B-4bit")

ap = argparse.ArgumentParser()
ap.add_argument("run")
args = ap.parse_args()

recs = defaultdict(dict)
for line in open(args.run):
    r = json.loads(line)
    recs[r["idx"]][r["arm"]] = r
arms = sorted({a for d in recs.values() for a in d}, key="APC".index)
complete = [i for i, d in recs.items() if all(a in d for a in arms)]
any_rec = next(iter(recs[complete[0]].values())) if complete else None


def sign_test(wins, losses):
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def redundancy(r):
    """Fraction of failed attempts whose canonical form a sibling had refuted at an earlier token."""
    import bisect
    atts = []
    first = defaultdict(dict)
    for k, text in enumerate(r["texts"]):
        offs = [a for a, _ in tok(text, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]]
        al = extract_attempts(text, r["numbers"], r["target"])
        for a in al:
            a["t"] = bisect.bisect_right(offs, a["offset"])  # token index = lockstep time
            if not a["correct"]:
                first[a["canon"]].setdefault(k, a["t"])
        atts.append(al)
    fail = sib = selfrep = 0
    for k, al in enumerate(atts):
        seen = set()
        for a in al:
            if a["correct"]:
                break
            fail += 1
            if a["canon"] in seen:
                selfrep += 1
            elif any(j != k and t < a["t"] for j, t in first[a["canon"]].items()):
                sib += 1
            seen.add(a["canon"])
    return fail, sib, selfrep


if not complete:
    raise SystemExit("no problem has all arms yet")
R = any_rec["rounds_max"]
budget = any_rec["k"] * any_rec["chunk"] * R
print(f"problems with all arms: {len(complete)}   K={any_rec['k']}  chunk={any_rec['chunk']}  rounds={R}  budget={budget} tokens/group")
print()
print("arm  found   mean_tokens_to_find   note_tokens/problem   fail  sib-redundant  self-repeat")
for a in arms:
    rs = [recs[i][a] for i in complete]
    found = sum(r["found"] for r in rs)
    ttf = [r["rounds"][r["found_round"]]["gen_tokens"] if r["found"] else budget for r in rs]
    notes = sum(r["note_tokens"] for r in rs) / len(rs)
    F = S = SR = 0
    for r in rs:
        f, s, sr = redundancy(r)
        F, S, SR = F + f, S + s, SR + sr
    print(f" {a}   {found:2d}/{len(rs)}   {sum(ttf) / len(ttf):10.0f}            {notes:8.0f}          "
          f"{F:5d}  {S / max(F, 1):8.1%}     {SR / max(F, 1):8.1%}")
print()
print("found by round (anytime curve)")
print("round " + " ".join(f"{r:>4d}" for r in range(R)))
for a in arms:
    row = []
    for rnd in range(R):
        row.append(sum(1 for i in complete if recs[i][a]["found"] and recs[i][a]["found_round"] <= rnd))
    print(f"  {a}   " + " ".join(f"{x:>4d}" for x in row))
print()
for x, y in (("C", "P"), ("C", "A"), ("P", "A")):
    if x not in arms or y not in arms:
        continue
    w = sum(1 for i in complete if recs[i][x]["found"] and not recs[i][y]["found"])
    l = sum(1 for i in complete if recs[i][y]["found"] and not recs[i][x]["found"])
    # among problems both found, who found first
    faster = sum(1 for i in complete if recs[i][x]["found"] and recs[i][y]["found"]
                 and recs[i][x]["found_round"] < recs[i][y]["found_round"])
    slower = sum(1 for i in complete if recs[i][x]["found"] and recs[i][y]["found"]
                 and recs[i][x]["found_round"] > recs[i][y]["found_round"])
    print(f"{x} vs {y}: {x} only {w}, {y} only {l}, sign-test p={sign_test(w, l):.3f};  "
          f"both found: {x} earlier {faster}, later {slower}")
print()
print("idx  " + "  ".join(f"{a}:round" for a in arms))
for i in sorted(complete):
    print(f"{i:3d}  " + "  ".join(f"{str(recs[i][a]['found_round']) if recs[i][a]['found'] else '-':>7}" for a in arms))
