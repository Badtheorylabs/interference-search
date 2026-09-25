"""Phase 1b analysis: rewind-based suppression, own-only vs coupled.

Tokens are *spent* tokens (discarded rewind tokens included), so arms are compared
at equal generated-token cost. Unfound problems count at the full group budget.
"""
import argparse
import json
from collections import defaultdict
from math import comb

ap = argparse.ArgumentParser()
ap.add_argument("run")
args = ap.parse_args()

ORDER = ["A", "Rself", "Rall"]
recs = defaultdict(dict)
for line in open(args.run):
    r = json.loads(line)
    recs[r["idx"]][r["arm"]] = r
arms = [a for a in ORDER if any(a in d for d in recs.values())]
complete = sorted(i for i, d in recs.items() if all(a in d for a in arms))
if not complete:
    raise SystemExit("no problem has all arms yet")
r0 = recs[complete[0]][arms[0]]
budget = r0["k"] * r0["budget"]
R = max(len(recs[i][a]["rounds"]) for i in complete for a in arms)


def sign_test(w, l):
    n = w + l
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(w, l) + 1)) / 2 ** n)


def spent_at_find(r):
    return r["rounds"][r["found_round"]]["spent"] if r["found"] else budget


print(f"problems with all arms: {len(complete)}   K={r0['k']}  per-stream budget={r0['budget']}  group budget={budget}")
print()
print("arm     found   mean spent tokens to find   rewinds/problem   discarded/problem")
for a in arms:
    rs = [recs[i][a] for i in complete]
    print(f" {a:6s} {sum(r['found'] for r in rs):2d}/{len(rs)}   {sum(map(spent_at_find, rs)) / len(rs):12.0f}"
          f"              {sum(r['rewinds'] for r in rs) / len(rs):8.1f}          {sum(r['discarded'] for r in rs) / len(rs):8.0f}")
print()
print("found by round")
print("round   " + " ".join(f"{x:>4d}" for x in range(R)))
for a in arms:
    print(f"  {a:6s}" + " ".join(f"{sum(1 for i in complete if recs[i][a]['found'] and recs[i][a]['found_round'] <= x):>4d}" for x in range(R)))
print()
for x, y in (("Rall", "Rself"), ("Rall", "A"), ("Rself", "A")):
    if x not in arms or y not in arms:
        continue
    w = sum(1 for i in complete if recs[i][x]["found"] and not recs[i][y]["found"])
    l = sum(1 for i in complete if recs[i][y]["found"] and not recs[i][x]["found"])
    cheaper = sum(1 for i in complete if spent_at_find(recs[i][x]) < spent_at_find(recs[i][y]))
    dearer = sum(1 for i in complete if spent_at_find(recs[i][x]) > spent_at_find(recs[i][y]))
    print(f"{x} vs {y}: found only by {x} {w}, only by {y} {l} (p={sign_test(w, l):.3f});  "
          f"{x} cheaper on {cheaper}, dearer on {dearer} (p={sign_test(cheaper, dearer):.3f})")
print()
print("idx  " + "  ".join(f"{a:>6s}" for a in arms))
for i in complete:
    print(f"{i:3d}  " + "  ".join(f"{(str(recs[i][a]['found_round']) if recs[i][a]['found'] else '-'):>6s}" for a in arms))
