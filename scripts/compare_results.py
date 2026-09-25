"""Compare a fresh Countdown run against the results saved in results/countdown.

    python scripts/compare_results.py <fresh dir>

Solve rates and rounds are deterministic, so they should match exactly. Timings are machine-dependent and
are not compared. Exits non-zero if anything differs.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAVED = ROOT / "results" / "countdown"
fresh = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "repro"
bad = 0


def check(label, old, new):
    global bad
    if old != new:
        bad += 1
        print(f"  DIFF  {label}: saved {old}, fresh {new}")


for name in ("countdown_benchmark.json", "frontier_vs_linear.json"):
    old, new = json.load(open(SAVED / name)), json.load(open(fresh / name))
    for group, arms in old.items():
        for arm, r in arms.items():
            got = new.get(group, {}).get(arm)
            if got is None:
                check(f"{name} / {group} / {arm}", "present", "missing")
                continue
            for b, v in r["solve"].items():
                check(f"{name} / {group} / {arm} / budget {b}", round(v, 4), round(got["solve"].get(b, -1), 4))
            check(f"{name} / {group} / {arm} / rounds", round(r["rounds_at_max_budget"], 2),
                  round(got["rounds_at_max_budget"], 2))
    print(f"{name}: checked")

old_c = (SAVED / "compression.txt").read_text().split()
new_c = (fresh / "compression.txt").read_text().split()
check("compression.txt", "identical", "identical" if old_c == new_c else "different")
print("compression.txt: checked")

print("all saved Countdown results reproduce exactly" if not bad else f"{bad} differences")
sys.exit(1 if bad else 0)
