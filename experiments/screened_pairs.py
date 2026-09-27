"""Appendix J. The screen on 28 public pairs without a code specialist.

Scores are recomputed from data/layer_stats, with c = 1/2 and min-norm scaling.
CPU only.

    python experiments/screened_pairs.py
"""

import csv
from collections import Counter

from common import DATA, fmt, score_str
from prism import THRESHOLD, load_stats


def main():
    with open(DATA / "screened_pairs.csv") as f:
        pairs = list(csv.DictReader(f))
    rows = []
    for p in pairs:
        a = load_stats(DATA / "layer_stats" / f"{p['id']}.json")
        rows.append((a.score, a.rho, p))
    rows.sort(key=lambda r: -r[0])
    print(f"{'family':10s} {'specialists':76s} {'rho':>6s} {'score':>6s}")
    for score, rho, p in rows:
        pair = f"{p['specialist_a']} + {p['specialist_b']}"
        print(f"{p['family']:10s} {pair:76s} {fmt(rho, 3):>6s} {score_str(score):>6s}")
    print("score in units of 1e-3")
    fam = Counter(p["family"] for p in pairs)
    print(f"\n{len(pairs)} pairs, " + ", ".join(f"{n} on {k}" for k, n in fam.items()))
    lo, hi = rows[-1][0], rows[0][0]
    print(f"scores from {score_str(lo)}e-3 to {score_str(hi)}e-3, the largest for "
          f"{rows[0][2]['specialist_a']} + {rows[0][2]['specialist_b']}")
    print(f"all below half the threshold ({THRESHOLD / 2:.2e}): {all(s < THRESHOLD / 2 for s, _, _ in rows)}")


if __name__ == "__main__":
    main()
