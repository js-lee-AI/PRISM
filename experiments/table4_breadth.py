"""Table 4. The interference score of 22 evaluated merges against their outcome.

rho and the score are recomputed from the layer statistics in data/layer_stats.
The six-task means come from the GPU evaluations (experiments/evaluate.py) and
decide each outcome. CPU only.

    python experiments/table4_breadth.py [--json results/table4.json]
"""

import argparse
import json
import os

from common import breadth, fmt, score_str, titration
from prism import THRESHOLD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None, help="also write the rows to this file")
    args = ap.parse_args()

    rows = breadth()
    print(f"{'setting':46s} {'K':>2s} {'c':>5s} {'rho':>7s} {'score':>7s} {'base':>6s} {'TA':>5s} {'PRISM':>6s}")
    for group, title in (("destructive", "naive averaging destroys the model"),
                         ("harmless", "merging is harmless")):
        print(title)
        for r in rows:
            if r["outcome"] != group:
                continue
            a = r["analysis"]
            g = "g" if a.gated else " "
            prism = f"{r['ta']:.1f}g" if a.gated else f"{r['prism']:.1f} "
            name = r["setting"] + (" *" if r["predicted"] else "")
            print(f"{name:46s} {a.K:2d} {fmt(a.coef, 2):>5s} {fmt(a.rho, 3):>6s}{g} {score_str(a.score):>7s} "
                  f"{r['base']:6.1f} {r['ta']:5.1f} {prism:>6s}")
    print("score in units of 1e-3, * outcome predicted before evaluation, "
          "g gated, so PRISM equals Task Arithmetic")

    dest = [r for r in rows if r["outcome"] == "destructive"]
    harm = [r for r in rows if r["outcome"] == "harmless"]
    above = [r for r in dest if r["analysis"].flagged]
    print(f"\n{len(rows)} configurations, {len(dest)} destructive and {len(harm)} harmless")
    print(f"score above the threshold of {THRESHOLD:.1e}: {len(above)} of {len(dest)} destructive, "
          f"{sum(r['analysis'].flagged for r in harm)} of {len(harm)} harmless")
    drop = [r["base"] - r["ta"] for r in above]
    gap = [abs(r["base"] - r["prism"]) for r in above]
    print(f"on those {len(above)}, plain averaging falls {min(drop):.1f} to {max(drop):.1f} points below "
          f"the base and PRISM stays within {max(gap):.1f} of it")

    # the prospective test counts the starred rows and the harmless end of the titration
    pred = [(r["analysis"].flagged, r["outcome"] == "destructive") for r in rows if r["predicted"]]
    end = next(t for t in titration() if t["lr"] == "1e-4" and t["base"] is not None)
    end_score = 0.5 * (end["I"] / 2) ** 0.5
    pred.append((end_score > THRESHOLD, end["base"] - end["ta"] > 10))
    print(f"predictions made before evaluation: {sum(p == o for p, o in pred)} of {len(pred)} correct")

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump([{"id": r["id"], "setting": r["setting"], "K": r["analysis"].K,
                        "c": r["analysis"].coef, "rho": r["analysis"].rho, "score": r["analysis"].score,
                        "gated": r["analysis"].gated, "base": r["base"], "ta": r["ta"],
                        "prism": r["ta"] if r["analysis"].gated else r["prism"],
                        "outcome": r["outcome"], "predicted": r["predicted"]} for r in rows], f, indent=1)
        print("wrote", args.json)


if __name__ == "__main__":
    main()
