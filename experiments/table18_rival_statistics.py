"""Table 18. Rival pre-merge statistics on the 22 configurations of Table 4.

Total power is recovered from each layer's interference and coherence,
P = (I + eps CR) / (1 - CR), and signal power is S = P - I. CPU only.

    python experiments/table18_rival_statistics.py
"""

from common import breadth, fmt
from prism import rival_statistics, separation

NAMES = [
    ("interference", "c sqrt(I/K) (used)"),
    ("total_power", "c sqrt(P/K) (total power)"),
    ("signal_power", "c sqrt(S/K) (signal power)"),
    ("incoherence", "1 - CR (mean incoherence)"),
    ("rho", "rho (cancellation ratio)"),
]


def main():
    rows = breadth()
    stats = [rival_statistics(r["analysis"]) for r in rows]
    destructive = [r["outcome"] == "destructive" for r in rows]
    print(f"{len(rows)} configurations, {sum(destructive)} destructive")
    print(f"{'statistic':28s} {'AUC':>5s} {'margin':>8s}  result")
    for key, name in NAMES:
        auc, margin = separation([s[key] for s in stats], destructive)
        result = "separates" if margin > 1 else "anti-predictive" if auc < 0.5 else "predictive"
        m = "<0.01" if margin < 0.01 else fmt(margin, 2)
        print(f"{name:28s} {fmt(auc, 2):>5s} {m + 'x':>8s}  {result}")
    print("AUC treats larger values as more destructive. "
          "Margin is the smallest destructive value over the largest harmless one.")


if __name__ == "__main__":
    main()
