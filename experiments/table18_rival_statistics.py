"""Table 18. Rival pre-merge statistics on the 22 configurations of Table 4.

Table 18 printed its margins from the inversion P = I / (1 - CR), so this script checks them with it.
Signal power is S = P - I. CPU only.

    python experiments/table18_rival_statistics.py
"""

from dataclasses import replace

from common import breadth, fmt
from prism import rival_statistics, separation

NAMES = [
    ("interference", "c sqrt(I/K) (used)"),
    ("total_power", "c sqrt(P/K) (total power)"),
    ("signal_power", "c sqrt(S/K) (signal power)"),
    ("incoherence", "1 - CR (mean incoherence)"),
    ("rho", "rho (cancellation ratio)"),
]

# AUC and margin as printed in Table 18
PRINTED = {
    "interference": ("0.98", "0.83"),
    "total_power": ("1.00", "21.2"),
    "signal_power": ("1.00", "23.3"),
    "incoherence": ("0.08", "<0.01"),
    "rho": ("0.25", "0.02"),
}


def table18_statistics(a):
    stats = []
    for s in a.stats:
        P = s["I"] / (1.0 - s["cr"])
        stats.append({**s, "P": P, "S": P - s["I"]})
    return rival_statistics(replace(a, stats=stats))


def main():
    rows = breadth()
    stats = [table18_statistics(r["analysis"]) for r in rows]
    destructive = [r["outcome"] == "destructive" for r in rows]
    print(f"{len(rows)} configurations, {sum(destructive)} destructive")
    print(f"{'statistic':28s} {'AUC':>5s} {'margin':>8s}  result")
    differ = []
    for key, name in NAMES:
        auc, margin = separation([s[key] for s in stats], destructive)
        result = "separates" if margin > 1 else "anti-predictive" if auc < 0.5 else "predictive"
        # the precision of Table 18
        m = "<0.01" if margin < 0.01 else fmt(margin, 1 if margin >= 10 else 2)
        print(f"{name:28s} {fmt(auc, 2):>5s} {m + 'x':>8s}  {result}")
        if (fmt(auc, 2), m) != PRINTED[key]:
            differ.append(name)
    print("AUC treats larger values as more destructive. "
          "Margin is the smallest destructive value over the largest harmless one.")
    print("All five rows match Table 18." if not differ else "Differs from Table 18 in " + ", ".join(differ))


if __name__ == "__main__":
    main()
