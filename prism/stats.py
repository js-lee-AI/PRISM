"""Layer-statistic files and the rival pre-merge statistics of the paper.

A layer-statistic file holds, for one candidate merge, the parameter count n,
mean interference I, mean cancellation C and coherence CR of every merged
layer. Scores, rho and the rival statistics are functions of these alone, so
they run on a CPU without the checkpoints. `prism-merge screen --out` writes
the same format.
"""

from __future__ import annotations

import json
import math

from .core import THRESHOLD, Analysis, cancellation_ratio, interference_score


def power(I, cr):
    # invert CR = S / (P + eps) with S = P - I
    return (I + 1e-10 * cr) / (1.0 - cr)


def load_stats(path, coef=None, threshold=THRESHOLD):
    """Read a layer-statistic file into an Analysis (c defaults to 1/K)."""
    with open(path) as f:
        d = json.load(f)
    stats = []
    for l in d["layers"]:
        P = l["P"] if "P" in l else power(l["I"], l["cr"])
        stats.append({"n": l["n"], "P": P, "S": P - l["I"], "I": l["I"], "C": l["C"], "cr": l["cr"]})
    K = d["K"]
    coef = 1.0 / K if coef is None else coef
    meta = {k: v for k, v in d.items() if k not in ("layers", "K", "scales")}
    return Analysis(K=K, coef=coef, scales=d.get("scales") or [1.0] * K, stats=stats,
                    rho=cancellation_ratio(stats), score=interference_score(stats, K, coef),
                    threshold=threshold, meta=meta)


def save_stats(analysis, path, **meta):
    """Write an Analysis as a layer-statistic file, one layer per line."""
    head = {**analysis.meta, **meta, "K": analysis.K, "norm_scaled": any(s != 1.0 for s in analysis.scales),
            "scales": analysis.scales}
    lines = ["{"] + [f"  {json.dumps(k)}: {json.dumps(v)}," for k, v in head.items()]
    rows = [json.dumps({"n": s["n"], "I": s["I"], "C": s["C"], "cr": s["cr"]}) for s in analysis.stats]
    lines += ['  "layers": [', ",\n".join("    " + r for r in rows), "  ]", "}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def rival_statistics(a):
    """The five pre-merge statistics compared in the paper's Table 18."""
    n = sum(s["n"] for s in a.stats)
    mean = {x: sum(s["n"] * s[x] for s in a.stats) / n for x in ("I", "P", "S", "cr")}
    c, K = a.coef, a.K
    return {
        "interference": c * math.sqrt(mean["I"] / K),
        "total_power": c * math.sqrt(mean["P"] / K),
        "signal_power": c * math.sqrt(max(mean["S"], 0.0) / K),
        "incoherence": 1.0 - mean["cr"],
        "rho": a.rho,
    }


def separation(scores, destructive):
    """AUC of larger-is-more-destructive (ties count half) and the margin
    min(destructive) / max(harmless)."""
    pos = [s for s, y in zip(scores, destructive) if y]
    neg = [s for s, y in zip(scores, destructive) if not y]
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return wins / (len(pos) * len(neg)), min(pos) / max(neg)
