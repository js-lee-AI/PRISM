"""Shared loaders for the experiment scripts."""

import csv
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from prism import load_stats

DATA = Path(__file__).resolve().parents[1] / "data"


def fmt(x, digits):
    # round half up, as the paper's tables do
    return str(Decimal(repr(x)).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP))


def score_str(x):
    # scores in units of 1e-3, as printed in Table 4
    x *= 1e3
    return fmt(x, 2) if x >= 1 else fmt(x, 3)


def breadth():
    """Table 4 rows with their layer statistics and outcome."""
    rows = []
    with open(DATA / "breadth.csv") as f:
        for r in csv.DictReader(f):
            a = load_stats(DATA / "layer_stats" / f"{r['id']}.json")
            base, ta = float(r["base"]), float(r["ta"])
            # destructive: mean more than 10 points below the base; harmless: within 1 point or better
            outcome = "destructive" if base - ta > 10 else "harmless" if ta >= base - 1 else "intermediate"
            rows.append({**r, "analysis": a, "base": base, "ta": ta,
                         "prism": float(r["prism"]) if r["prism"] else None,
                         "predicted": r["predicted"] == "1", "outcome": outcome})
    return rows


def titration():
    with open(DATA / "titration.csv") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["step"] = int(r["step"])
        r["tokens_b"] = float(r["tokens_b"])
        r["I"] = float(r["I"])
        r["rho"] = float(r["rho"])
        for k in ("base", "ta", "prism"):
            r[k] = float(r[k]) if r[k] else None
    return rows
