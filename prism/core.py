"""PRISM on in-memory arrays, with numpy only.

prism.checkpoints runs the same steps on safetensors checkpoints with torch.
The statistics and the threshold rule below are shared by both.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

THRESHOLD = 1.9206e-3   # frozen screening threshold on c*sqrt(I/K), printed as 1.9e-3
GATE = 0.1              # rho-gate tau_g
PROJS = ("gate_proj", "up_proj", "down_proj")


def mlp_layers(keys):
    """Group Hugging Face MLP projection keys by decoder layer."""
    ids = sorted({int(k.split(".")[2]) for k in keys
                  if k.startswith("model.layers.") and ".mlp.gate_proj." in k})
    return [[f"model.layers.{i}.mlp.{p}.weight" for p in PROJS] for i in ids]


def cancellation_ratio(stats):
    num = sum(s["n"] * s["C"] for s in stats)
    den = sum(s["n"] * s["I"] for s in stats)
    return num / (den + 1e-12)


def interference_score(stats, K, coef):
    # screening score c * sqrt(I / K), I the parameter-weighted mean over merged layers
    I = sum(s["n"] * s["I"] for s in stats) / sum(s["n"] for s in stats)
    return coef * math.sqrt(I / K)


def thresholds(stats, kappa=1.0):
    return [kappa * math.sqrt(max(s["I"], 1e-12)) * math.sqrt(2 * math.log(max(s["n"], 2)))
            for s in stats]


def soft_threshold(x, lam, min_keep=1e-3):
    x = np.asarray(x)
    y = np.sign(x) * np.clip(np.abs(x) - lam, 0, None)
    if np.count_nonzero(y) / y.size < min_keep:
        # keep-rate floor: lower the threshold to the k-th largest magnitude
        k = max(int(min_keep * x.size), 1)
        a = np.abs(x).ravel()
        q = np.partition(a, a.size - k)[a.size - k]
        y = np.sign(x) * np.clip(np.abs(x) - q, 0, None)
    return y


def layer_stats(d):
    """Mean P, S, I = P - S and C over the matrices of one layer.

    d is a list with one array per matrix, each of shape (K, ...) and holding
    the scaled task vectors s_k * tau_k.
    """
    K = d[0].shape[0]
    P = S = C = 0.0
    n = 0
    for x in d:
        p = np.mean(x * x, axis=0)
        s = np.mean(x, axis=0) ** 2
        pos = np.clip(x, 0, None).sum(0)
        neg = np.clip(-x, 0, None).sum(0)
        c = np.minimum(2.0 / K * np.minimum(pos, neg) ** 2, np.clip(p - s, 0, None))
        P += float(p.sum())
        S += float(s.sum())
        C += float(c.sum())
        n += p.size
    P, S, C = P / n, S / n, C / n
    return {"n": n, "P": P, "S": S, "I": max(P - S, 0.0), "C": C, "cr": S / (P + 1e-10)}


@dataclass
class Analysis:
    """Pre-merge statistics of one candidate merge."""

    K: int
    coef: float
    scales: list
    stats: list
    rho: float
    score: float
    threshold: float = THRESHOLD
    gate: float = GATE
    meta: dict = field(default_factory=dict)

    @property
    def flagged(self):
        return self.score > self.threshold

    @property
    def gated(self):
        return self.rho < self.gate

    def __str__(self):
        scales = ", ".join(f"{s:.3f}" for s in self.scales)
        verdict = ("above the threshold, merge with PRISM" if self.flagged
                   else "below the threshold, keep the plain average")
        gate = "gate returns the plain average" if self.gated else "thresholding on"
        return (f"K={self.K}  c={self.coef:.3f}  scales=[{scales}]\n"
                f"score={self.score:.3e}  threshold={self.threshold:.1e}  "
                f"rho={self.rho:.3f} ({gate})\n"
                f"-> {verdict}")


def _groups(base, layers):
    if layers is None:
        layers = mlp_layers(base) or [[k] for k in base]
    return [list(g) for g in layers]


def _task_vectors(base, specialists, key):
    w0 = np.asarray(base[key], dtype=np.float32)
    return w0, [np.asarray(s[key], dtype=np.float32) - w0 for s in specialists]


def _norm_scales(base, specialists, layers):
    # s_k = min_j ||tau_j|| / ||tau_k||, norms over the merged coordinates
    sq = [0.0] * len(specialists)
    for keys in layers:
        for key in keys:
            _, taus = _task_vectors(base, specialists, key)
            for k, t in enumerate(taus):
                sq[k] += float(np.square(t).sum())
    norms = [math.sqrt(x) for x in sq]
    if max(norms) == 0:
        return [1.0] * len(norms)
    return [min(norms) / (n + 1e-10) for n in norms]


def _analyze(base, specialists, layers, coef, norm_scaling):
    K = len(specialists)
    coef = 1.0 / K if coef is None else coef
    scales = _norm_scales(base, specialists, layers) if norm_scaling else [1.0] * K
    stats = []
    for keys in layers:
        d = []
        for key in keys:
            _, taus = _task_vectors(base, specialists, key)
            d.append(np.stack([np.float32(s) * t for s, t in zip(scales, taus)]))
        stats.append(layer_stats(d))
    return Analysis(K=K, coef=coef, scales=scales, stats=stats,
                    rho=cancellation_ratio(stats), score=interference_score(stats, K, coef))


def analyze(base, specialists, layers=None, coef=None, norm_scaling=True):
    """Score a candidate merge before building it.

    base and each specialist map parameter names to arrays of equal shapes.
    layers groups the keys to merge; by default the MLP projections of each
    decoder layer, or every key on its own when the names are not HF names.
    """
    return _analyze(base, specialists, _groups(base, layers), coef, norm_scaling)


def merge(base, specialists, layers=None, coef=None, kappa=1.0, gate=GATE, min_keep=1e-3,
          norm_scaling=True, denoise=True):
    """PRISM merge. Returns the merged weights (float32) and a summary dict.

    Keys outside the merged layers keep their base values. denoise=False gives
    the plain average with the same scaling and coefficient.
    """
    layers = _groups(base, layers)
    a = _analyze(base, specialists, layers, coef, norm_scaling)
    K, coef, scales = a.K, a.coef, a.scales
    gated = a.rho < gate
    lams = [0.0] * len(layers) if gated else thresholds(a.stats, kappa)

    merged = {k: np.array(v, dtype=np.float32) for k, v in base.items()}
    keep = []
    for keys, lam in zip(layers, lams):
        for key in keys:
            w0, taus = _task_vectors(base, specialists, key)
            delta = sum(np.float32(s) * t for s, t in zip(scales, taus)) / np.float32(K)
            if denoise:
                delta = soft_threshold(delta, lam, min_keep)
            keep.append(np.count_nonzero(delta) / delta.size)
            w = w0 + np.float32(coef) * delta
            merged[key] = np.where(np.isfinite(w), w, w0).astype(np.float32)

    info = {"K": K, "coef": coef, "scales": scales, "stats": a.stats, "rho": a.rho,
            "score": a.score, "gated": gated, "denoise": denoise, "kappa": kappa,
            "gate": gate, "thresholds": lams, "keep_rates": keep,
            "mean_keep_rate": sum(keep) / len(keep)}
    return merged, info
