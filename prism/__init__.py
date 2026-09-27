"""PRISM: predict and repair merge collapse in large language models.

    import prism
    report = prism.analyze(base, specialists)          # score and rho, before merging
    merged, info = prism.merge(base, specialists)      # average, then soft-threshold

base and each specialist map parameter names to arrays. The core needs only
numpy. The checkpoint functions import torch the first time they are used.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .core import (GATE, THRESHOLD, Analysis, analyze, cancellation_ratio, interference_score,
                   merge, mlp_layers, soft_threshold, thresholds)
from .stats import load_stats, rival_statistics, save_stats, separation

# Heavy names, imported on first attribute access (PEP 562).
_LAZY = {
    "Checkpoint": "checkpoints",
    "screen_checkpoints": "checkpoints",
    "merge_checkpoints": "checkpoints",
    "save_merged": "checkpoints",
}

__all__ = [
    "__version__", "analyze", "merge", "Analysis", "THRESHOLD", "GATE", "interference_score",
    "cancellation_ratio", "thresholds", "soft_threshold", "mlp_layers", "load_stats", "save_stats",
    "rival_statistics", "separation", *_LAZY,
]


def __getattr__(name):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module
    return getattr(import_module(f".{module}", __name__), name)


def __dir__():
    return sorted(__all__)
