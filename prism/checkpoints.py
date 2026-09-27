"""PRISM on safetensors checkpoints, with torch. Needs the `merge` extra.

Weights stream one matrix at a time, so a 7B merge runs on a CPU or one GPU.
This is the code that produced the paper's merges.
"""

from __future__ import annotations

import json
import math
import os

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from .core import THRESHOLD, Analysis, cancellation_ratio, interference_score, mlp_layers, thresholds


class Checkpoint:
    """Reads single tensors from a (possibly sharded) safetensors checkpoint."""

    def __init__(self, name_or_path):
        if os.path.isdir(name_or_path):
            path = name_or_path
        else:
            path = snapshot_download(name_or_path, allow_patterns=["*.json", "*.safetensors"])
        index = os.path.join(path, "model.safetensors.index.json")
        if os.path.exists(index):
            with open(index) as f:
                self.weight_map = json.load(f)["weight_map"]
        else:
            with safe_open(os.path.join(path, "model.safetensors"), "pt") as f:
                self.weight_map = {k: "model.safetensors" for k in f.keys()}
        self.name = name_or_path
        self.path = path
        self._files = {}

    def get(self, key):
        fn = self.weight_map[key]
        if fn not in self._files:
            self._files[fn] = safe_open(os.path.join(self.path, fn), "pt")
        return self._files[fn].get_tensor(key)


def _task_vectors(base, specialists, key, device):
    w0 = base.get(key).to(device, torch.float32)
    return w0, [s.get(key).to(device, torch.float32) - w0 for s in specialists]


def norm_scales(base, specialists, layers, device="cpu"):
    # s_k = min_j ||tau_j|| / ||tau_k||, norms over the merged (MLP) coordinates
    sq = [0.0] * len(specialists)
    for keys in layers:
        for key in keys:
            _, taus = _task_vectors(base, specialists, key, device)
            for k, t in enumerate(taus):
                sq[k] += t.pow(2).sum().item()
    norms = [math.sqrt(x) for x in sq]
    if max(norms) == 0:
        return [1.0] * len(norms)
    return [min(norms) / (n + 1e-10) for n in norms]


def layer_stats(base, specialists, layers, scales, device="cpu"):
    """Layer means of total power P, signal power S, interference I = P - S and cancellation C."""
    K = len(specialists)
    stats = []
    for keys in layers:
        P = S = C = 0.0
        n = 0
        for key in keys:
            _, taus = _task_vectors(base, specialists, key, device)
            d = torch.stack([s * t for s, t in zip(scales, taus)])
            del taus
            p = d.pow(2).mean(0)
            s = d.mean(0).pow(2)
            pos = d.clamp(min=0).sum(0)
            neg = (-d).clamp(min=0).sum(0)
            c = torch.minimum(2.0 / K * torch.minimum(pos, neg).pow(2), (p - s).clamp(min=0))
            P += p.sum().item()
            S += s.sum().item()
            C += c.sum().item()
            n += p.numel()
            del d, p, s, pos, neg, c
        P, S, C = P / n, S / n, C / n
        stats.append({"n": n, "P": P, "S": S, "I": max(P - S, 0.0), "C": C,
                      "cr": S / (P + 1e-10)})
    return stats


def soft_threshold(x, lam, min_keep=1e-3):
    y = x.sign() * (x.abs() - lam).clamp(min=0)
    if (y != 0).float().mean().item() < min_keep:
        # keep-rate floor: lower the threshold to the k-th largest magnitude
        k = max(int(min_keep * x.numel()), 1)
        q = x.abs().flatten().topk(k).values[-1]
        y = x.sign() * (x.abs() - q).clamp(min=0)
    return y


def analyze(base, specialists, coef=None, norm_scaling=True, device="cpu"):
    K = len(specialists)
    coef = 1.0 / K if coef is None else coef
    layers = mlp_layers(base.weight_map)
    scales = norm_scales(base, specialists, layers, device) if norm_scaling else [1.0] * K
    stats = layer_stats(base, specialists, layers, scales, device)
    return {
        "K": K,
        "coef": coef,
        "scales": scales,
        "layers": layers,
        "stats": stats,
        "rho": cancellation_ratio(stats),
        "score": interference_score(stats, K, coef),
    }


def merge(base, specialists, coef=None, kappa=1.0, gate=0.1, min_keep=1e-3,
          norm_scaling=True, denoise=True, device="cpu"):
    """Returns the merged MLP weights (bf16, cpu) and a summary dict.

    denoise=False gives the plain average with the same scaling and coefficient.
    """
    info = analyze(base, specialists, coef, norm_scaling, device)
    K, coef, scales = info["K"], info["coef"], info["scales"]
    gated = info["rho"] < gate
    lams = thresholds(info["stats"], kappa)
    if gated:
        lams = [0.0] * len(lams)

    merged, keep = {}, []
    for keys, lam in zip(info.pop("layers"), lams):
        for key in keys:
            w0, taus = _task_vectors(base, specialists, key, device)
            delta = sum(s * t for s, t in zip(scales, taus)) / K
            del taus
            if denoise:
                delta = soft_threshold(delta, lam, min_keep)
            keep.append((delta != 0).float().mean().item())
            w = w0 + coef * delta
            w = torch.where(torch.isfinite(w), w, w0)
            merged[key] = w.to(torch.bfloat16).cpu()
            del w0, delta, w
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    info.update(gated=gated, denoise=denoise, kappa=kappa, gate=gate,
                thresholds=lams, keep_rates=keep, mean_keep_rate=sum(keep) / len(keep))
    return merged, info


def _open(m):
    return m if isinstance(m, Checkpoint) else Checkpoint(m)


def screen_checkpoints(base, specialists, coef=None, norm_scaling=True, threshold=THRESHOLD,
                       device="cpu"):
    """Interference score and rho of a candidate merge. Nothing is merged."""
    base, specialists = _open(base), [_open(s) for s in specialists]
    info = analyze(base, specialists, coef, norm_scaling, device)
    meta = {"base": base.name, "specialists": [s.name for s in specialists]}
    return Analysis(K=info["K"], coef=info["coef"], scales=info["scales"], stats=info["stats"],
                    rho=info["rho"], score=info["score"], threshold=threshold, meta=meta)


def merge_checkpoints(base, specialists, out=None, device="cpu", **kwargs):
    """PRISM merge of Hub ids or local folders. With out, also saves a full
    Hugging Face checkpoint there, together with merge_info.json."""
    base, specialists = _open(base), [_open(s) for s in specialists]
    merged, info = merge(base, specialists, device=device, **kwargs)
    if out is not None:
        save_merged(merged, base.name, out, dict(info, base=base.name,
                                                 specialists=[s.name for s in specialists]))
    return merged, info


def save_merged(merged, base, out, info=None):
    """Load the base model, replace the merged tensors and save it."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
    _, unexpected = model.load_state_dict(merged, strict=False)
    assert not unexpected, unexpected
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out)
    AutoTokenizer.from_pretrained(base).save_pretrained(out)
    if info is not None:
        with open(os.path.join(out, "merge_info.json"), "w") as f:
            json.dump(info, f, indent=1)
