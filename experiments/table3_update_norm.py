"""Table 3, Norm and Coords columns. The size of the update each operator writes.

Norm is ||delta|| / ||delta_bar|| over the merged MLP coordinates, where delta_bar is
the norm-scaled average, and Coords is the share of merged weights that differ from
the base once the merge is stored in bfloat16.
Streams the three Qwen2.5-7B checkpoints one matrix at a time on a CPU, in about
eight minutes once they are downloaded. Needs the merge extra.

    python experiments/table3_update_norm.py
"""

import argparse
import math

import torch

from prism import checkpoints as ck


def hard_threshold(x, lam, min_keep=1e-3):
    # same threshold and floor as PRISM, but kept coordinates are not shrunk
    y = x * (x.abs() > lam)
    if (y != 0).float().mean().item() < min_keep:
        k = max(int(min_keep * x.numel()), 1)
        q = x.abs().flatten().topk(k).values[-1]
        y = x * (x.abs() > q)
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--specialists", nargs="+", default=["Qwen/Qwen2.5-Math-7B", "Qwen/Qwen2.5-Coder-7B"])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    base = ck.Checkpoint(args.base)
    specialists = [ck.Checkpoint(s) for s in args.specialists]
    info = ck.analyze(base, specialists, device=args.device)
    lams = ck.thresholds(info["stats"], 1.0)
    ops = {"Task Arithmetic": lambda x, lam: x,
           "Hard threshold, same lambda": hard_threshold,
           "PRISM": lambda x, lam: ck.soft_threshold(x, lam, 1e-3)}
    sq = {k: 0.0 for k in ops}
    nnz = {k: 0 for k in ops}
    total = 0
    for keys, lam in zip(info["layers"], lams):
        for key in keys:
            w0, taus = ck._task_vectors(base, specialists, key, args.device)
            avg = sum(s * t for s, t in zip(info["scales"], taus)) / len(taus)
            for name, op in ops.items():
                d = op(avg, lam)
                sq[name] += d.double().pow(2).sum().item()
                w = (w0 + info["coef"] * d).to(torch.bfloat16).float()
                nnz[name] += int((w != w0).sum().item())
            total += avg.numel()
    print(f"rho={info['rho']:.3f}  score={info['score']:.3e}")
    print(f"{'method':30s} {'norm':>6s} {'coords %':>9s}")
    for name in ops:
        print(f"{name:30s} {math.sqrt(sq[name] / sq['Task Arithmetic']):6.3f} {100 * nnz[name] / total:9.1f}")


if __name__ == "__main__":
    torch.set_num_threads(min(16, torch.get_num_threads()))
    main()
