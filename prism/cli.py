"""The `prism-merge` command. `python -m prism` runs the same thing."""

from __future__ import annotations

import argparse

from . import __version__
from .core import GATE, THRESHOLD


def _demo(args):
    import numpy as np

    from . import analyze, merge

    # the toy merge of examples/quickstart.py
    rng = np.random.default_rng(args.seed)
    base = {"w": rng.normal(0, 0.02, (512, 1024))}
    shared = (rng.random((512, 1024)) < 0.005) * 0.2
    specialists = [{"w": base["w"] + shared + rng.normal(0, 0.02, (512, 1024))} for _ in range(2)]
    print(analyze(base, specialists))
    target = 0.5 * shared
    for name, denoise in [("plain average", False), ("PRISM", True)]:
        merged, info = merge(base, specialists, denoise=denoise)
        err = np.linalg.norm(merged["w"] - base["w"] - target) / np.linalg.norm(target)
        print(f"{name:14s} error {err:.2f}   weights changed {100 * info['mean_keep_rate']:.1f}%")
    return 0


def _score(args):
    from .stats import load_stats

    for path in args.files:
        a = load_stats(path, coef=args.coef, threshold=args.threshold)
        if len(args.files) == 1:
            print(a)
        else:
            flag = "PRISM" if a.flagged else "average"
            print(f"{a.meta.get('id', path):32s} K={a.K}  c={a.coef:.3f}  rho={a.rho:.3f}  "
                  f"score={a.score:.3e}  -> {flag}")
    return 0


def _screen(args):
    from .checkpoints import screen_checkpoints
    from .stats import save_stats

    a = screen_checkpoints(args.base, args.specialists, coef=args.coef,
                           norm_scaling=not args.no_norm_scaling, threshold=args.threshold,
                           device=args.device)
    if args.per_layer:
        print("layer        P            I            C         CR")
        for i, s in enumerate(a.stats):
            print(f"{i:5d}  {s['P']:.4e}  {s['I']:.4e}  {s['C']:.4e}  {s['cr']:.3f}")
    print(a)
    if args.out:
        save_stats(a, args.out)
        print("wrote", args.out)
    return 0


def _merge(args):
    from .checkpoints import merge_checkpoints

    _, info = merge_checkpoints(
        args.base, args.specialists, out=args.out, coef=args.coef, kappa=args.kappa,
        gate=args.gate, min_keep=args.min_keep, norm_scaling=not args.no_norm_scaling,
        denoise=not args.average, device=args.device)
    print(f"scales={['%.3f' % s for s in info['scales']]} rho={info['rho']:.3f} "
          f"gated={info['gated']} score={info['score']:.3e} "
          f"mean keep rate={info['mean_keep_rate']:.4f}")
    print("wrote", args.out)
    return 0


def _checkpoint_args(p):
    p.add_argument("--base", required=True, help="Hub id or local folder of the shared base")
    p.add_argument("--specialists", nargs="+", required=True, help="the K fine-tuned models")
    p.add_argument("--coef", type=float, default=None, help="global coefficient c (default 1/K)")
    p.add_argument("--no-norm-scaling", action="store_true", help="raw merge, s_k = 1")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="prism-merge", description="Predict and repair merge collapse in large language models")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run the CPU quickstart on a toy merge")
    demo.add_argument("--seed", type=int, default=0)
    demo.set_defaults(func=_demo)

    score = sub.add_parser("score", help="score and rho from layer-statistic files, on CPU")
    score.add_argument("files", nargs="+", help="json files as written by `screen --out`")
    score.add_argument("--coef", type=float, default=None, help="global coefficient c (default 1/K)")
    score.add_argument("--threshold", type=float, default=THRESHOLD)
    score.set_defaults(func=_score)

    screen = sub.add_parser("screen", help="score a candidate merge of real checkpoints (merge extra)")
    _checkpoint_args(screen)
    screen.add_argument("--threshold", type=float, default=THRESHOLD)
    screen.add_argument("--device", default="cpu")
    screen.add_argument("--per-layer", action="store_true", help="print layer statistics")
    screen.add_argument("--out", default=None, help="write the layer statistics to this json file")
    screen.set_defaults(func=_screen)

    mrg = sub.add_parser("merge", help="merge real checkpoints with PRISM and save (merge extra)")
    _checkpoint_args(mrg)
    mrg.add_argument("--out", required=True, help="folder for the merged checkpoint")
    mrg.add_argument("--kappa", type=float, default=1.0, help="threshold scale")
    mrg.add_argument("--gate", type=float, default=GATE, help="rho-gate threshold tau_g")
    mrg.add_argument("--min-keep", type=float, default=1e-3, help="keep-rate floor for each matrix")
    mrg.add_argument("--average", action="store_true", help="plain average, no thresholding")
    mrg.add_argument("--device", default=None, help="cuda when available, else cpu")
    mrg.set_defaults(func=_merge)

    args = parser.parse_args(argv)
    if getattr(args, "device", "") is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args.func(args)
