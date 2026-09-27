"""Figure 4. Screening densities (a-d) and the continued-pretraining titration (e).

Prints the titration numbers of Section 5.7 and Appendix J. With --plot it also
draws the figure, which needs matplotlib (pip install matplotlib). CPU only.

    python experiments/figure4_screening.py [--plot results/figure4.png]
"""

import argparse
import os

import numpy as np

from common import breadth, score_str, titration
from prism import THRESHOLD, interference_score


def fit(rows):
    """Power-law fit of the score against training tokens on the 1e-4 ladder."""
    t = np.array([r["tokens_b"] for r in rows])
    y = np.array([r["score"] for r in rows])
    b, log_a = np.polyfit(np.log(t), np.log(y), 1)
    return b, np.exp(log_a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", default=None, help="write the figure to this png or pdf")
    args = ap.parse_args()

    ladder = titration()
    for r in ladder:
        r["score"] = interference_score([{"n": 1, "I": r["I"]}], 2, 0.5)
    low = [r for r in ladder if r["lr"] == "1e-4"]
    high = [r for r in ladder if r["lr"] == "5e-4"]

    print("Two Qwen2.5-1.5B specialists continued on Japanese web text and PubMed abstracts, K = 2, c = 1/2")
    print(f"{'lr':5s} {'step':>5s} {'tokens (B)':>11s} {'score':>6s}")
    for r in ladder:
        print(f"{r['lr']:5s} {r['step']:5d} {r['tokens_b']:11.2f} {score_str(r['score']):>6s}")
    print("score in units of 1e-3")

    rows = breadth()
    cluster = min(r["analysis"].score for r in rows if r["outcome"] == "destructive" and r["analysis"].flagged)
    b, A = fit(low)
    x_thr = (THRESHOLD / A) ** (1 / b)
    x_dst = (cluster / A) ** (1 / b)
    print(f"\nfit on the 1e-4 ladder: score ~ tokens^{b:.2f}")
    print(f"crosses the threshold of {THRESHOLD:.1e} near {x_thr:.1f}B tokens")
    print(f"reaches {score_str(cluster)}e-3, the lower edge of the destructive cluster, near {x_dst:.0f}B tokens")
    first = next(r for r in high if r["score"] >= cluster)
    print(f"at 5e-4 the ladder reaches that edge {first['tokens_b'] - low[-1]['tokens_b']:.2f}B tokens "
          f"after the 1e-4 endpoint")
    for r in (low[-1], high[-1]):
        verdict = "destructive" if r["base"] - r["ta"] > 10 else "harmless"
        print(f"endpoint at {r['tokens_b']:.2f}B tokens (lr {r['lr']}): score {score_str(r['score'])}e-3, "
              f"{verdict}, six-task mean TA {r['ta']:.1f}, PRISM {r['prism']:.1f}, base {r['base']:.1f}")

    if args.plot:
        plot(rows, low, high, b, A, cluster, args.plot)
        print("wrote", args.plot)


def plot(rows, low, high, b, A, cluster, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    })
    harm_c, dest_c, thr_c, lo_c, hi_c, fit_c = "#5B84B1", "#D1605E", "#4D4D4D", "#3A3A3A", "#8E6BB0", "#A8A8A8"
    grid = dict(color="#E6E6E6", lw=0.5, zorder=0)

    K = np.array([r["analysis"].K for r in rows], float)
    c = np.array([r["analysis"].coef for r in rows])
    rho = np.array([r["analysis"].rho for r in rows])
    score = np.array([r["analysis"].score for r in rows])
    dest = np.array([r["outcome"] == "destructive" for r in rows])

    def kde(v, g, bw):
        z = (g[:, None] - v[None, :]) / bw
        return np.exp(-0.5 * z ** 2).sum(1)

    fig = plt.figure(figsize=(5.5, 1.78))
    outer = GridSpec(1, 2, figure=fig, width_ratios=[4.35, 1.32], wspace=0.25,
                     left=0.068, right=0.992, top=0.77, bottom=0.20)
    left = outer[0, 0].subgridspec(1, 4, width_ratios=[1.3, 1.0, 1.0, 1.0], wspace=0.34)
    axes = [fig.add_subplot(left[0, i]) for i in range(4)]
    ax_e = fig.add_subplot(outer[0, 1])
    panels = [
        (np.log10(score), np.linspace(-6.5, -1.5, 600), (0.22, 0.10), (-6.3, -1.7), r"(a) $c\sqrt{\overline{I}/K}$"),
        (rho, np.linspace(-0.15, 0.95, 600), (0.045, 0.045), (-0.1, 0.85), r"(b) $\rho$"),
        (K, np.linspace(0.5, 9.5, 600), (0.32, 0.32), (0.8, 9.2), r"(c) $K$"),
        (c / np.sqrt(K), np.linspace(-0.02, 0.45, 600), (0.017, 0.017), (0.0, 0.42), r"(d) $c/\sqrt{K}$"),
    ]
    for ax, (v, g, bws, xl, title) in zip(axes, panels):
        for m, col, bw in ((~dest, harm_c, bws[0]), (dest, dest_c, bws[1])):
            d = kde(v[m], g, bw)
            d /= d.max()
            ax.fill_between(g, d, color=col, alpha=0.30, lw=0)
            ax.plot(g, d, color=col, lw=1.0)
        ax.set_xlim(*xl)
        ax.set_ylim(0, 1.12)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x", **grid)
        ax.set_title(title, loc="left", pad=3, fontweight="bold")
    axes[0].axvline(np.log10(THRESHOLD), color=thr_c, lw=0.8, ls=(0, (4, 2)))
    axes[0].set_xticks([-6, -4, -2])
    axes[0].set_xticklabels([r"$10^{-6}$", r"$10^{-4}$", r"$10^{-2}$"])
    axes[0].set_ylabel("density")
    axes[1].set_xticks([0, 0.4, 0.8])
    axes[2].set_xticks([2, 4, 6, 8])
    axes[3].set_xticks([0, 0.2, 0.4])

    t1 = np.array([r["tokens_b"] for r in low])
    y1 = np.array([r["score"] for r in low])
    t2 = np.array([r["tokens_b"] for r in high])
    y2 = np.array([r["score"] for r in high])
    t_in = np.logspace(np.log10(t1[0]), np.log10(t1[-1]), 50)
    t_out = np.logspace(np.log10(t1[-1]), np.log10(60), 80)
    ax_e.plot(t_in, A * t_in ** b, "-", color=fit_c, lw=1.0)
    ax_e.plot(t_out, A * t_out ** b, "--", color=fit_c, lw=1.0, dashes=(3, 2))
    ax_e.axhline(THRESHOLD, color=thr_c, lw=0.8, ls=(0, (4, 2)))
    ax_e.axhline(cluster, color=dest_c, lw=0.8, ls=(0, (1, 1.5)))
    ax_e.plot(t1, y1, "-o", color=lo_c, lw=0.9, ms=2.4, mew=0)
    ax_e.plot(np.r_[t1[-1], t2], np.r_[y1[-1], y2], "-", color=hi_c, lw=0.9)
    ax_e.plot(t2, y2, "D", color=hi_c, ms=2.3, mew=0)
    ax_e.plot([t1[-1]], [y1[-1]], "o", ms=5.2, mfc=harm_c, mec="white", mew=0.6, zorder=5)
    ax_e.plot([t2[-1]], [y2[-1]], "o", ms=5.2, mfc=dest_c, mec="white", mew=0.6, zorder=5)
    ax_e.set_xscale("log")
    ax_e.set_yscale("log")
    ax_e.set_xlim(0.1, 60)
    ax_e.set_ylim(1.8e-4, 1.0e-2)
    ax_e.set_xlabel("tokens (billions)")
    ax_e.set_ylabel(r"$c\sqrt{\overline{I}/K}$")
    ax_e.grid(True, which="major", **grid)
    ax_e.set_title("(e) Titration", loc="left", pad=3, fontweight="bold")

    fig.legend(handles=[
        Patch(facecolor=harm_c + "4D", edgecolor=harm_c, label="harmless"),
        Patch(facecolor=dest_c + "4D", edgecolor=dest_c, label="destructive"),
        Line2D([], [], color=thr_c, lw=0.8, ls=(0, (4, 2)), label="frozen threshold"),
        Line2D([], [], color=lo_c, lw=0.9, marker="o", ms=2.4, mew=0, label=r"LR $10^{-4}$"),
        Line2D([], [], color=hi_c, lw=0.9, marker="D", ms=2.3, mew=0, label=r"LR $5{\times}10^{-4}$"),
        Line2D([], [], color=fit_c, lw=1.0, label="fit"),
        Line2D([], [], color=dest_c, lw=0.8, ls=(0, (1, 1.5)), label="destructive cluster"),
    ], loc="lower center", ncol=7, bbox_to_anchor=(0.515, 0.885), handlelength=1.5,
        handletextpad=0.4, columnspacing=0.8, borderaxespad=0)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=300)


if __name__ == "__main__":
    main()
