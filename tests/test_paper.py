"""The experiment scripts print the numbers of the paper."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(script):
    return subprocess.run([sys.executable, str(ROOT / "experiments" / script)], capture_output=True,
                          text=True, check=True, timeout=120, cwd=ROOT).stdout


def test_table4():
    out = run("table4_breadth.py")
    for row in [
        "Qwen2.5-1.5B ja + med, constructed *            2  0.50  0.342     5.19   56.1  35.8  55.9",
        "Qwen2.5-7B Math + Coder                         2  0.50  0.293     4.25   68.0  49.5  67.8",
        "Qwen2.5-7B Math + Finance *                     2  0.50  0.036g   0.653   68.1  37.0  37.0g",
        "Qwen2.5-7B self-SFT, K = 8                      8  0.13  0.720    0.003   68.1  68.0  68.0",
        "Llama-3.1-8B Tulu-3 + Tulu-3-DPO *              2  0.50  0.003g   0.002   59.3  64.5  64.5g",
    ]:
        assert row in out
    assert "5 of 7 destructive, 0 of 15 harmless" in out
    assert "falls 14.4 to 20.3 points below the base and PRISM stays within 0.5 of it" in out
    assert "12 of 14 correct" in out


def test_table18():
    out = run("table18_rival_statistics.py")
    assert "c sqrt(I/K) (used)            0.98    0.83x  predictive" in out
    assert "c sqrt(P/K) (total power)     1.00   21.21x  separates" in out
    assert "1 - CR (mean incoherence)     0.08   <0.01x  anti-predictive" in out
    assert "rho (cancellation ratio)      0.25    0.02x  anti-predictive" in out


def test_screened_pairs():
    out = run("screened_pairs.py")
    assert "28 pairs, 14 on Qwen2.5, 11 on Llama-3.1, 3 on Mistral" in out
    assert "scores from 0.002e-3 to 0.853e-3" in out
    assert "DeepSeek-R1-Distill-Qwen-7B + Qwen/Qwen2.5-Math-PRM-7B" in out
    assert "all below half the threshold (9.60e-04): True" in out


def test_figure4_titration():
    out = run("figure4_screening.py")
    assert "crosses the threshold of 1.9e-03 near 7.5B tokens" in out
    assert "reaches 4.17e-3, the lower edge of the destructive cluster, near 36B tokens" in out
    assert "endpoint at 1.00B tokens (lr 1e-4): score 0.699e-3, harmless" in out
    assert "endpoint at 2.05B tokens (lr 5e-4): score 5.19e-3, destructive" in out
