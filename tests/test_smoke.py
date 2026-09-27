import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import prism

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    return subprocess.run([sys.executable, *args], capture_output=True, text=True,
                          check=True, timeout=120, cwd=ROOT).stdout


def test_import_does_not_pull_torch():
    # a fresh interpreter, since pytest plugins may have imported torch already
    out = run("-c", "import sys, prism; print('torch' in sys.modules)")
    assert out.strip() == "False"


def test_power_splits_into_signal_and_interference():
    x = np.random.default_rng(0).normal(size=(3, 50, 40))
    s = prism.core.layer_stats([x])
    assert s["P"] == pytest.approx(s["S"] + s["I"])
    # I is the population variance across specialists (Eq. 1)
    assert s["I"] == pytest.approx(x.var(axis=0).mean())


def test_opposite_signs_are_pure_cancellation():
    x = np.stack([np.full(10, 0.3), np.full(10, -0.3)])
    s = prism.core.layer_stats([x])
    assert s["S"] == 0 and s["C"] == pytest.approx(s["I"]) == pytest.approx(0.09)
    assert prism.cancellation_ratio([s]) == pytest.approx(1.0)


def test_one_sided_drift_is_dispersion():
    # one specialist moves, the other stays at the base: no opposing signs, rho = 0
    x = np.stack([np.full(10, 0.3), np.zeros(10)])
    s = prism.core.layer_stats([x])
    assert s["I"] > 0 and s["C"] == 0
    assert prism.cancellation_ratio([s]) == 0


def test_score_threshold_and_soft_threshold_closed_forms():
    stats = [{"n": 100, "I": 4e-4}, {"n": 300, "I": 1e-4}]
    assert prism.interference_score(stats, K=2, coef=0.5) == pytest.approx(0.5 * math.sqrt(1.75e-4 / 2))
    assert prism.thresholds(stats[:1])[0] == pytest.approx(0.02 * math.sqrt(2 * math.log(100)))
    y = prism.soft_threshold(np.array([0.5, -0.2, 0.05, -0.01]), 0.1, min_keep=0)
    assert y.tolist() == pytest.approx([0.4, -0.1, 0.0, 0.0])


def test_keep_rate_floor_keeps_the_largest_coordinates():
    x = np.arange(1, 2001, dtype=np.float32)
    y = prism.soft_threshold(x, 1e9, min_keep=1e-3)
    # the floor lowers the threshold to the 2nd largest magnitude, so only the largest survives
    assert np.count_nonzero(y) == 1 and y[-1] == 1.0


def test_gate_returns_the_plain_average():
    rng = np.random.default_rng(1)
    base = {"w": rng.normal(size=(64, 64))}
    # one-sided drift gives rho = 0 < 0.1, so lambda = 0 and PRISM is Task Arithmetic
    specialists = [{"w": base["w"] + np.abs(rng.normal(size=(64, 64)))}, {"w": base["w"].copy()}]
    merged, info = prism.merge(base, specialists, norm_scaling=False)
    plain, _ = prism.merge(base, specialists, norm_scaling=False, denoise=False)
    assert info["gated"] and np.array_equal(merged["w"], plain["w"])


def test_merge_writes_the_consensus_and_drops_the_noise():
    rng = np.random.default_rng(0)
    base = {"w": rng.normal(0, 0.02, (256, 512))}
    shared = (rng.random((256, 512)) < 0.005) * 0.2
    specialists = [{"w": base["w"] + shared + rng.normal(0, 0.02, (256, 512))} for _ in range(2)]
    merged, info = prism.merge(base, specialists)
    update = merged["w"] - base["w"].astype(np.float32)   # merged weights are float32
    assert not info["gated"] and info["mean_keep_rate"] < 0.01
    assert np.all(update[shared == 0] == 0)
    assert np.all(update[shared > 0] > 0)


def test_hf_names_are_grouped_by_layer():
    keys = [f"model.layers.{i}.mlp.{p}.weight" for i in (1, 0) for p in prism.core.PROJS]
    keys += ["model.layers.0.self_attn.q_proj.weight", "lm_head.weight"]
    layers = prism.mlp_layers(keys)
    assert len(layers) == 2 and layers[0][0] == "model.layers.0.mlp.gate_proj.weight"


def test_layer_statistic_files_recompute_their_recorded_rho():
    for path in sorted((ROOT / "data" / "layer_stats").glob("*.json")):
        a = prism.load_stats(path)
        assert a.rho == pytest.approx(a.meta["recorded_rho"], rel=1e-9), path.name


def test_save_and_load_round_trip(tmp_path):
    rng = np.random.default_rng(2)
    base = {"w": rng.normal(size=(32, 32))}
    specialists = [{"w": base["w"] + rng.normal(0, 0.1, (32, 32))} for _ in range(3)]
    a = prism.analyze(base, specialists)
    prism.save_stats(a, tmp_path / "s.json", base="toy", specialists=["a", "b", "c"])
    b = prism.load_stats(tmp_path / "s.json")
    assert (b.K, b.score, b.rho) == (3, pytest.approx(a.score), pytest.approx(a.rho))


def test_quickstart_prints_what_the_readme_shows():
    out = run(str(ROOT / "examples" / "quickstart.py"))
    assert "score=4.989e-03  threshold=1.9e-03  rho=0.361 (thresholding on)" in out
    assert "plain average  error 1.00   weights changed 100.0%" in out
    assert "PRISM          error 0.37   weights changed 0.5%" in out


def test_cli():
    assert "demo" in run("-m", "prism", "--help")
    assert run("-m", "prism", "demo") == run(str(ROOT / "examples" / "quickstart.py"))
    out = run("-m", "prism", "score", "data/layer_stats/qwen7b_math_coder.json")
    assert "score=4.251e-03" in out and "rho=0.293" in out and "merge with PRISM" in out


def _tiny_checkpoints(root, K=2, layers=2, seed=0):
    torch = pytest.importorskip("torch")
    st = pytest.importorskip("safetensors.torch")
    g = torch.Generator().manual_seed(seed)
    shapes = {"gate_proj": (48, 16), "up_proj": (48, 16), "down_proj": (16, 48)}
    base = {f"model.layers.{i}.mlp.{p}.weight": torch.randn(s, generator=g) * 0.02
            for i in range(layers) for p, s in shapes.items()}
    base["model.layers.0.self_attn.q_proj.weight"] = torch.randn(16, 16, generator=g)
    shared = {k: (torch.rand(v.shape, generator=g) < 0.02) * 0.2 for k, v in base.items()}
    dirs = []
    for j in range(K + 1):
        w = base if j == 0 else {k: v + shared[k] + 0.02 * torch.randn(v.shape, generator=g)
                                 for k, v in base.items()}
        d = root / f"m{j}"
        d.mkdir()
        st.save_file({k: v.to(torch.bfloat16).contiguous() for k, v in w.items()}, str(d / "model.safetensors"))
        dirs.append(str(d))
    return dirs


def test_torch_path_matches_the_numpy_core(tmp_path):
    torch = pytest.importorskip("torch")
    from safetensors.torch import load_file

    dirs = _tiny_checkpoints(tmp_path)
    a = prism.screen_checkpoints(dirs[0], dirs[1:])
    w = [{k: v.float().numpy() for k, v in load_file(f"{d}/model.safetensors").items()} for d in dirs]
    b = prism.analyze(w[0], w[1:])
    assert a.score == pytest.approx(b.score, rel=1e-5) and a.rho == pytest.approx(b.rho, rel=1e-5)
    merged, info = prism.merge_checkpoints(dirs[0], dirs[1:])
    ref, ref_info = prism.merge(w[0], w[1:])
    assert info["keep_rates"] == pytest.approx(ref_info["keep_rates"])
    for k, v in merged.items():
        np.testing.assert_allclose(v.float().numpy(), ref[k], rtol=1e-2, atol=1e-4)


@pytest.mark.gpu
def test_merge_on_cuda_matches_cpu(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    dirs = _tiny_checkpoints(tmp_path)
    cpu, _ = prism.merge_checkpoints(dirs[0], dirs[1:], device="cpu")
    gpu, _ = prism.merge_checkpoints(dirs[0], dirs[1:], device="cuda")
    for k in cpu:
        torch.testing.assert_close(cpu[k], gpu[k], rtol=1e-2, atol=1e-4)
