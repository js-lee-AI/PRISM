"""PRISM on a toy merge. CPU only, no downloads, well under a second."""

import numpy as np

import prism

rng = np.random.default_rng(0)
base = {"w": rng.normal(0, 0.02, (512, 1024))}
shared = (rng.random((512, 1024)) < 0.005) * 0.2      # an update both specialists agree on
specialists = [{"w": base["w"] + shared + rng.normal(0, 0.02, (512, 1024))} for _ in range(2)]

print(prism.analyze(base, specialists))               # screen before merging

target = 0.5 * shared                                 # what a clean merge adds at c = 1/K
for name, denoise in [("plain average", False), ("PRISM", True)]:
    merged, info = prism.merge(base, specialists, denoise=denoise)
    err = np.linalg.norm(merged["w"] - base["w"] - target) / np.linalg.norm(target)
    print(f"{name:14s} error {err:.2f}   weights changed {100 * info['mean_keep_rate']:.1f}%")
