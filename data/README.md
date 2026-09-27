# Layer statistics and evaluated outcomes

The screening results of the paper are functions of the files in this folder, so they can be recomputed on a CPU without downloading any checkpoint.

## `layer_stats/*.json`

One file per candidate merge, in the format that `prism-merge screen --out` writes. The header names the base, the specialists, K and the norm-scaling factors s<sub>k</sub>. Each entry of `layers` is one decoder layer and holds four numbers over its merged MLP coordinates (gate, up and down projections).

| field | meaning |
|---|---|
| `n` | number of merged coordinates in the layer |
| `I` | mean interference, the variance of the scaled task vectors across specialists (paper Eq. 1) |
| `C` | mean cancellation power (paper Eq. 3) |
| `cr` | coherence ratio S / (P + 10<sup>-10</sup>) |

```json
{"n": 203685888, "I": 0.00011013453032826209, "C": 3.069451263359001e-05, "cr": 0.6548292784561065}
```

The score c√(Ī/K) uses the parameter-weighted mean of `I`, and ρ is the weighted sum of `C` over the weighted sum of `I`. `recorded_rho` is the ρ that the screen logged when the file was made, kept as a checksum. Specialists marked "(ours)" were trained for the paper and are not released.

## `breadth.csv`

The 22 merges of paper Table 4, in the paper's order. Each `id` names its file in `layer_stats/`.

| column | meaning |
|---|---|
| `setting` | the row label of Table 4 |
| `K` | number of specialists, merged with c = 1/K |
| `predicted` | 1 when the outcome was predicted before evaluation |
| `base`, `ta`, `prism` | six-task means of the base model, Task Arithmetic and PRISM, from `experiments/evaluate.py`. `prism` is empty where the ρ-gate returns Task Arithmetic |

A merge is destructive when Task Arithmetic falls more than 10 points below the base and harmless when it stays within 1 point or better, which splits the 22 rows into 7 and 15 with nothing in between.

## `screened_pairs.csv`

The 28 public pairs without a code specialist that the screen scored in Appendix J, 14 on Qwen2.5-7B, 11 on Llama-3.1-8B and 3 on Mistral-7B. Eight of them are also rows of Table 4 and share its file.

## `titration.csv`

The continued-pretraining ladder of Figure 4(e). Two Qwen2.5-1.5B specialists, one on Japanese web text and one on PubMed abstracts, are scored as one merge with K = 2, c = 1/2 and min-norm scaling at every measured step. `lr` is the learning rate, `tokens_b` the tokens seen by each specialist in billions, `I` the parameter-weighted mean interference and `rho` the cancellation ratio. The two evaluated endpoints carry their six-task means.
