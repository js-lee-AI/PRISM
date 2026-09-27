#!/bin/bash
# Six-task means of the base, Task Arithmetic and PRISM rows of Tables 1 and 2 and of
# the matching Table 4 rows. Needs one GPU and the merge and eval extras.
#
#   bash experiments/main_merges.sh qwen7b      # Table 1 and Table 4, Qwen2.5-7B Math + Coder
#   bash experiments/main_merges.sh qwen1.5b    # Table 2 and Table 4, Qwen2.5-1.5B Math + Coder
#   bash experiments/main_merges.sh qwen7b_k3   # Table 4, Qwen2.5-7B Math + Coder + Instruct
set -e
OUT=${OUT:-outputs}
RES=${RES:-results}
mkdir -p "$OUT" "$RES"

run() {
    name=$1; base=$2; shift 2
    prism-merge screen --base "$base" --specialists "$@" --out "$RES/${name}_stats.json"
    prism-merge merge --base "$base" --specialists "$@" --out "$OUT/${name}_prism"
    prism-merge merge --base "$base" --specialists "$@" --average --out "$OUT/${name}_ta"
    python experiments/evaluate.py --model "$base" --out "$RES/${name}_base.json"
    python experiments/evaluate.py --model "$OUT/${name}_ta" --force --out "$RES/${name}_ta.json"
    python experiments/evaluate.py --model "$OUT/${name}_prism" --force --out "$RES/${name}_prism.json"
}

case "$1" in
    qwen7b)
        run qwen7b_math_coder Qwen/Qwen2.5-7B Qwen/Qwen2.5-Math-7B Qwen/Qwen2.5-Coder-7B ;;
    qwen1.5b)
        run qwen1.5b_math_coder Qwen/Qwen2.5-1.5B \
            Qwen/Qwen2.5-Math-1.5B-Instruct Qwen/Qwen2.5-Coder-1.5B-Instruct ;;
    qwen7b_k3)
        run qwen7b_math_coder_instruct Qwen/Qwen2.5-7B \
            Qwen/Qwen2.5-Math-7B Qwen/Qwen2.5-Coder-7B Qwen/Qwen2.5-7B-Instruct ;;
    *)
        echo "usage: bash experiments/main_merges.sh qwen7b|qwen1.5b|qwen7b_k3"; exit 1 ;;
esac
