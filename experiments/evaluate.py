"""Six-task lm-evaluation-harness mean and WikiText-2 perplexity for a (merged) model.

A model with perplexity above 100 counts as collapsed and the harness is skipped
unless --force is given. Needs a GPU and the `eval` extra.

    python experiments/evaluate.py --model outputs/qwen7b_math_coder_prism --out results/prism.json
"""

import argparse
import json
import math

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

TASKS = {
    "gsm8k": "exact_match,strict-match",
    "arc_challenge": "acc_norm,none",
    "mmlu": "acc,none",
    "truthfulqa_mc2": "acc,none",
    "hellaswag": "acc_norm,none",
    "winogrande": "acc,none",
}


@torch.no_grad()
def wikitext2_ppl(model, tok, n_texts=100, max_len=512):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [r["text"] for r in ds if len(r["text"].strip()) > 50][:n_texts]
    dtype = model.dtype
    model.float()
    nll, ntok = 0.0, 0
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=max_len).input_ids
        ids = ids.to(model.device)
        loss = model(ids, labels=ids).loss.item()
        if math.isfinite(loss):
            nll += loss * ids.shape[1]
            ntok += ids.shape[1]
    model.to(dtype)
    return math.exp(nll / ntok) if ntok else float("inf")


def harness(model, tok, limit, batch_size, seed):
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=batch_size)
    out = lm_eval.simple_evaluate(model=lm, tasks=list(TASKS), limit=limit, random_seed=seed)
    return {t: 100 * out["results"][t][m] for t, m in TASKS.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="merged checkpoint dir or HF model id")
    ap.add_argument("--limit", type=int, default=500, help="examples for each task")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="run the harness even if collapsed")
    ap.add_argument("--out", default=None, help="write results to this json file")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=args.device).eval()

    res = {"model": args.model, "ppl": wikitext2_ppl(model, tok)}
    res["collapsed"] = res["ppl"] > 100
    print(f"wikitext-2 ppl: {res['ppl']:.2f}")

    if not res["collapsed"] or args.force:
        res["tasks"] = harness(model, tok, args.limit, args.batch_size, args.seed)
        res["mean"] = sum(res["tasks"].values()) / len(res["tasks"])
        for t, v in res["tasks"].items():
            print(f"{t:15s} {v:5.1f}")
        print(f"{'mean':15s} {res['mean']:5.1f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
