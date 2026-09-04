"""Perplexity evaluation for compressed MoE models.

Uses load_quantizable_moe to properly handle linearized expert weights.

Usage:
    python eval_moe.py --model Qwen/Qwen3-30B-A3B --configs fouroversix mse-1x-1.5x
"""

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from llmcompressor.modeling.moe.linearize import load_quantizable_moe
from transformers import AutoModelForCausalLM, AutoTokenizer

COMPRESSED_DIR = Path("compressed-models")


def evaluate_ppl(model, tokenizer, max_length=2048, stride=512):
    testdata = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(testdata["text"])
    encodings = tokenizer(text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        device = next(model.parameters()).device
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
        nlls.append(outputs.loss)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    return torch.exp(torch.stack(nlls).mean()).item()


def model_short_name(model_id: str) -> str:
    return model_id.rstrip("/").split("/")[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--compressed-dir", type=str, default=str(COMPRESSED_DIR))
    parser.add_argument("--no-baseline", action="store_true")
    args = parser.parse_args()

    compressed_dir = Path(args.compressed_dir)
    name = model_short_name(args.model)
    results = []
    baseline_ppl = None

    if not args.no_baseline:
        print(f"  Loading unquantized {args.model}...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, device_map="auto", torch_dtype="auto"
        )
        model.eval()
        baseline_ppl = evaluate_ppl(model, tokenizer)
        print(f"  Unquantized: {baseline_ppl:.3f}")
        results.append(("Unquantized", baseline_ppl))
        del model
        torch.cuda.empty_cache()

    for config_key in args.configs:
        path = compressed_dir / f"{name}-{config_key}"
        if not path.exists():
            print(f"  SKIP (not found): {path}")
            results.append((config_key, None))
            continue

        print(f"  Loading {config_key} from {path}...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(path)
        with load_quantizable_moe():
            model = AutoModelForCausalLM.from_pretrained(
                path, device_map="auto", torch_dtype="auto"
            )
        model.eval()
        ppl = evaluate_ppl(model, tokenizer)
        print(f"  {config_key}: {ppl:.3f}")
        results.append((config_key, ppl))
        del model
        torch.cuda.empty_cache()

    print(f"\n\n{'='*60}")
    print(f"Results: {name}")
    print(f"{'='*60}\n")
    print("| Config | word_perplexity | delta vs unquantized |")
    print("|---|---|---|")
    for label, ppl in results:
        if ppl is None:
            print(f"| {label} | — | — |")
        elif label == "Unquantized":
            print(f"| {label} | **{ppl:.3f}** | — |")
        elif baseline_ppl is not None:
            delta = ppl - baseline_ppl
            print(f"| {label} | {ppl:.3f} | +{delta:.3f} |")
        else:
            print(f"| {label} | {ppl:.3f} | — |")


if __name__ == "__main__":
    main()
