import argparse
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Constants ────────────────────────────────────────────────────────

DEFAULT_MODELS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3-70B-Instruct",
]

DEFAULT_CONFIGS = [
    "fouroversix",
    "mse-1x-1.5x",
    "nvfp4_expanded_mse",
    "expand-3.4",
]

COMPRESSED_DIR = Path("/tmp/fouroversix-sanity")

CONFIG_LABELS = {
    "fouroversix": "FourOverSix",
    "mse-1x-1.5x": "MSE 1x+1.5x",
    "nvfp4_expanded_mse": "nvfp4_expanded_mse",
    "expand-3.4": "expand=3.4 (original ablation)",
    "default-mse": "Default MSE",
    "minmax": "MinMax",
    "expanded-gs-prior": "Expanded MSE + gs prior",
}


# ── Evaluation ───────────────────────────────────────────────────────


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


def load_and_eval(path: str, label: str) -> float:
    print(f"  Loading {label} from {path}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, device_map="auto", torch_dtype="auto"
    )
    model.eval()
    ppl = evaluate_ppl(model, tokenizer)
    print(f"  {label}: {ppl:.3f}")
    del model
    torch.cuda.empty_cache()
    return ppl


# ── Main ─────────────────────────────────────────────────────────────


def model_short_name(model_id: str) -> str:
    return model_id.rstrip("/").split("/")[-1]


def main():
    parser = argparse.ArgumentParser(
        description="PPL evaluation for NVFP4 observer comparison"
    )
    parser.add_argument(
        "--models", nargs="+", default=DEFAULT_MODELS,
        help="HuggingFace model IDs (baselines)",
    )
    parser.add_argument(
        "--configs", nargs="+", default=DEFAULT_CONFIGS,
        help="Observer config names to evaluate",
    )
    parser.add_argument(
        "--compressed-dir", type=str, default=str(COMPRESSED_DIR),
        help="Directory containing compressed models",
    )
    parser.add_argument(
        "--no-baseline", action="store_true",
        help="Skip unquantized baseline evaluation",
    )
    args = parser.parse_args()

    compressed_dir = Path(args.compressed_dir)
    all_results = {}

    for model_id in args.models:
        name = model_short_name(model_id)
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        results = []
        baseline_ppl = None

        if not args.no_baseline:
            baseline_ppl = load_and_eval(model_id, "Unquantized")
            results.append(("Unquantized", baseline_ppl))

        for config_key in args.configs:
            path = compressed_dir / f"{name}-{config_key}"
            label = CONFIG_LABELS.get(config_key, config_key)
            if not path.exists():
                print(f"  SKIP (not found): {path}")
                results.append((label, None))
                continue
            ppl = load_and_eval(str(path), label)
            results.append((label, ppl))

        all_results[name] = (baseline_ppl, results)

    # ── Print markdown tables ────────────────────────────────────
    print(f"\n\n{'='*60}")
    print("Results (markdown)")
    print(f"{'='*60}\n")

    for name, (baseline_ppl, results) in all_results.items():
        print(f"### {name}\n")
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
        print()


if __name__ == "__main__":
    main()
