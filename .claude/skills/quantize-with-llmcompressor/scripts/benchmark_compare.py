#!/usr/bin/env python3
"""OPTIONAL DEV/EVAL TOOL -- not part of the core quantization skill.

Apples-to-apples comparison: LLM Compressor vs NVIDIA TensorRT Model Optimizer
on the *same* model and *same* target scheme.

This is deliberately kept OUT of the core skill path because it pulls in
out-of-tree, separately-versioned dependencies (``nvidia-modelopt``) that the
upstream llm-compressor repo neither declares nor imports. Install it only if you
want the comparison, ideally in its own virtualenv (modelopt pins an older
``transformers`` than llmcompressor needs -- see references/modelopt_comparison.md).

For each tool it reports three numbers a practitioner cares about:

* **Accuracy**  -- WikiText-2 perplexity of the quantized model (lower better).
* **Footprint** -- on-disk size of the exported checkpoint (smaller better).
* **Cost**      -- wall-clock time to quantize (lower better).

The built-in ``perplexity()`` here is a lightweight proxy for a quick read. For a
rigorous, citable accuracy number use ``lm-eval`` (the wikitext task), which is
the evaluation path used throughout the upstream examples/ READMEs.

This is intentionally a *small-model* harness so it runs on a single consumer GPU
in minutes. Scale ``--model`` / ``--num-samples`` up for a heavier comparison.

Usage (run each tool in its own venv):
    python benchmark_compare.py --model meta-llama/Llama-3.2-1B-Instruct \
        --scheme FP8_DYNAMIC --tools llmcompressor
"""

import argparse
import importlib.util
import json
import os
import sys
import time

# The FP4 cast is torch.compile-decorated and needs Triton; on platforms
# without it (e.g. Windows) fall back to eager. Must happen before importing
# torch, so we sniff argv for an FP4 scheme.
if any("FP4" in a.upper() for a in sys.argv) and (
    importlib.util.find_spec("triton") is None
):
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch  # noqa: E402

# llmcompressor scheme  ->  modelopt config attribute name
MODELOPT_CFG = {
    "FP8_DYNAMIC": "FP8_DEFAULT_CFG",
    "FP8": "FP8_DEFAULT_CFG",
    "NVFP4": "NVFP4_DEFAULT_CFG",
    "NVFP4A16": "NVFP4_DEFAULT_CFG",
    "W8A8": "INT8_DEFAULT_CFG",
    "INT8": "INT8_DEFAULT_CFG",
    "W4A16": "INT4_AWQ_CFG",
}


def dir_size_gb(path):
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return round(total / (1024**3), 3)


def get_calib_texts(tokenizer, num_samples, max_seq_len):
    from datasets import load_dataset

    ds = load_dataset(
        "HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{num_samples}]"
    )
    texts = [tokenizer.apply_chat_template(ex["messages"], tokenize=False) for ex in ds]
    return texts


@torch.no_grad()
def perplexity(model, tokenizer, max_seq_len=2048, stride=1024, limit=40):
    """Sliding-window WikiText-2 perplexity. `limit` caps the number of windows
    for speed; raise it for a tighter estimate."""
    from datasets import load_dataset

    test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(test["text"])
    enc = tokenizer(text, return_tensors="pt").input_ids
    device = next(model.parameters()).device
    nlls, count, n_tokens, prev_end = [], 0, 0, 0
    for begin in range(0, enc.size(1), stride):
        end = min(begin + max_seq_len, enc.size(1))
        # Only score tokens not already scored by the previous window, so
        # overlapping context is conditioned on but never double-counted.
        trg_len = end - prev_end
        input_ids = enc[:, begin:end].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        out = model(input_ids, labels=target_ids)
        nlls.append(out.loss.float() * trg_len)
        n_tokens += trg_len
        prev_end = end
        count += 1
        if count >= limit or end == enc.size(1):
            break
    return float(torch.exp(torch.stack(nlls).sum() / n_tokens))


def run_llmcompressor(model_id, scheme, num_samples, max_seq_len, outdir):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype="auto", device_map="cuda"
    )
    recipe = QuantizationModifier(targets="Linear", scheme=scheme, ignore=["lm_head"])

    kwargs = {"model": model, "recipe": recipe}
    data_free = scheme.upper() in {
        "FP8_DYNAMIC",
        "FP8",
        "FP8_BLOCK",
        "W4A16",
        "W8A16",
        "NVFP4A16",
        "MXFP4A16",
    }
    if not data_free:
        from datasets import load_dataset

        ds = load_dataset(
            "HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{num_samples}]"
        ).shuffle(seed=42)
        ds = ds.map(
            lambda e: {
                "text": tokenizer.apply_chat_template(e["messages"], tokenize=False)
            }
        )
        ds = ds.map(
            lambda s: tokenizer(
                s["text"],
                padding=False,
                max_length=max_seq_len,
                truncation=True,
                add_special_tokens=False,
            ),
            remove_columns=ds.column_names,
        )
        kwargs.update(
            dataset=ds, max_seq_length=max_seq_len, num_calibration_samples=num_samples
        )

    t0 = time.time()
    oneshot(**kwargs)
    elapsed = time.time() - t0

    ppl = perplexity(model, tokenizer)
    os.makedirs(outdir, exist_ok=True)
    model.save_pretrained(outdir, save_compressed=True)
    tokenizer.save_pretrained(outdir)
    return {
        "tool": "llmcompressor",
        "scheme": scheme,
        "seconds": round(elapsed, 1),
        "perplexity": round(ppl, 4),
        "size_gb": dir_size_gb(outdir),
        "vllm_servable": True,
        "output": outdir,
    }


def run_modelopt(model_id, scheme, num_samples, max_seq_len, outdir):
    import modelopt.torch.quantization as mtq
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg_name = MODELOPT_CFG.get(scheme.upper())
    if cfg_name is None:
        return {
            "tool": "modelopt",
            "scheme": scheme,
            "error": f"no modelopt config mapped for {scheme}",
        }
    config = getattr(mtq, cfg_name)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype="auto", device_map="cuda"
    )
    texts = get_calib_texts(tokenizer, num_samples, max_seq_len)

    def forward_loop(m):
        for t in texts:
            ids = tokenizer(
                t, return_tensors="pt", truncation=True, max_length=max_seq_len
            ).input_ids.to(m.device)
            m(ids)

    t0 = time.time()
    model = mtq.quantize(model, config, forward_loop)
    elapsed = time.time() - t0

    ppl = perplexity(model, tokenizer)
    os.makedirs(outdir, exist_ok=True)
    size_gb = None
    try:
        from modelopt.torch.export import export_hf_checkpoint

        export_hf_checkpoint(model, export_dir=outdir)
        size_gb = dir_size_gb(outdir)
    except Exception as exc:  # noqa: BLE001 - export support varies by scheme
        print(f"[modelopt] export_hf_checkpoint unavailable for {scheme}: {exc}")
    return {
        "tool": "modelopt",
        "scheme": scheme,
        "seconds": round(elapsed, 1),
        "perplexity": round(ppl, 4),
        "size_gb": size_gb,
        "vllm_servable": "checkpoint export scheme-dependent",
        "output": outdir,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--scheme", default="FP8_DYNAMIC")
    parser.add_argument(
        "--tools", default="both", choices=["both", "llmcompressor", "modelopt"]
    )
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--out-prefix", default="bench")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    results = []
    if args.tools in ("both", "llmcompressor"):
        print("\n### Running LLM Compressor ...")
        results.append(
            run_llmcompressor(
                args.model,
                args.scheme,
                args.num_samples,
                args.max_seq_len,
                f"{args.out_prefix}-llmcompressor-{args.scheme}",
            )
        )
    if args.tools in ("both", "modelopt"):
        print("\n### Running NVIDIA Model Optimizer ...")
        try:
            results.append(
                run_modelopt(
                    args.model,
                    args.scheme,
                    args.num_samples,
                    args.max_seq_len,
                    f"{args.out_prefix}-modelopt-{args.scheme}",
                )
            )
        except Exception as exc:  # noqa: BLE001
            results.append(
                {"tool": "modelopt", "scheme": args.scheme, "error": repr(exc)}
            )

    print("\n" + "=" * 78)
    print(f"COMPARISON  model={args.model}  scheme={args.scheme}")
    print("=" * 78)
    hdr = f"{'tool':<16}{'ppl (WikiText2)':>18}{'size GB':>12}{'time s':>10}"
    print(hdr)
    print("-" * 78)
    for r in results:
        if "error" in r:
            print(f"{r['tool']:<16}  ERROR: {r['error']}")
            continue
        print(
            f"{r['tool']:<16}{r['perplexity']:>18}{str(r['size_gb']):>12}"
            f"{r['seconds']:>10}"
        )
    print("=" * 78)
    print("Note: LLM Compressor emits a compressed-tensors checkpoint that runs")
    print("directly with `vllm serve`. Lower perplexity = better accuracy.")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(
                {"model": args.model, "scheme": args.scheme, "results": results},
                f,
                indent=2,
            )
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
