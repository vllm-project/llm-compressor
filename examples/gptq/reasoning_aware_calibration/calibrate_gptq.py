"""GPTQ calibration for three token-matched MATH-500 data conditions.

- full: prompt + dataset solution, every token contributes to the Hessian
  (regular GPTQ calibration)
- dataset_masked: prompt + dataset solution, only solution tokens contribute
- onpolicy_masked: prompt + the model's own trace from
  generate_on_policy_traces.py, only generated tokens contribute (RAC)

Masked prompt tokens are still forwarded as context. All three conditions use
the same 500 problems and exactly the same number of active Hessian tokens: for
every problem the response is cut to the shorter of the dataset solution and the
generated trace, and the full condition shortens its solutions so that prompt
plus solution tokens add up to the same total.
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def proportional_quotas(weights, total):
    """Allocate an exact integer budget proportional to nonnegative weights."""
    denominator = sum(weights)
    if total < 0 or total > denominator:
        raise ValueError(f"Cannot allocate {total} tokens from capacity {denominator}")
    if denominator == 0:
        if total:
            raise ValueError("Cannot allocate a nonzero budget over empty weights")
        return [0] * len(weights)
    quotas = [(total * weight) // denominator for weight in weights]
    remainder = total - sum(quotas)
    order = sorted(
        range(len(weights)),
        key=lambda index: (-(total * weights[index] % denominator), index),
    )
    for index in order[:remainder]:
        quotas[index] += 1
    if sum(quotas) != total or any(q > w for q, w in zip(quotas, weights)):
        raise AssertionError("Token quota allocation failed")
    return quotas


def build_rows(prompts, traces, arm):
    if len(prompts) != 500 or len(traces) != 500:
        raise ValueError(
            f"Expected 500 prompts and traces; got {len(prompts)}, {len(traces)}"
        )
    for index, (prompt, trace) in enumerate(zip(prompts, traces, strict=True)):
        if prompt["index"] != index or trace["index"] != index:
            raise ValueError(
                "Prompt and trace rows are not in the fixed MATH-500 order"
            )
        if prompt["unique_id"] != trace["unique_id"]:
            raise ValueError(f"Prompt/trace ID mismatch at MATH-500 row {index}")
        if prompt["prompt_ids"] != trace["prompt_ids"]:
            raise ValueError(f"Prompt token ID mismatch at MATH-500 row {index}")

    matched_response_lengths = [
        min(len(prompt["gold_response_ids"]), len(trace["response_ids"]))
        for prompt, trace in zip(prompts, traces, strict=True)
    ]
    active_token_budget = sum(matched_response_lengths)
    prompt_token_count = sum(len(prompt["prompt_ids"]) for prompt in prompts)
    if active_token_budget <= prompt_token_count:
        raise ValueError(
            f"Matched answer budget ({active_token_budget}) must exceed prompt tokens "
            f"({prompt_token_count}) so the unmasked baseline can retain every prompt"
        )

    if arm == "full":
        # Every row keeps its complete prompt. The remaining global budget is
        # allocated over gold responses, then the whole sequence is unmasked.
        response_quotas = proportional_quotas(
            matched_response_lengths, active_token_budget - prompt_token_count
        )
        rows = []
        for prompt, response_quota in zip(prompts, response_quotas, strict=True):
            sequence_length = len(prompt["prompt_ids"]) + response_quota
            rows.append(
                {
                    "input_ids": prompt["prompt_ids"]
                    + prompt["gold_response_ids"][:response_quota],
                    "attention_mask": [1] * sequence_length,
                }
            )
    else:
        rows = []
        for prompt, trace, response_quota in zip(
            prompts, traces, matched_response_lengths, strict=True
        ):
            response_ids = (
                prompt["gold_response_ids"]
                if arm == "dataset_masked"
                else trace["response_ids"]
            )
            prompt_ids = prompt["prompt_ids"]
            rows.append(
                {
                    "input_ids": prompt_ids + response_ids[:response_quota],
                    "attention_mask": [1] * (len(prompt_ids) + response_quota),
                    "loss_mask": [0] * len(prompt_ids) + [1] * response_quota,
                }
            )

    active_tokens = (
        sum(len(row["input_ids"]) for row in rows)
        if arm == "full"
        else sum(sum(row["loss_mask"]) for row in rows)
    )
    if active_tokens != active_token_budget:
        raise AssertionError(
            f"{arm} has {active_tokens} active tokens, expected {active_token_budget}"
        )
    return rows, active_token_budget, prompt_token_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--traces", required=True)
    parser.add_argument(
        "--arm",
        choices=["full", "dataset_masked", "onpolicy_masked"],
        required=True,
    )
    parser.add_argument(
        "--scheme", choices=["FP8_DYNAMIC", "MXFP4XFP8", "W4A16"], required=True
    )
    parser.add_argument("--save-dir", required=True)
    args = parser.parse_args()

    prompts = read_jsonl(args.prompts)
    traces = read_jsonl(args.traces)
    generation_metadata = json.loads(
        (Path(args.traces).parent / "generation_complete.json").read_text(
            encoding="utf-8"
        )
    )
    if generation_metadata["model"] != args.model:
        raise ValueError(
            "Trace-generation and calibration model paths must match: "
            f"{generation_metadata['model']} != {args.model}"
        )
    rows, active_tokens, prompt_tokens = build_rows(prompts, traces, args.arm)
    dataset = Dataset.from_list(rows)
    max_seq_length = max(len(row["input_ids"]) for row in rows)
    print(
        f"arm={args.arm} scheme={args.scheme} rows={len(rows)} "
        f"active_hessian_tokens={active_tokens:,} "
        f"prompt_context_tokens={prompt_tokens:,} "
        f"max_seq_length={max_seq_length}"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    oneshot(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        recipe=GPTQModifier(
            targets=["Linear"],
            ignore=["lm_head"],
            scheme=args.scheme,
        ),
        num_calibration_samples=len(rows),
        shuffle_calibration_samples=False,
        batch_size=1,
        max_seq_length=max_seq_length,
        use_loss_mask=args.arm != "full",
        pipeline="sequential",
        sequential_offload_device="cpu",
        output_dir=args.save_dir,
        save_compressed=True,
    )

    stats = {
        "arm": args.arm,
        "scheme": args.scheme,
        "num_samples": len(rows),
        "active_hessian_tokens": active_tokens,
        "prompt_context_tokens": prompt_tokens,
        "matched_response_token_lengths": [
            min(len(prompt["gold_response_ids"]), len(trace["response_ids"]))
            for prompt, trace in zip(prompts, traces, strict=True)
        ],
        "max_seq_length": max_seq_length,
        "model": args.model,
        "dataset": generation_metadata["dataset"],
        "dataset_revision": generation_metadata["dataset_revision"],
        "generation_config": {
            key: generation_metadata[key]
            for key in (
                "temperature",
                "top_p",
                "top_k",
                "seed",
                "max_new_tokens",
            )
        },
    }
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    with open(
        Path(args.save_dir) / "calibration_stats.json", "w", encoding="utf-8"
    ) as f:
        json.dump(stats, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
