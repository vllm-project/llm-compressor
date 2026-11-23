"""Generate fixed-seed on-policy reasoning traces for MATH-500 with vLLM.

Reasoning-Aware Compression (RAC, https://arxiv.org/abs/2509.12464) calibrates
on the uncompressed model's own reasoning traces. This script samples those
traces in thinking mode and stores, for every MATH-500 problem, the prompt token
IDs, the dataset solution token IDs and the generated response token IDs.

The saved prompt token IDs are also used to build both dataset-response
calibration arms. Generated token IDs are retained directly from vLLM so the
on-policy response is not changed by a second tokenization pass.

Run this in an environment with vLLM installed; vLLM and llm-compressor may pin
incompatible versions of torch/transformers.
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

DATASET = "HuggingFaceH4/MATH-500"
DATASET_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
# Qwen3 recommended sampling settings for thinking mode
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20


def chat_ids(tokenizer, messages, **kwargs):
    ids = tokenizer.apply_chat_template(messages, tokenize=True, **kwargs)
    return ids["input_ids"] if hasattr(ids, "input_ids") else list(ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset = load_dataset(DATASET, split="test", revision=DATASET_REVISION)

    samples = []
    for index, example in enumerate(dataset):
        user = [{"role": "user", "content": example["problem"]}]
        prompt_ids = chat_ids(
            tokenizer, user, add_generation_prompt=True, enable_thinking=True
        )
        assistant = "<think>\n" + example["solution"].strip() + "\n</think>"
        completed_ids = chat_ids(
            tokenizer,
            user + [{"role": "assistant", "content": assistant}],
            enable_thinking=True,
        )
        if completed_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(f"Chat template changed the prompt prefix for row {index}")
        samples.append(
            {
                "index": index,
                "unique_id": example["unique_id"],
                "prompt": example["problem"],
                "gold_solution": example["solution"],
                "prompt_ids": prompt_ids,
                "gold_response_ids": completed_ids[len(prompt_ids) :],
            }
        )

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max(len(s["prompt_ids"]) for s in samples) + args.max_new_tokens,
        seed=args.seed,
        enable_prefix_caching=True,
    )
    sampling = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    outputs = llm.generate(
        [{"prompt_token_ids": s["prompt_ids"]} for s in samples], sampling
    )

    trace_path = output_dir / "traces.jsonl"
    prompt_path = output_dir / "prompts.jsonl"
    trace_tmp = output_dir / "traces.jsonl.tmp"
    prompt_tmp = output_dir / "prompts.jsonl.tmp"
    generated_tokens = 0
    truncated = 0
    with (
        trace_tmp.open("w", encoding="utf-8") as traces,
        prompt_tmp.open("w", encoding="utf-8") as prompts,
    ):
        for sample, output in zip(samples, outputs, strict=True):
            completion = output.outputs[0]
            prompt_ids = list(output.prompt_token_ids)
            response_ids = list(completion.token_ids)
            if prompt_ids != sample["prompt_ids"]:
                raise ValueError(
                    f"vLLM changed the prompt token IDs for row {sample['index']}"
                )
            if not response_ids:
                raise ValueError(
                    f"vLLM returned an empty trace for row {sample['index']}"
                )
            trace = {
                "index": sample["index"],
                "unique_id": sample["unique_id"],
                "prompt_ids": prompt_ids,
                "response_ids": response_ids,
                "finish_reason": completion.finish_reason,
                "text": completion.text,
            }
            traces.write(json.dumps(trace) + "\n")
            prompts.write(json.dumps(sample) + "\n")
            generated_tokens += len(response_ids)
            truncated += completion.finish_reason == "length"

    trace_tmp.replace(trace_path)
    prompt_tmp.replace(prompt_path)
    (output_dir / "generation_complete.json").write_text(
        json.dumps(
            {
                "dataset": DATASET,
                "dataset_revision": DATASET_REVISION,
                "num_samples": len(samples),
                "model": args.model,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "top_k": TOP_K,
                "seed": args.seed,
                "max_new_tokens": args.max_new_tokens,
                "generated_response_tokens": generated_tokens,
                "truncated_at_max_new_tokens": truncated,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"Generated {len(samples)} traces ({generated_tokens:,} response tokens; "
        f"{truncated} reached max_new_tokens={args.max_new_tokens})."
    )
    print(f"Prompts: {prompt_path}\nTraces: {trace_path}")


if __name__ == "__main__":
    main()
