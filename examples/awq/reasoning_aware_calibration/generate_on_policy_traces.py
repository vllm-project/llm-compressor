"""
Generate on-policy chain-of-thought calibration traces with vLLM.

Reasoning-Aware Compression (RAC, https://arxiv.org/abs/2509.12464) calibrates
compression on the model's *own* reasoning traces rather than on prompts or on
responses written by another model. This script samples those traces with the
uncompressed model and stores, per sample:

- input_ids: prompt tokens followed by the generated tokens, exactly as vLLM
  produced them (no re-tokenization, so the chat template cannot drop or alter
  the `<think>` block of the response)
- loss_mask: 0 for prompt tokens, 1 for generated tokens

It also writes the selected prompts together with the dataset's original
responses, so that calibrate_awq.py can build baselines on identical prompts.

Run this in an environment with vLLM installed; vLLM is not a dependency of
llm-compressor and the two may pin incompatible versions of torch/transformers.
"""

import argparse
import json
import random

from datasets import load_dataset
from transformers import AutoTokenizer


def prompt_response_pair(example):
    # first user turn and first assistant turn, from either openai style
    # {"role", "content"} messages (e.g. OpenR1-Math) or sharegpt style
    # {"from", "value"} conversations (e.g. open-perfectblend)
    if "messages" in example:
        turns = [(t["role"], t["content"]) for t in example["messages"]]
    else:
        turns = [(t["from"], t["value"]) for t in example["conversations"]]
    prompt = next((text for role, text in turns if role in ("user", "human")), None)
    response = next(
        (text for role, text in turns if role in ("assistant", "gpt")), None
    )
    return prompt, response


def chat_ids(tokenizer, messages, **kwargs):
    ids = tokenizer.apply_chat_template(messages, tokenize=True, **kwargs)
    return ids["input_ids"] if hasattr(ids, "input_ids") else list(ids)


def sample_prompts(
    dataset,
    split,
    tokenizer,
    num_samples,
    max_prompt_tokens,
    seed,
    enable_thinking=True,
    exclude=(),
):
    """Randomly pick prompt/response pairs whose rendered prompt fits
    max_prompt_tokens. With num_samples=None, walk the whole shuffled dataset
    lazily; otherwise draw from 4x num_samples random candidates."""
    ds = load_dataset(dataset, split=split)
    rng = random.Random(seed)
    num_candidates = len(ds) if num_samples is None else min(len(ds), 4 * num_samples)
    for idx in rng.sample(range(len(ds)), num_candidates):
        if idx in exclude:
            continue
        prompt, response = prompt_response_pair(ds[idx])
        if not prompt or not response:
            continue
        prompt_ids = chat_ids(
            tokenizer,
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        if len(prompt_ids) > max_prompt_tokens:
            continue
        yield {"index": idx, "prompt": prompt, "response": response}
        if num_samples is not None:
            num_samples -= 1
            if num_samples == 0:
                return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--dataset", default="mlabonne/open-perfectblend")
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    # Qwen3 recommended sampling settings for thinking mode
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="traces.jsonl")
    parser.add_argument("--prompts-output", default="prompts.jsonl")
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    selected = list(
        sample_prompts(
            args.dataset,
            args.split,
            tokenizer,
            args.num_samples,
            args.max_prompt_tokens,
            args.seed,
            enable_thinking=not args.no_thinking,
        )
    )
    # render the prompt exactly as it will be served, including the assistant
    # turn opener (and, for thinking mode, whatever the template adds there)
    prompt_ids = [
        chat_ids(
            tokenizer,
            [{"role": "user", "content": s["prompt"]}],
            add_generation_prompt=True,
            enable_thinking=not args.no_thinking,
        )
        for s in selected
    ]

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_prompt_tokens + args.max_new_tokens,
        seed=args.seed,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    outputs = llm.generate(
        [{"prompt_token_ids": ids} for ids in prompt_ids], sampling_params
    )

    num_truncated = 0
    num_generated = 0
    with open(args.output, "w") as f:
        for sample, output in zip(selected, outputs):
            completion = output.outputs[0]
            prompt = list(output.prompt_token_ids)
            generated = list(completion.token_ids)
            num_truncated += completion.finish_reason == "length"
            num_generated += len(generated)
            record = {
                "index": sample["index"],
                "input_ids": prompt + generated,
                "loss_mask": [0] * len(prompt) + [1] * len(generated),
                "num_prompt_tokens": len(prompt),
                "num_generated_tokens": len(generated),
                "finish_reason": completion.finish_reason,
                "text": completion.text,
            }
            f.write(json.dumps(record) + "\n")

    with open(args.prompts_output, "w") as f:
        for sample in selected:
            f.write(json.dumps(sample) + "\n")

    print(
        f"wrote {len(selected)} traces to {args.output}: "
        f"{num_generated} generated tokens "
        f"(mean {num_generated / len(selected):.0f}), "
        f"{num_truncated} truncated at max_new_tokens"
    )


if __name__ == "__main__":
    main()
