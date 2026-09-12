"""
Measure wikitext-2 perplexity using vLLM with prompt_logprobs.

No lm_eval dependency — uses vLLM directly for model loading and inference.
Computes word-level perplexity matching lm_eval's wikitext metric.

Usage:
    python measure_ppl.py --model ~/hf_hub/Llama-3-8B-HIGGS-5.0bit-all-A8
    python measure_ppl.py --model meta-llama/Meta-Llama-3-8B --max-model-len 4096
"""

import argparse
import json
import math
import os
import sys

import torch


def load_wikitext():
    from datasets import load_dataset

    ds = load_dataset(
        "EleutherAI/wikitext_document_level",
        "wikitext-2-raw-v1",
        split="test",
        cache_dir=os.path.expanduser("~/hf_hub"),
    )
    texts = [doc for doc in ds["page"] if doc.strip()]
    full_text = "\n\n".join(texts)
    return full_text


def measure_ppl_vllm(model_path, max_model_len=4096, gpu_mem=0.85, tp=1):
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        dtype="auto",
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_mem,
        tensor_parallel_size=tp,
        trust_remote_code=True,
        enforce_eager=True,
    )

    tokenizer = llm.get_tokenizer()
    text = load_wikitext()
    tokens = tokenizer.encode(text)
    print(f"Total tokens: {len(tokens)}", flush=True)

    stride = max_model_len - 1
    total_nll = 0.0
    total_tokens = 0
    total_words = 0

    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        prompt_logprobs=0,
    )

    chunks = []
    for start in range(0, len(tokens), stride):
        chunk = tokens[start : start + stride]
        if len(chunk) < 2:
            continue
        chunks.append(chunk)

    print(f"Processing {len(chunks)} chunks of up to {stride} tokens...", flush=True)

    batch_size = 4
    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start : batch_start + batch_size]
        prompt_token_ids = [{"prompt_token_ids": c} for c in batch]

        outputs = llm.generate(
            prompt_token_ids,
            sampling_params=sampling_params,
        )

        for output, chunk in zip(outputs, batch):
            if output.prompt_logprobs is None:
                continue
            for i, lp_dict in enumerate(output.prompt_logprobs):
                if lp_dict is None:
                    continue
                token_id = chunk[i]
                if token_id in lp_dict:
                    logprob = lp_dict[token_id].logprob
                    total_nll -= logprob
                    total_tokens += 1

        done = min(batch_start + batch_size, len(chunks))
        print(
            f"  [{done}/{len(chunks)}] tokens={total_tokens}, "
            f"running_ppl={math.exp(total_nll / max(total_tokens, 1)):.4f}",
            flush=True,
        )

    byte_nll = total_nll / max(total_tokens, 1)
    byte_ppl = math.exp(byte_nll)

    word_count = len(text.split())
    word_nll = total_nll / max(word_count, 1)
    word_ppl = math.exp(word_nll)

    return {
        "word_perplexity": word_ppl,
        "byte_perplexity": byte_ppl,
        "total_tokens": total_tokens,
        "total_words": word_count,
        "total_nll": total_nll,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-mem", type=float, default=0.85)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: <model>/eval_results.json)")
    args = parser.parse_args()

    result = measure_ppl_vllm(
        args.model,
        max_model_len=args.max_model_len,
        gpu_mem=args.gpu_mem,
        tp=args.tp,
    )

    out_path = args.output
    if out_path is None and os.path.isdir(args.model):
        out_path = os.path.join(args.model, "eval_results.json")

    print(f"\nResults:", flush=True)
    print(f"  Word perplexity: {result['word_perplexity']:.4f}", flush=True)
    print(f"  Byte perplexity: {result['byte_perplexity']:.4f}", flush=True)
    print(f"  Tokens: {result['total_tokens']}", flush=True)
    print(f"  Words: {result['total_words']}", flush=True)

    if out_path:
        save_data = {"ppl": result["word_perplexity"]}
        save_data.update(result)
        with open(out_path, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"  Saved to: {out_path}", flush=True)

    return result


if __name__ == "__main__":
    main()
