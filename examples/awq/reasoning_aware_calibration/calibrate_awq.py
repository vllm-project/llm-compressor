"""
AWQ calibrated on the model's own reasoning traces, compared against baselines
built from the same prompts:

- default:    prompt + dataset response, full text, no mask (regular AWQ)
- offpolicy:  prompt + dataset response, loss on the response tokens only
- onpolicy:   prompt + the model's own CoT from generate_on_policy_traces.py,
              loss on the generated tokens only (RAC)

Long CoT traces carry many more loss tokens than dataset responses, so a fair
comparison also needs equal token budgets. --loss-token-budget sets the number
of loss tokens (response tokens, for default) of a set:

- with a smaller budget, a set keeps whole samples, in order, until the
  budget is reached
- default/offpolicy with a larger budget draws a fresh set of dataset
  prompt/response pairs (disjoint from the trace prompts) until the budget is
  reached

The budget may be a number or the name of the other set ("offpolicy" or
"onpolicy"), meaning "as many loss tokens as that set has".
"""

import argparse
import json

import torch
from datasets import Dataset
from generate_on_policy_traces import chat_ids, sample_prompts
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQModifier


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def row(input_ids, loss_mask, max_seq_length):
    input_ids = input_ids[:max_seq_length]
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "loss_mask": loss_mask[:max_seq_length],
    }


def dataset_response_row(tokenizer, sample, max_seq_length):
    # render responses with a reasoning block (e.g. DeepSeek-R1 traces) in
    # thinking form and plain responses in non-thinking form; the prompt is
    # rendered the same way so the prompt/response boundary is exact
    thinking = "</think>" in sample["response"]
    user = [{"role": "user", "content": sample["prompt"]}]
    prompt_ids = chat_ids(
        tokenizer, user, add_generation_prompt=True, enable_thinking=thinking
    )
    input_ids = chat_ids(
        tokenizer,
        user + [{"role": "assistant", "content": sample["response"]}],
        enable_thinking=thinking,
    )
    if input_ids[: len(prompt_ids)] != prompt_ids:
        return None
    loss_mask = [0] * len(prompt_ids) + [1] * (len(input_ids) - len(prompt_ids))
    return row(input_ids, loss_mask, max_seq_length)


def offpolicy_rows(tokenizer, prompts, max_seq_length):
    rows = (dataset_response_row(tokenizer, s, max_seq_length) for s in prompts)
    return [r for r in rows if r is not None]


def onpolicy_rows(traces, max_seq_length):
    return [row(t["input_ids"], t["loss_mask"], max_seq_length) for t in traces]


def num_loss_tokens(rows):
    return sum(sum(r["loss_mask"]) for r in rows)


def take_until(rows, budget):
    taken, total = [], 0
    for r in rows:
        taken.append(r)
        total += sum(r["loss_mask"])
        if total >= budget:
            break
    return taken


def build_rows(args, tokenizer):
    traces = read_jsonl(args.traces)
    prompts = read_jsonl(args.prompts)
    assert [t["index"] for t in traces] == [p["index"] for p in prompts]

    sets = {
        "offpolicy": lambda: offpolicy_rows(tokenizer, prompts, args.max_seq_length),
        "onpolicy": lambda: onpolicy_rows(traces, args.max_seq_length),
    }
    # default uses the same rows as offpolicy, it only drops the mask later
    source = "offpolicy" if args.arm == "default" else args.arm
    rows = sets[source]()
    budget = args.loss_token_budget
    if budget is None:
        return rows
    if budget in sets:
        budget = num_loss_tokens(sets[budget]())
    budget = int(budget)

    if budget <= num_loss_tokens(rows):
        return take_until(rows, budget)
    if source == "onpolicy":
        raise ValueError(
            f"only {num_loss_tokens(rows)} on-policy loss tokens available, "
            "generate more traces to reach the budget"
        )
    fresh = sample_prompts(
        args.dataset,
        args.split,
        tokenizer,
        num_samples=None,
        max_prompt_tokens=args.max_prompt_tokens,
        seed=args.extra_seed,
        enable_thinking=False,
        exclude={p["index"] for p in prompts},
    )
    rows = (dataset_response_row(tokenizer, s, args.max_seq_length) for s in fresh)
    return take_until((r for r in rows if r is not None), budget)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--arm", choices=["default", "offpolicy", "onpolicy"])
    parser.add_argument("--traces", default="traces.jsonl")
    parser.add_argument("--prompts", default="prompts.jsonl")
    parser.add_argument("--loss-token-budget", default=None)
    parser.add_argument("--scheme", default="W4A16")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    # only used to draw extra off-policy pairs; keep in sync with the generator
    parser.add_argument("--dataset", default="mlabonne/open-perfectblend")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--extra-seed", type=int, default=1234)
    parser.add_argument("--save-dir", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)

    ds = Dataset.from_list(build_rows(args, tokenizer))
    num_tokens = sum(len(ids) for ids in ds["input_ids"])
    num_masked = sum(sum(mask) for mask in ds["loss_mask"])
    print(f"[{args.arm}] {len(ds)} samples, {num_tokens} tokens, {num_masked} in mask")
    if args.arm == "default":
        ds = ds.remove_columns("loss_mask")

    recipe = [
        # long reasoning traces make the cached layer inputs too large for the
        # GPU, so keep them in CPU memory
        AWQModifier(offload_device=torch.device("cpu")),
        QuantizationModifier(
            ignore=["lm_head"], scheme=args.scheme, targets=["Linear"]
        ),
    ]
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=args.max_seq_length,
        num_calibration_samples=len(ds),
        use_loss_mask=args.arm != "default",
    )

    model.save_pretrained(args.save_dir, save_compressed=True)
    tokenizer.save_pretrained(args.save_dir)


if __name__ == "__main__":
    main()
