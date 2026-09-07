"""RTN/GPTQ + subgraph QAD, saved as NVFP4 weights with FP16 activations."""

import argparse
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.qad import QADModifier
from llmcompressor.modifiers.quantization import QuantizationModifier


def prepare_dataset(tokenizer, name, split, samples, length, text_column, offset=0):
    stream = load_dataset(name, split=split, streaming=True)
    rows = []
    for example in stream.skip(offset).take(samples):
        text = (
            tokenizer.apply_chat_template(example["messages"], tokenize=False)
            if "messages" in example
            else example[text_column]
        )
        rows.append(
            tokenizer(
                text,
                truncation=True,
                max_length=length,
                padding=False,
                add_special_tokens=False,
            )
        )
    return Dataset.from_list(rows).shuffle(seed=42)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="RedHatAI/Llama-3.1-8B-Instruct")
    parser.add_argument("--quantizer", choices=["rtn", "gptq"], default="gptq")
    parser.add_argument("--teacher-mode", choices=["local", "full"], default="local")
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--dataset", default="HuggingFaceH4/ultrachat_200k")
    parser.add_argument("--split", default="train_sft")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--qad-dataset", default=None)
    parser.add_argument("--qad-split", default="train_sft")
    parser.add_argument("--qad-text-column", default="text")
    parser.add_argument("--qad-samples", type=int, default=512)
    parser.add_argument("--qad-max-seq-length", type=int, default=2048)
    parser.add_argument("--qad-batch-size", type=int, default=1)
    parser.add_argument("--qad-offset", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--targets-per-subgraph", type=int, default=1)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}")
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, attn_implementation="sdpa"
    ).to("cuda")
    dataset = prepare_dataset(
        tokenizer,
        args.dataset,
        args.split,
        args.samples,
        args.max_seq_length,
        args.text_column,
    )
    extra = {}
    if args.qad_dataset is not None:
        extra["qad_dataset"] = prepare_dataset(
            tokenizer,
            args.qad_dataset,
            args.qad_split,
            args.qad_samples,
            args.qad_max_seq_length,
            args.qad_text_column,
            args.qad_offset,
        )
        extra["qad_dataset_args"] = dict(
            num_calibration_samples=args.qad_samples,
            max_seq_length=args.qad_max_seq_length,
            batch_size=args.qad_batch_size,
            shuffle_calibration_samples=False,
        )
    kwargs = dict(targets="Linear", scheme="NVFP4A16", ignore=["lm_head"])
    quantizer = (
        GPTQModifier(**kwargs, actorder="static")
        if args.quantizer == "gptq"
        else QuantizationModifier(**kwargs)
    )
    qad = QADModifier(
        teacher_mode=args.teacher_mode,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.accumulation_steps,
    )
    oneshot(
        model=model,
        processor=tokenizer,
        dataset=dataset,
        recipe=[quantizer, qad],
        pipeline="sequential",
        sequential_targets=["LlamaDecoderLayer"],
        sequential_targets_per_subgraph=args.targets_per_subgraph,
        propagate_error=True,
        sequential_offload_device="cpu",
        batch_size=1,
        num_calibration_samples=args.samples,
        max_seq_length=args.max_seq_length,
        shuffle_calibration_samples=False,
        **extra,
    )
    # Avoid Transformers 5.14's offloaded multi-shard name-conversion issue.
    model.save_pretrained(
        output, save_compressed=True, max_shard_size="4GB", save_original_format=False
    )
    tokenizer.save_pretrained(output)


if __name__ == "__main__":
    main()
