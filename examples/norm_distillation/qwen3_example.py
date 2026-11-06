"""Norm-only distillation of an already-compressed W4A16 checkpoint.

Trains only the normalization gains of RedHatAI/Qwen3-0.6B-quantized.w4a16 against
the unquantized Qwen/Qwen3-0.6B. Packed weights and qparams are saved unchanged.
"""

import argparse

from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.qad import NormDistillationModifier


def wikitext_blocks(tokenizer, length):
    rows = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(row for row in rows["text"] if row.strip())
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    blocks = [ids[i : i + length] for i in range(0, len(ids) - length + 1, length)]
    return Dataset.from_dict(
        {"input_ids": blocks, "attention_mask": [[1] * length for _ in blocks]}
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="RedHatAI/Qwen3-0.6B-quantized.w4a16")
    parser.add_argument("--teacher", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output", default="Qwen3-0.6B-quantized.w4a16-norm-distilled")
    parser.add_argument("--samples", type=int, default=1760)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


def main():
    args = parse_args()
    # Load the checkpoint as is. A path passed to oneshot would be decompressed
    # and saved dense; a loaded compressed model is saved compressed again.
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype="auto", device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.teacher)

    oneshot(
        model=model,
        processor=tokenizer,
        dataset=wikitext_blocks(tokenizer, args.max_seq_length),
        recipe=NormDistillationModifier(teacher=args.teacher, lr=args.learning_rate),
        num_calibration_samples=args.samples,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
