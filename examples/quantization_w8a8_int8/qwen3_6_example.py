"""W8A8 INT8 quantization for Qwen3.6-35B-A3B (Qwen3.5 MoE architecture).

Requires transformers >= v5.

Examples:
  # Data-free RTN (default)
  python qwen3_6_example.py

  # SmoothQuant + GPTQ with calibration data
  python qwen3_6_example.py --algorithm gptq

Quantizes Linear layers in MoE experts and full self-attention.
Skips linear-attention (Gated DeltaNet), router gates, embeddings,
vision tower, and lm_head (same as NVFP4 MoE examples).
"""

from __future__ import annotations

import argparse

import torch
from compressed_tensors.utils import save_mtp_tensors_to_checkpoint
from datasets import load_dataset
from transformers import AutoProcessor, Qwen3_5MoeForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier

MODEL_ID = "Qwen/Qwen3.6-35B-A3B"

NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 4096
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"

IGNORE = [
    "re:.*lm_head",
    "re:visual.*",
    "re:model.visual.*",
    "re:.*mlp.gate$",
    "re:.*embed_tokens$",
    "re:.*shared_expert_gate$",
    "re:.*linear_attn.*",
]


def _build_calibration_dataset(processor):
    ds = load_dataset(
        DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]"
    ).select_columns(["messages"])
    ds = ds.shuffle(seed=42)

    def preprocess(example):
        messages = [
            {
                "role": m["role"],
                "content": [{"type": "text", "text": m["content"]}],
            }
            for m in example["messages"]
        ]
        return processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=False,
            processor_kwargs={
                "return_tensors": "pt",
                "padding": False,
                "truncation": True,
                "max_length": MAX_SEQUENCE_LENGTH,
                "add_special_tokens": False,
            },
        )

    return ds.map(preprocess, batched=False, remove_columns=ds.column_names)


def _data_collator(batch):
    assert len(batch) == 1
    return {key: torch.tensor(value) for key, value in batch[0].items()}


def _build_recipe(algorithm: str):
    if algorithm == "rtn":
        return QuantizationModifier(
            targets="Linear",
            scheme="W8A8",
            ignore=IGNORE,
        )
    if algorithm == "gptq":
        return [
            SmoothQuantModifier(smoothing_strength=0.8),
            GPTQModifier(
                targets="Linear",
                scheme="W8A8",
                ignore=IGNORE,
            ),
        ]
    raise ValueError(f"Unknown algorithm: {algorithm}")


def _save_suffix(algorithm: str) -> str:
    if algorithm == "rtn":
        return "-W8A8-RTN"
    return "-W8A8-Dynamic-Per-Token"


def main():
    parser = argparse.ArgumentParser(
        description="W8A8 quantization for Qwen3.6 MoE"
    )
    parser.add_argument(
        "--algorithm",
        choices=["rtn", "gptq"],
        default="rtn",
        help=(
            "rtn: data-free round-to-nearest (default). "
            "gptq: SmoothQuant + GPTQ with calibration data."
        ),
    )
    parser.add_argument(
        "--model-id",
        default=MODEL_ID,
        help=f"Hugging Face model id (default: {MODEL_ID})",
    )
    args = parser.parse_args()

    model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        args.model_id, dtype="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model_id)
    recipe = _build_recipe(args.algorithm)

    oneshot_kwargs: dict = dict(
        model=model,
        recipe=recipe,
        moe_calibrate_all_experts=True,
    )
    if args.algorithm == "gptq":
        oneshot_kwargs.update(
            dataset=_build_calibration_dataset(processor),
            max_seq_length=MAX_SEQUENCE_LENGTH,
            num_calibration_samples=NUM_CALIBRATION_SAMPLES,
            data_collator=_data_collator,
        )

    oneshot(**oneshot_kwargs)

    save_dir = args.model_id.rstrip("/").split("/")[-1] + _save_suffix(
        args.algorithm
    )
    model.save_pretrained(save_dir, save_compressed=True)
    processor.save_pretrained(save_dir)
    save_mtp_tensors_to_checkpoint(source_model=args.model_id, dest_dir=save_dir)
    print(f"Saved quantized model to {save_dir}")


if __name__ == "__main__":
    main()
