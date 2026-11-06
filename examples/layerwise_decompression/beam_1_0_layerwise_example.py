import os

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, BeamForCausalLM

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context

MODEL_ID = "reflection/Beam-1.0-501B-A23B-FP8"
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4"
OFFLOAD_DIR = "./beam-offload"

DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"
NUM_CALIBRATION_SAMPLES = 100
MAX_SEQUENCE_LENGTH = 2048

IGNORE = [
    "lm_head",
    "model.embed_tokens",
    r"re:.*mlp\.gate$",
]


def main():
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        from compressed_tensors.offload import init_dist

        init_dist()

    os.makedirs(OFFLOAD_DIR, exist_ok=True)

    with load_context(BeamForCausalLM):
        model = BeamForCausalLM.from_pretrained(
            MODEL_ID,
            # This model has enough expert tensors to exceed the OS mmap
            # limit when offloaded to CPU shared memory. Use disk offload.
            device_map="disk",
            offload_folder=OFFLOAD_DIR,
            dtype="auto",
        )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
    ds = ds.shuffle(seed=42)

    def preprocess(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
            )
        }

    ds = ds.map(preprocess, remove_columns=ds.column_names)

    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=MAX_SEQUENCE_LENGTH,
            truncation=True,
            add_special_tokens=False,
        )

    ds = ds.map(tokenize, remove_columns=ds.column_names)

    recipe = QuantizationModifier(
        targets="Linear",
        scheme="NVFP4",
        ignore=IGNORE,
    )

    oneshot(
        model=model,
        tokenizer=tokenizer,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
        batch_size=1,
        pipeline="sequential",
        sequential_targets=["BeamDecoderLayer"],
        sequential_targets_per_subgraph=1,
        layerwise_decompression=True,
        layerwise_compression=True,
    )

    os.makedirs(SAVE_DIR, exist_ok=True)
    model.save_pretrained(SAVE_DIR, save_compressed=True)
    tokenizer.save_pretrained(SAVE_DIR)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
