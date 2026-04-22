import argparse
import time

import torch
from datasets import load_dataset
from transformers import Llama4ForConditionalGeneration, Llama4Processor

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier

MODEL_ID = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

DATASET_ID = "neuralmagic/calibration"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 8192


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actorder", default=None)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--num-samples", type=int, default=NUM_CALIBRATION_SAMPLES)
    args = parser.parse_args()

    num_samples = args.num_samples

    model = Llama4ForConditionalGeneration.from_pretrained(MODEL_ID, dtype="auto")
    processor = Llama4Processor.from_pretrained(MODEL_ID)

    ds = load_dataset(DATASET_ID, name="LLM", split=f"train[:{num_samples}]")

    def preprocess_function(example):
        messages = []
        for message in example["messages"]:
            messages.append(
                {
                    "role": message["role"],
                    "content": [{"type": "text", "text": message["content"]}],
                }
            )

        return processor.apply_chat_template(
            messages,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH,
            tokenize=True,
            add_special_tokens=False,
            return_dict=True,
            add_generation_prompt=False,
        )

    ds = ds.map(preprocess_function, batched=False, remove_columns=ds.column_names)

    def data_collator(batch):
        assert len(batch) == 1
        return {
            key: (
                torch.tensor(value)
                if key != "pixel_values"
                else torch.tensor(value, dtype=torch.bfloat16).squeeze(0)
            )
            for key, value in batch[0].items()
        }

    gptq_kwargs = dict(
        targets="Linear",
        ignore=[
            "re:.*lm_head",
            "re:.*self_attn",
            "re:.*router",
            "re:.*vision_model.*",
            "re:.*multi_modal_projector.*",
            "Llama4TextAttention",
        ],
        scheme="FP8_BLOCK",
    )
    if args.actorder:
        gptq_kwargs["actorder"] = args.actorder

    recipe = GPTQModifier(**gptq_kwargs)

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=num_samples,
        data_collator=data_collator,
        sequential_targets=["Llama4TextMLP"],
    )

    elapsed_time = time.time() - start_time
    peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
    print("Quantization Complete")
    print(f"Time: {elapsed_time / 60:.2f} minutes ({elapsed_time:.2f} seconds)")
    print(f"Peak GPU Memory: {peak_memory_gb:.2f} GB")

    save_dir = args.save_dir or (
        MODEL_ID.rstrip("/").split("/")[-1]
        + f"-FP8-BLOCK-GPTQ-{args.actorder or 'no-actorder'}"
    )
    model.save_pretrained(save_dir, save_compressed=True)
    processor.save_pretrained(save_dir)
    print(f"Model saved to {save_dir}")


if __name__ == "__main__":
    main()
