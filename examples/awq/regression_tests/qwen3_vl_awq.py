import argparse
import base64
import time
from io import BytesIO

import torch
from compressed_tensors.offload import dispatch_model
from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

DATASET_ID = "lmms-lab/flickr30k"
DATASET_SPLIT = "test"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", default="W4A16_ASYM")
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--num-samples", type=int, default=NUM_CALIBRATION_SAMPLES)
    args = parser.parse_args()

    num_samples = args.num_samples

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{num_samples}]")
    ds = ds.shuffle(seed=42)

    def preprocess_and_tokenize(example):
        buffered = BytesIO()
        example["image"].save(buffered, format="PNG")
        encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
        base64_qwen = f"data:image;base64,{encoded_image}"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": base64_qwen},
                    {"type": "text", "text": "What does the image show?"},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        return processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            max_length=MAX_SEQUENCE_LENGTH,
            truncation=True,
        )

    ds = ds.map(preprocess_and_tokenize, remove_columns=ds.column_names)

    def data_collator(batch):
        assert len(batch) == 1
        return {key: torch.tensor(value) for key, value in batch[0].items()}

    recipe = AWQModifier(
        scheme=args.scheme,
        ignore=["re:.*lm_head", "re:.*visual.*"],
        duo_scaling=False,
    )

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    oneshot(
        model=model,
        tokenizer=MODEL_ID,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=num_samples,
        data_collator=data_collator,
        sequential_targets=["Qwen3VLTextDecoderLayer"],
    )

    elapsed_time = time.time() - start_time
    peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
    print("Quantization Complete")
    print(f"Time: {elapsed_time / 60:.2f} minutes ({elapsed_time:.2f} seconds)")
    print(f"Peak GPU Memory: {peak_memory_gb:.2f} GB")

    print("\n\n")
    print("========== SAMPLE GENERATION ==============")
    dispatch_model(model)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "http://images.cocodataset.org/train2017/000000231895.jpg",
                },
                {"type": "text", "text": "Please describe the animal in this image\n"},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        return_tensors="pt",
    ).to(model.device)
    output = model.generate(**inputs, max_new_tokens=100)
    print(processor.decode(output[0], skip_special_tokens=True))
    print("==========================================\n\n")

    save_dir = args.save_dir or (
        MODEL_ID.rstrip("/").split("/")[-1] + f"-{args.scheme}"
    )
    model.save_pretrained(save_dir, save_compressed=True)
    processor.save_pretrained(save_dir)
    print(f"Model saved to {save_dir}")


if __name__ == "__main__":
    main()
