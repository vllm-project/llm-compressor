"""IterSmooth + dynamic W8A8 for dense Qwen3.6-27B on CUDA or Ascend.

Read image records from a local directory in filename order, using the
checkpoint's processor defaults without an additional image-size or
sequence-length limit.
JSON/JSONL with an ``image`` path and an optional ``text`` prompt is also supported;
relative image paths are resolved against the JSON/JSONL file's directory.
Set TEXT_CALIBRATION_DATA to a JSON/JSONL file of raw ``text`` prompts to reuse
text-only calibration data alongside the images, or set CALIBRATION_DATA to None
for text-only calibration. Mixed inputs use eager basic calibration with native
CPU offload, since a single sequential trace cannot represent both input branches.

This recipe adds IterSmooth preprocessing for the text decoder, uses native
compressed-tensors INT8 numerics, and preserves MTP weights in floating point.
The native saver exports a compressed-tensors checkpoint. Adjust the model,
calibration and output paths below.
On Ascend, expose one NPU with ASCEND_RT_VISIBLE_DEVICES=0.
"""

from importlib.util import find_spec
from pathlib import Path

import torch
from compressed_tensors.offload import set_onload_device
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.itersmooth import IterSmoothModifier
from llmcompressor.utils import load_context
from llmcompressor.utils.dev import get_main_device

MODEL_ID = "Qwen/Qwen3.6-27B"
# Set this to an image directory or a JSON/JSONL file of image prompts.
# Set to None for text-only calibration using TEXT_CALIBRATION_DATA.
CALIBRATION_DATA = "calibImages"
TEXT_CALIBRATION_DATA = None  # E.g. "/data/calib/qwen36.jsonl"; None disables it.
TEXT_MAX_SEQUENCE_LENGTH = None  # Text-only truncation; None keeps the full prompt.
DEFAULT_TEXT = "Describe this image in detail."
SAVE_DIR = "Qwen3.6-27B-IterSmooth-W8A8"
MAX_SHARD_SIZE = "4GB"
ALPHA = 0.9

if not 0 <= ALPHA <= 1:
    raise ValueError("ALPHA must be in [0, 1]")
if TEXT_MAX_SEQUENCE_LENGTH is not None and TEXT_MAX_SEQUENCE_LENGTH < 1:
    raise ValueError("TEXT_MAX_SEQUENCE_LENGTH must be positive or None")
if CALIBRATION_DATA is None and TEXT_CALIBRATION_DATA is None:
    raise ValueError("Set CALIBRATION_DATA and/or TEXT_CALIBRATION_DATA")
if Path(SAVE_DIR).resolve() == Path(MODEL_ID).resolve():
    raise ValueError("SAVE_DIR must differ from the original model directory")

# Register the Ascend backend when installed; CUDA does not need torch_npu.
if find_spec("torch_npu") is not None:
    import torch_npu  # noqa: F401

device = get_main_device()
if device.type == "npu":
    if torch.npu.device_count() != 1:
        raise RuntimeError("Expose one NPU with ASCEND_RT_VISIBLE_DEVICES")
    torch.npu.set_device(device)
elif device.type != "cuda":
    raise RuntimeError("This example requires a CUDA GPU or an Ascend NPU")
print(f"Calibration device: {device}")

# Use the checkpoint's processor settings for both image and text inputs.
processor = AutoProcessor.from_pretrained(MODEL_ID)


def tokenize(sample, image_root=None):
    is_image = image_root is not None
    prompt = (sample.get("text") or DEFAULT_TEXT) if is_image else sample["text"]
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Each calibration text must be a nonempty user prompt")
    content = []
    if is_image:
        if not isinstance(sample["image"], str) or not sample["image"].strip():
            raise ValueError("Each image calibration sample must contain an image path")
        image_path = Path(sample["image"]).expanduser()
        if not image_path.is_absolute():
            image_path = image_root / image_path
        content.append({"type": "image", "image": str(image_path.resolve(strict=True))})
    content.append({"type": "text", "text": prompt})
    text_options = {}
    if not is_image and TEXT_MAX_SEQUENCE_LENGTH is not None:
        text_options = {"truncation": True, "max_length": TEXT_MAX_SEQUENCE_LENGTH}
    inputs = processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs=text_options,
    )
    return {key: value.tolist() for key, value in inputs.items()}


datasets = []
num_image_samples = 0
num_text_samples = 0
if CALIBRATION_DATA is not None:
    calibration_path = Path(CALIBRATION_DATA).expanduser().resolve(strict=True)
    if calibration_path.is_dir():
        image_paths = sorted(
            path
            for path in calibration_path.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        # Require an explicit prompt file rather than silently ignoring its text.
        if any(
            path.suffix.lower() in {".json", ".jsonl"}
            for path in calibration_path.iterdir()
            if path.is_file()
        ):
            raise ValueError(
                "Set CALIBRATION_DATA to the JSON/JSONL prompt file in this directory"
            )
        image_dataset = Dataset.from_dict(
            {
                "image": [str(path) for path in image_paths],
                "text": [DEFAULT_TEXT] * len(image_paths),
            }
        )
    else:
        image_dataset = load_dataset(
            "json", data_files=str(calibration_path), split="train"
        )
    if "image" not in image_dataset.column_names or len(image_dataset) == 0:
        raise ValueError("Image calibration data must be nonempty with an image column")
    num_image_samples = len(image_dataset)
    datasets.append(
        image_dataset.map(
            tokenize,
            fn_kwargs={"image_root": calibration_path.parent},
            remove_columns=image_dataset.column_names,
        )
    )

if TEXT_CALIBRATION_DATA is not None:
    text_path = Path(TEXT_CALIBRATION_DATA).expanduser().resolve(strict=True)
    text_dataset = load_dataset("json", data_files=str(text_path), split="train")
    if "text" not in text_dataset.column_names or len(text_dataset) == 0:
        raise ValueError("Text calibration data must be nonempty with a text column")
    if "image" in text_dataset.column_names and any(
        image is not None for image in text_dataset["image"]
    ):
        raise ValueError(
            "Use CALIBRATION_DATA for image records, not TEXT_CALIBRATION_DATA"
        )
    num_text_samples = len(text_dataset)
    datasets.append(
        text_dataset.map(tokenize, remove_columns=text_dataset.column_names)
    )

# Tokenize each modality separately so Arrow schemas can be aligned at concatenation.
# Missing image columns become None for text samples and are omitted by the collator.
dataset = concatenate_datasets(datasets)
mixed_calibration = num_image_samples > 0 and num_text_samples > 0
print(f"Calibration samples: {num_text_samples} text + {num_image_samples} image")


def data_collator(batch):
    if len(batch) != 1:
        raise ValueError("This example requires batch_size=1")
    # The processor already added the text batch dimension. Keep pixel_values
    # and image_grid_thw in their original packed shapes, without stacking.
    inputs = {
        key: torch.as_tensor(value)
        for key, value in batch[0].items()
        if value is not None
    }
    if mixed_calibration:
        # Basic calibration runs the real model; avoid allocating all prompt logits.
        # Its input mover accepts tensors only. A 1-D LongTensor is interpreted by
        # Qwen as token indices, so [-1] selects the last token (not index [1]).
        inputs["logits_to_keep"] = torch.as_tensor([-1], dtype=torch.long)
    return inputs


# Keep weights on CPU; the native offload lifecycle loads them as needed.
# Qwen3.6-27B uses the Qwen3.5 architecture and its standard Transformers class.
# See https://huggingface.co/Qwen/Qwen3.6-27B/blob/main/config.json.
with load_context(Qwen3_5ForConditionalGeneration):
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype="auto",
        device_map="cpu",
        attn_implementation="eager",
    )
model.eval()

# Smooth norm -> Q/K/V (full-attention blocks) and norm -> gate/up (all blocks).
# Quantize full-attention, MLP, vision-block and QKV/Z input projections.
# Linear attention's in_proj_a/in_proj_b, out_proj and conv1d stay floating point.
# Do not extend smoothing to QKV/Z alone: a/b share the same normalized input.
recipe = [
    IterSmoothModifier(alpha=ALPHA, enable_subgraph_type=["norm-linear"]),
    QuantizationModifier(
        targets=[
            r"re:.*self_attn\.(q_proj|k_proj|v_proj|o_proj)$",
            r"re:.*mlp\.(gate_proj|up_proj|down_proj)$",
            r"re:.*linear_attn\.(in_proj_qkv|in_proj_z)$",
            r"re:.*visual\.blocks\.\d+\.attn\.(qkv|proj)$",
            r"re:.*visual\.blocks\.\d+\.mlp\.linear_fc[12]$",
        ],
        ignore=[r"re:.*linear_fc2$"],
        # Symmetric INT8: per-channel weights and dynamic per-token activations.
        # Keep native INT8 scaling (currently absmax/127.5 in compressed-tensors).
        scheme="W8A8",
    ),
]

calibration_args = dict(
    model=model,
    processor=processor,
    dataset=dataset,
    sequential_targets=(
        ["Qwen3_5VisionBlock", "Qwen3_5DecoderLayer"]
        if num_image_samples
        else ["Qwen3_5DecoderLayer"]
    ),
    sequential_targets_per_subgraph=1,
    sequential_offload_device="cpu",
    batch_size=1,
    data_collator=data_collator,
    num_calibration_samples=len(dataset),
    shuffle_calibration_samples=False,
    max_seq_length=None,
)
if mixed_calibration:
    # Sequential traces specialize to the first batch's image/no-image branch.
    # Eager forwards collect both modalities in ONE smoothing pass; native offload
    # keeps weights on CPU and onloads them for execution, at extra transfer cost.
    set_onload_device(model, device)
    print("Mixed calibration: basic IterSmooth with CPU offload, then data-free W8A8")
    oneshot(**calibration_args, recipe=recipe[0], pipeline="basic")
    oneshot(model=model, processor=processor, recipe=recipe[1], pipeline="datafree")
else:
    # Homogeneous batches retain the native sequential/data-free independent flow.
    oneshot(**calibration_args, recipe=recipe, pipeline="independent")

# Save in the native compressed-tensors format with the configured shard size.
# The native saver carries over unloaded MTP tensors in floating point and adds
# them to quantization_config.ignore.
model.save_pretrained(SAVE_DIR, save_compressed=True, max_shard_size=MAX_SHARD_SIZE)
processor.save_pretrained(SAVE_DIR)
