"""IterSmooth + dynamic W8A8 for dense Qwen3.6-27B on CUDA or Ascend.

By default, calibration uses a public dataset of instruction prompts.
To use local JSON/JSONL, set CALIBRATION_DATA to a file with raw user prompts
in a ``text`` column. Adjust the model and output paths below as needed.
On Ascend, expose one NPU with ASCEND_RT_VISIBLE_DEVICES=0.
"""

from importlib.util import find_spec
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.itersmooth import IterSmoothModifier
from llmcompressor.utils import load_context
from llmcompressor.utils.dev import get_main_device

MODEL_ID = "Qwen/Qwen3.6-27B"
DATASET_ID = "neuralmagic/LLM_compression_calibration"
CALIBRATION_DATA = None  # Set to "calib.jsonl" to use local calibration prompts.
SAVE_DIR = "Qwen3.6-27B-IterSmooth-W8A8"
NUM_CALIBRATION_SAMPLES = 128
MAX_SEQUENCE_LENGTH = 2048
ALPHA = 0.9

if NUM_CALIBRATION_SAMPLES < 1 or MAX_SEQUENCE_LENGTH < 1:
    raise ValueError("NUM_CALIBRATION_SAMPLES and MAX_SEQUENCE_LENGTH must be positive")
if not 0 <= ALPHA <= 1:
    raise ValueError("ALPHA must be in [0, 1]")
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

# Prepare calibration data using the model's chat template.
processor = AutoProcessor.from_pretrained(MODEL_ID)
tokenizer = processor.tokenizer
if CALIBRATION_DATA is None:
    dataset = load_dataset(DATASET_ID, split="train")
    # Use the raw user prompt, rather than the dataset's preformatted text.
    dataset = dataset.map(lambda sample: {"text": sample["messages"][0]["content"]})
else:
    dataset = load_dataset("json", data_files=CALIBRATION_DATA, split="train")
if "text" not in dataset.column_names or len(dataset) == 0:
    raise ValueError("Calibration data must be nonempty and contain a text column")
dataset = dataset.shuffle(seed=42).select(
    range(min(NUM_CALIBRATION_SAMPLES, len(dataset)))
)


def tokenize(sample):
    prompt = sample["text"]
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Each calibration text must be a nonempty user prompt")
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
    )


dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

# Keep weights on CPU; the native offload lifecycle loads them as needed.
# Qwen3.6-27B uses the Qwen3.5 architecture and its standard Transformers class.
# See https://huggingface.co/Qwen/Qwen3.6-27B/blob/main/config.json.
with load_context(Qwen3_5ForConditionalGeneration):
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="cpu",
        attn_implementation="eager",
    )
model.eval()

# Smooth norm -> Q/K/V (full-attention blocks) and norm -> gate/up (all blocks).
# Quantize full-attention and MLP projections; linear attention stays floating point.
recipe = [
    IterSmoothModifier(alpha=ALPHA, enable_subgraph_type=["norm-linear"]),
    QuantizationModifier(
        targets=[
            r"re:.*self_attn\.(q_proj|k_proj|v_proj|o_proj)$",
            r"re:.*mlp\.(gate_proj|up_proj|down_proj)$",
        ],
        scheme="W8A8",
    ),
]

# Run smoothing first, then derive weight scales from the smoothed weights.
# Dynamic W8A8 uses a data-free second stage, without another forward pass.
# Qwen3.6-27B's text decoder layers are Qwen3_5DecoderLayer; target them explicitly
# to keep the vision tower out of the calibration subgraphs.
oneshot(
    model=model,
    processor=processor,
    dataset=dataset,
    recipe=recipe,
    pipeline="independent",
    sequential_targets=["Qwen3_5DecoderLayer"],
    sequential_targets_per_subgraph=1,
    sequential_offload_device="cpu",
    batch_size=1,
    num_calibration_samples=len(dataset),
    max_seq_length=MAX_SEQUENCE_LENGTH,
    save_compressed=True,
    output_dir=SAVE_DIR,
)
