from compressed_tensors.quantization.quant_scheme import (
    FP8_BLOCK,
    NVFP4,
    QuantizationScheme,
)
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4PreTrainedModel,
)

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context

# Upstream BUG: norms should be loaded in float32, but usually aren't due to the base
# model having a quant_config which overrides this. Loading in float32 actually
# breaks the model definition (it expects bfloat16). Let's force load in bfloat16.
DeepseekV4PreTrainedModel._keep_in_fp32_modules_strict = set()

# Select model and load it.
# MODEL_ID = "RedHatAI/DeepSeek-V4-Flash-BF16"
MODEL_ID = "/mnt/nvme-data/engine/kylesayrs/DeepSeek-V4-Pro-BF16"
# MODEL_ID = "RedHatAI/DeepSeek-V4-Flash-BF16"

with load_context():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto_offload",
        max_memory={},
        offload_folder="/mnt/nvme-data/engine/kylesayrs/offload_folder",
    )

# kluge for the way I saved the decompressed checkpoint
# mds = model.model.layers[-1].self_attn.wq_a._hf_hook.weights_map.dataset.index
# mds["model.hc_head.base"] = mds['model.hc_head.hc_base']
# mds["model.hc_head.fn"] = mds['model.hc_head.hc_fn']
# mds["model.hc_head.scale"] = mds['model.hc_head.hc_scale']

tokenizer = AutoTokenizer.from_pretrained("RedHatAI/DeepSeek-V4-Flash-BF16")

# Select calibration dataset.
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"

# Select number of samples. 512 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# Load dataset and preprocess.
ds = load_dataset(
    DATASET_ID,
    split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]",
)
ds = ds.shuffle(seed=42)


def preprocess(example):
    # DeepSeek-V4 does not have a traditional chat template.
    # Encode manually per https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/tree/main/encoding
    BOS = "<｜begin▁of▁sentence｜>"
    EOS = "<｜end▁of▁sentence｜>"
    text = BOS
    for message in example["messages"]:
        role = message["role"]
        content = message["content"]
        if role == "system":
            text += content
        elif role == "user":
            text += f"<｜User｜>{content}"
        elif role == "assistant":
            text += f"<｜Assistant｜></think>{content}{EOS}"

    return {"text": text}


ds = ds.map(preprocess)


def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


ds = ds.map(tokenize, remove_columns=ds.column_names)

# Configure the quantization algorithm to run.
#   * quantize mlp/expert weights to NVFP4
#   * quantize attention projection weights to FP8_BLOCK
# model.model.layers.0.self_attn.q_a_proj
#
# wq_a  | q_a_proj
# wq_b  | q_b_proj
# wkv   | kv_proj
# wo_a  | o_a_proj
# wo_b  | o_b_proj

recipe = QuantizationModifier(
    config_groups={
        "attention": QuantizationScheme(
            targets=[
                r"re:.*attn\.(q_a_proj|q_b_proj|kv_proj|o_a_proj|o_b_proj)$",
                r"re:.*attn\.compressor\.indexer\.q_b_proj$",
            ],
            **FP8_BLOCK,
        ),
        "experts": QuantizationScheme(
            targets=[
                r"re:.*mlp\..*(gate|up|down)_proj$",
            ],
            **NVFP4,
        ),
    },
    ignore=[],
)

# Apply algorithms.
# due to the large size of DeepSeek-V4, we specify sequential targets such that
# only one block is loaded into GPU memory at a time
oneshot(
    model=model,
    processor=tokenizer,
    dataset=ds,
    # recipe=recipe,
    pipeline="sequential",
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    sequential_targets=["DeepseekV4DecoderLayer"],
    batch_size=1,
    shuffle_calibration_samples=True,
    propagate_error=False,  # work around reliance on transformers cache
    # something weird happens with the cache and propagation
)

# Save to disk compressed.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4-FP8-BLOCK"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
