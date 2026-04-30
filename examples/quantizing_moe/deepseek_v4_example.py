from compressed_tensors.quantization.quant_scheme import (
    FP8_BLOCK,
    NVFP4,
    QuantizationScheme,
)
from compressed_tensors.offload import load_offloaded_model, init_dist
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.datasets.utils import get_rank_partition
from llmcompressor.modeling.moe.linearize import linearize_moe_model
from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier

# Select model and load it.
#MODEL_ID = "/mnt/nfs-preprod-1/engine/kylesayrs/DeepSeek-V4-Flash-bf16-dequantized"
#MODEL_ID = "/mnt/nfs-preprod-1/engine/kylesayrs/DeepSeek-V4-Flash-bf16-dequantized-5layers"
MODEL_ID = "DeepSeek-V4-Flash-bf16"

init_dist()
with load_offloaded_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map="auto_offload",
        #device_map="cpu",
        max_memory={"cpu": 3e10},
        offload_folder="offload_folder",
    )
linearize_moe_model(model)

# kluge for the way I saved the decompressed checkpoint
# mds = model.model.layers[-1].self_attn.wq_a._hf_hook.weights_map.dataset.index
# mds["model.hc_head.base"] = mds['model.hc_head.hc_base']
# mds["model.hc_head.fn"] = mds['model.hc_head.hc_fn']
# mds["model.hc_head.scale"] = mds['model.hc_head.hc_scale']

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Select calibration dataset.
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"

# Select number of samples. 512 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 1024
MAX_SEQUENCE_LENGTH = 512

# Load dataset and preprocess.
ds = load_dataset(
    DATASET_ID, split=get_rank_partition(DATASET_SPLIT, NUM_CALIBRATION_SAMPLES)
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
recipe = GPTQModifier(
    config_groups={
        "attention": QuantizationScheme(
            targets=[
                r"re:model.*attn.*(wkv|wo_a|wo_b|wq_a|wq_b)$",
                r"re:model.*attn\.compressor.*(wgate|wkv)$",
            ],
            **FP8_BLOCK,
        ),
        "experts": QuantizationScheme(
            targets=[
                r"re:model.*mlp.*(gate|up|down)_proj$",
            ],
            **NVFP4,
        ),
    },
    ignore=[
        "lm_head",
        #r"re:model.*self_attn.*"
        #r"re:model.*ffn_hc$"
    ],
)
# model.layers.4.self_attn.compressor.indexer.weights_proj
# model.layers.3.ffn_hc

# Apply algorithms.
# due to the large size of DeepSeek-V4, we specify sequential targets such that
# only one block is loaded into GPU memory at a time
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    sequential_targets=["DeepseekV4DecoderLayer"],
    batch_size=32,
)

# Save to disk compressed.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4-FP8-BLOCK-GPTQ"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
