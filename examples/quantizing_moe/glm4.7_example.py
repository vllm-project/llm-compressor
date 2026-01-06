import os
from pathlib import Path

from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier, AWQMapping
from llmcompressor.modeling.glm4_moe import CalibrationGlm4MoeMoE

# This script does W4A16 AWQ quantization of the GLM-4.7 model.  It uses Group Size of 32 and two datasets (one specific for quantization and one for reasoning models)
# Running this script on an RTX PRO 6000 Workstation cards sees up to 40GB of VRAM used and roughly ~3.5 hours of run time.
# This model script uses the glm4 modeling file to make sure that for each calibration sample, all experts are engaged.
# This script also uses a local .ENV file, for Source and Destination.  Change as needed.
# GLM 4.7 has Dense layers for the first three layers, so we skip multiple sections of those layers.  We then need to add all of that to a mapping, to apply it during quantization.


# =========================
# Load ENV Variables
# =========================
from dotenv import load_dotenv

# Load the .env that sits next to this script (works regardless of where you run it)
# The .env file should be in the directory this script is run from and should look like the following:
# SRC_DIR=/media/fmodels/zai-org/GLM-4.7/
# DST_DIR=/media/fmodels/TheHouseOfTheDude/GLM-4.7_Compressed-Tensors/W4A16_GS32
# Those two lines are all that's needed.
load_dotenv(Path(__file__).with_name(".env"))

def require_env(key: str) -> str:
    val = os.getenv(key)
    if not val or not val.strip():
        raise RuntimeError(f"Missing environment variable: {key}")
    return val.strip()

SRC_DIR = require_env("SRC_DIR")
DST_DIR = require_env("DST_DIR")

# =========================
# Model (GLM / GLM-MoE)
# =========================
MODEL_ID = require_env("SRC_DIR")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# =========================
# Calibration data (Neural Magic + Rombo Optimized Reasoning)
# =========================
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# Calculate sample distribution: 60% Neural Magic, 40% Rombo
NUM_NEURALMAGIC = int(NUM_CALIBRATION_SAMPLES * 0.6)  # ~307 samples
NUM_ROMBO = NUM_CALIBRATION_SAMPLES - NUM_NEURALMAGIC  # ~205 samples

print(f"Loading calibration datasets: {NUM_NEURALMAGIC} from Neural Magic, {NUM_ROMBO} from Rombo")

# Load Neural Magic dataset
neuralmagic_dataset_id = "neuralmagic/LLM_compression_calibration"
neuralmagic_split = "train"
ds_neuralmagic = load_dataset(neuralmagic_dataset_id, split=neuralmagic_split)

# Sample from Neural Magic dataset
n_nm = min(NUM_NEURALMAGIC, len(ds_neuralmagic))
ds_neuralmagic = ds_neuralmagic.shuffle(seed=42).select(range(n_nm))

# Render messages to chat-style text (batch)
# The neuralmagic dataset has "messages" field with user/assistant roles
def preprocess_neuralmagic(batch):
    rendered = []
    for messages in batch["messages"]:
        # Apply chat template to the messages directly
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        rendered.append(text)
    return {"text": rendered}

ds_neuralmagic = ds_neuralmagic.map(preprocess_neuralmagic, batched=True, num_proc=4)

# Load Rombo Optimized Reasoning dataset
rombo_dataset_id = "Rombo-Org/Optimized_Reasoning"
rombo_split = "train"
ds_rombo = load_dataset(rombo_dataset_id, split=rombo_split)

# Sample from Rombo dataset
n_rombo = min(NUM_ROMBO, len(ds_rombo))
ds_rombo = ds_rombo.shuffle(seed=43).select(range(n_rombo))

# Preprocess Rombo dataset
# Format: {"instruction": "", "input": [""], "output": [""]}
def preprocess_rombo(batch):
    rendered = []
    for instruction, inputs, outputs in zip(batch["instruction"], batch["input"], batch["output"]):
        # Construct text from instruction, input, and output
        # Combine instruction with all input/output pairs
        text_parts = [instruction]
        
        # Handle input array (may contain multiple items)
        if isinstance(inputs, list) and len(inputs) > 0:
            for inp in inputs:
                if inp and inp.strip():
                    text_parts.append(f"\n\nInput: {inp}")
        
        # Handle output array (may contain multiple items)
        if isinstance(outputs, list) and len(outputs) > 0:
            for out in outputs:
                if out and out.strip():
                    text_parts.append(f"\n\nOutput: {out}")
        
        # Join all parts
        text = "".join(text_parts)
        rendered.append(text)
    return {"text": rendered}

ds_rombo = ds_rombo.map(preprocess_rombo, batched=True, num_proc=4)

# Combine both datasets
ds = concatenate_datasets([ds_neuralmagic, ds_rombo])

# Shuffle the combined dataset
ds = ds.shuffle(seed=44)

# Tokenize in batches
ds = ds.map(
    lambda batch: tokenizer(
        batch["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    ),
    batched=True,
    remove_columns=ds.column_names,
    num_proc=4,
)

print(f"Combined calibration dataset: {len(ds)} samples")

# =========================
# AWQ recipe with config_groups
#  - Weight-only INT4 (W4A16 **symmetric**)
#  - group_size: 32
#  - IMPORTANT: do NOT ignore mlp.gate / gate_up_proj (merged layer)
#  - Keep router and output head unquantized
# =========================

moe_ignores = [
    # Layers 0-2: Dense layer - ignore attention and MLP
    "model.layers.[0-2].self_attn.(q|k|v|o)_proj",
    "model.layers.[0-2].mlp.(gate|up|down)_proj",

    # Layers 3-91: MoE layers - ignore shared_experts
    "model.layers.3.mlp.shared_experts.gate_proj",
    "model.layers.3.mlp.shared_experts.up_proj",
    "model.layers.3.mlp.shared_experts.down_proj",
    "model.layers.4.mlp.shared_experts.gate_proj",
    "model.layers.4.mlp.shared_experts.up_proj",
    "model.layers.4.mlp.shared_experts.down_proj",
    "model.layers.5.mlp.shared_experts.gate_proj",
    "model.layers.5.mlp.shared_experts.up_proj",
    "model.layers.5.mlp.shared_experts.down_proj",
    "model.layers.6.mlp.shared_experts.gate_proj",
    "model.layers.6.mlp.shared_experts.up_proj",
    "model.layers.6.mlp.shared_experts.down_proj",
    "model.layers.7.mlp.shared_experts.gate_proj",
    "model.layers.7.mlp.shared_experts.up_proj",
    "model.layers.7.mlp.shared_experts.down_proj",
    "model.layers.8.mlp.shared_experts.gate_proj",
    "model.layers.8.mlp.shared_experts.up_proj",
    "model.layers.8.mlp.shared_experts.down_proj",
    "model.layers.9.mlp.shared_experts.gate_proj",
    "model.layers.9.mlp.shared_experts.up_proj",
    "model.layers.9.mlp.shared_experts.down_proj",
    "model.layers.10.mlp.shared_experts.gate_proj",
    "model.layers.10.mlp.shared_experts.up_proj",
    "model.layers.10.mlp.shared_experts.down_proj",
    "model.layers.11.mlp.shared_experts.gate_proj",
    "model.layers.11.mlp.shared_experts.up_proj",
    "model.layers.11.mlp.shared_experts.down_proj",
    "model.layers.12.mlp.shared_experts.gate_proj",
    "model.layers.12.mlp.shared_experts.up_proj",
    "model.layers.12.mlp.shared_experts.down_proj",
    "model.layers.13.mlp.shared_experts.gate_proj",
    "model.layers.13.mlp.shared_experts.up_proj",
    "model.layers.13.mlp.shared_experts.down_proj",
    "model.layers.14.mlp.shared_experts.gate_proj",
    "model.layers.14.mlp.shared_experts.up_proj",
    "model.layers.14.mlp.shared_experts.down_proj",
    "model.layers.15.mlp.shared_experts.gate_proj",
    "model.layers.15.mlp.shared_experts.up_proj",
    "model.layers.15.mlp.shared_experts.down_proj",
    "model.layers.16.mlp.shared_experts.gate_proj",
    "model.layers.16.mlp.shared_experts.up_proj",
    "model.layers.16.mlp.shared_experts.down_proj",
    "model.layers.17.mlp.shared_experts.gate_proj",
    "model.layers.17.mlp.shared_experts.up_proj",
    "model.layers.17.mlp.shared_experts.down_proj",
    "model.layers.18.mlp.shared_experts.gate_proj",
    "model.layers.18.mlp.shared_experts.up_proj",
    "model.layers.18.mlp.shared_experts.down_proj",
    "model.layers.19.mlp.shared_experts.gate_proj",
    "model.layers.19.mlp.shared_experts.up_proj",
    "model.layers.19.mlp.shared_experts.down_proj",
    "model.layers.20.mlp.shared_experts.gate_proj",
    "model.layers.20.mlp.shared_experts.up_proj",
    "model.layers.20.mlp.shared_experts.down_proj",
    "model.layers.21.mlp.shared_experts.gate_proj",
    "model.layers.21.mlp.shared_experts.up_proj",
    "model.layers.21.mlp.shared_experts.down_proj",
    "model.layers.22.mlp.shared_experts.gate_proj",
    "model.layers.22.mlp.shared_experts.up_proj",
    "model.layers.22.mlp.shared_experts.down_proj",
    "model.layers.23.mlp.shared_experts.gate_proj",
    "model.layers.23.mlp.shared_experts.up_proj",
    "model.layers.23.mlp.shared_experts.down_proj",
    "model.layers.24.mlp.shared_experts.gate_proj",
    "model.layers.24.mlp.shared_experts.up_proj",
    "model.layers.24.mlp.shared_experts.down_proj",
    "model.layers.25.mlp.shared_experts.gate_proj",
    "model.layers.25.mlp.shared_experts.up_proj",
    "model.layers.25.mlp.shared_experts.down_proj",
    "model.layers.26.mlp.shared_experts.gate_proj",
    "model.layers.26.mlp.shared_experts.up_proj",
    "model.layers.26.mlp.shared_experts.down_proj",
    "model.layers.27.mlp.shared_experts.gate_proj",
    "model.layers.27.mlp.shared_experts.up_proj",
    "model.layers.27.mlp.shared_experts.down_proj",
    "model.layers.28.mlp.shared_experts.gate_proj",
    "model.layers.28.mlp.shared_experts.up_proj",
    "model.layers.28.mlp.shared_experts.down_proj",
    "model.layers.29.mlp.shared_experts.gate_proj",
    "model.layers.29.mlp.shared_experts.up_proj",
    "model.layers.29.mlp.shared_experts.down_proj",
    "model.layers.30.mlp.shared_experts.gate_proj",
    "model.layers.30.mlp.shared_experts.up_proj",
    "model.layers.30.mlp.shared_experts.down_proj",
    "model.layers.31.mlp.shared_experts.gate_proj",
    "model.layers.31.mlp.shared_experts.up_proj",
    "model.layers.31.mlp.shared_experts.down_proj",
    "model.layers.32.mlp.shared_experts.gate_proj",
    "model.layers.32.mlp.shared_experts.up_proj",
    "model.layers.32.mlp.shared_experts.down_proj",
    "model.layers.33.mlp.shared_experts.gate_proj",
    "model.layers.33.mlp.shared_experts.up_proj",
    "model.layers.33.mlp.shared_experts.down_proj",
    "model.layers.34.mlp.shared_experts.gate_proj",
    "model.layers.34.mlp.shared_experts.up_proj",
    "model.layers.34.mlp.shared_experts.down_proj",
    "model.layers.35.mlp.shared_experts.gate_proj",
    "model.layers.35.mlp.shared_experts.up_proj",
    "model.layers.35.mlp.shared_experts.down_proj",
    "model.layers.36.mlp.shared_experts.gate_proj",
    "model.layers.36.mlp.shared_experts.up_proj",
    "model.layers.36.mlp.shared_experts.down_proj",
    "model.layers.37.mlp.shared_experts.gate_proj",
    "model.layers.37.mlp.shared_experts.up_proj",
    "model.layers.37.mlp.shared_experts.down_proj",
    "model.layers.38.mlp.shared_experts.gate_proj",
    "model.layers.38.mlp.shared_experts.up_proj",
    "model.layers.38.mlp.shared_experts.down_proj",
    "model.layers.39.mlp.shared_experts.gate_proj",
    "model.layers.39.mlp.shared_experts.up_proj",
    "model.layers.39.mlp.shared_experts.down_proj",
    "model.layers.40.mlp.shared_experts.gate_proj",
    "model.layers.40.mlp.shared_experts.up_proj",
    "model.layers.40.mlp.shared_experts.down_proj",
    "model.layers.41.mlp.shared_experts.gate_proj",
    "model.layers.41.mlp.shared_experts.up_proj",
    "model.layers.41.mlp.shared_experts.down_proj",
    "model.layers.42.mlp.shared_experts.gate_proj",
    "model.layers.42.mlp.shared_experts.up_proj",
    "model.layers.42.mlp.shared_experts.down_proj",
    "model.layers.43.mlp.shared_experts.gate_proj",
    "model.layers.43.mlp.shared_experts.up_proj",
    "model.layers.43.mlp.shared_experts.down_proj",
    "model.layers.44.mlp.shared_experts.gate_proj",
    "model.layers.44.mlp.shared_experts.up_proj",
    "model.layers.44.mlp.shared_experts.down_proj",
    "model.layers.45.mlp.shared_experts.gate_proj",
    "model.layers.45.mlp.shared_experts.up_proj",
    "model.layers.45.mlp.shared_experts.down_proj",
    "model.layers.46.mlp.shared_experts.gate_proj",
    "model.layers.46.mlp.shared_experts.up_proj",
    "model.layers.46.mlp.shared_experts.down_proj",
    "model.layers.47.mlp.shared_experts.gate_proj",
    "model.layers.47.mlp.shared_experts.up_proj",
    "model.layers.47.mlp.shared_experts.down_proj",
    "model.layers.48.mlp.shared_experts.gate_proj",
    "model.layers.48.mlp.shared_experts.up_proj",
    "model.layers.48.mlp.shared_experts.down_proj",
    "model.layers.49.mlp.shared_experts.gate_proj",
    "model.layers.49.mlp.shared_experts.up_proj",
    "model.layers.49.mlp.shared_experts.down_proj",
    "model.layers.50.mlp.shared_experts.gate_proj",
    "model.layers.50.mlp.shared_experts.up_proj",
    "model.layers.50.mlp.shared_experts.down_proj",
    "model.layers.51.mlp.shared_experts.gate_proj",
    "model.layers.51.mlp.shared_experts.up_proj",
    "model.layers.51.mlp.shared_experts.down_proj",
    "model.layers.52.mlp.shared_experts.gate_proj",
    "model.layers.52.mlp.shared_experts.up_proj",
    "model.layers.52.mlp.shared_experts.down_proj",
    "model.layers.53.mlp.shared_experts.gate_proj",
    "model.layers.53.mlp.shared_experts.up_proj",
    "model.layers.53.mlp.shared_experts.down_proj",
    "model.layers.54.mlp.shared_experts.gate_proj",
    "model.layers.54.mlp.shared_experts.up_proj",
    "model.layers.54.mlp.shared_experts.down_proj",
    "model.layers.55.mlp.shared_experts.gate_proj",
    "model.layers.55.mlp.shared_experts.up_proj",
    "model.layers.55.mlp.shared_experts.down_proj",
    "model.layers.56.mlp.shared_experts.gate_proj",
    "model.layers.56.mlp.shared_experts.up_proj",
    "model.layers.56.mlp.shared_experts.down_proj",
    "model.layers.57.mlp.shared_experts.gate_proj",
    "model.layers.57.mlp.shared_experts.up_proj",
    "model.layers.57.mlp.shared_experts.down_proj",
    "model.layers.58.mlp.shared_experts.gate_proj",
    "model.layers.58.mlp.shared_experts.up_proj",
    "model.layers.58.mlp.shared_experts.down_proj",
    "model.layers.59.mlp.shared_experts.gate_proj",
    "model.layers.59.mlp.shared_experts.up_proj",
    "model.layers.59.mlp.shared_experts.down_proj",
    "model.layers.60.mlp.shared_experts.gate_proj",
    "model.layers.60.mlp.shared_experts.up_proj",
    "model.layers.60.mlp.shared_experts.down_proj",
    "model.layers.61.mlp.shared_experts.gate_proj",
    "model.layers.61.mlp.shared_experts.up_proj",
    "model.layers.61.mlp.shared_experts.down_proj",
    "model.layers.62.mlp.shared_experts.gate_proj",
    "model.layers.62.mlp.shared_experts.up_proj",
    "model.layers.62.mlp.shared_experts.down_proj",
    "model.layers.63.mlp.shared_experts.gate_proj",
    "model.layers.63.mlp.shared_experts.up_proj",
    "model.layers.63.mlp.shared_experts.down_proj",
    "model.layers.64.mlp.shared_experts.gate_proj",
    "model.layers.64.mlp.shared_experts.up_proj",
    "model.layers.64.mlp.shared_experts.down_proj",
    "model.layers.65.mlp.shared_experts.gate_proj",
    "model.layers.65.mlp.shared_experts.up_proj",
    "model.layers.65.mlp.shared_experts.down_proj",
    "model.layers.66.mlp.shared_experts.gate_proj",
    "model.layers.66.mlp.shared_experts.up_proj",
    "model.layers.66.mlp.shared_experts.down_proj",
    "model.layers.67.mlp.shared_experts.gate_proj",
    "model.layers.67.mlp.shared_experts.up_proj",
    "model.layers.67.mlp.shared_experts.down_proj",
    "model.layers.68.mlp.shared_experts.gate_proj",
    "model.layers.68.mlp.shared_experts.up_proj",
    "model.layers.68.mlp.shared_experts.down_proj",
    "model.layers.69.mlp.shared_experts.gate_proj",
    "model.layers.69.mlp.shared_experts.up_proj",
    "model.layers.69.mlp.shared_experts.down_proj",
    "model.layers.70.mlp.shared_experts.gate_proj",
    "model.layers.70.mlp.shared_experts.up_proj",
    "model.layers.70.mlp.shared_experts.down_proj",
    "model.layers.71.mlp.shared_experts.gate_proj",
    "model.layers.71.mlp.shared_experts.up_proj",
    "model.layers.71.mlp.shared_experts.down_proj",
    "model.layers.72.mlp.shared_experts.gate_proj",
    "model.layers.72.mlp.shared_experts.up_proj",
    "model.layers.72.mlp.shared_experts.down_proj",
    "model.layers.73.mlp.shared_experts.gate_proj",
    "model.layers.73.mlp.shared_experts.up_proj",
    "model.layers.73.mlp.shared_experts.down_proj",
    "model.layers.74.mlp.shared_experts.gate_proj",
    "model.layers.74.mlp.shared_experts.up_proj",
    "model.layers.74.mlp.shared_experts.down_proj",
    "model.layers.75.mlp.shared_experts.gate_proj",
    "model.layers.75.mlp.shared_experts.up_proj",
    "model.layers.75.mlp.shared_experts.down_proj",
    "model.layers.76.mlp.shared_experts.gate_proj",
    "model.layers.76.mlp.shared_experts.up_proj",
    "model.layers.76.mlp.shared_experts.down_proj",
    "model.layers.77.mlp.shared_experts.gate_proj",
    "model.layers.77.mlp.shared_experts.up_proj",
    "model.layers.77.mlp.shared_experts.down_proj",
    "model.layers.78.mlp.shared_experts.gate_proj",
    "model.layers.78.mlp.shared_experts.up_proj",
    "model.layers.78.mlp.shared_experts.down_proj",
    "model.layers.79.mlp.shared_experts.gate_proj",
    "model.layers.79.mlp.shared_experts.up_proj",
    "model.layers.79.mlp.shared_experts.down_proj",
    "model.layers.80.mlp.shared_experts.gate_proj",
    "model.layers.80.mlp.shared_experts.up_proj",
    "model.layers.80.mlp.shared_experts.down_proj",
    "model.layers.81.mlp.shared_experts.gate_proj",
    "model.layers.81.mlp.shared_experts.up_proj",
    "model.layers.81.mlp.shared_experts.down_proj",
    "model.layers.82.mlp.shared_experts.gate_proj",
    "model.layers.82.mlp.shared_experts.up_proj",
    "model.layers.82.mlp.shared_experts.down_proj",
    "model.layers.83.mlp.shared_experts.gate_proj",
    "model.layers.83.mlp.shared_experts.up_proj",
    "model.layers.83.mlp.shared_experts.down_proj",
    "model.layers.84.mlp.shared_experts.gate_proj",
    "model.layers.84.mlp.shared_experts.up_proj",
    "model.layers.84.mlp.shared_experts.down_proj",
    "model.layers.85.mlp.shared_experts.gate_proj",
    "model.layers.85.mlp.shared_experts.up_proj",
    "model.layers.85.mlp.shared_experts.down_proj",
    "model.layers.86.mlp.shared_experts.gate_proj",
    "model.layers.86.mlp.shared_experts.up_proj",
    "model.layers.86.mlp.shared_experts.down_proj",
    "model.layers.87.mlp.shared_experts.gate_proj",
    "model.layers.87.mlp.shared_experts.up_proj",
    "model.layers.87.mlp.shared_experts.down_proj",
    "model.layers.88.mlp.shared_experts.gate_proj",
    "model.layers.88.mlp.shared_experts.up_proj",
    "model.layers.88.mlp.shared_experts.down_proj",
    "model.layers.89.mlp.shared_experts.gate_proj",
    "model.layers.89.mlp.shared_experts.up_proj",
    "model.layers.89.mlp.shared_experts.down_proj",
    "model.layers.90.mlp.shared_experts.gate_proj",
    "model.layers.90.mlp.shared_experts.up_proj",
    "model.layers.90.mlp.shared_experts.down_proj",
    "model.layers.91.mlp.shared_experts.gate_proj",
    "model.layers.91.mlp.shared_experts.up_proj",
    "model.layers.91.mlp.shared_experts.down_proj",

    # Ignore the output head
    "lm_head",
]

# Create explicit mappings that skip layers 0-2
mappings = []
for layer_idx in range(3, 92):  # Skip dense layers 0-2
    mappings.append(
        AWQMapping(
            smooth_layer=f"model.layers.{layer_idx}.input_layernorm",
            balance_layers=[
                f"model.layers.{layer_idx}.self_attn.q_proj",
                f"model.layers.{layer_idx}.self_attn.k_proj",
                f"model.layers.{layer_idx}.self_attn.v_proj",
            ]
        )
    )

recipe = [
    AWQModifier(
        ignore=moe_ignores,
        mappings=mappings,  # Provide explicit mappings
        config_groups={
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "symmetric": True,   # W4A16 (symmetric)
                    "strategy": "group",
                    "group_size": 32,
                    "dynamic": False,
                },
            },
        },
    ),
]

# =========================
# Quantize + save (writes quantization_config for vLLM)
# =========================
SAVE_DIR = require_env("DST_DIR")

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
#    output_dir=SAVE_DIR,
)

# Fix generation config validation issue before saving
if hasattr(model, 'generation_config') and model.generation_config is not None:
    # If temperature is set but do_sample is False, either enable do_sample or remove temperature
    if hasattr(model.generation_config, 'temperature') and model.generation_config.temperature is not None:
        if not getattr(model.generation_config, 'do_sample', False):
            # Set do_sample=True to make temperature valid, or remove temperature
            model.generation_config.do_sample = True

# (Optional redundant save)
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

print("Saved to:", SAVE_DIR)

