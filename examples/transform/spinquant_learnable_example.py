"""Learn SpinQuant rotations (the "learned rotations" contribution of SpinQuant).

This example learns the rotation matrices via AdamW/SGD over a calibration set, per
"SpinQuant: LLM quantization with learned rotations" (https://arxiv.org/abs/2405.16406),
instead of using fixed Hadamard rotations (QuaRot-style, which is what
``spinquant_example.py`` demonstrates).

Pipeline:

  1. Apply R1 and R2 rotations (offline, fused into weights at runtime).
  2. LEARN the rotation matrices to minimize the language modeling loss over a
     calibration set. Only the rotations are trained; the model is frozen.
  3. Fuse the learned rotations back into the weights.
  4. Quantize the (now rotated) weights with a QuantizationModifier.

Rotations are applied, learned, and fused ONE AT A TIME internally, because the
compressed-tensors transform factories cannot compose multiple parametrized
(requires_grad) transforms on the same Linear (R1 and R2 both target attn_v/attn_o).
``learn_steps`` is split evenly across the requested rotations.

Env vars (all optional):

  MODEL_ID=meta-llama/Meta-Llama-3-8B-Instruct
  ROTATIONS=R1,R2            rotations to apply and learn
  SCHEME=W4A16               quantization scheme for the QuantizationModifier
  LEARN_STEPS=100            total gradient steps, split across rotations
  LEARN_LR=1e-3              learning rate
  LEARN_OPTIMIZER=adamw      adamw | sgd (SGD uses momentum 0.9)
  NUM_CALIBRATION_SAMPLES=512
  MAX_SEQUENCE_LENGTH=2048

Run (on the GPU box):

  source /home/rzhao/envs/vllm_compress/vllm_compress/bin/activate
  python -u spinquant_learnable_example.py 2>&1 \
    | tee /home/rzhao/vllm_exp/spinquant_learn.log
"""

import os

# Keep the HF cache on /local rather than a quota'd NFS home. Must be set before the
# transformers/huggingface_hub imports below.
os.environ.setdefault("HF_HOME", "/local/models/ruhui/hf_cache")

from compressed_tensors.offload import dispatch_model
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform import SpinQuantModifier

# --- configuration (env-var overridable) ---
MODEL_ID = os.environ.get("MODEL_ID", "meta-llama/Meta-Llama-3-8B-Instruct")
ROTATIONS = os.environ.get("ROTATIONS", "R1,R2").split(",")
SCHEME = os.environ.get("SCHEME", "W4A16")
LEARN_STEPS = int(os.environ.get("LEARN_STEPS", "100"))
LEARN_LR = float(os.environ.get("LEARN_LR", "1e-3"))
LEARN_OPTIMIZER = os.environ.get("LEARN_OPTIMIZER", "adamw")
NUM_CALIBRATION_SAMPLES = int(os.environ.get("NUM_CALIBRATION_SAMPLES", "512"))
MAX_SEQUENCE_LENGTH = int(os.environ.get("MAX_SEQUENCE_LENGTH", "2048"))

print(
    f"SpinQuant(learnable) | model={MODEL_ID} rotations={ROTATIONS} "
    f"scheme={SCHEME} steps={LEARN_STEPS} lr={LEARN_LR} opt={LEARN_OPTIMIZER}"
)

# --- model + tokenizer ---
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# --- calibration data (needed to LEARN the rotations) ---
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"

ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)


def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(example["messages"], tokenize=False)
    }


def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


ds = ds.map(preprocess)
ds = ds.map(tokenize, remove_columns=ds.column_names)

# --- recipe: learn rotations, then quantize the rotated weights ---
recipe = [
    SpinQuantModifier(
        rotations=ROTATIONS,
        learnable=True,
        learn_steps=LEARN_STEPS,
        learn_lr=LEARN_LR,
        learn_optimizer=LEARN_OPTIMIZER,
    ),
    QuantizationModifier(
        targets="Linear",
        scheme=SCHEME,
        ignore=["lm_head", "re:.*mlp.gate$"],
    ),
]

# The `independent` (default) pipeline runs each modifier with its own sub-pipeline:
# SpinQuant gets a data-free pass (where the rotation training happens in
# on_calibration_start) and QuantizationModifier gets its own. If you pass
# `pipeline="sequential"`, the whole recipe shares one sequential run instead.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# --- sanity generation ---
print("\n\n")
print("========== SAMPLE GENERATION ==============")
dispatch_model(model)
sample = tokenizer("Hello my name is", return_tensors="pt")
sample = {key: value.to(model.device) for key, value in sample.items()}
output = model.generate(**sample, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")

# --- save compressed (offline rotations are already fused into the weights) ---
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + (
    f"-spinquant-learn{''.join(ROTATIONS)}-{SCHEME.lower()}"
)
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
print(f"\nSaved to {SAVE_DIR}")
