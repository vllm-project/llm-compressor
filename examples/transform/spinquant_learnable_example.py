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

# Keep the HF cache on /local when this box has it (the NFS home is quota'd at ~100GB),
# but fall back to the default ~/.cache/huggingface elsewhere so the script is portable
# across machines. Must be set before the transformers/huggingface_hub imports below.
_LOCAL_HF_HOME = "/local/models/ruhui/hf_cache"
_USER_HF_HOME = os.path.expanduser("~/.cache/huggingface")
os.environ.setdefault(
    "HF_HOME", _LOCAL_HF_HOME if os.path.isdir(_LOCAL_HF_HOME) else _USER_HF_HOME
)

# huggingface_hub derives HF_TOKEN_PATH from HF_HOME, so redirecting the cache to
# /local hides ~/.cache/huggingface/token and every gated repo (Llama, etc.) 401s even
# when you are logged in. Point the token path at whichever location actually holds a
# token, preferring the private home copy -- do NOT copy the token onto /local, which
# is world-readable on a shared node. An explicit HF_TOKEN env var still wins over this.
for _token_path in (
    os.path.join(_USER_HF_HOME, "token"),
    os.path.join(_LOCAL_HF_HOME, "token"),
):
    if os.path.isfile(_token_path):
        os.environ.setdefault("HF_TOKEN_PATH", _token_path)
        break

# Checkpoints land next to the HF cache on /local when this box has it, otherwise in
# ~/models, so the script is portable across machines and does not dump a multi-GB
# checkpoint into whatever directory it happens to be launched from. Override with
# OUTPUT_ROOT=...
_LOCAL_OUTPUT_ROOT = "/local/models/ruhui"
OUTPUT_ROOT = os.environ.get(
    "OUTPUT_ROOT",
    _LOCAL_OUTPUT_ROOT
    if os.path.isdir(_LOCAL_OUTPUT_ROOT)
    else os.path.expanduser("~/models"),
)

from compressed_tensors.offload import dispatch_model, remove_dispatch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform import SpinQuantModifier
from llmcompressor.utils.dev import get_main_device

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
    tokenized = tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )
    # Learning the rotations is a gradient descent on the language modeling loss, so the
    # batches must carry `labels` -- unlike the purely observational calibration the
    # other examples do. With labels present the model computes `outputs.loss` itself
    # (standard causal-LM shift-by-one). Without them `outputs.loss` is None and
    # SpinQuantModifier._learn_rotations falls through to a branch that itself reads
    # batch["labels"], raising KeyError: 'labels'.
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


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

# --- consolidate onto one device before saving ---
# `dispatch_model` above spreads the model across every visible GPU, which lands
# embed_tokens and lm_head on different devices. save_pretrained -> _retie_embeddings
# then compares them with torch.equal, which requires a common device and raises
# "Expected all tensors to be on the same device".
#
# Order matters: `model.to(...)` on a dispatched model is silently swallowed by the
# offload hooks and moves nothing, so the dispatch must be removed FIRST. Verified:
#   .to() alone                  -> params stay on cuda:0..3
#   remove_dispatch() alone      -> params stay on cuda:0..3
#   remove_dispatch() then .to() -> all params on one device
remove_dispatch(model, onload_tensors=True)
model.to(get_main_device())

# --- save compressed (offline rotations are already fused into the weights) ---
SAVE_DIR = os.path.join(
    OUTPUT_ROOT,
    MODEL_ID.rstrip("/").split("/")[-1]
    + f"-spinquant-learn{''.join(ROTATIONS)}-{SCHEME.lower()}",
)
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
print(f"\nSaved to {SAVE_DIR}")
