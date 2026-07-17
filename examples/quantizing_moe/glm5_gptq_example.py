from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modeling.moe.linearize import linearize_moe
from llmcompressor.modifiers.quantization import GPTQModifier

# Select model and load it.
# load_context() fast-path fails for glm_moe_dsa (inherits glm4_moe checkpoint
# conversion mappings, which try to set gate_up_proj on LinearExperts2D during
# weight loading). Load normally then linearize explicitly.
model_id = "zai-org/GLM-5.2"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
linearize_moe(model)
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Select calibration dataset.
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))


def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
        )
    }


ds = ds.map(preprocess)


def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding="max_length",
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


ds = ds.map(tokenize, remove_columns=ds.column_names)

# Layers 0-2 are dense; skip them and the output head.
ignore = [
    "re:model.layers.0.*",
    "re:model.layers.1.*",
    "re:model.layers.2.*",
    "lm_head",
    "re:.*mlp.gate$",
]

recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=ignore)

# Two-target strategy:
#   - "GlmMoeDsaAttention": keeps GlmMoeDsaIndexer (data-dependent top-k control flow)
#     inside a leaf so the fx tracer never enters it.
#   - "ExpertMLP": each linearized expert is its own subgraph boundary so only one
#     expert's Hessian is resident in GPU memory at a time.
#
# sequential_targets_per_subgraph batches multiple ExpertMLP modules per subgraph,
# balancing memory usage against calibration runtime.
# Value = num_experts_per_group // batch_size + buffer  (384 // 4 + 10 = 106).
num_experts = getattr(model.config, "n_routed_experts", 384)
oneshot(
    model=model,
    dataset=ds,
    batch_size=4,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    sequential_targets=["GlmMoeDsaAttention", "ExpertMLP"],
    sequential_targets_per_subgraph=(num_experts // 4 + 10),
)

SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-W4A16-G128"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
