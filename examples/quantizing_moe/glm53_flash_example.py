import torch
from compressed_tensors.offload import init_dist
from transformers import AutoTokenizer, Glm5NextForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context

# torchrun --nproc-per-node N ...
init_dist()

# Load the model. Swap to "RedHatAI/GLM-5.3-Flash-BF16" after confirming small model
MODEL_ID = "inference-optimization/GLM-5.3-Flash-0.1B-A0.1B"
with load_context(Glm5NextForConditionalGeneration):
    # GLM-5.3-Flash is a vision-language MoE model, so it must be loaded with its
    # `Glm5NextForConditionalGeneration` class (not `AutoModelForCausalLM`) in order
    # to keep the vision tower.
    model = Glm5NextForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto_offload",
    )
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Configure the quantization algorithm to run.
recipe = QuantizationModifier(
    scheme="NVFP4",
    targets=[r"re:.*mlp\.experts\..*(gate|up|down)_proj$"],
    ignore=[
        r"re:.*visual.*",  # vision tower stays full precision
        "lm_head",
        r"re:.*mlp\.gate$",  # MoE router
        r"re:.*self_attn\.indexer\..*",  # sensitive to quantization
    ],
)

# Apply algorithms.
oneshot(
    model=model,
    processor=tokenizer,
    recipe=recipe,
    dataset="perfectblend",
    splits="train[:512]",
    max_seq_length=2048,
    num_calibration_samples=512,
)

# Save to disk compressed. MTP tensors (not built by transformers) are copied
# over automatically by the save utility.
model.generation_config.top_p = None
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

torch.distributed.destroy_process_group()
