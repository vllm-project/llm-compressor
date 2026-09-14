# NOTE: to use a custom dataset, see examples/custom_dataset_example.py
from compressed_tensors.offload import init_dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.pruning import REAPPruningModifier
from llmcompressor.utils import load_context

init_dist()
MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"

with load_context():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto_offload",
        max_memory={},
        offload_folder="offload_folder",
    )
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Prune 25% of the experts in each MoE layer, based on saliency, then quantize.
recipe = [
    REAPPruningModifier(sparsity=0.25),
    GPTQModifier(
        targets="Linear",
        scheme="NVFP4",
        ignore=[
            r"re:.*conv1d.*",
            r"backbone\.embeddings",
            r"re:.*_latent_proj.*",  # sensitive to quantization
            r"re:.*mixer.gate\..*",
            r"re:mtp.layers.*",
            "backbone.norm_f",
            "lm_head",
        ],
    ),
]

oneshot(
    model=model,
    dataset="perfectblend",
    splits="train[:512]",
    recipe=recipe,
    max_seq_length=2048,
    num_calibration_samples=1024,
    batch_size=16,
    pipeline="sequential",
)

SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4-REAP-25"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
