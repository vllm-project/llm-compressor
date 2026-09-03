# NOTE: to use a custom dataset, see examples/custom_dataset_example.py
from compressed_tensors.offload import init_dist
from compressed_tensors.quantization.quant_scheme import (
    NVFP4,
    QuantizationScheme,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.pruning import REAPPruningModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context

init_dist()
MODEL_ID = "tencent/Hy3"

with load_context():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto_offload",
        max_memory={},
        offload_folder="offload_folder",
    )
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

recipe = [
    REAPPruningModifier(sparsity=0.25),
    QuantizationModifier(
        config_groups={
            "experts": QuantizationScheme(
                targets=[
                    r"re:.*mlp.*$",
                ],
                **NVFP4,
            ),
        },
        ignore=[
            "re:.*lm_head",
            "re:.*mlp.gate$",
        ],
    ),
]

oneshot(
    model=model,
    dataset="perfectblend",
    recipe=recipe,
    max_seq_length=2048,
    num_calibration_samples=1024,
    batch_size=1,
    shuffle_calibration_samples=True,
    propagate_error=False,
)

SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4-REAP-25"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
