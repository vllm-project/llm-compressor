# from compressed_tensors.entrypoints.convert import (
#     ModelOptNvfp4Converter,
# )
# from compressed_tensors.quantization import (
#     QuantizationScheme,
# )
# from compressed_tensors.quantization.quant_scheme import FP8_BLOCK

# from llmcompressor import model_free_ptq

# MODEL_ID = "thinkingmachines/Inkling"
# SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-FP8-BLOCK"

# # Convert modelopt NVFP4 format to compressed-tensors format and
# # apply FP8-Block to the model's compatible self_attn Linear layers
# # Once quantized, the model is saved to SAVE_DIR.
# model_free_ptq(
#     model_stub=MODEL_ID,
#     save_directory=SAVE_DIR,
#     scheme=QuantizationScheme(
#         **FP8_BLOCK,
#         targets=[
#             "Linear",
#         ],
#     ),
#     ignore=["re:.*sconv$"],
#     max_workers=8,
#     device=["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
# )


from transformers import AutoProcessor, InklingForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context

MODEL_ID = "thinkingmachines/Inkling"
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-FP8-BLOCK"

# Select model and load it in the `load_context` context
with load_context(InklingForConditionalGeneration):
    model = InklingForConditionalGeneration.from_pretrained(
        MODEL_ID,
        max_memory={"cpu": 500e9},
        device_map="auto_offload",  # fit as much as possible on cpu, rest goes on disk
        offload_folder="./offload_folder",
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)

# Configure the quantization algorithm to run.
#   * quantize the weights to NVFP4
recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_BLOCK",
    ignore=[
        "lm_head",
        "model.llm.unembed",
        "model.llm.embed",
        "re:.*sconv$",
        "re:.*norm.*",
        "re:.*bias$",
        "re:.*gate.*",
        "re:.*global_scale$",
        "re:.*shared_experts.*",
        "re:.*visual.*",
        "re:.*audio.*",
    ],
)

# Apply algorithms.
oneshot(
    model=model,
    processor=processor,
    recipe=recipe,
)

# Save to disk compressed.
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)
