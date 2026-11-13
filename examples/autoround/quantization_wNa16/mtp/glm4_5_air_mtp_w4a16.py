"""Quantize GLM-4.5-Air and its MTP layer to W4A16 with AutoRound."""

import torch
from auto_round.calib_dataset import get_dataset
from compressed_tensors.offload import set_onload_device
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.autoround import AutoRoundModifier
from llmcompressor.utils import load_context

MODEL_ID = "zai-org/GLM-4.5-Air"
SAVE_DIR = "GLM-4.5-Air-W4A16-G128-AutoRound-MTP"

# Load the MTP layer with the backbone so the recipe can quantize it.
with load_context(load_mtp=True):
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto_offload")
set_onload_device(model, "cuda")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
assert hasattr(model, "mtp"), "MTP was not attached; check the Transformers version"

NUM_CALIBRATION_SAMPLES = 128
MAX_SEQUENCE_LENGTH = 2048
ds = get_dataset(
    tokenizer=tokenizer,
    seqlen=MAX_SEQUENCE_LENGTH,
    nsamples=NUM_CALIBRATION_SAMPLES,
)

# During calibration, the MTP layer runs after the backbone on the calibration
# tokens and is tuned as one block like a decoder layer.
#   * `eh_proj` is ignored because vLLM's MTP layer does not load a quantized one
recipe = AutoRoundModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=["lm_head", "re:.*mlp.gate$", r"re:.*\.eh_proj$"],
    iters=200,
    enable_torch_compile=False,
)

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    shuffle_calibration_samples=False,
)

# Every targeted MTP linear is quantized; `eh_proj` stays unquantized
mtp_linears = {
    name: module
    for name, module in model.mtp.named_modules()
    if isinstance(module, torch.nn.Linear) and not name.endswith("mlp.gate")
}
for name, module in mtp_linears.items():
    quantized = hasattr(module, "quantization_scheme")
    assert quantized != name.endswith("eh_proj"), f"unexpected state for mtp.{name}"
print(f"Quantized {len(mtp_linears) - 1} MTP linears")

model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

# Serve with MTP speculative decoding:
#   vllm serve GLM-4.5-Air-W4A16-G128-AutoRound-MTP \
#       --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}'
