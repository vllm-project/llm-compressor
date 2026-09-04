"""GPTQ with the LUT-B (3-bit codebook) scheme on Llama-3-8B-Instruct.

This is the calibrated counterpart to `llama3_example.py` (data-free RTN LUT-B).
GPTQLutBModifier fits a per-tile non-uniform codebook and uses the GPTQ
Hessian-based error compensation to snap weights to codebook centers, targeting
the same MLP projections as the RTN baseline so the numbers are comparable.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQLutBModifier

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Same targets as the data-free LUT-B example, now with GPTQ calibration.
recipe = GPTQLutBModifier(
    targets="re:.*layers.*mlp.*_proj$",
    scheme="LUTB",
    ignore=["lm_head"],
)

oneshot(
    model=model,
    dataset="perfectblend",
    # Slice the split up front so only the first 2048 examples are tokenized,
    # instead of tokenizing the full ~1.4M-example dataset to select 2048.
    splits="train[:2048]",
    recipe=recipe,
    max_seq_length=2048,
    num_calibration_samples=2048,
)

SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-LUTB-GPTQ"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
