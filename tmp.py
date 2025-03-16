from llmcompressor.core.llmcompressor.llmcompressor import LLMCompressor
from llmcompressor.modifiers.quantization.gptq import GPTQModifier
from llmcompressor.modifiers.smoothquant.base import SmoothQuantModifier

model_id = "meta-llama/Llama-3.2-1B-instruct"
recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),
    GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])
]

compressor = LLMCompressor(model_id, recipe)
compressor.set_calibration_dataset("ultrachat_200k", split="train_sft[:512]")
compressor.post_train()