from llmcompressor import LLMCompressor
from llmcompressor.modifiers.quantization.gptq import GPTQModifier
from llmcompressor.modifiers.smoothquant.base import SmoothQuantModifier

#model_id = "meta-llama/Llama-3.2-1B-instruct"
model_id = "meta-llama/Llama-3.1-8B-Instruct"
recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),
    GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"])
]
output_dir = model_id.split("/")[1] + "-W8A8"

compressor = LLMCompressor(model_id, recipe)
compressor.set_calibration_dataset("ultrachat_200k", split="train_sft[:512]")
compressor.post_train(output_dir=output_dir)
