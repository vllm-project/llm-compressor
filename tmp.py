from llmcompressor.core.llmcompressor.llmcompressor import LLMCompressor
from llmcompressor.modifiers.quantization.gptq import GPTQModifier

recipe = [
    GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])
]

compressor = LLMCompressor("meta-llama/Llama-3.2-1B-instruct", recipe)
compressor.set_calibration_dataset("ultrachat_200k", split="train_sft[:512]")
compressor.post_train()