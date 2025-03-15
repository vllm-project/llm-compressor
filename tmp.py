from llmcompressor.core.llmcompressor import LLMCompressor
from llmcompressor.modifiers.quantization.gptq import GPTQModifier

recipe = [
    GPTQModifier()
]

compressor = LLMCompressor("meta-llama/Llama-3.2-1B-instruct", recipe)
compressor.set_calibration_dataset("ultrachat_200k", split="train_sft[:1%]")
compressor.post_train()