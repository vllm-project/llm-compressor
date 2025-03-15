from llmcompressor.core.llmcompressor.llmcompressor import LLMCompressor

compressor = LLMCompressor("meta-llama/Llama-3.2-1B-instruct", "")
compressor.set_calibration_dataset("ultrachat_200k", split="train_sft[:1%]")