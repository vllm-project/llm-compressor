from llmcompressor.core import LLMCompressor
from llmcompressor.modifiers.quantization.gptq import GPTQModifier
from llmcompressor.modifiers.smoothquant.base import SmoothQuantModifier
from llmcompressor.modifiers.pruning import ConstantPruningModifier

model_id = "meta-llama/Llama-3.1-8B-Instruct"
recipe = ConstantPruningModifier(targets="__ALL__")
save_path = model_id.split("/")[1] + "-W8A8"

compressor = LLMCompressor(model_id, recipe)
compressor.set_train_dataset("ultrachat_200k", split="train_sft[:512]")
compressor.train(output_dir="tmp_output")
