from llmcompressor.core import LLMCompressor
from llmcompressor.modifiers.quantization.gptq import GPTQModifier
from llmcompressor.modifiers.smoothquant.base import SmoothQuantModifier
from llmcompressor.modifiers.pruning import ConstantPruningModifier
from llmcompressor.modifiers.distillation import OutputDistillationModifier

model_id = "meta-llama/Llama-3.2-1B-Instruct"
#recipe = ConstantPruningModifier(targets="__ALL__")
recipe = [
    ConstantPruningModifier(targets="__ALL__"),
    OutputDistillationModifier(targets="__ALL__")
]
save_path = model_id.split("/")[1] + "-W8A8"

compressor = LLMCompressor(model_id, recipe, distill_teacher=model_id)
compressor.set_train_dataset("ultrachat_200k", split="train_sft[:512]")
compressor.train(output_dir="tmp_output", per_device_train_batch_size=1)
