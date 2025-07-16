# NOTE: Fine tuning can require more steps than is shown in the example
# See the Axolotl integration blog post for best fine tuning practices
# https://developers.redhat.com/articles/2025/06/17/axolotl-meets-llm-compressor-fast-sparse-open

from sft_trainer import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor.args import DatasetArguments, ModelArguments
from llmcompressor.transformers import TextGenerationDataset

model_path = "neuralmagic/Llama-2-7b-pruned50-retrained"
teacher_path = "neuralmagic/Llama-2-7b-gsm8k"
output_dir = "./output_trl_sft_test_7b_gsm8k"

model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype="auto", device_map="auto"
)
teacher = AutoModelForCausalLM.from_pretrained(
    teacher_path, torch_dtype="auto", device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_path)
max_seq_length = 512

# Load gsm8k using SparseML dataset tools
dataset_args = DatasetArguments(
    dataset="gsm8k", dataset_config_name="main", max_seq_length=max_seq_length
)
dataset_manager = TextGenerationDataset.load_from_registry(
    dataset_args.dataset,
    dataset_args=dataset_args,
    split="train",
    processor=tokenizer,
)
train_dataset = dataset_manager()
print(f"--> Training Set Length = {len(train_dataset)}")

# recipe for maintaining model sparsity during finetuning
recipe = """
test_stage:
  pruning_modifiers:
    ConstantPruningModifier:
      targets: ['re:.*q_proj.weight', 're:.*k_proj.weight', 're:.*v_proj.weight',
      're:.*o_proj.weight', 're:.*gate_proj.weight', 're:.*up_proj.weight',
      're:.*down_proj.weight']
      start: 0
    OutputDistillationModifier:
      targets: ['re:model.layers.\\d+$']
      comparison: "square_head"
      start: 0
      orig_scale: 1.0
      distill_scale: 1.0
"""

trl_sft_config_args = dict(
    output_dir=output_dir,
    num_train_epochs=0.6,
    logging_steps=50,
    gradient_checkpointing=True,
    bf16=True,
    save_safetensors=False,  # workaround for shared tensors
    max_seq_length=max_seq_length,
    packing=True,
)
model_args = ModelArguments(model=model, distill_teacher=teacher)

# This step can be supplanted by fine tuning via integrated FT libraries such as Axolotl
trainer = SFTTrainer(
    model=model,
    teacher=teacher,
    processing_class=tokenizer,
    recipe=recipe,
    train_dataset=train_dataset,
    trl_sft_config_args=trl_sft_config_args,
    dataset_args=dataset_args,
    model_args=model_args,
)
trainer.train()
trainer.save_model(output_dir)
