from sft_trainer import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, DefaultDataCollator

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
data_args = DatasetArguments(
    dataset="gsm8k", dataset_config_name="main", max_seq_length=max_seq_length
)
dataset_manager = TextGenerationDataset.load_from_registry(
    data_args.dataset,
    data_args=data_args,
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

data_collator = DefaultDataCollator()
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

trainer = SFTTrainer(
    model=model,
    teacher=teacher,
    processing_class=tokenizer,
    recipe=recipe,
    train_dataset=train_dataset,
    data_collator=data_collator,
    trl_sft_config_args=trl_sft_config_args,
    data_args=data_args,
    model_args=model_args,
)
trainer.train()
trainer.save_model(output_dir)
