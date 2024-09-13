from sft_trainer import SFTTrainer
from transformers import AutoTokenizer, DefaultDataCollator

from llmcompressor.transformers import (
    DataTrainingArguments,
    SparseAutoModelForCausalLM,
    TextGenerationDataset,
    TrainingArguments,
)

model_path = "neuralmagic/Llama-2-7b-pruned50-retrained"
teacher_path = "neuralmagic/Llama-2-7b-gsm8k"

model = SparseAutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype="auto", device_map="auto"
)
teacher = SparseAutoModelForCausalLM.from_pretrained(
    teacher_path, torch_dtype="auto", device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load gsm8k using SparseML dataset tools
data_args = DataTrainingArguments(
    dataset="gsm8k", dataset_config_name="main", max_seq_length=512
)
dataset_manager = TextGenerationDataset.load_from_registry(
    data_args.dataset,
    data_args=data_args,
    split="train",
    tokenizer=tokenizer,
)
train_dataset = dataset_manager.tokenize_and_process()
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
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=0.6,
    logging_steps=50,
    gradient_checkpointing=True,
    bf16=True,
    save_safetensors=False,  # workaround for shared tensors
)
trainer = SFTTrainer(
    model=model,
    teacher=teacher,
    tokenizer=tokenizer,
    recipe=recipe,
    train_dataset=train_dataset,
    data_collator=data_collator,
    args=training_args,
    data_args=data_args,
    max_seq_length=data_args.max_seq_length,
    packing=True,
    max_steps=4,
)
trainer.train()
trainer.save_model()
