from datasets import load_dataset
from sft_trainer import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM

from llmcompressor.transformers import TrainingArguments

model_path = "neuralmagic/Llama-2-7b-pruned50-retrained"
output_dir = "./output_trl_sft_test_7b_gsm8k_sft_data"
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype="auto", device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

# recipe for maintaining model sparsity during finetuning
recipe = """
test_stage:
  pruning_modifiers:
    ConstantPruningModifier:
      targets: ['re:.*q_proj.weight', 're:.*k_proj.weight', 're:.*v_proj.weight',
      're:.*o_proj.weight','re:.*gate_proj.weight', 're:.*up_proj.weight',
      're:.*down_proj.weight']
      start: 0
"""

# Load gsm8k using TRL dataset tools
dataset = load_dataset("gsm8k", "main", split="train")


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example["question"])):
        text = f"Question: {example['question'][i]}\n Answer: {example['answer'][i]}"
        output_texts.append(text)
    return output_texts


response_template = "Answer:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

training_args = TrainingArguments(
    num_train_epochs=0.6,
    logging_steps=50,
    gradient_checkpointing=True,
)

trainer = SFTTrainer(
    output_dir=output_dir,
    model=model,
    tokenizer=tokenizer,
    recipe=recipe,
    train_dataset=dataset,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
    args=training_args,
    max_seq_length=512,
)
trainer.train()
