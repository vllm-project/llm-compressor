# NOTE: Fine tuning can require more steps than is shown in the example
# See the Axolotl integration blog post for best fine tuning practices
# https://developers.redhat.com/articles/2025/06/17/axolotl-meets-llm-compressor-fast-sparse-open

from datasets import load_dataset
from sft_trainer import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM

from llmcompressor.args import ModelArguments

model_path = "neuralmagic/Llama-2-7b-pruned50-retrained"
output_dir = "./output_trl_sft_test_7b_gsm8k_sft_data"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto")
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

trl_sft_config_args = dict(
    output_dir=output_dir,
    num_train_epochs=0.6,
    logging_steps=50,
    gradient_checkpointing=True,
    max_seq_length=512,
)
model_args = ModelArguments(model=model)

# This step can be supplanted by fine tuning via integrated FT libraries such as Axolotl
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    recipe=recipe,
    train_dataset=dataset,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
    trl_sft_config_args=trl_sft_config_args,
    model_args=model_args,
)
trainer.train()
