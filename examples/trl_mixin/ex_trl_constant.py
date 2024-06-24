# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from datasets import load_dataset
from trl import DataCollatorForCompletionOnlyLM

from llmcompressor.transformers import (
    SFTTrainer,
    SparseAutoModelForCausalLM,
    TrainingArguments,
)

from transformers import AutoTokenizer

model_path = "neuralmagic/Llama-2-7b-pruned50-retrained"
output_dir = "./output_trl_sft_test_7b_gsm8k_sft_data"
model = SparseAutoModelForCausalLM.from_pretrained(
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
    output_dir=output_dir,
    num_train_epochs=0.6,
    logging_steps=50,
    gradient_checkpointing=True,
)

trainer = SFTTrainer(
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
trainer.save_model()
