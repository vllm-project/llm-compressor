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

from transformers import DefaultDataCollator, AutoTokenizer

from llmcompressor.transformers import (
    DataTrainingArguments,
    SFTTrainer,
    SparseAutoModelForCausalLM,
    TextGenerationDataset,
    TrainingArguments,
)

model_path = "neuralmagic/Llama-2-7b-pruned50-retrained"
teacher_path = "zoo:llama2-7b-gsm8k_llama2_pretrain-base"
output_dir = "./output_trl_sft_test_7b_gsm8k"

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
)
trainer.train()
trainer.save_model()
