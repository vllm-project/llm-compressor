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

from sparseml.transformers import oneshot, SparseAutoModelForCausalLM
from sparseml.transformers.finetune.data.data_args import DataTrainingArguments
from sparseml.transformers.finetune.data.base import TextGenerationDataset
from transformers import AutoTokenizer
import torch

recipe = "example_quant_recipe.yaml"
model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
dataset_name = "open_platypus"
split = "train"
num_calibration_samples = 512
max_seq_length = 1024
pad_to_max_length = False
output_dir = "./llama1.1b_old_quant_out"
device = "cuda:0" if torch.cuda_is_available() else "cpu"

model = SparseAutoModelForCausalLM.from_pretrained(model_name, device_map=device)

tokenizer = AutoTokenizer.from_pretrained(model_name)
data_args = DataTrainingArguments(
    dataset=dataset_name,
    max_seq_length=max_seq_length,
    pad_to_max_length=pad_to_max_length,
)
dataset_manager = TextGenerationDataset.load_from_registry(
    data_args.dataset,
    data_args=data_args,
    split=split,
    tokenizer=tokenizer,
)
calib_dataset = dataset_manager.tokenize_and_process(
    dataset_manager.get_raw_dataset()
)

oneshot(
    model=model_name,
    dataset=dataset_name,
    output_dir=output_dir,
    overwrite_output_dir=True,
    max_seq_length = max_seq_length,
    num_calibration_samples=num_calibration_samples,
    recipe=recipe,
    pad_to_max_length=pad_to_max_length
)