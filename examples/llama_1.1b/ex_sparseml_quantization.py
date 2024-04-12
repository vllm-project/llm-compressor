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

dataset_name = "open_platypus"
overwrite_output_dir = True
splits = {"calibration": "train"}
seed = 42
output_dir = "./llama_1.1b_quant_mod_only"
num_calibration_samples = 1024
recipe = "example_quant_recipe.yaml"
model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
max_seq_length = 1024
pad_to_max_length = False

model = SparseAutoModelForCausalLM.from_pretrained(model_name, device_map="cuda:0")

oneshot(
    model=model_name,
    dataset=dataset_name,
    output_dir=output_dir,
    overwrite_output_dir=overwrite_output_dir,
    splits = splits,
    max_seq_length = max_seq_length,
    seed=seed,
    num_calibration_samples=num_calibration_samples,
    recipe=recipe,
    pad_to_max_length=pad_to_max_length
)