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

####
#
# The following example shows how the example in `ex_config_quantization.py`
# can be done within vllm's llm-compressor project
# Be sure to `pip install llmcompressor` before running
# See https://github.com/vllm-project/llm-compressor for more information
#
####

from pathlib import Path

import torch
from llmcompressor.transformers import oneshot


recipe = str(Path(__file__).parent / "example_quant_recipe.yaml")
model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
dataset_name = "open_platypus"
split = "train"
num_calibration_samples = 512
max_seq_length = 1024
pad_to_max_length = False
output_dir = "./llama1.1b_llmcompressor_quant_out"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

oneshot(
    model=model_name,
    dataset=dataset_name,
    output_dir=output_dir,
    overwrite_output_dir=True,
    max_seq_length=max_seq_length,
    num_calibration_samples=num_calibration_samples,
    recipe=recipe,
    pad_to_max_length=pad_to_max_length,
)
