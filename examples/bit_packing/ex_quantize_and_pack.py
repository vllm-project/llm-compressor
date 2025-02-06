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
# The following example shows how to run QDQ inside `compressed-tensors`
# QDQ (quantize & de-quantize) is a way to evaluate quantized model
# accuracy but will not lead to a runtime speedup.
# See `../llama_1.1b/ex_config_quantization.py` to go beyond QDQ
# and quantize models that will run more performantly.
#
####

from pathlib import Path

import torch
from compressed_tensors.compressors import ModelCompressor
from compressed_tensors.quantization import (
    QuantizationConfig,
    QuantizationStatus,
    apply_quantization_config,
)
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


config_file = Path(__file__).parent / "int4_config.json"
model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
dataset_name = "garage-bAInd/Open-Platypus"
split = "train"
num_calibration_samples = 128
max_seq_length = 512
pad_to_max_length = False
output_dir = "./llama1.1b_new_quant_out_test_packing"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map=device, torch_dtype="auto"
)
model.eval()  # no grad or updates needed for base model
config = QuantizationConfig.model_validate_json(config_file.read_text())

# set status to calibration
config.quantization_status = QuantizationStatus.CALIBRATION

# initialize quantization
apply_quantization_config(model, config)

# create dataset
dataset = load_dataset(dataset_name, split=f"train[:{num_calibration_samples}]")
tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize_function(examples):
    return tokenizer(
        examples["output"], padding=False, truncation=True, max_length=1024
    )


tokenized_dataset = dataset.map(tokenize_function, batched=True)

data_loader = DataLoader(
    tokenized_dataset,
    batch_size=1,
)

with torch.no_grad():
    for idx, sample in tqdm(enumerate(data_loader), desc="Running calibration"):
        sample = {k: v.to(model.device) for k, v in sample.items()}
        _ = model(**sample)

        if idx >= num_calibration_samples:
            break

# convert model to QDQ model
compressor = ModelCompressor(quantization_config=config)
compressed_state_dict = compressor.compress(model)

# save QDQ model
model.save_pretrained(output_dir, state_dict=compressed_state_dict)
compressor.update_config(output_dir)
