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

from tqdm import tqdm
from torch.utils.data import RandomSampler
from compressed_tensors.quantization import (
    apply_quantization_config,
    freeze_module_quantization,
    QuantizationConfig,
    QuantizationStatus,
)
from sparseml.transformers.finetune.data.data_args import DataTrainingArguments
from sparseml.transformers.finetune.data.base import TextGenerationDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DefaultDataCollator
from torch.utils.data import DataLoader
from sparseml.pytorch.utils import tensors_to_device

config_file = "example_quant_config.json"
model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
dataset_name = "open_platypus"
split = "train"
num_calibration_samples = 512
max_seq_length = 1024
pad_to_max_length = False
output_dir = "./llama1.1b_new_quant_out"

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda:0")
model.eval()  # no grad or updates needed for base model
config = QuantizationConfig.parse_file(config_file)

# set status to calibration
config.quantization_status = QuantizationStatus.CALIBRATION

# initialize quantization
apply_quantization_config(model, config)

# create dataset
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
data_loader = DataLoader(
    calib_dataset, batch_size=1, collate_fn=DefaultDataCollator(), sampler=RandomSampler(calib_dataset)
)

# run calibration
for idx, sample in tqdm(enumerate(data_loader), desc="Running calibration"):
    sample = tensors_to_device(sample, "cuda:0")
    _ = model(**sample)

    if idx >= num_calibration_samples:
        break

# freeze params after calibration
model.apply(freeze_module_quantization)

# this functionality will move but for now we need to get the save override from
# SparseML in order to save the config
from sparseml.transformers.compression import modify_save_pretrained
modify_save_pretrained(model) 
model.save_pretrained(output_dir)