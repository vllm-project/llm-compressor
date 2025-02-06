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
# The following example shows how a model can be calibrated and
# compressed entirely with primitives within `compressed-tensors`
# using PyTorch hooks.
# The resulting model's .safetensors file should be 1.2GB,
# whereas the original model's .safetensors file is 4.1GB.
# See `./ex_llmcompressor_quantization.py` for how this can be
# simplified using the vllm's `llm-compressor` package
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
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DefaultDataCollator


config_file = Path(__file__).parent / "example_quant_config.json"
model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
dataset_name = "garage-bAInd/Open-Platypus"
split = "train"
num_calibration_samples = 512
max_seq_length = 1024
pad_to_max_length = False
output_dir = "./llama1.1b_new_quant_out"
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


# create hook to keep track of scales and zero points on each module with a quantization_scheme
def update_scale_zp_hook(
    module: torch.nn.Module, input: torch.Tensor, _output: torch.Tensor
):
    from compressed_tensors.quantization.utils import calculate_qparams
    from compressed_tensors.utils import update_parameter_data

    quantization_scheme = getattr(module, "quantization_scheme", None)
    if not quantization_scheme:
        # no quantization scheme nothing to do
        return

    # update weight scale / zero-point
    quantization_args = getattr(quantization_scheme, "weights", None)
    min_val, max_val = torch.aminmax(module.weight.data)
    scale, zp = calculate_qparams(min_val, max_val, quantization_args)
    update_parameter_data(module, scale, "weight_scale")
    update_parameter_data(module, zp, "weight_zero_point")

    # update input_activations scale / zero-point
    quantization_args = getattr(quantization_scheme, "input_activations", None)
    min_val, max_val = torch.aminmax(input[0])
    scale, zp = calculate_qparams(min_val, max_val, quantization_args)
    update_parameter_data(module, scale, "input_scale")
    update_parameter_data(module, zp, "input_zero_point")

    return


# register hook on each submodule in model (recursively)
model.apply(lambda module: module.register_forward_hook(update_scale_zp_hook))

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
    collate_fn=DefaultDataCollator(),
    sampler=RandomSampler(tokenized_dataset),
)

# run calibration, hook will update scales and zero points where applicable
with torch.no_grad():
    for idx, sample in tqdm(enumerate(data_loader), desc="Running calibration"):
        sample = {k: v.to(model.device) for k, v in sample.items()}
        _ = model(**sample)

        if idx >= num_calibration_samples:
            break

# apply compression
compressor = ModelCompressor(quantization_config=config)
compressed_state_dict = compressor.compress(model)

# save quantized model
model.save_pretrained(output_dir, state_dict=compressed_state_dict)
compressor.update_config(output_dir)
