import math

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from llmcompressor.utils.dev import skip_weights_download

model_id = "Qwen/Qwen3-30B-A3B"

config = AutoConfig.from_pretrained(model_id)
config.num_hidden_layers = 3

# Check if the model is multimodal. If so, change `AutoModelForCausalLM` to `...ForConditionalGeneration`
with skip_weights_download(AutoModelForCausalLM):
    model = AutoModelForCausalLM.from_pretrained(model_id, config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

# Initialize weights
for m in model.modules():
    m._is_hf_initialized = False
model.init_weights()

# Fix initialization of weights which were not properly implemented by `init_weights`
with torch.no_grad():
    for name, param in model.named_parameters():
        if not torch.isfinite(param).all() or param.abs().max() > 1e6:
            if "norm" in name.lower() and "weight" in name:
                param.fill_(1.0)
            elif "bias" in name:
                param.zero_()
            else:
                if param.ndim >= 2:
                    torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                else:
                    torch.nn.init.normal_(param, std=0.02)

num_parameters = model.num_parameters()
num_parameters_b = num_parameters / 1e9
assert num_parameters_b <= 1.0  # ideally the model is ~1b parameters. It's okay to use more parameters if they are required by the model architecture

save_path = f"Qwen3-{num_parameters_b:.1f}B-{num_parameters_b/10:.1f}B"  # for moe models, an estimate of the number of activated parameters should be included in the name
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
