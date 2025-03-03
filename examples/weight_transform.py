from compressed_tensors.transforms import Hadamard, RandomHadamard, Transforms
from compressed_tensors.transforms.transform_args import (
    ModuleTarget,
    TransformationArgs,
)
from compressed_tensors.transforms.transform_config import TransformationConfig
from compressed_tensors.transforms.transform_data import TransformData
from compressed_tensors.transforms.transform_scheme import TransformationScheme
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

ignore = ["re:*.mlp.down_proj$"]
module_targets = [ModuleTarget.WEIGHTS]

# Start with a processed 
targets = ["Linear"] # 2048 * 2048
v_linear_args = TransformationArgs(
    targets=targets, module_targets=module_targets, ignore=ignore, call_args={"transpose": True, "first": False}
)

targets = ["re:*.mlp.down_proj$"] # 5632 * 5632
v_down_proj = TransformationArgs(
    targets=targets, module_targets=module_targets, call_args={"transpose": True, "first": False}
)

targets = ["re:*.attn.q_proj$", "re:*.attn.o_proj$", "re:*.mlp.down_proj$"] # 2048 * 2048
u_q_o_down_proj = TransformationArgs(
    targets=targets, module_targets=module_targets,
)

targets = ["re:*.attn.gate_proj$", "re:*.mlp.up_proj$"]  # 5632 * 5632
u_gate_up_proj = TransformationArgs(
    targets=targets, module_targets=module_targets,
)

targets = ["re:*.attn.k_proj$", "re:*.attn.v_proj$"] # 256 * 256
u_k_v_proj = TransformationArgs(
    targets=targets, module_targets=module_targets,
)


# This will apply the random_had to the first set of args
# It will then apply the second set of args
# any overalp will be applied in order
v_scheme = TransformationScheme(
    transform_type="random-hadamard",
    groups=[v_linear_args],
    transform_creation_args={"size": 2048},
)

v_scheme_down_proj = TransformationScheme(
    transform_type="random-hadamard",
    groups=[v_down_proj],
    transform_creation_args={"size": 5632},
)

# We could combine multiple args to the same scheme but then would make it more difficult to consolidate order of transforms
u_scheme_q_o_down_proj = TransformationScheme(
    transform_type="random-hadamard",
    groups=[u_q_o_down_proj],
    transform_creation_args={"size": 2048},
)

u_scheme_gate_up_proj = TransformationScheme(
    transform_type="random-hadamard",
    groups=[u_gate_up_proj],
    transform_creation_args={"size": 5632},
)

u_scheme_k_v_proj = TransformationScheme(
    transform_type="random-hadamard",
    groups=[u_k_v_proj],
    transform_creation_args={"size": 256},
)

# QuIP Recipe with weight only quantization
config = TransformationConfig(
    transform_groups={
        "u_transform_q_o_down_proj": u_scheme_q_o_down_proj,
        "u_transform_gate_up_proj": u_scheme_gate_up_proj,
        "u_transform_k_v_proj": u_scheme_k_v_proj,
        "v_transform_linear": v_scheme,
        "v_transform_down_proj": v_scheme_down_proj
    }
)

#MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
x = model.model.layers[0]
attn = x.self_attn
mlp = x.mlp

layers = [
    attn.q_proj,
    attn.k_proj,
    attn.v_proj,
    attn.o_proj,
    mlp.gate_proj,
    mlp.down_proj,
    mlp.up_proj
]

for layer in layers:

    current_weight = layer.weight
    (n, m) = current_weight.shape
    U = torch.eye(n).to("cuda").to(torch.bfloat16)
    V = torch.eye(m).to("cuda").to(torch.bfloat16)
    print(n, layer)

    output = torch.matmul(U, current_weight)
    output = torch.matmul(output, V.T)
