import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStrategy,
)
from compressed_tensors.transforms import Hadamard, RandomHadamard, Transforms
from compressed_tensors.transforms.transform_args import (
    ModuleTarget,
    TransformationArgs,
)
from compressed_tensors.transforms.transform_config import TransformationConfig
from compressed_tensors.transforms.transform_scheme import TransformationScheme
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
# U(W)V.T

ignore = ["re:.*.mlp.down_proj$", "lm_head"]
module_targets = [ModuleTarget.WEIGHT.value]

# Start with a processed
targets = ["Linear"]  # 2048 * 2048
v_linear_args = TransformationArgs(
    targets=targets,
    module_targets=module_targets,
    ignore=ignore,
    call_args={"transpose": True, "first": False},
)

targets = ["re:.*.mlp.down_proj$"]  # 8192 * 8192
v_down_proj = TransformationArgs(
    targets=targets,
    module_targets=module_targets,
    call_args={"transpose": True, "first": False},
)

targets = [
    "re:.*.attn.q_proj$",
    "re:.*.attn.o_proj$",
    "re:.*.mlp.down_proj$",
]  # 2048 * 2048
u_q_o_down_proj = TransformationArgs(
    targets=targets,
    module_targets=module_targets,
)

targets = ["re:.*.mlp.gate_proj$", "re:.*.mlp.up_proj$"]  # 8192 * 8192
u_gate_up_proj = TransformationArgs(
    targets=targets,
    module_targets=module_targets,
)

targets = ["re:.*.attn.k_proj$", "re:.*.attn.v_proj$"]  # 512 * 512
u_k_v_proj = TransformationArgs(
    targets=targets,
    module_targets=module_targets,
)


# This will apply the random_had to the first set of args
# It will then apply the second set of args
# any overalp will be applied in order
v_scheme = TransformationScheme(
    transform_type="hadamard",
    groups=[v_linear_args],
    transform_creation_args={"size": 2048},
)

v_scheme_down_proj = TransformationScheme(
    transform_type="hadamard",
    groups=[v_down_proj],
    transform_creation_args={"size": 8192},
)

# We could combine multiple args to the same scheme but then would make it more difficult to consolidate order of transforms
u_scheme_q_o_down_proj = TransformationScheme(
    transform_type="hadamard",
    groups=[u_q_o_down_proj],
    transform_creation_args={"size": 2048},
)

u_scheme_gate_up_proj = TransformationScheme(
    transform_type="hadamard",
    groups=[u_gate_up_proj],
    transform_creation_args={"size": 8192},
)

u_scheme_k_v_proj = TransformationScheme(
    transform_type="hadamard",
    groups=[u_k_v_proj],
    transform_creation_args={"size": 512},
)

# QuIP Recipe with weight only quantization
config = TransformationConfig(
    transform_groups={
        "u_transform_q_o_down_proj": u_scheme_q_o_down_proj,
        "u_transform_k_v_proj": u_scheme_k_v_proj,
        "u_transform_gate_up_proj": u_scheme_gate_up_proj,
        "v_transform_linear": v_scheme,
        "v_transform_down_proj": v_scheme_down_proj,
    }
)

recipe = QuantizationModifier(
    targets="Linear",
    ignore=["lm_head"],
    config_groups={
        "group_0": QuantizationScheme(
            targets=["Linear"],
            weights=QuantizationArgs(
                num_bits=4,
                symmetric=True,
                strategy=QuantizationStrategy.GROUP,
                group_size=128,
                observer="mse"
            ),
        )
    },
    transforms_config=config,
)

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

oneshot(model=model, recipe=recipe)

print("\n\n")
print("========== SAMPLE GENERATION ==============")
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to("cuda")
output = model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")

# Save to disk compressed.
SAVE_DIR = MODEL_ID.split("/")[1] + "-W4A16-uncompressed-hadamard-random-debug"

model.save_pretrained(SAVE_DIR, save_compressed=False)
tokenizer.save_pretrained(SAVE_DIR)