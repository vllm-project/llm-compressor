import shutil

import torch.distributed as dist
from compressed_tensors.offload import get_device_map, init_dist, load_offloaded_model
from compressed_tensors.utils import patch_attr
from transformers import AutoModelForCausalLM

from llmcompressor.modeling.moe import conversion_mappings
from llmcompressor.modeling.moe.linearize import load_quantizable_moe
from tests.testing_utils import requires_gpu, torchrun


@requires_gpu(2)
@torchrun(world_size=2)
def test_load_quantizable_moe():
    MODEL_ID = "inference-optimization/Qwen3-1.6B-A0.9B"
    OFFLOAD_DIR = "./offload_folder"

    init_dist()
    with load_offloaded_model(), load_quantizable_moe():
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto_offload",
            max_memory={"cpu": 2e9},
            offload_folder=OFFLOAD_DIR,
        )

    device_map = {
        name: str(offload) for name, (_, offload) in get_device_map(model).items()
    }

    for index in range(0, 4):
        assert device_map[f"model.layers.{index}.self_attn.q_proj"] == "cpu"
        assert device_map[f"model.layers.{index}.mlp.experts.0.down_proj"] == "cpu"

    for index in range(4, 10):
        assert device_map[f"model.layers.{index}.self_attn.q_proj"] == "disk"
        assert device_map[f"model.layers.{index}.mlp.experts.0.down_proj"] == "disk"

    dist.barrier()
    shutil.rmtree(OFFLOAD_DIR, ignore_errors=True)
    dist.barrier()


@requires_gpu(2)
@torchrun(world_size=2)
def test_linearize_moe_model():
    # clear all loading mappings; must use `linearize_moe` pathway
    with patch_attr(conversion_mappings, "ARCH_TO_MOE_MAPPINGS", {}):
        test_load_quantizable_moe()
