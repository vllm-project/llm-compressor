"""Quantize MTP layers and exercise vLLM speculative decoding."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from compressed_tensors.offload import set_onload_device
from compressed_tensors.utils.safetensors_load import get_weight_mappings
from transformers import AutoProcessor, Glm4MoeForCausalLM

from llmcompressor import oneshot, reset_session
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context
from tests.testing_utils import requires_cadence, requires_gpu


@requires_gpu(1)
@requires_cadence("nightly")
@pytest.mark.parametrize(
    "model_id,revision,model_cls,mtp_prefix",
    [
        pytest.param(
            "inference-optimization/GLM-4.5-Air-0.72B-MTP",
            "317504c08f5d3628bdcfed44a5016cdd70c73b74",
            Glm4MoeForCausalLM,
            "model.layers.46.",
            id="trailing-layers",
        ),
    ],
)
def test_mtp_vllm(tmp_path, model_id, revision, model_cls, mtp_prefix):
    pytest.importorskip("vllm")
    with load_context(model_cls, load_mtp=True):
        model = model_cls.from_pretrained(
            model_id, revision=revision, dtype=torch.bfloat16, device_map="cpu"
        )
    set_onload_device(model, "cuda")
    oneshot(
        model=model,
        recipe=QuantizationModifier(
            scheme="FP8_DYNAMIC",
            ignore=["lm_head", r"re:.*\.eh_proj$"],
        ),
    )
    model.save_pretrained(tmp_path)
    AutoProcessor.from_pretrained(model_id, revision=revision).save_pretrained(tmp_path)
    assert any(
        name.startswith(mtp_prefix) and name.endswith(".weight_scale")
        for name in get_weight_mappings(tmp_path)
    )
    reset_session()
    del model

    llm_kwargs = {
        "model": str(tmp_path),
        "dtype": "bfloat16",
        "gpu_memory_utilization": 0.25,
        "max_model_len": 256,
        "max_num_seqs": 2,
        "enforce_eager": True,
        "enable_prefix_caching": False,
        "disable_log_stats": False,
        "speculative_config": {"method": "mtp", "num_speculative_tokens": 1},
    }
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("run_vllm.py")),
            json.dumps("FP8_DYNAMIC"),
            json.dumps(llm_kwargs),
            json.dumps(["The capital of France is", "Actions speak louder than"]),
        ],
        env={**os.environ, "VLLM_WORKER_MULTIPROC_METHOD": "spawn"},
        check=True,
        timeout=300,
    )
