"""Quantize MTP layers and exercise supported vLLM speculative decoding."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from compressed_tensors.offload import set_onload_device
from compressed_tensors.utils.safetensors_load import get_weight_mappings
from safetensors import safe_open
from transformers import (
    AutoProcessor,
    Glm4MoeForCausalLM,
    InklingForConditionalGeneration,
    Qwen3_5ForConditionalGeneration,
)

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
            "nm-testing/GLM-4.5-Air-0.72B-MTP",
            "f6bc34dce3f6447db3a93a843d4bb659cd1f95e0",
            Glm4MoeForCausalLM,
            "model.layers.46.",
            id="trailing-layers",
        ),
        pytest.param(
            "Qwen/Qwen3.5-0.8B",
            "2fc06364715b967f1860aea9cf38778875588b17",
            Qwen3_5ForConditionalGeneration,
            "mtp.layers.0.",
            id="qwen3_5_0_8b",
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
            ignore=[
                "lm_head",
                r"re:.*\.eh_proj$",
                r"re:.*\.linear_attn\..*",
                r"re:.*\.visual\..*",
            ],
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
    kwargs = (
        {"limit_mm_per_prompt": {"image": 0, "video": 0}}
        if model_cls is Qwen3_5ForConditionalGeneration
        else {}
    )
    _run_mtp_vllm(tmp_path, **kwargs)


@requires_gpu(1)
@requires_cadence("nightly")
@pytest.mark.parametrize("method", ["gptq", "autoround"])
def test_calibrated_mtp_vllm(tmp_path, method):
    """MTP calibrated during oneshot loads and drafts tokens in vLLM"""
    pytest.importorskip("vllm")
    from llmcompressor.modifiers.autoround import AutoRoundModifier
    from llmcompressor.modifiers.gptq import GPTQModifier

    model_id = "nm-testing/GLM-4.5-Air-0.72B-MTP"
    revision = "f6bc34dce3f6447db3a93a843d4bb659cd1f95e0"
    with load_context(Glm4MoeForCausalLM, load_mtp=True):
        model = Glm4MoeForCausalLM.from_pretrained(
            model_id, revision=revision, dtype=torch.bfloat16, device_map="cpu"
        )
    set_onload_device(model, "cuda")
    # vLLM builds `eh_proj` as an unquantized nn.Linear
    kwargs = {"scheme": "W4A16", "ignore": ["lm_head", r"re:.*\.eh_proj$"]}
    if method == "gptq":
        recipe = GPTQModifier(**kwargs)
    else:
        recipe = AutoRoundModifier(**kwargs, iters=10)
    processor = AutoProcessor.from_pretrained(model_id, revision=revision)
    oneshot(
        model=model,
        recipe=recipe,
        processor=processor,
        dataset="open_platypus",
        num_calibration_samples=16,
        max_seq_length=256,
    )
    model.save_pretrained(tmp_path)
    processor.save_pretrained(tmp_path)
    mtp_prefix = f"model.layers.{model.config.num_hidden_layers}."
    assert any(
        name.startswith(mtp_prefix) and name.endswith(".weight_packed")
        for name in get_weight_mappings(tmp_path)
    )
    reset_session()
    del model
    _run_mtp_vllm(tmp_path)


def _run_mtp_vllm(model_path, **kwargs):
    llm_kwargs = {
        "model": str(model_path),
        "dtype": "bfloat16",
        "gpu_memory_utilization": 0.25,
        "max_model_len": 256,
        "max_num_seqs": 2,
        "enforce_eager": True,
        "enable_prefix_caching": False,
        "disable_log_stats": False,
        "speculative_config": {"method": "mtp", "num_speculative_tokens": 1},
        **kwargs,
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


@requires_gpu(1)
@requires_cadence("nightly")
def test_inkling_mtp_vllm(tmp_path):
    pytest.importorskip("vllm")
    model_id = "nm-testing/Inkling-Dense-54M-MTP"
    revision = "1e837c74be11b84604a5ff7b5ce0c5a89d42ff9a"
    with load_context(InklingForConditionalGeneration, load_mtp=True):
        model = InklingForConditionalGeneration.from_pretrained(
            model_id,
            revision=revision,
            dtype=torch.bfloat16,
            device_map="cpu",
        )
    assert len(model.mtp.layers) == 1
    set_onload_device(model, "cuda")
    oneshot(
        model=model,
        recipe=QuantizationModifier(
            scheme="FP8_DYNAMIC",
            ignore=["lm_head", r"re:.*\.eh_proj$", r"re:.*(vision|audio).*"],
        ),
        output_dir=str(tmp_path),
    )
    AutoProcessor.from_pretrained(model_id, revision=revision).save_pretrained(tmp_path)
    weights = get_weight_mappings(tmp_path)
    mtp_scales = [
        name
        for name in weights
        if name.startswith("model.mtp.layers.0.") and name.endswith(".weight_scale")
    ]
    assert len(mtp_scales) == 7
    assert not any(name.startswith("mtp.") for name in weights)
    for scale in mtp_scales:
        weight = scale.removesuffix("_scale")
        with safe_open(weights[weight], framework="pt") as checkpoint:
            assert checkpoint.get_tensor(weight).dtype == torch.float8_e4m3fn
        with safe_open(weights[scale], framework="pt") as checkpoint:
            assert torch.isfinite(checkpoint.get_tensor(scale)).all()
    reset_session()
    del model
    _run_mtp_vllm(tmp_path, limit_mm_per_prompt={"image": 0, "audio": 0, "video": 0})
