"""
Checkpoint conversion e2e smoke tests.

Each parametrized case converts a checkpoint using a specific converter, then
validates that vLLM can load the converted model and produce coherent output.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from loguru import logger
from compressed_tensors.entrypoints.convert import convert_checkpoint
from compressed_tensors.entrypoints.convert.converters import (
    AutoAWQConverter,
    CompressedTensorsDequantizer,
    FP8BlockDequantizer,
    ModelOptNvfp4Converter,
)
from compressed_tensors.quantization import QuantizationArgs, QuantizationType

from tests.testing_utils import requires_cadence, requires_gpu

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

VLLM_PYTHON_ENV = os.environ.get("VLLM_PYTHON_ENV", "same")

PROMPT = "The capital of France is"

QWEN_TARGETS = [
    r"re:.*mlp.*\.(gate_up|gate|up|down)_proj$",
    r"re:.*self_attn.*\.(q|k|v|o)_proj$",
]


@requires_gpu(1)
@requires_cadence("nightly")
@pytest.mark.parametrize(
    "model_id, make_converter, expected_quant_method",
    [
        pytest.param(
            "Qwen/Qwen2.5-0.5B-Instruct-AWQ",
            lambda model_id: AutoAWQConverter.from_pretrained(model_id),
            "compressed-tensors",
            id="autoawq",
        ),
        pytest.param(
            "nvidia/Qwen3-8B-NVFP4",
            lambda model_id: ModelOptNvfp4Converter(
                targets=QWEN_TARGETS,
                kv_cache_scheme=QuantizationArgs(
                    num_bits=8, dynamic=False, type=QuantizationType.FLOAT
                ),
            ),
            "compressed-tensors",
            id="nvfp4",
        ),
        pytest.param(
            "qwen-community/Qwen3-4B-FP8",
            lambda model_id: FP8BlockDequantizer(
                targets=QWEN_TARGETS,
                weight_block_size=(128, 128),
            ),
            None,
            id="fp8block-dequant",
        ),
        pytest.param(
            "nm-testing/SmolLM-1.7B-Instruct-quantized.w4a16",
            lambda model_id: CompressedTensorsDequantizer(
                model_stub=model_id, ignore=["lm_head", "re:.*embed_tokens$"]
            ),
            None,
            id="ct-dequant-smollm",
        ),
        pytest.param(
            "nm-testing/tinyllama-w4a16-compressed",
            lambda model_id: CompressedTensorsDequantizer(
                model_stub=model_id, ignore=["lm_head", "re:.*embed_tokens$"]
            ),
            None,
            id="ct-dequant-tinyllama",
        ),
        pytest.param(
            "nm-testing/Meta-Llama-3-8B-Instruct-nonuniform-test",
            lambda model_id: CompressedTensorsDequantizer(
                model_stub=model_id, ignore=["lm_head", "re:.*embed_tokens$"]
            ),
            None,
            id="ct-dequant-llama3",
        ),
    ],
)
def test_conversion(
    tmp_path: Path,
    model_id: str,
    make_converter,
    expected_quant_method: str | None,
) -> None:
    convert_outdir = tmp_path / "converted"
    convert_checkpoint(
        model_stub=model_id,
        save_directory=convert_outdir,
        converter=make_converter(model_id),
    )

    with (convert_outdir / "config.json").open() as f:
        config = json.load(f)
    if expected_quant_method is not None:
        assert config["quantization_config"]["quant_method"] == expected_quant_method
    else:
        assert "quantization_config" not in config

    llm_kwargs = {
        "model": str(convert_outdir),
        "max_model_len": 64,
        "enforce_eager": True,
    }

    run_vllm_path = str(Path(__file__).parent / "run_vllm.py")
    vllm_env = sys.executable if VLLM_PYTHON_ENV.lower() == "same" else VLLM_PYTHON_ENV

    env = os.environ.copy()
    venv_bin = os.path.dirname(vllm_env)
    env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")

    result = subprocess.run(
        [
            vllm_env,
            run_vllm_path,
            json.dumps(None),
            json.dumps(llm_kwargs),
            json.dumps([PROMPT]),
        ],
        capture_output=True,
        text=True,
        env=env,
    )

    logger.info(result.stdout)

    assert (
        result.returncode == 0
    ), f"vLLM failed with exit code {result.returncode}:\n{result.stderr}"
    assert (
        "Paris" in result.stdout
    ), f"Expected 'Paris' in generated output:\n{result.stdout}"
