"""
AutoAWQ to compressed-tensors conversion e2e smoke test.

Converts an AutoAWQ checkpoint to compressed-tensors format, then validates
that vLLM can load the converted model and produce coherent output.

Requires AutoAWQConverter from compressed-tensors:
https://github.com/vishnuprasanth-j/compressed-tensors/commit/cc834757807cf188ffe8557b4b6e168e38ffdbfb

Takes ~25s on a single A100-80GB.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from compressed_tensors.entrypoints.convert import convert_checkpoint
from compressed_tensors.entrypoints.convert.converters import AutoAWQConverter

from tests.testing_utils import requires_cadence, requires_gpu

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

VLLM_PYTHON_ENV = os.environ.get("VLLM_PYTHON_ENV", "same")

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct-AWQ"
PROMPT = "The capital of France is"


@requires_gpu(1)
@requires_cadence("nightly")
def test_autoawq_to_compressed_tensors(tmp_path: Path) -> None:
    convert_outdir = tmp_path / "converted"
    convert_checkpoint(
        model_stub=MODEL_ID,
        save_directory=convert_outdir,
        converter=AutoAWQConverter.from_pretrained(MODEL_ID),
    )

    with (convert_outdir / "config.json").open() as f:
        config = json.load(f)
    assert config["quantization_config"]["quant_method"] == "compressed-tensors"

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

    print(result.stdout)

    assert (
        result.returncode == 0
    ), f"vLLM failed with exit code {result.returncode}:\n{result.stderr}"
    assert (
        "Paris" in result.stdout
    ), f"Expected 'Paris' in generated output:\n{result.stdout}"
