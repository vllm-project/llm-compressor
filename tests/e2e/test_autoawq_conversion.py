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
from pathlib import Path

from compressed_tensors.entrypoints.convert import convert_checkpoint
from compressed_tensors.entrypoints.convert.converters import AutoAWQConverter
from vllm import LLM, SamplingParams

from tests.testing_utils import requires_cadence, requires_gpu

# vLLM spawns engine in a subprocess; fork fails if CUDA is already initialized
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

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

    llm = LLM(
        model=str(convert_outdir),
        max_model_len=64,
        enforce_eager=True,  # skip torch.compile + CUDA graphs to speed up test
    )
    outputs = llm.generate([PROMPT], SamplingParams(temperature=0.0, max_tokens=16))
    generated_text = outputs[0].outputs[0].text

    print(f"PROMPT: {PROMPT}")
    print(f"GENERATED: {generated_text}")

    assert (
        "Paris" in generated_text
    ), f"Expected 'Paris' in generated output, got: {generated_text}"
