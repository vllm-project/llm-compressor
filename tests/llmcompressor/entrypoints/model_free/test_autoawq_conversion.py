"""
AutoAWQ to compressed-tensors conversion smoke test.

Requires AutoAWQConverter from compressed-tensors:
https://github.com/vishnuprasanth-j/compressed-tensors/commit/cc834757807cf188ffe8557b4b6e168e38ffdbfb

Uses vLLM for baseline inference (loads AWQ models natively).
Takes ~35s on a single A100-80GB.
"""

import json
import os
from pathlib import Path

import torch
from compressed_tensors.entrypoints.convert import convert_checkpoint
from compressed_tensors.entrypoints.convert.converters import AutoAWQConverter
from vllm import LLM, SamplingParams

from tests.testing_utils import requires_cadence, requires_gpu

# vLLM spawns engine in a subprocess; fork fails if CUDA is already initialized
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct-AWQ"
PROMPT = "The capital of Switzerland is"
TOP_K = 10


def _get_prompt_logprobs(model_path: str | Path) -> list[dict[int, float]]:
    llm = LLM(
        model=str(model_path),
        max_model_len=64,
        enforce_eager=True,  # skip torch.compile + CUDA graphs to speed up test
    )
    params = SamplingParams(max_tokens=1, prompt_logprobs=TOP_K)
    outputs = llm.generate([PROMPT], params)
    prompt_logprobs = outputs[0].prompt_logprobs
    assert prompt_logprobs is not None, "Expected prompt logprobs to be available"

    result = []
    for position_logprobs in prompt_logprobs:
        if position_logprobs is None:
            continue
        result.append({tok_id: lp.logprob for tok_id, lp in position_logprobs.items()})

    torch.cuda.empty_cache()

    return result


@requires_gpu
@requires_cadence("nightly")
def test_autoawq_to_compressed_tensors(tmp_path: Path) -> None:
    baseline_logprobs = _get_prompt_logprobs(MODEL_ID)
    assert len(baseline_logprobs)

    convert_outdir = tmp_path / "converted"
    convert_checkpoint(
        model_stub=MODEL_ID,
        save_directory=convert_outdir,
        converter=AutoAWQConverter.from_pretrained(MODEL_ID),
    )

    with (convert_outdir / "config.json").open() as f:
        config = json.load(f)
    assert config["quantization_config"]["quant_method"] == "compressed-tensors"

    converted_logprobs = _get_prompt_logprobs(convert_outdir)

    assert len(baseline_logprobs) == len(converted_logprobs)
    for pos, (baseline, converted) in enumerate(
        zip(baseline_logprobs, converted_logprobs)
    ):
        assert set(baseline.keys()) == set(converted.keys()), (
            f"Token ID mismatch at position {pos}: "
            f"baseline={sorted(baseline.keys())}, converted={sorted(converted.keys())}"
        )

        tokens = baseline.keys()
        assert tokens
        b_vals = torch.tensor([baseline[t] for t in tokens])
        c_vals = torch.tensor([converted[t] for t in tokens])
        assert torch.allclose(b_vals, c_vals), (
            f"Logprob mismatch at position {pos}: "
            f"max diff = {(b_vals - c_vals).abs().max().item()}"
        )
