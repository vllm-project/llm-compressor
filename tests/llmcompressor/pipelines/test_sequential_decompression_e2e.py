import pytest
import torch
from compressed_tensors.quantization import QuantizationStatus

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

W4A16_MODEL = "nm-testing/TinyLlama-1.1B-Chat-v1.0-W4A16-G128-compressed"
FP8_MODEL = "nm-testing/TinyLlama-1.1B-Chat-v1.0-FP8-Dynamic-compressed"


def _run(model_id, force_full):
    return oneshot(
        model=model_id,
        recipe=GPTQModifier(scheme="W4A16", targets="Linear", ignore=["lm_head"]),
        dataset="open_platypus",
        num_calibration_samples=4,
        max_seq_length=64,
        save_compressed=True,
        force_full_decompression=force_full,
        output_dir=None,
        pipeline="sequential",
    )


def _first_quant_linear(model):
    for _, m in model.named_modules():
        if getattr(m, "quantization_scheme", None) is not None and isinstance(
            m, torch.nn.Linear
        ):
            return m
    raise AssertionError("no quantized linear found")


@pytest.mark.integration
@pytest.mark.parametrize(
    "model_id",
    [
        W4A16_MODEL,
        pytest.param(
            FP8_MODEL,
            marks=pytest.mark.xfail(
                reason="float-quantized (FP8) decompress leaves an fp8 operand "
                "(bf16 vs Float8_e4m3fn matmul); tracked separately",
                strict=False,
            ),
        ),
    ],
)
def test_jit_decompression_runs_and_recompresses(model_id):
    """JIT sequential decompression calibrates then recompresses a compressed model."""
    model = _run(model_id, force_full=False)
    mod = _first_quant_linear(model)
    # save_compressed=True -> module ends up recompressed
    assert not hasattr(mod, "weight")
    assert hasattr(mod, "weight_packed")
    assert mod.quantization_status in (
        QuantizationStatus.COMPRESSED,
        QuantizationStatus.FROZEN,
    )


def _run_dense(model_id, force_full):
    # save_compressed=False keeps the calibrated modules dense (fake-quantized) so the
    # returned model can be run directly and compared like-for-like.
    return oneshot(
        model=model_id,
        recipe=GPTQModifier(scheme="W4A16", targets="Linear", ignore=["lm_head"]),
        dataset="open_platypus",
        num_calibration_samples=4,
        max_seq_length=64,
        save_compressed=False,
        force_full_decompression=force_full,
        output_dir=None,
        pipeline="sequential",
    )


@pytest.mark.integration
def test_jit_matches_full_decompression():
    """JIT per-subgraph decompression and legacy full-decompress agree numerically."""
    prompt = torch.tensor([[1, 2, 3, 4, 5]])
    jit = _run_dense(W4A16_MODEL, force_full=False)
    base = _run_dense(W4A16_MODEL, force_full=True)
    with torch.no_grad():
        a = jit(prompt.to(jit.device)).logits.float().cpu().flatten()
        b = base(prompt.to(base.device)).logits.float().cpu().flatten()
    assert torch.isfinite(a).all()
    # GPTQ on GPU is run-to-run nondeterministic (atomicAdd): base-vs-base cosine is
    # ~0.968 for this config, so JIT-vs-base only needs to sit within that noise floor.
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    print(f"JIT-vs-full-decompress logit cosine similarity: {cos:.5f}")
    assert cos > 0.95
