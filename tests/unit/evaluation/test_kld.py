"""
Unit tests for :mod:`llmcompressor.evaluation.kld`.

These tests exercise the offline / pure-PyTorch portions of the evaluator
(architecture helpers, the KLD math, error handling). The vLLM hidden-state
extraction path requires a GPU and is covered by the Colab notebook in
``examples/kld_evaluation/``.
"""

import math

import pytest
import torch
from torch import nn

from llmcompressor.evaluation.kld import (
    KLDivergenceEvaluator,
    KLDivergenceResult,
    _find_lm_head,
    _find_logits_processor,
    evaluate_kl_divergence,
)

# ---------------------------------------------------------------------
# _find_lm_head
# ---------------------------------------------------------------------


class _ModelWithLmHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.lm_head = nn.Linear(32, 1000, bias=False)


class _ModelWithEmbedOut(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_out = nn.Linear(32, 1000, bias=False)


class _ModelWithOutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.output = nn.Linear(32, 1000, bias=False)


class _ParallelLMHeadStub(nn.Module):
    """Mimics vLLM's ParallelLMHead — non-Linear nn.Module with .weight."""

    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab_size, hidden_size))


class _ModelWithParallelLMHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.lm_head = _ParallelLMHeadStub(1000, 32)


class _ModelNoHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(32, 1000)


def test_find_lm_head_named_lm_head():
    model = _ModelWithLmHead()
    found = _find_lm_head(model)
    assert found is model.lm_head


def test_find_lm_head_named_embed_out():
    model = _ModelWithEmbedOut()
    found = _find_lm_head(model)
    assert found is model.embed_out


def test_find_lm_head_named_output():
    model = _ModelWithOutput()
    found = _find_lm_head(model)
    assert found is model.output


def test_find_lm_head_accepts_non_linear_with_weight():
    """vLLM's ParallelLMHead is not nn.Linear — must still be found."""
    model = _ModelWithParallelLMHead()
    found = _find_lm_head(model)
    assert found is model.lm_head
    assert hasattr(found, "weight")


def test_find_lm_head_missing_returns_none():
    model = _ModelNoHead()
    assert _find_lm_head(model) is None


# ---------------------------------------------------------------------
# _find_logits_processor
# ---------------------------------------------------------------------


class LogitsProcessor(nn.Module):
    """Local stub matching vLLM's class name."""

    def forward(self, lm_head, hidden_states):  # pragma: no cover - unused in tests
        return hidden_states


class _ModelWithProcessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits_processor = LogitsProcessor()
        self.lm_head = nn.Linear(32, 1000)


def test_find_logits_processor_direct_attribute():
    model = _ModelWithProcessor()
    found = _find_logits_processor(model)
    assert found is model.logits_processor


def test_find_logits_processor_via_class_name_fallback():
    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = LogitsProcessor()

    found = _find_logits_processor(Wrapper())
    assert isinstance(found, LogitsProcessor)


def test_find_logits_processor_missing_returns_none():
    assert _find_logits_processor(_ModelWithLmHead()) is None


# ---------------------------------------------------------------------
# KLDivergenceEvaluator._compute_kld
# ---------------------------------------------------------------------


def _make_lm_head(hidden: int = 16, vocab: int = 64) -> nn.Linear:
    head = nn.Linear(hidden, vocab, bias=False, dtype=torch.float32)
    torch.manual_seed(0)
    head.weight.data.normal_()
    return head


def test_compute_kld_same_hidden_states_is_zero():
    head = _make_lm_head()
    h = [torch.randn(10, 16), torch.randn(7, 16)]
    result = KLDivergenceEvaluator._compute_kld(h, h, head)
    assert result.mean_kld == pytest.approx(0.0, abs=1e-6)
    assert result.num_prompts == 2
    assert result.num_tokens == 17
    assert result.skipped == 0


def test_compute_kld_different_hidden_states_is_positive():
    head = _make_lm_head()
    torch.manual_seed(1)
    base = [torch.randn(10, 16), torch.randn(8, 16)]
    quant = [torch.randn(10, 16), torch.randn(8, 16)]
    result = KLDivergenceEvaluator._compute_kld(base, quant, head)
    assert result.mean_kld > 0.0
    assert math.isfinite(result.mean_kld)
    assert result.num_prompts == 2


def test_compute_kld_shape_mismatch_skipped():
    head = _make_lm_head()
    base = [torch.randn(10, 16), torch.randn(8, 16)]
    quant = [torch.randn(10, 16), torch.randn(7, 16)]  # second mismatched
    result = KLDivergenceEvaluator._compute_kld(base, quant, head)
    assert result.num_prompts == 1
    assert result.skipped == 1


def test_compute_kld_empty_captures_skipped():
    head = _make_lm_head()
    base = [torch.empty(0), torch.randn(8, 16)]
    quant = [torch.empty(0), torch.randn(8, 16)]
    result = KLDivergenceEvaluator._compute_kld(base, quant, head)
    assert result.num_prompts == 1
    assert result.skipped == 1


def test_compute_kld_all_skipped_raises():
    head = _make_lm_head()
    base = [torch.empty(0)]
    quant = [torch.empty(0)]
    with pytest.raises(RuntimeError, match="No valid prompts"):
        KLDivergenceEvaluator._compute_kld(base, quant, head)


def test_compute_kld_mismatched_prompt_counts_raises():
    head = _make_lm_head()
    with pytest.raises(ValueError, match="Mismatched prompt counts"):
        KLDivergenceEvaluator._compute_kld(
            [torch.randn(5, 16)],
            [torch.randn(5, 16), torch.randn(5, 16)],
            head,
        )


def test_compute_kld_with_bias_lm_head():
    head = nn.Linear(16, 64, bias=True, dtype=torch.float32)
    h = [torch.randn(5, 16)]
    result = KLDivergenceEvaluator._compute_kld(h, h, head)
    assert result.mean_kld == pytest.approx(0.0, abs=1e-6)


def test_compute_kld_handles_dtype_promotion():
    """Hidden states in bf16/fp16 must be promoted to fp32 for KLD math."""
    head = _make_lm_head()
    h = [torch.randn(5, 16, dtype=torch.float16)]
    # Same input → KLD ≈ 0 even with low-precision input.
    result = KLDivergenceEvaluator._compute_kld(h, h, head)
    assert result.mean_kld == pytest.approx(0.0, abs=1e-3)


# ---------------------------------------------------------------------
# Construction & validation
# ---------------------------------------------------------------------


def test_evaluator_rejects_tensor_parallel():
    with pytest.raises(ValueError, match="tensor_parallel_size"):
        KLDivergenceEvaluator(
            base_model_id="dummy",
            tensor_parallel_size=2,
        )


def test_evaluator_default_quantized_falls_back_to_base():
    e = KLDivergenceEvaluator(base_model_id="foo")
    assert e.quantized_model_id == "foo"


def test_evaluator_max_tokens_clamped_to_one():
    e = KLDivergenceEvaluator(base_model_id="foo", max_tokens=0)
    assert e.max_tokens == 1


def test_evaluator_evaluate_empty_prompts_raises():
    e = KLDivergenceEvaluator(base_model_id="foo")
    with pytest.raises(ValueError, match="non-empty"):
        e.evaluate(prompts=[])


# ---------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------


def test_result_str_contains_key_fields():
    r = KLDivergenceResult(
        mean_kld=0.123456,
        per_prompt_kld=[0.1, 0.15],
        num_prompts=2,
        num_tokens=20,
        skipped=1,
    )
    s = str(r)
    assert "0.123456" in s
    assert "num_prompts=2" in s
    assert "num_tokens=20" in s
    assert "skipped=1" in s


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------


def test_evaluate_kl_divergence_function_exists_and_validates():
    # Should fail validation before touching vLLM.
    with pytest.raises(ValueError, match="tensor_parallel_size"):
        evaluate_kl_divergence(
            base_model_id="foo",
            quantized_model_id="bar",
            tensor_parallel_size=4,
        )
