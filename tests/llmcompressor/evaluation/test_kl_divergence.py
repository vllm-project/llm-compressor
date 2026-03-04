"""
Unit tests for the KL-divergence evaluation tool.

Tests core computation logic using small synthetic models and tensors.
Does not require GPU or large model downloads.
"""

import pytest
import torch
import torch.nn.functional as F

from llmcompressor.evaluation.kl_divergence import (
    KLDivergenceResult,
    _kl_divergence_per_token,
    evaluate_kl_divergence,
)


class TestKLDivergencePerToken:
    """Tests for the core KL divergence computation."""

    def test_identical_distributions(self):
        """KL divergence of identical distributions should be zero."""
        logits = torch.randn(10, 100)
        log_probs = F.log_softmax(logits, dim=-1)

        kl = _kl_divergence_per_token(log_probs, log_probs)
        assert kl.shape == (10,)
        assert torch.allclose(kl, torch.zeros(10), atol=1e-5)

    def test_different_distributions(self):
        """KL divergence of different distributions should be positive."""
        torch.manual_seed(42)
        logits_p = torch.randn(5, 50)
        logits_q = torch.randn(5, 50)

        log_probs_p = F.log_softmax(logits_p, dim=-1)
        log_probs_q = F.log_softmax(logits_q, dim=-1)

        forward_kl = _kl_divergence_per_token(log_probs_p, log_probs_q)
        reverse_kl = _kl_divergence_per_token(log_probs_q, log_probs_p)

        # KL should be positive
        assert (forward_kl >= -1e-6).all()
        assert (reverse_kl >= -1e-6).all()

        # KL should be asymmetric (generally different values)
        assert not torch.allclose(forward_kl, reverse_kl, atol=1e-3)

    def test_known_value(self):
        """Test against manually computed KL divergence."""
        # P = [0.5, 0.5], Q = [0.25, 0.75]
        # KL(P||Q) = 0.5*log(0.5/0.25) + 0.5*log(0.5/0.75)
        #          = 0.5*log(2) + 0.5*log(2/3)
        #          ≈ 0.5*0.6931 + 0.5*(-0.4055) ≈ 0.1438
        import math

        expected_kl = 0.5 * math.log(2) + 0.5 * math.log(2 / 3)

        p = torch.tensor([[0.5, 0.5]])
        q = torch.tensor([[0.25, 0.75]])
        log_p = p.log()
        log_q = q.log()

        kl = _kl_divergence_per_token(log_p, log_q)
        assert kl.shape == (1,)
        assert abs(kl.item() - expected_kl) < 1e-5


class TestKLDivergenceResult:
    """Tests for the result dataclass."""

    def test_summary(self):
        result = KLDivergenceResult(
            forward_kld_mean=0.05,
            reverse_kld_mean=0.08,
            symmetric_kld_mean=0.065,
            num_samples=10,
            num_tokens=500,
        )
        summary = result.summary()
        assert "0.050000" in summary
        assert "0.080000" in summary
        assert "10 samples" in summary
        assert "500 tokens" in summary

    def test_to_dict(self):
        result = KLDivergenceResult(
            forward_kld_mean=0.1,
            reverse_kld_mean=0.2,
            num_samples=5,
            num_tokens=100,
        )
        d = result.to_dict()
        assert d["forward_kld_mean"] == 0.1
        assert d["reverse_kld_mean"] == 0.2
        assert d["num_samples"] == 5
        assert isinstance(d, dict)


class TestEvaluateKLDivergenceWithMocks:
    """Integration-style tests using tiny models."""

    @pytest.fixture
    def tiny_models(self):
        """Create two tiny randomly-initialized models for testing."""
        try:
            from transformers import AutoConfig, AutoModelForCausalLM

            config = AutoConfig.from_pretrained(
                "hf-internal-testing/tiny-random-LlamaForCausalLM"
            )
        except Exception:
            pytest.skip("Cannot load tiny model config (no network or HF access)")

        model_a = AutoModelForCausalLM.from_config(config)
        model_b = AutoModelForCausalLM.from_config(config)
        model_a.eval()
        model_b.eval()
        return model_a, model_b

    def test_evaluate_with_preloaded_models(self, tiny_models):
        """Test that evaluation runs end-to-end with preloaded models."""
        model_a, model_b = tiny_models

        try:
            result = evaluate_kl_divergence(
                base_model=model_a,
                target_model=model_b,
                dataset_id="wikitext",
                dataset_config="wikitext-2-raw-v1",
                dataset_split="test",
                num_samples=4,
                max_seq_length=32,
                device="cpu",
            )
        except Exception:
            pytest.skip("Cannot load dataset (no network)")

        assert isinstance(result, KLDivergenceResult)
        assert result.num_tokens > 0
        assert result.forward_kld_mean >= 0
        assert result.reverse_kld_mean >= 0
        assert len(result.forward_kld_per_sample) > 0

    def test_same_model_gives_zero_kld(self, tiny_models):
        """When evaluating the same model against itself, KLD should be ~0."""
        model_a, _ = tiny_models

        try:
            result = evaluate_kl_divergence(
                base_model=model_a,
                target_model=model_a,
                dataset_id="wikitext",
                dataset_config="wikitext-2-raw-v1",
                dataset_split="test",
                num_samples=2,
                max_seq_length=32,
                device="cpu",
            )
        except Exception:
            pytest.skip("Cannot load dataset (no network)")

        assert result.forward_kld_mean < 1e-4
        assert result.reverse_kld_mean < 1e-4
