"""
KL Divergence evaluation for measuring quantization quality.

Computes the KL Divergence between a baseline model and a quantized model by
extracting pre-lm_head hidden states via buffer-based capture inside vLLM's
worker process, then reconstructing log-probabilities offline.  This avoids
the bottleneck of transferring full vocabulary logprob tensors (~120k tokens)
through vLLM by working with the hidden dimension (~4096) instead — roughly a
30x reduction in data volume.

Pipeline
--------
1. Load baseline and quantized models via vLLM **simultaneously** (both live
   in GPU memory throughout evaluation — no teardown between prompts).
2. Before the first ``generate()`` call, patch each model's ``LogitsProcessor``
   via ``LLM.apply_model`` to write pre-lm_head hidden states into a
   pre-allocated CUDA buffer.  Buffer writes are CUDA memory operations and
   are recorded in CUDA graphs, unlike Python forward-hook callbacks which
   are silently skipped during graph replay.
3. Run inference one prompt at a time on each model; retrieve captures from
   the worker via a second ``LLM.apply_model`` call after each generate.
4. Build a plain ``nn.Linear`` from the baseline ``lm_head.weight`` and apply
   it to both hidden state sets to obtain log-probabilities.
5. Compute token-weighted ``KL(P_base || P_quant)`` online per prompt,
   accumulating the weighted sum across the dataset.

Key improvements over the hook-based approach
----------------------------------------------
- No ``enforce_eager=True``: buffer writes (``Tensor.copy_``) are CUDA ops
  that are captured and replayed by CUDA graphs correctly.
- Both models remain resident in GPU memory for the full evaluation run,
  eliminating GPU teardown and reload overhead between models.
- KLD is computed online after each prompt; no intermediate results are
  written to disk.

References
----------
- https://vllm.ai/blog/extract-hidden-states
- https://github.com/vllm-project/llm-compressor/issues/2646
- https://github.com/vllm-project/llm-compressor/issues/2667
"""

from __future__ import annotations

import contextlib
import gc
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn

if TYPE_CHECKING:
    from vllm import LLM

__all__ = [
    "KLDivergenceEvaluator",
    "KLDivergenceResult",
    "evaluate_kl_divergence",
]

_DEFAULT_DATASET = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models require large amounts of training data.",
    "Quantization reduces model size by lowering the precision of weights.",
    "Natural language processing has advanced significantly in recent years.",
    "Efficient inference is critical for deploying large language models.",
]

# Maximum number of tokens the capture buffer can hold.  Prompts longer than
# this are truncated to the first _MAX_CAPTURE_TOKENS prefill positions.
_MAX_CAPTURE_TOKENS = 8192


# ------------------------------------------------------------------
# Module-level worker functions
#
# These must be module-level (not closures) so vLLM v1's serializer
# can send them to the worker process by reference via pickle.  Any
# mutable state they need is stored on the model object itself.
# ------------------------------------------------------------------


def _worker_setup_buffer_capture(model: nn.Module) -> None:
    """
    Patch ``LogitsProcessor.forward`` to capture pre-lm_head hidden states
    into a pre-allocated CUDA buffer.

    Buffer writes (``Tensor.copy_``) are CUDA memory operations — they are
    recorded in CUDA graphs and replayed correctly on each forward pass,
    unlike Python forward-hook callbacks which are skipped during graph replay.

    Must be called via ``apply_model`` **before** the first ``generate()`` so
    that the patched forward is included in any CUDA-graph capture.

    :param model: The vLLM-loaded model running inside the worker.
    :raises RuntimeError: If ``LogitsProcessor`` or ``lm_head`` cannot be
        found, or if ``lm_head`` has no ``.weight`` attribute.
    """
    lp = _find_logits_processor(model)
    if lp is None:
        raise RuntimeError(
            "Could not locate LogitsProcessor in the vLLM model. "
            "KLDivergenceEvaluator requires a decoder model that exposes a "
            "'logits_processor' attribute (Llama, OPT, Mistral, Qwen, ...)."
        )

    head = _find_lm_head(model)
    if head is None:
        raise RuntimeError("Could not locate lm_head in the vLLM model.")

    weight = getattr(head, "weight", None)
    if weight is None:
        raise RuntimeError("lm_head has no .weight attribute.")

    hidden_dim = weight.shape[1]
    device = weight.device
    buf_dtype = weight.dtype

    # Pre-allocate a fixed-size CUDA buffer.  Writes to this buffer are CUDA
    # operations, so they are captured and replayed by CUDA graphs correctly.
    buf = torch.zeros(_MAX_CAPTURE_TOKENS, hidden_dim, device=device, dtype=buf_dtype)
    n_tok = torch.zeros(1, dtype=torch.int64, device=device)

    model.__kld_buf__ = buf
    model.__kld_n__ = n_tok

    original_fwd = lp.forward

    def _capturing_forward(lm_head_w, hidden_states, *args, **kwargs):
        n = hidden_states.shape[0]
        cap = min(n, _MAX_CAPTURE_TOKENS)
        # CUDA copy — captured in CUDA graph and replayed on each call.
        buf[:cap].copy_(hidden_states[:cap].to(buf_dtype))
        n_tok.fill_(cap)
        return original_fwd(lm_head_w, hidden_states, *args, **kwargs)

    # Instance-level replacement: only this LogitsProcessor is patched,
    # not the whole class.  Created inside the worker, so no pickle needed.
    lp.forward = _capturing_forward


def _worker_collect_buffer(model: nn.Module) -> dict:
    """
    Read the captured hidden states from the CUDA buffer and return to caller.

    Called via ``apply_model`` after each ``generate()`` to retrieve the
    pre-lm_head hidden states written by ``_worker_setup_buffer_capture``.

    :param model: The vLLM-loaded model running inside the worker.
    :return: ``{"hidden_states": Tensor[n, hidden_dim], "n_tokens": int}``
        where hidden states are CPU float32.
    """
    buf: torch.Tensor = model.__kld_buf__
    n: int = int(model.__kld_n__[0].item())
    return {
        "hidden_states": buf[:n].detach().float().cpu(),
        "n_tokens": n,
    }


def _worker_copy_lm_head(model: nn.Module) -> dict:
    """
    Return the lm_head weight and bias as CPU float32 tensors.

    Returns a ``dict`` (rather than a tuple) to avoid ambiguity in how
    vLLM's ``apply_model`` aggregates multi-worker return values.

    :param model: The vLLM-loaded model running inside the worker.
    :return: ``{"weight": Tensor, "bias": Tensor | None}``.
    :raises RuntimeError: If lm_head or its weight cannot be found.
    """
    head = _find_lm_head(model)
    if head is None:
        raise RuntimeError("Could not locate lm_head in the vLLM model.")
    weight = getattr(head, "weight", None)
    if weight is None:
        raise RuntimeError("lm_head has no .weight attribute.")
    bias = getattr(head, "bias", None)
    return {
        "weight": weight.detach().float().cpu(),
        "bias": bias.detach().float().cpu() if bias is not None else None,
    }


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


@dataclass
class KLDivergenceResult:
    """
    Result of a KL Divergence evaluation run.

    :param mean_kld: Token-weighted mean KL Divergence over all evaluated
        prompts.  Prompts with more tokens contribute proportionally more
        to this value than shorter prompts.
    :param per_prompt_kld: Per-prompt mean KLD (averaged over each prompt's
        tokens).  Entries are ``float("nan")`` for skipped prompts so that
        list indices stay aligned with the original input prompt list.
    :param num_prompts: Number of prompts successfully evaluated (excludes
        skipped prompts).
    :param num_tokens: Total number of tokens evaluated.
    :param skipped: Number of prompts skipped (empty capture or shape
        mismatch between baseline and quantized hidden states).
    """

    mean_kld: float
    per_prompt_kld: list[float] = field(default_factory=list)
    num_prompts: int = 0
    num_tokens: int = 0
    skipped: int = 0

    def __str__(self) -> str:
        return (
            f"KLDivergenceResult("
            f"mean_kld={self.mean_kld:.6f}, "
            f"num_prompts={self.num_prompts}, "
            f"num_tokens={self.num_tokens}, "
            f"skipped={self.skipped})"
        )


class KLDivergenceEvaluator:
    """
    Evaluates KL Divergence between a baseline and quantized model using
    vLLM hidden state extraction.

    Both models are loaded into GPU memory simultaneously and remain resident
    for the full evaluation run.  Pre-lm_head hidden states are captured via
    a pre-allocated CUDA buffer patched onto each model's ``LogitsProcessor``
    before the first ``generate()`` call — this avoids the ``enforce_eager``
    requirement of Python forward-hook approaches, since buffer writes are
    CUDA memory operations recorded and replayed by CUDA graphs.

    KLD is computed online after each prompt pair; no intermediate data is
    written to disk.

    Tensor parallelism is not supported (``tensor_parallel_size`` must be 1)
    because the lm_head weight is sharded across ranks under TP, which would
    require gather logic the offline KLD step does not implement.

    :param base_model_id: HuggingFace model ID or local path for the baseline
        (unquantized) model.
    :param quantized_model_id: HuggingFace model ID or local path for the
        quantized model. If ``None``, defaults to ``base_model_id`` (useful
        for sanity checks — result should be ~0).
    :param dtype: Model dtype passed to vLLM. Defaults to ``"auto"``.
    :param max_tokens: Maximum number of tokens to generate per prompt.
        Must be >= 1. Set to 1 to evaluate only on the prompt prefill
        positions (recommended for pure quantization quality assessment).
    :param gpu_memory_utilization: Fraction of GPU memory each model may use.
        Defaults to 0.45 so that two concurrent models use ~0.90 total.
        Reduce if two models do not fit in GPU memory simultaneously.
    :param tensor_parallel_size: Must be 1 (TP not yet supported).

    Example::

        evaluator = KLDivergenceEvaluator(
            base_model_id="meta-llama/Meta-Llama-3-8B",
            quantized_model_id="meta-llama/Meta-Llama-3-8B-W4A16",
        )
        result = evaluator.evaluate(prompts=["Hello world", ...])
        print(result.mean_kld)
    """

    def __init__(
        self,
        base_model_id: str,
        quantized_model_id: Optional[str] = None,
        dtype: str = "auto",
        max_tokens: int = 1,
        gpu_memory_utilization: float = 0.45,
        tensor_parallel_size: int = 1,
    ):
        if tensor_parallel_size != 1:
            raise ValueError(
                "KLDivergenceEvaluator only supports tensor_parallel_size=1; "
                f"got {tensor_parallel_size}. Sharded lm_head requires gather "
                "logic that is not yet implemented for offline KLD."
            )
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1; got {max_tokens}.")

        self.base_model_id = base_model_id
        self.quantized_model_id = quantized_model_id or base_model_id
        self.dtype = dtype
        self.max_tokens = max_tokens
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tensor_parallel_size = tensor_parallel_size

    def evaluate(
        self,
        prompts: Optional[list[str]] = None,
    ) -> KLDivergenceResult:
        """
        Run KL Divergence evaluation over the given prompts.

        Both models are loaded at the start and remain in GPU memory for the
        full run.  KLD is computed online after each prompt pair; no data is
        written to disk.

        :param prompts: List of text prompts to evaluate on. Defaults to a
            small built-in calibration set if not provided.
        :return: :class:`KLDivergenceResult` with mean KLD and diagnostics.
        :raises ValueError: If *prompts* is an empty list.
        """
        import os

        from vllm import LLM, SamplingParams

        os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        if prompts is None:
            prompts = list(_DEFAULT_DATASET)
            logger.info(
                "No prompts provided; using built-in calibration set "
                f"({len(prompts)} prompts)."
            )

        if not prompts:
            raise ValueError("prompts must be a non-empty list.")

        logger.info(
            f"Loading base model: {self.base_model_id} "
            f"(gpu_memory_utilization={self.gpu_memory_utilization})"
        )
        base_llm = self._build_llm(self.base_model_id)

        logger.info(
            f"Loading quantized model: {self.quantized_model_id} "
            f"(gpu_memory_utilization={self.gpu_memory_utilization})"
        )
        quant_llm = self._build_llm(self.quantized_model_id)

        try:
            # Patch both models before any generate() triggers CUDA-graph capture.
            logger.info("Setting up buffer capture on both models.")
            base_llm.apply_model(_worker_setup_buffer_capture)
            quant_llm.apply_model(_worker_setup_buffer_capture)

            lm_head_data = base_llm.apply_model(_worker_copy_lm_head)[0]
            lm_head = self._build_cpu_lm_head(lm_head_data)

            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=self.max_tokens,
                prompt_logprobs=1,
            )

            base_hidden: list[torch.Tensor] = []
            quant_hidden: list[torch.Tensor] = []

            logger.info(f"Evaluating {len(prompts)} prompts...")
            for idx, prompt in enumerate(prompts):
                base_llm.generate([prompt], sampling_params=sampling_params, use_tqdm=False)
                base_data = base_llm.apply_model(_worker_collect_buffer)[0]

                quant_llm.generate([prompt], sampling_params=sampling_params, use_tqdm=False)
                quant_data = quant_llm.apply_model(_worker_collect_buffer)[0]

                h_base = base_data["hidden_states"]
                h_quant = quant_data["hidden_states"]

                if h_base.numel() == 0 or h_quant.numel() == 0:
                    logger.warning(f"Empty capture for prompt {idx}; skipping.")

                base_hidden.append(h_base)
                quant_hidden.append(h_quant)

            logger.info("Computing KL Divergence from hidden states...")
            return self._compute_kld(base_hidden, quant_hidden, lm_head)

        finally:
            self._teardown_llm(base_llm)
            self._teardown_llm(quant_llm)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_llm(self, model_id: str) -> "LLM":
        """
        Instantiate a vLLM ``LLM`` for *model_id*.

        ``VLLM_ALLOW_INSECURE_SERIALIZATION=1`` must be set before calling
        ``LLM()`` so that ``apply_model`` can transmit module-level callables
        to the worker process via pickle.

        :param model_id: HuggingFace model ID or local path.
        :return: Configured ``LLM`` instance.
        """
        from vllm import LLM

        return LLM(
            model=model_id,
            dtype=self.dtype,
            gpu_memory_utilization=self.gpu_memory_utilization,
            tensor_parallel_size=self.tensor_parallel_size,
        )

    @staticmethod
    def _build_cpu_lm_head(lm_head_data: dict) -> nn.Linear:
        """Build a CPU ``nn.Linear`` from copied lm_head weight/bias tensors."""
        weight: torch.Tensor = lm_head_data["weight"]
        bias: Optional[torch.Tensor] = lm_head_data["bias"]
        cpu_head = nn.Linear(
            weight.shape[1],
            weight.shape[0],
            bias=bias is not None,
            dtype=torch.float32,
        )
        cpu_head.weight.data.copy_(weight)
        if bias is not None:
            cpu_head.bias.data.copy_(bias)
        return cpu_head

    @staticmethod
    def _teardown_llm(llm: "LLM") -> None:
        """
        Explicitly release all GPU resources held by *llm*.

        ``del llm`` alone is insufficient — vLLM retains engine allocations,
        KV cache, and distributed state that persist until explicitly torn down.

        :param llm: The ``LLM`` instance to destroy.
        """
        with contextlib.suppress(Exception):
            from vllm.distributed.parallel_state import (
                destroy_distributed_environment,
                destroy_model_parallel,
            )

            destroy_model_parallel()
            destroy_distributed_environment()
        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _compute_kld(
        base_hidden: list[torch.Tensor],
        quant_hidden: list[torch.Tensor],
        lm_head: nn.Linear,
    ) -> KLDivergenceResult:
        """
        Apply *lm_head* to both hidden state lists and compute KL Divergence.

        ``KL(P_base || P_quant)`` is computed per token then averaged with
        token-count weighting so that longer prompts contribute proportionally
        more to the reported ``mean_kld``.

        Skipped prompts (empty or shape-mismatched) insert ``float("nan")``
        into ``per_prompt_kld`` so that list indices stay aligned with the
        original input prompt list.

        :param base_hidden: Pre-lm_head hidden states from the baseline model.
        :param quant_hidden: Pre-lm_head hidden states from the quantized model.
        :param lm_head: Plain ``nn.Linear`` built from the baseline lm_head.
        :return: :class:`KLDivergenceResult`.
        :raises ValueError: If list lengths differ.
        :raises RuntimeError: If every prompt is skipped.
        """
        if len(base_hidden) != len(quant_hidden):
            raise ValueError(
                f"Mismatched prompt counts: base={len(base_hidden)}, "
                f"quant={len(quant_hidden)}"
            )

        lm_head.eval()
        per_prompt_kld: list[float] = []
        total_tokens = 0
        total_kld_sum = 0.0
        skipped = 0

        with torch.no_grad():
            for idx, (h_base, h_quant) in enumerate(
                zip(base_hidden, quant_hidden, strict=True)
            ):
                if h_base.numel() == 0 or h_quant.numel() == 0:
                    skipped += 1
                    per_prompt_kld.append(float("nan"))
                    continue

                if h_base.shape != h_quant.shape:
                    logger.warning(
                        f"Prompt {idx} hidden state shape mismatch: "
                        f"base={tuple(h_base.shape)}, "
                        f"quant={tuple(h_quant.shape)}. Skipping."
                    )
                    skipped += 1
                    per_prompt_kld.append(float("nan"))
                    continue

                logits_base = lm_head(h_base.float())
                logits_quant = lm_head(h_quant.float())

                log_p = F.log_softmax(logits_base, dim=-1)
                log_q = F.log_softmax(logits_quant, dim=-1)

                kld_sum = F.kl_div(
                    log_q, log_p, reduction="sum", log_target=True
                ).item()
                n_tokens = h_base.shape[0]

                per_prompt_kld.append(kld_sum / n_tokens)
                total_kld_sum += kld_sum
                total_tokens += n_tokens

        valid = [v for v in per_prompt_kld if not math.isnan(v)]
        if not valid:
            raise RuntimeError(
                "No valid prompts were evaluated. "
                f"All {skipped} prompts were skipped."
            )

        mean_kld = total_kld_sum / total_tokens
        return KLDivergenceResult(
            mean_kld=mean_kld,
            per_prompt_kld=per_prompt_kld,
            num_prompts=len(valid),
            num_tokens=total_tokens,
            skipped=skipped,
        )


def evaluate_kl_divergence(
    base_model_id: str,
    quantized_model_id: str,
    prompts: Optional[list[str]] = None,
    dtype: str = "auto",
    max_tokens: int = 1,
    gpu_memory_utilization: float = 0.45,
    tensor_parallel_size: int = 1,
) -> KLDivergenceResult:
    """
    Convenience wrapper around :class:`KLDivergenceEvaluator`.

    :param base_model_id: HuggingFace model ID or path for the baseline model.
    :param quantized_model_id: HuggingFace model ID or path for the quantized
        model.
    :param prompts: Calibration prompts. Defaults to a small built-in set.
    :param dtype: Model dtype for vLLM (e.g. ``"auto"``, ``"bfloat16"``).
    :param max_tokens: Max tokens to generate per prompt (must be >= 1).
    :param gpu_memory_utilization: Fraction of GPU memory each model may use.
        Defaults to 0.45 for two concurrent models (~0.90 total).
    :param tensor_parallel_size: Must be 1 (TP not supported).
    :return: :class:`KLDivergenceResult` with mean KLD and diagnostics.

    Example::

        from llmcompressor.evaluation import evaluate_kl_divergence

        result = evaluate_kl_divergence(
            base_model_id="meta-llama/Meta-Llama-3-8B",
            quantized_model_id="./Meta-Llama-3-8B-W4A16",
        )
        print(f"Mean KLD: {result.mean_kld:.6f}")
    """
    evaluator = KLDivergenceEvaluator(
        base_model_id=base_model_id,
        quantized_model_id=quantized_model_id,
        dtype=dtype,
        max_tokens=max_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
    )
    return evaluator.evaluate(prompts=prompts)


# ------------------------------------------------------------------
# Architecture helpers
# ------------------------------------------------------------------


def _find_logits_processor(model: nn.Module) -> Optional[nn.Module]:
    """
    Locate the ``LogitsProcessor`` module in a vLLM model.

    vLLM decoder models attach a ``logits_processor`` attribute on the
    top-level ``ForCausalLM`` class. Fallback: scan ``model.modules()``
    for any module whose class is named ``LogitsProcessor``.

    :param model: The vLLM-loaded model.
    :return: The ``LogitsProcessor`` module if found, else ``None``.
    """
    direct = getattr(model, "logits_processor", None)
    if isinstance(direct, nn.Module):
        return direct

    for module in model.modules():
        if type(module).__name__ == "LogitsProcessor":
            return module
    return None


def _find_lm_head(model: nn.Module) -> Optional[nn.Module]:
    """
    Locate the lm_head projection in *model*.

    Tries common attribute names (``lm_head``, ``output``, ``embed_out``).
    Returns the module as-is regardless of type — the caller reads
    ``.weight`` directly, which handles both ``nn.Linear`` and vLLM's
    ``ParallelLMHead``.

    :param model: Model to search.
    :return: The lm_head module if found, else ``None``.
    """
    for attr in ("lm_head", "output", "embed_out"):
        candidate = getattr(model, attr, None)
        if isinstance(candidate, nn.Module) and hasattr(candidate, "weight"):
            return candidate
    return None
