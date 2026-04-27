"""
KL Divergence evaluation for measuring quantization quality.

Computes the KL Divergence between a baseline model and a quantized model by
extracting pre-lm_head hidden states via vLLM's apply_model hook, then
reconstructing log-probabilities offline.  This avoids the bottleneck of
transferring full vocabulary logprob tensors (~120k tokens) through vLLM by
working with the hidden dimension (~4096) instead — roughly a 30x reduction
in data volume.

Pipeline
--------
1. Load baseline and quantized models via vLLM (sequentially, one model in
   GPU memory at a time).
2. Register a forward hook on the model's ``LogitsProcessor`` via
   ``LLM.apply_model`` with ``enforce_eager=True`` to guarantee hooks fire
   on every forward call (CUDA-graph replay skips Python callbacks).
   Hook state is stored on the model object itself so it survives the
   serialization boundary imposed by vLLM v1's spawn-based worker isolation.
3. Run inference one prompt at a time; retrieve captures from the worker
   via a second ``LLM.apply_model`` call after each generate.
4. Build a plain ``nn.Linear`` from the baseline ``lm_head.weight`` and apply
   it to both hidden state sets to obtain log-probabilities.
5. Compute token-weighted ``KL(P_base || P_quant)`` averaged over the dataset.

References
----------
- https://vllm.ai/blog/extract-hidden-states
- https://github.com/vllm-project/llm-compressor/issues/2646
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

# Attribute names used to stash state on the model inside the worker.
_CAPTURES_ATTR = "__kld_captures__"
_HOOK_ATTR = "__kld_hook_handle__"


# ------------------------------------------------------------------
# Module-level worker functions
#
# These must be module-level (not closures) so vLLM v1's serializer
# can send them to the worker process by reference via pickle.  Any
# mutable state they need is stored on the model object itself.
# ------------------------------------------------------------------


def _worker_register_hook(model: nn.Module) -> None:
    """
    Register a forward hook on the model's ``LogitsProcessor``.

    Stores the hook handle and an empty capture buffer as attributes on
    *model* so they survive the worker-process boundary.

    :param model: The vLLM-loaded model running inside the worker.
    :raises RuntimeError: If ``LogitsProcessor`` cannot be found.
    """
    processor = _find_logits_processor(model)
    if processor is None:
        raise RuntimeError(
            "Could not locate LogitsProcessor in the vLLM model. "
            "KLDivergenceEvaluator requires a decoder model that exposes a "
            "'logits_processor' attribute (Llama, OPT, Mistral, Qwen, ...)."
        )
    setattr(model, _CAPTURES_ATTR, [])

    def _hook(
        module: nn.Module,
        inputs: tuple,
        output: torch.Tensor,
    ) -> None:
        # LogitsProcessor.forward(lm_head, hidden_states, embedding_bias=None)
        # inputs[1] is the pre-projection hidden state tensor.
        if len(inputs) < 2 or not isinstance(inputs[1], torch.Tensor):
            return
        getattr(model, _CAPTURES_ATTR).append(
            inputs[1].detach().to(torch.float32).cpu()
        )

    handle = processor.register_forward_hook(_hook)
    setattr(model, _HOOK_ATTR, handle)


def _worker_collect_captures(model: nn.Module) -> list[torch.Tensor]:
    """
    Return and clear the hidden-state captures stored on *model*.

    :param model: The vLLM-loaded model running inside the worker.
    :return: List of captured hidden-state tensors (CPU, float32).
    """
    captures: list[torch.Tensor] = list(getattr(model, _CAPTURES_ATTR, []))
    setattr(model, _CAPTURES_ATTR, [])
    return captures


def _worker_remove_hook(model: nn.Module) -> None:
    """
    Remove the registered hook and clear capture storage from *model*.

    :param model: The vLLM-loaded model running inside the worker.
    """
    handle = getattr(model, _HOOK_ATTR, None)
    if handle is not None:
        handle.remove()
    setattr(model, _CAPTURES_ATTR, [])
    setattr(model, _HOOK_ATTR, None)


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
        """Return a compact human-readable summary."""
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

    Extracts pre-lm_head hidden states by hooking the model's
    ``LogitsProcessor`` inside the vLLM worker process via ``LLM.apply_model``.
    Captures are stored on the model object itself (to survive vLLM v1's
    spawn-based worker isolation) and retrieved after each generate call.
    KL Divergence is then computed offline using the baseline model's lm_head
    weight matrix — ~30x more storage-efficient than full logprob extraction.

    ``enforce_eager`` is always forced to ``True`` regardless of the caller's
    value: CUDA-graph replay skips Python callbacks, so forward hooks would
    silently produce no captures under graph-captured execution.

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
    :param gpu_memory_utilization: Fraction of GPU memory vLLM may use.
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
        gpu_memory_utilization: float = 0.85,
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

        Loads baseline and quantized models sequentially to avoid holding
        both in GPU memory at once.  GPU memory is explicitly released after
        each model via ``_teardown_llm``.

        :param prompts: List of text prompts to evaluate on. Defaults to a
            small built-in calibration set if not provided.
        :return: :class:`KLDivergenceResult` with mean KLD and diagnostics.
        :raises ValueError: If *prompts* is an empty list.
        """
        if prompts is None:
            prompts = list(_DEFAULT_DATASET)
            logger.info(
                "No prompts provided; using built-in calibration set "
                f"({len(prompts)} prompts)."
            )

        if not prompts:
            raise ValueError("prompts must be a non-empty list.")

        logger.info(f"Extracting hidden states from base model: {self.base_model_id}")
        base_hidden, base_lm_head = self._extract_hidden_states(
            self.base_model_id, prompts
        )

        logger.info(
            f"Extracting hidden states from quantized model: "
            f"{self.quantized_model_id}"
        )
        quant_hidden, _ = self._extract_hidden_states(
            self.quantized_model_id, prompts
        )

        logger.info("Computing KL Divergence from hidden states...")
        return self._compute_kld(base_hidden, quant_hidden, base_lm_head)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_llm(self, model_id: str) -> "LLM":
        """
        Instantiate a vLLM ``LLM`` for *model_id* with eager execution forced.

        ``enforce_eager=True`` is mandatory: CUDA-graph replay skips Python
        forward-hook callbacks, which would silently produce empty captures.

        Also sets ``VLLM_ALLOW_INSECURE_SERIALIZATION=1`` so that
        ``apply_model`` can transmit module-level callables to the worker
        process via pickle (vLLM v1's default serializer rejects functions).

        :param model_id: HuggingFace model ID or local path.
        :return: Configured ``LLM`` instance.
        """
        import os

        from vllm import LLM

        os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        return LLM(
            model=model_id,
            dtype=self.dtype,
            gpu_memory_utilization=self.gpu_memory_utilization,
            enforce_eager=True,  # hooks require eager execution
            tensor_parallel_size=self.tensor_parallel_size,
        )

    def _extract_hidden_states(
        self,
        model_id: str,
        prompts: list[str],
    ) -> tuple[list[torch.Tensor], nn.Linear]:
        """
        Load *model_id* via vLLM, register a hook on LogitsProcessor, run
        inference one prompt at a time, then return per-prompt hidden states
        and a CPU copy of the lm_head projection.

        :param model_id: Model to load.
        :param prompts: Prompts to run inference on.
        :return: ``(hidden_states_per_prompt, lm_head_cpu)``.  Each tensor
            in the list has shape ``[prompt_len, hidden_dim]`` on CPU.
        """
        from vllm import SamplingParams

        llm = self._build_llm(model_id)

        llm.apply_model(_worker_register_hook)

        # prompt_logprobs=1 forces vLLM to compute logits for every prompt
        # token position, not just the final prefill token.
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=self.max_tokens,
            prompt_logprobs=1,
        )

        per_prompt_hidden: list[torch.Tensor] = []

        for idx, prompt in enumerate(prompts):
            outputs = llm.generate(
                [prompt],
                sampling_params=sampling_params,
                use_tqdm=False,
            )

            captures: list[torch.Tensor] = llm.apply_model(
                _worker_collect_captures
            )[0]

            if not outputs:
                logger.warning(
                    f"vLLM returned no output for prompt {idx}; skipping."
                )
                per_prompt_hidden.append(torch.empty(0))
                continue

            if not captures:
                logger.warning(
                    f"No hidden states captured for prompt {idx}. "
                    "The LogitsProcessor hook may not have fired — check "
                    "the model architecture."
                )
                per_prompt_hidden.append(torch.empty(0))
                continue

            request = outputs[0]
            prompt_len = len(request.prompt_token_ids or [])

            # Concatenate prefill + decode captures, keep only prefill tokens.
            full = torch.cat(captures, dim=0)
            keep = full[:prompt_len] if full.shape[0] >= prompt_len else full
            per_prompt_hidden.append(keep)

        llm.apply_model(_worker_remove_hook)

        lm_head_data = llm.apply_model(_worker_copy_lm_head)[0]
        weight: torch.Tensor = lm_head_data["weight"]
        bias: Optional[torch.Tensor] = lm_head_data["bias"]

        self._teardown_llm(llm)

        cpu_head = nn.Linear(
            weight.shape[1],
            weight.shape[0],
            bias=bias is not None,
            dtype=torch.float32,
        )
        cpu_head.weight.data.copy_(weight)
        if bias is not None:
            cpu_head.bias.data.copy_(bias)

        return per_prompt_hidden, cpu_head

    @staticmethod
    def _teardown_llm(llm: "LLM") -> None:
        """
        Explicitly release all GPU resources held by *llm*.

        ``del llm`` alone is insufficient — vLLM retains engine allocations,
        KV cache, and distributed state that persist until explicitly torn down.
        This method calls vLLM's distributed teardown helpers, then forces a
        Python GC pass and clears the CUDA allocator cache.

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

                # Sum KLD over all tokens; divide by token count for per-prompt
                # average, but accumulate raw sum for token-weighted global mean.
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
    gpu_memory_utilization: float = 0.85,
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
    :param gpu_memory_utilization: Fraction of GPU memory vLLM may allocate.
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
