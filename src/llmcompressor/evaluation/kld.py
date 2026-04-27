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
1. Load baseline and quantized models via vLLM (sequentially, to fit one model
   in GPU memory at a time).
2. Register a forward hook on the model's ``LogitsProcessor`` via
   ``LLM.apply_model``.  The processor receives ``(lm_head, hidden_states)``
   as positional inputs, so the hook captures ``inputs[1]`` — the
   pre-projection activations.
3. Run inference one prompt at a time so each capture maps to a single
   prompt's token sequence.
4. Build a plain ``nn.Linear`` from the baseline ``lm_head.weight`` and apply
   it to both hidden state sets to obtain log-probabilities.
5. Compute per-token ``KL(P_base || P_quant)`` and average over the dataset.

References
----------
- https://vllm.ai/blog/extract-hidden-states
- https://github.com/vllm-project/llm-compressor/issues/2646
"""

from __future__ import annotations

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


@dataclass
class KLDivergenceResult:
    """
    Result of a KL Divergence evaluation run.

    :param mean_kld: Mean KL Divergence averaged over all prompts.
    :param per_prompt_kld: Per-prompt KL Divergence values.
    :param num_prompts: Number of prompts successfully evaluated.
    :param num_tokens: Total number of tokens evaluated.
    :param skipped: Number of prompts skipped due to capture mismatches.
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

    Extracts pre-lm_head hidden states from both models via ``LLM.apply_model``
    hooks, then computes KL Divergence offline using the baseline model's
    lm_head weight matrix. This is ~30x more storage-efficient than extracting
    full logprobs.

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
        Set to 1 to evaluate only on the prompt prefill positions.
    :param gpu_memory_utilization: Fraction of GPU memory vLLM may use.
    :param enforce_eager: Disable CUDA graph capture (useful for debugging).
    :param tensor_parallel_size: Must be 1.

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
        enforce_eager: bool = False,
        tensor_parallel_size: int = 1,
    ):
        if tensor_parallel_size != 1:
            raise ValueError(
                "KLDivergenceEvaluator only supports tensor_parallel_size=1; "
                f"got {tensor_parallel_size}. Sharded lm_head requires gather "
                "logic that is not yet implemented for offline KLD."
            )

        self.base_model_id = base_model_id
        self.quantized_model_id = quantized_model_id or base_model_id
        self.dtype = dtype
        self.max_tokens = max(1, max_tokens)
        self.gpu_memory_utilization = gpu_memory_utilization
        self.enforce_eager = enforce_eager
        self.tensor_parallel_size = tensor_parallel_size

    def evaluate(
        self,
        prompts: Optional[list[str]] = None,
    ) -> KLDivergenceResult:
        """
        Run KL Divergence evaluation over the given prompts.

        Loads baseline and quantized models sequentially to avoid holding
        both in GPU memory at once. Hidden states are moved to CPU after
        extraction so the GPU can be released before loading the next model.

        :param prompts: List of text prompts to evaluate on. Defaults to a
            small built-in calibration set if not provided.
        :return: :class:`KLDivergenceResult` with mean KLD and diagnostics.
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
        from vllm import LLM

        return LLM(
            model=model_id,
            dtype=self.dtype,
            gpu_memory_utilization=self.gpu_memory_utilization,
            enforce_eager=self.enforce_eager,
            tensor_parallel_size=self.tensor_parallel_size,
        )

    def _extract_hidden_states(
        self,
        model_id: str,
        prompts: list[str],
    ) -> tuple[list[torch.Tensor], nn.Linear]:
        """
        Load *model_id* via vLLM, register a hook on the LogitsProcessor to
        capture pre-projection hidden states, run inference one prompt at a
        time, then return the captures along with a CPU copy of the
        ``lm_head`` projection.

        :param model_id: Model to load.
        :param prompts: Prompts to run inference on.
        :return: ``(hidden_states_per_prompt, lm_head_cpu)``. Each hidden
            state tensor has shape ``[seq_len, hidden_dim]`` and lives on CPU.
        """
        from vllm import SamplingParams

        llm = self._build_llm(model_id)
        captures: list[torch.Tensor] = []

        def _hook(
            module: nn.Module,
            inputs: tuple,
            output: torch.Tensor,
        ) -> None:
            # LogitsProcessor.forward(lm_head, hidden_states, embedding_bias=None)
            # so inputs[1] is the pre-projection hidden state tensor.
            if len(inputs) < 2:
                return
            hidden = inputs[1]
            if not isinstance(hidden, torch.Tensor):
                return
            captures.append(hidden.detach().to(torch.float32).cpu())

        hook_handles: list = []

        def _register(model: nn.Module) -> None:
            processor = _find_logits_processor(model)
            if processor is None:
                raise RuntimeError(
                    f"Could not locate LogitsProcessor in model {model_id}. "
                    "This evaluator requires a vLLM decoder model with a "
                    "logits_processor attribute."
                )
            hook_handles.append(processor.register_forward_hook(_hook))

        llm.apply_model(_register)

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=self.max_tokens,
            # Force vLLM to compute logits for every prompt position.
            prompt_logprobs=0,
        )

        # Process prompts one at a time so each prompt maps cleanly to its
        # own set of captures (vLLM otherwise concatenates batched prompts
        # into a single flat hidden-state tensor).
        per_prompt_hidden: list[torch.Tensor] = []
        for idx, prompt in enumerate(prompts):
            captures.clear()
            outputs = llm.generate(
                [prompt],
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            if not outputs:
                logger.warning(f"vLLM returned no output for prompt {idx}; skipping.")
                per_prompt_hidden.append(torch.empty(0))
                continue

            request = outputs[0]
            prompt_len = len(request.prompt_token_ids or [])

            if not captures:
                logger.warning(
                    f"No hidden states captured for prompt {idx}. "
                    "The hook may not have fired — check vLLM model architecture."
                )
                per_prompt_hidden.append(torch.empty(0))
                continue

            # Concatenate all captures from this prompt's forward calls.
            # Prefill yields [prompt_len, hidden]; subsequent decode steps yield
            # [1, hidden] each. We only keep the prompt-prefill positions to
            # ensure base/quant captures align by token index.
            full = torch.cat(captures, dim=0)
            keep = full[:prompt_len] if full.shape[0] >= prompt_len else full
            per_prompt_hidden.append(keep)

        # Remove hooks
        for h in hook_handles:
            h.remove()
        hook_handles.clear()

        # Copy lm_head weight to CPU as a plain Linear so the GPU can be
        # released before loading the next model.
        lm_head_holder: list[nn.Linear] = []

        def _copy_lm_head(model: nn.Module) -> None:
            head = _find_lm_head(model)
            if head is None:
                raise RuntimeError(
                    f"Could not locate lm_head in model {model_id}."
                )
            weight = getattr(head, "weight", None)
            if weight is None:
                raise RuntimeError(
                    f"lm_head in {model_id} has no .weight attribute."
                )
            bias = getattr(head, "bias", None)

            cpu_head = nn.Linear(
                weight.shape[1],
                weight.shape[0],
                bias=bias is not None,
                dtype=torch.float32,
            )
            cpu_head.weight.data.copy_(weight.detach().float().cpu())
            if bias is not None:
                cpu_head.bias.data.copy_(bias.detach().float().cpu())
            lm_head_holder.append(cpu_head)

        llm.apply_model(_copy_lm_head)

        # Release GPU memory before next model loads.
        del llm

        if not lm_head_holder:
            raise RuntimeError(f"Failed to copy lm_head from model {model_id}.")

        return per_prompt_hidden, lm_head_holder[0]

    @staticmethod
    def _compute_kld(
        base_hidden: list[torch.Tensor],
        quant_hidden: list[torch.Tensor],
        lm_head: nn.Linear,
    ) -> KLDivergenceResult:
        """
        Apply *lm_head* to both hidden state lists and compute KL Divergence.

        KLD is computed as ``KL(P_base || P_quant)`` where P_base is the
        baseline distribution. Averaged over all prompts, weighted equally.

        :param base_hidden: Pre-lm_head hidden states from the baseline model.
        :param quant_hidden: Pre-lm_head hidden states from the quantized model.
        :param lm_head: Plain ``nn.Linear`` built from the baseline model's
            lm_head weight.
        :return: :class:`KLDivergenceResult`.
        """
        if len(base_hidden) != len(quant_hidden):
            raise ValueError(
                f"Mismatched prompt counts: base={len(base_hidden)}, "
                f"quant={len(quant_hidden)}"
            )

        lm_head.eval()
        per_prompt_kld: list[float] = []
        total_tokens = 0
        skipped = 0

        with torch.no_grad():
            for idx, (h_base, h_quant) in enumerate(zip(base_hidden, quant_hidden)):
                if h_base.numel() == 0 or h_quant.numel() == 0:
                    skipped += 1
                    continue

                if h_base.shape != h_quant.shape:
                    logger.warning(
                        f"Prompt {idx} hidden state shape mismatch: "
                        f"base={tuple(h_base.shape)}, "
                        f"quant={tuple(h_quant.shape)}. Skipping."
                    )
                    skipped += 1
                    continue

                # Both hidden state sets pass through baseline's lm_head.
                logits_base = lm_head(h_base.float())
                logits_quant = lm_head(h_quant.float())

                # KL(P || Q) = sum(P * (log P - log Q))
                # batchmean averages over the leading (token) dim.
                log_p = F.log_softmax(logits_base, dim=-1)
                log_q = F.log_softmax(logits_quant, dim=-1)
                kld = F.kl_div(log_q, log_p, reduction="batchmean", log_target=True)

                per_prompt_kld.append(kld.item())
                total_tokens += h_base.shape[0]

        if not per_prompt_kld:
            raise RuntimeError(
                "No valid prompts were evaluated. "
                f"All {skipped} prompts were skipped."
            )

        mean_kld = sum(per_prompt_kld) / len(per_prompt_kld)
        return KLDivergenceResult(
            mean_kld=mean_kld,
            per_prompt_kld=per_prompt_kld,
            num_prompts=len(per_prompt_kld),
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
    enforce_eager: bool = False,
    tensor_parallel_size: int = 1,
) -> KLDivergenceResult:
    """
    Convenience wrapper around :class:`KLDivergenceEvaluator`.

    :param base_model_id: HuggingFace model ID or path for the baseline model.
    :param quantized_model_id: HuggingFace model ID or path for the quantized
        model.
    :param prompts: Calibration prompts. Defaults to a small built-in set.
    :param dtype: Model dtype for vLLM (e.g. ``"auto"``, ``"bfloat16"``).
    :param max_tokens: Max tokens to generate per prompt. Use ``1`` for
        prefill-only evaluation.
    :param gpu_memory_utilization: Fraction of GPU memory vLLM may allocate.
    :param enforce_eager: Disable CUDA graph optimizations.
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
        enforce_eager=enforce_eager,
        tensor_parallel_size=tensor_parallel_size,
    )
    return evaluator.evaluate(prompts=prompts)


# ------------------------------------------------------------------
# Architecture helpers
# ------------------------------------------------------------------


def _find_logits_processor(model: nn.Module) -> Optional[nn.Module]:
    """
    Locate the ``LogitsProcessor`` module in a vLLM model.

    vLLM decoder models (Llama, OPT, Mistral, Qwen, ...) attach a
    ``logits_processor`` attribute on the top-level ``ForCausalLM`` class.
    Fallback: scan ``model.modules()`` for any module whose class is named
    ``LogitsProcessor``.

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
    Returns the module as-is regardless of whether it is ``nn.Linear`` or
    a vLLM-specific class such as ``ParallelLMHead`` — the caller reads
    ``.weight`` and ``.bias`` directly.

    :param model: Model to search.
    :return: The lm_head module if found, else ``None``.
    """
    for attr in ("lm_head", "output", "embed_out"):
        candidate = getattr(model, attr, None)
        if isinstance(candidate, nn.Module) and hasattr(candidate, "weight"):
            return candidate
    return None
