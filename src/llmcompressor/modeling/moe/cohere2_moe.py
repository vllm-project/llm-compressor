import torch
from compressed_tensors import align_module_device, get_execution_device
from loguru import logger
from transformers import PreTrainedModel
from transformers.models.cohere2_moe.modeling_cohere2_moe import Cohere2MoeTopKRouter

from llmcompressor.modeling.fuse import fuse_norm_linears
from llmcompressor.modeling.moe.linearize import linearize_moe


class LinearRouter(torch.nn.Module):
    """
    Drop-in replacement for ``Cohere2MoeTopKRouter`` which performs the router
    projection through a child ``nn.Linear`` rather than a raw ``nn.Parameter``.

    SpinQuant (and the underlying ``compressed_tensors`` transform engine) can only
    fuse rotations into ``nn.Linear`` / ``nn.Embedding`` modules. Exposing the router
    projection as an ``nn.Linear`` allows the R1 rotation to be applied to it, which is
    required for routing to remain invariant under R1 (the router reads the same
    normalized residual stream as the q/k/v and mlp projections).

    The forward pass is numerically identical to the original router.

    :param router: original ``Cohere2MoeTopKRouter`` module to replace
    """

    def __init__(self, router: Cohere2MoeTopKRouter):
        super().__init__()
        self.top_k = router.top_k
        self.expert_selection_fn = router.expert_selection_fn
        self.norm_topk_prob = router.norm_topk_prob

        num_experts, hidden_size = router.weight.shape
        exec_device = get_execution_device(router)
        with align_module_device(router, exec_device):
            self.linear = torch.nn.Linear(
                hidden_size,
                num_experts,
                bias=False,
                dtype=router.weight.dtype,
                device=exec_device,
            )
            self.linear.weight.data.copy_(router.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        router_logits = self.linear(hidden_states)
        router_scores, selected_experts = torch.topk(router_logits, self.top_k, dim=-1)
        if self.expert_selection_fn == "softmax":
            router_scores = torch.nn.functional.softmax(
                router_scores, dim=1, dtype=torch.float
            )
        elif self.expert_selection_fn == "sigmoid":
            router_scores = torch.nn.functional.sigmoid(router_scores)
            if self.norm_topk_prob:
                router_scores = router_scores / torch.sum(
                    router_scores, dim=-1, keepdims=True
                )
        else:
            raise ValueError("`expert_selection_fn` can only be `softmax` or `sigmoid`")

        router_scores = router_scores.to(hidden_states.dtype)
        return router_logits, router_scores, selected_experts


def _layer_norm_consumers(layer: torch.nn.Module) -> list[torch.nn.Linear]:
    """
    Collect every ``nn.Linear`` that consumes the output of a decoder layer's
    ``input_layernorm``. Cohere2MoE uses a parallel block, so this is the union of the
    attention q/k/v projections and all MLP input projections (gate/up for the dense
    layer or every expert, plus the router).
    """
    linears = [
        layer.self_attn.q_proj,
        layer.self_attn.k_proj,
        layer.self_attn.v_proj,
    ]

    mlp = layer.mlp
    router = getattr(mlp, "gate", None)
    if isinstance(router, LinearRouter):
        linears.append(router.linear)
        for expert in mlp.experts:
            # experts is an nn.ModuleList that also holds the shared act_fn
            if hasattr(expert, "gate_proj"):
                linears += [expert.gate_proj, expert.up_proj]
    else:
        # dense layer (first_k_dense_replace)
        linears += [mlp.gate_proj, mlp.up_proj]

    return linears


def prepare_cohere2_moe_for_spinquant(model: PreTrainedModel):
    """
    Prepare a ``Cohere2MoeForCausalLM`` model so that SpinQuant rotations can be applied.

    This performs the architecture-specific transformations that the generic SpinQuant
    pipeline cannot, all of which are mathematically lossless:

    1. Linearizes the batched MoE experts (3D ``nn.Parameter`` -> per-expert
       ``nn.Linear``) via :func:`linearize_moe`, so R1/R4 can target the expert
       projections.
    2. Replaces each MoE router (:class:`Cohere2MoeTopKRouter`) with a
       :class:`LinearRouter` (exposing an ``nn.Linear``), so the R1 rotation can be
       applied to it (the router reads the rotated residual stream and must be rotated
       for routing to remain invariant).
    3. Fuses every ``input_layernorm`` scale into all of its consumers and resets the
       norm to ones. Cohere2MoE uses a parallel block: a single ``input_layernorm``
       feeds attention (q/k/v), the MLP (gate/up), AND the router. These consumers span
       different parents and the router is absent in the dense first layer, so they
       cannot be grouped by the generic norm-fusing pass (``match_modules_set``); we
       therefore fuse them here, per layer. The final ``model.norm`` -> ``lm_head``
       fusion is left to the modifier (it must run after embeddings are untied).

    This should be called before running ``SpinQuantModifier`` (e.g. before
    ``oneshot``).

    :param model: ``Cohere2MoeForCausalLM`` model to prepare in-place
    """
    linearize_moe(model)

    num_routers = 0
    for layer in model.model.layers:
        mlp = getattr(layer, "mlp", None)
        router = getattr(mlp, "gate", None)
        if isinstance(router, Cohere2MoeTopKRouter):
            mlp.gate = LinearRouter(router)
            num_routers += 1

    # fuse each parallel-block input_layernorm into all of its consumers (lossless)
    for layer in model.model.layers:
        fuse_norm_linears(layer.input_layernorm, _layer_norm_consumers(layer))

    logger.info(f"Prepared {num_routers} Cohere2MoE router(s) for SpinQuant")
    return model
