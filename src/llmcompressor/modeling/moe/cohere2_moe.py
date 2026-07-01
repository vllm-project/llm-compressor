import torch
from compressed_tensors import align_module_device, get_execution_device
from loguru import logger
from transformers import PreTrainedModel
from transformers.models.cohere2_moe.modeling_cohere2_moe import Cohere2MoeTopKRouter

from llmcompressor.modeling.fuse import fuse_norm_linears
from llmcompressor.modeling.moe.linearize import linearize_moe


class LinearRouter(torch.nn.Module):
    """
    Drop-in replacement for ``Cohere2MoeTopKRouter`` that wraps the router weight in an
    ``nn.Linear``. SpinQuant can only fuse rotations into ``nn.Linear``/``nn.Embedding``,
    so this exposes the router projection to R1 (keeping routing invariant). Output is
    numerically identical to the original router.

    :param router: original ``Cohere2MoeTopKRouter`` to replace
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
                router_scores, dim=-1, dtype=torch.float
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
    All ``nn.Linear`` consumers of a decoder layer's ``input_layernorm``. Cohere2MoE
    uses a parallel block, so this is the q/k/v projections plus the MLP inputs
    (gate/up for the dense layer or each expert, plus the router).
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
    Prepare a ``Cohere2MoeForCausalLM`` model for SpinQuant rotations.

    Applies architecture-specific transformations the generic pipeline cannot, all
    lossless:

    1. Linearize batched MoE experts (3D ``nn.Parameter`` -> per-expert ``nn.Linear``)
       via :func:`linearize_moe`, exposing them to R1/R4.
    2. Replace each MoE router with :class:`LinearRouter` so R1 can be applied (the
       router reads the rotated residual stream and must rotate with it).
    3. Fuse each ``input_layernorm`` into all of its consumers and reset the norm to
       ones. Cohere2MoE's single ``input_layernorm`` feeds attention, the MLP, and the
       router; these span different parents (and the router is absent in the dense first
       layer), so the generic ``match_modules_set`` pass can't group them. The final
       ``model.norm`` -> ``lm_head`` fusion is left to the modifier (must run after
       embeddings are untied).

    Call before ``SpinQuantModifier`` (e.g. before ``oneshot``).

    :param model: ``Cohere2MoeForCausalLM`` to prepare in-place
    """
    # SpinQuant requires RMSNorm: Cohere2MoE uses Cohere2MoeLayerNorm (mean-centering)
    # when `rms_norm_eps is None`, which breaks rotation invariance and norm fusion.
    assert model.config.rms_norm_eps is not None, (
        "SpinQuant requires RMSNorm, but this Cohere2MoE config uses "
        "Cohere2MoeLayerNorm (rms_norm_eps is None); mean-centering breaks "
        "rotation invariance and norm fusion."
    )
    # Shared experts also consume `input_layernorm` in the parallel block but are not
    # handled by `_layer_norm_consumers`, so their norm would not be fused.
    assert getattr(model.config, "num_shared_experts", 0) == 0, (
        "prepare_cohere2_moe_for_spinquant does not support shared experts yet"
        "(num_shared_experts > 0); their input_layernorm consumers are not fused."
    )

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
