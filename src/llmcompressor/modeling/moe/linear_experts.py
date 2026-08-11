from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar

import torch
from compressed_tensors.offload import get_cache_init_kwargs, offload_module
from transformers import PreTrainedConfig
from transformers.activations import ACT2FN
from transformers.integrations.moe import _default_apply_gate

from llmcompressor.utils.dev import skip_weights_initialize

from .context import get_calibrate_all_experts_flag
from .helpers import (
    FusedExpertsProtocol,
    MoEConfig,
    get_use_experts_implementation_args,
)

# Keep in sync with compressed_tensors QuantizationMetadata weight_* names.
_WEIGHT_QPARAM_NAMES = [
    f"weight_{suffix}"
    for suffix in ("global_scale", "scale", "shape", "zero_point", "g_idx")
]


class ExpertMLP(torch.nn.Module, ABC):
    @abstractmethod
    def copy_from_experts_module(self, experts: FusedExpertsProtocol, index: int):
        raise NotImplementedError()


class ExpertMLPWithGate(ExpertMLP):
    up_proj: torch.nn.Linear
    gate_proj: torch.nn.Linear
    down_proj: torch.nn.Linear
    _apply_gate: Callable[[torch.Tensor], torch.Tensor]

    def __init__(
        self,
        hidden_dim: int,
        intermediate_size: int,
        mlp_bias: bool,
        _apply_gate: Callable[[torch.Tensor], torch.Tensor],
        dtype: torch.dtype,
    ):
        super().__init__()
        self.intermediate_size = intermediate_size
        self.up_proj = torch.nn.Linear(
            hidden_dim, intermediate_size, bias=mlp_bias, dtype=dtype
        )
        self.gate_proj = torch.nn.Linear(
            hidden_dim, intermediate_size, bias=mlp_bias, dtype=dtype
        )
        self.down_proj = torch.nn.Linear(
            intermediate_size, hidden_dim, bias=mlp_bias, dtype=dtype
        )
        self._apply_gate = _apply_gate

    def copy_from_experts_module(self, experts: FusedExpertsProtocol, index: int):
        # load weights
        if not experts.is_transposed:
            gate_weight = experts.gate_up_proj[index, : self.intermediate_size]
            up_weight = experts.gate_up_proj[index, self.intermediate_size :]
            down_weight = experts.down_proj[index]

        else:
            gate_weight = experts.gate_up_proj[index, :, : self.intermediate_size].T
            up_weight = experts.gate_up_proj[index, :, self.intermediate_size :].T
            down_weight = experts.down_proj[index].T

        self.gate_proj.weight.copy_(gate_weight)
        self.up_proj.weight.copy_(up_weight)
        self.down_proj.weight.copy_(down_weight)

        # load biases
        if experts.has_bias:
            gate_bias = experts.gate_up_proj_bias[index, : self.intermediate_size]
            up_bias = experts.gate_up_proj_bias[index, self.intermediate_size :]
            down_bias = experts.down_proj_bias[index]

            self.gate_proj.bias.copy_(gate_bias)
            self.up_proj.bias.copy_(up_bias)
            self.down_proj.bias.copy_(down_bias)

    def copy_to_experts_module(self, experts: FusedExpertsProtocol, index: int):
        """Inverse of :meth:`copy_from_experts_module` for weight (and bias) tensors."""
        if not experts.is_transposed:
            experts.gate_up_proj[index, : self.intermediate_size].copy_(
                self.gate_proj.weight
            )
            experts.gate_up_proj[index, self.intermediate_size :].copy_(
                self.up_proj.weight
            )
            experts.down_proj[index].copy_(self.down_proj.weight)
        else:
            experts.gate_up_proj[index, :, : self.intermediate_size].copy_(
                self.gate_proj.weight.T
            )
            experts.gate_up_proj[index, :, self.intermediate_size :].copy_(
                self.up_proj.weight.T
            )
            experts.down_proj[index].copy_(self.down_proj.weight.T)

        if experts.has_bias:
            experts.gate_up_proj_bias[index, : self.intermediate_size].copy_(
                self.gate_proj.bias
            )
            experts.gate_up_proj_bias[index, self.intermediate_size :].copy_(
                self.up_proj.bias
            )
            experts.down_proj_bias[index].copy_(self.down_proj.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            self._apply_gate(
                torch.cat(
                    [self.gate_proj(hidden_states), self.up_proj(hidden_states)], dim=-1
                )
            )
        )


class ExpertMLPWithoutGate(ExpertMLP):
    up_proj: torch.nn.Linear
    down_proj: torch.nn.Linear
    act_fn: torch.nn.Module

    def __init__(
        self,
        hidden_dim: int,
        intermediate_size: int,
        mlp_bias: bool,
        act_fn: torch.nn.Module,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.intermediate_size = intermediate_size
        self.up_proj = torch.nn.Linear(
            hidden_dim, intermediate_size, bias=mlp_bias, dtype=dtype
        )
        self.down_proj = torch.nn.Linear(
            intermediate_size, hidden_dim, bias=mlp_bias, dtype=dtype
        )
        self.act_fn = act_fn

    def copy_from_experts_module(self, experts: FusedExpertsProtocol, index: int):
        # load weights
        if not experts.is_transposed:
            up_weight = experts.up_proj[index]
            down_weight = experts.down_proj[index]

        else:
            up_weight = experts.up_proj[index].T
            down_weight = experts.down_proj[index].T

        self.up_proj.weight.copy_(up_weight)
        self.down_proj.weight.copy_(down_weight)

        # load biases
        if experts.has_bias:
            up_bias = experts.up_proj_bias[index]
            down_bias = experts.down_proj_bias[index]

            self.up_proj.bias.copy_(up_bias)
            self.down_proj.bias.copy_(down_bias)

    def copy_to_experts_module(self, experts: FusedExpertsProtocol, index: int):
        """Inverse of :meth:`copy_from_experts_module` for weight (and bias) tensors."""
        if not experts.is_transposed:
            experts.up_proj[index].copy_(self.up_proj.weight)
            experts.down_proj[index].copy_(self.down_proj.weight)
        else:
            experts.up_proj[index].copy_(self.up_proj.weight.T)
            experts.down_proj[index].copy_(self.down_proj.weight.T)

        if experts.has_bias:
            experts.up_proj_bias[index].copy_(self.up_proj.bias)
            experts.down_proj_bias[index].copy_(self.down_proj.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.up_proj(hidden_states)))


class LinearExperts2D(torch.nn.ModuleList):
    """

    # 1. try for mappings (efficient load)
    # 2. try for standardized moe, convert after load
    # 3. Explicit replacement (GraniteMoeLinearExperts)

    """

    is_concatenated: ClassVar[bool]
    is_transposed: ClassVar[bool]
    has_bias: ClassVar[bool]
    has_gate: ClassVar[bool]
    _apply_gate: ClassVar[Callable[[torch.Tensor], torch.Tensor]]

    num_experts: int
    intermediate_size: int

    # custom model definitions
    _registry: ClassVar[dict[type[torch.nn.Module], type["LinearExperts2D"]]] = dict()

    @classmethod
    def get_registration(
        cls, key: type[torch.nn.Module], default: Any = None
    ) -> type["LinearExperts2D"]:
        from .granitemoe import GraniteMoeLinearExperts  # noqa: F401
        from .llama4 import Llama4LinearExperts  # noqa: F401

        return cls._registry.get(key, default)

    @classmethod
    def get_linear_experts_cls(
        cls, experts_cls: type[FusedExpertsProtocol]
    ) -> type["LinearExperts2D"]:
        if linear_experts_cls := cls.get_registration(experts_cls):
            return linear_experts_cls

        experts_cls_args = get_use_experts_implementation_args(experts_cls)
        if experts_cls_args is None:
            raise ValueError(
                "Cannot create linear experts class from a class which does not have "
                "the `use_experts_implementation` argument. "
            )

        experts_cls_args["_apply_gate"] = getattr(
            experts_cls, "_apply_gate", _default_apply_gate
        )

        # reuse existing classes to avoid creating excessive types
        linear_experts_cls = type("LinearExperts2D", (cls,), experts_cls_args)
        cls._registry[experts_cls] = linear_experts_cls
        return linear_experts_cls

    @classmethod
    @torch.no_grad()
    def from_experts_module(
        cls, experts: FusedExpertsProtocol, config: PreTrainedConfig
    ):
        with skip_weights_initialize():
            self = cls(config)

        for index in range(self.num_experts):
            expert: ExpertMLP = self[index]
            expert.copy_from_experts_module(experts, index)

        # Needed by :meth:`to_experts_module` / ``repack_moe`` to restore the native
        # fused experts class and config (see issue #2699).
        self._source_experts_cls = experts.__class__
        self._source_config = config

        # copy offloading from original
        offload_kwargs = get_cache_init_kwargs(experts)
        for module in self.modules():
            offload_module(module, **offload_kwargs)

        return self

    @torch.no_grad()
    def to_experts_module(self) -> FusedExpertsProtocol:
        """
        Pack this linearized experts module back into the native fused 3D experts
        module it was created from.

        Restores ``gate_up_proj`` / ``down_proj`` (and ``weight_*`` qparams when
        present) so ``save_pretrained`` writes HF-native 3D keys. This is an
        explicit repack step and does not rely on transformers WeightConverter
        one-to-many mappings.
        """
        experts_cls = getattr(self, "_source_experts_cls", None)
        config = getattr(self, "_source_config", None)
        if experts_cls is None or config is None:
            raise RuntimeError(
                f"{type(self).__name__} is missing source experts metadata for "
                "repack. It must be created via from_experts_module()."
            )

        with skip_weights_initialize():
            fused: FusedExpertsProtocol = experts_cls(config)

        first_param = next(self.parameters(), None)
        if first_param is not None:
            fused.to(device=first_param.device, dtype=first_param.dtype)

        for index in range(self.num_experts):
            expert: ExpertMLP = self[index]
            expert.copy_to_experts_module(fused, index)

        self._pack_weight_qparams(fused)

        offload_kwargs = get_cache_init_kwargs(self)
        offload_module(fused, **offload_kwargs)
        return fused

    def _pack_weight_qparams(self, fused: FusedExpertsProtocol) -> None:
        """
        Pack per-expert Linear ``weight_*`` qparams onto the fused experts module
        as ``{gate_up,up,down}_proj_{suffix}`` (HF / CT native key layout).
        """
        first = self[0]
        has_gate = isinstance(first, ExpertMLPWithGate)

        for qparam in _WEIGHT_QPARAM_NAMES:
            suffix = qparam.removeprefix("weight_")
            if has_gate:
                gate_vals = [
                    getattr(self[i].gate_proj, qparam, None)
                    for i in range(self.num_experts)
                ]
                up_vals = [
                    getattr(self[i].up_proj, qparam, None)
                    for i in range(self.num_experts)
                ]
                if all(g is not None for g in gate_vals) and all(
                    u is not None for u in up_vals
                ):
                    # Match weight packing: concat gate/up on the out-feature axis,
                    # then stack experts. Scalars (global_scale) are stacked only.
                    if gate_vals[0].ndim == 0:
                        packed = torch.stack(
                            [torch.stack([g, u]) for g, u in zip(gate_vals, up_vals)],
                            dim=0,
                        )
                    else:
                        packed = torch.stack(
                            [
                                torch.cat([g, u], dim=-1)
                                for g, u in zip(gate_vals, up_vals)
                            ],
                            dim=0,
                        )
                    setattr(
                        fused,
                        f"gate_up_proj_{suffix}",
                        torch.nn.Parameter(packed, requires_grad=False),
                    )

            down_vals = [
                getattr(self[i].down_proj, qparam, None)
                for i in range(self.num_experts)
            ]
            if all(d is not None for d in down_vals):
                packed = torch.stack(down_vals, dim=0)
                setattr(
                    fused,
                    f"down_proj_{suffix}",
                    torch.nn.Parameter(packed, requires_grad=False),
                )

            if not has_gate:
                up_vals = [
                    getattr(self[i].up_proj, qparam, None)
                    for i in range(self.num_experts)
                ]
                if all(u is not None for u in up_vals):
                    packed = torch.stack(up_vals, dim=0)
                    setattr(
                        fused,
                        f"up_proj_{suffix}",
                        torch.nn.Parameter(packed, requires_grad=False),
                    )

    def __init__(self, config: PreTrainedConfig, *args, **kwargs):
        moe_config = MoEConfig.from_config(config)

        # store num_experts before appending `act_fn` to module list
        self.num_experts = moe_config.num_experts
        self.intermediate_size = moe_config.intermediate_size
        act_fn: torch.nn.Module = ACT2FN[moe_config.hidden_act]

        expert_cls = ExpertMLPWithGate if self.has_gate else ExpertMLPWithoutGate
        post_up_fn = self._apply_gate if self.has_gate else act_fn.forward
        super().__init__(
            [
                expert_cls(
                    moe_config.hidden_dim,
                    moe_config.intermediate_size,
                    moe_config.use_bias,
                    post_up_fn,
                    moe_config.dtype,
                )
                for _ in range(moe_config.num_experts)
            ]
        )

        self.act_fn = act_fn
        self.alpha = moe_config.alpha
        self.limit = moe_config.limit

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)

        # create tokens mask
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)

        for expert_index in range(self.num_experts):
            # select tokens for this expert
            top_k_pos, token_indices = torch.where(expert_mask[expert_index])

            # apply expert
            expert = self[expert_index]
            if get_calibrate_all_experts_flag():
                expert_output = expert(hidden_states)[token_indices]
            else:
                expert_output = expert(hidden_states[token_indices])

            # apply weighting to outputs
            expert_weights = top_k_weights[token_indices, top_k_pos, None]
            weighted_output = expert_output * expert_weights

            # accumulate using index_add_ to match eager implementation exactly
            final_hidden_states.index_add_(
                0, token_indices, weighted_output.to(final_hidden_states.dtype)
            )

        return final_hidden_states
