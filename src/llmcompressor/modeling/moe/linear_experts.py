from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, Callable, ClassVar

import torch
from compressed_tensors.offload import get_cache_init_kwargs, offload_module
from compressed_tensors.quantization import QuantizationStatus
from compressed_tensors.utils import get_direct_state_dict, replace_direct_state_dict
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


class _ExpertPackMode(Enum):
    DENSE = auto()
    COMPRESSED = auto()


class CompressedFusedLinear(torch.nn.Module):
    """Serialization-only fused projection that owns compressed 3D tensors.

    Native fused experts store ``gate_up_proj`` / ``down_proj`` as Parameters.
    Compressed checkpoints instead need nested keys such as
    ``experts.down_proj.weight_packed``, so this module replaces those
    Parameters before ``save_pretrained``. It is not intended for fused
    expert forward.
    """

    def __init__(self, state: dict[str, torch.Tensor]):
        super().__init__()
        replace_direct_state_dict(self, state)


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

        self.copy_bias_to_experts_module(experts, index)

    def copy_bias_to_experts_module(
        self, experts: FusedExpertsProtocol, index: int
    ) -> None:
        if not experts.has_bias:
            return
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

        self.copy_bias_to_experts_module(experts, index)

    def copy_bias_to_experts_module(
        self, experts: FusedExpertsProtocol, index: int
    ) -> None:
        if not experts.has_bias:
            return
        experts.up_proj_bias[index].copy_(self.up_proj.bias)
        experts.down_proj_bias[index].copy_(self.down_proj.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.up_proj(hidden_states)))


class LinearExperts2D(torch.nn.ModuleList):
    """

    # 1. try for mappings (efficient load)
    # 2. try for standardized moe, convert after load
    # 3. Explicit replacement (Llama4LinearExperts)

    """

    is_concatenated: ClassVar[bool]
    is_transposed: ClassVar[bool]
    has_bias: ClassVar[bool]
    has_gate: ClassVar[bool]
    _apply_gate: ClassVar[Callable[[torch.Tensor], torch.Tensor]]

    # override when the generic gate_up_proj split does not fit the model
    expert_cls_with_gate: ClassVar[type[ExpertMLP]] = ExpertMLPWithGate
    expert_cls_without_gate: ClassVar[type[ExpertMLP]] = ExpertMLPWithoutGate

    num_experts: int
    intermediate_size: int

    # custom model definitions
    _registry: ClassVar[dict[type[torch.nn.Module], type["LinearExperts2D"]]] = dict()

    @classmethod
    def get_registration(
        cls, key: type[torch.nn.Module], default: Any = None
    ) -> type["LinearExperts2D"]:
        from .gpt_oss import GptOssLinearExperts  # noqa: F401
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
        self._record_source_metadata(experts, config)

        # copy offloading from original
        offload_kwargs = get_cache_init_kwargs(experts)
        for module in self.modules():
            offload_module(module, **offload_kwargs)

        return self

    def _record_source_metadata(
        self, experts: FusedExpertsProtocol, config: PreTrainedConfig
    ) -> None:
        self._source_experts_cls = experts.__class__
        self._source_config = config

    @torch.no_grad()
    def to_experts_module(self) -> FusedExpertsProtocol:
        """
        Pack this linearized experts module back into the native fused 3D experts
        module it was created from.

        Dense experts restore native ``gate_up_proj`` / ``down_proj`` Parameters
        (and sibling ``weight_*`` qparams when present). Compressed experts
        replace those Parameters with serialization-only nested modules that
        own packed tensors and qparams. Call after
        ``ModelCompressor.compress_model`` when saving compressed 3D
        checkpoints.
        """
        experts_cls, config = self._require_source_metadata()
        pack_mode = self._expert_pack_mode()

        with skip_weights_initialize():
            fused: FusedExpertsProtocol = experts_cls(config)

        first_param = next(self.parameters(), None)
        if first_param is not None:
            fused.to(device=first_param.device)
        float_param = next(
            (
                param
                for param in self.parameters()
                if param.is_floating_point() or param.is_complex()
            ),
            None,
        )
        if float_param is not None:
            fused.to(dtype=float_param.dtype)

        if pack_mode is _ExpertPackMode.DENSE:
            for index in range(self.num_experts):
                expert: ExpertMLP = self[index]
                expert.copy_to_experts_module(fused, index)
            self._pack_weight_qparams(fused)
        else:
            self._pack_compressed_projections(fused)
            if self.has_bias:
                for index in range(self.num_experts):
                    self[index].copy_bias_to_experts_module(fused, index)

        offload_kwargs = get_cache_init_kwargs(self)
        offload_module(fused, **offload_kwargs)
        for child in fused.children():
            if isinstance(child, CompressedFusedLinear):
                offload_module(child, **offload_kwargs)
        return fused

    def _require_source_metadata(self) -> tuple[type, PreTrainedConfig]:
        experts_cls = getattr(self, "_source_experts_cls", None)
        config = getattr(self, "_source_config", None)
        if experts_cls is None or config is None:
            raise RuntimeError(
                f"{type(self).__name__} is missing source experts metadata for "
                "repack. It must be created via from_experts_module() or "
                "load_quantizable_moe()."
            )
        return experts_cls, config

    def _expert_pack_mode(self) -> _ExpertPackMode:
        linears = [
            linear
            for expert_index in range(self.num_experts)
            for linear in self[expert_index].modules()
            if isinstance(linear, torch.nn.Linear)
        ]
        statuses = {getattr(linear, "quantization_status", None) for linear in linears}
        if not linears or statuses == {None}:
            return _ExpertPackMode.DENSE
        if QuantizationStatus.COMPRESSED in statuses and statuses <= {
            QuantizationStatus.COMPRESSED,
            None,
        }:
            if None in statuses:
                raise RuntimeError(
                    "Cannot repack a mix of compressed and uncompressed expert "
                    "Linears. Compress all targeted experts before calling "
                    "repack_moe()."
                )
            return _ExpertPackMode.COMPRESSED
        if statuses & {
            QuantizationStatus.INITIALIZED,
            QuantizationStatus.CALIBRATION,
            QuantizationStatus.FROZEN,
        }:
            raise RuntimeError(
                "Cannot repack quantized expert Linears before they are "
                "compressed. Call ModelCompressor.compress_model(model) "
                "before repack_moe()."
            )
        raise RuntimeError(
            f"Cannot repack experts with quantization statuses {statuses}."
        )

    def _pack_weight_qparams(self, fused: FusedExpertsProtocol) -> None:
        """
        Pack per-expert Linear ``weight_*`` qparams onto the fused experts module
        as ``{gate_up,up,down}_proj_{suffix}`` (HF / CT native key layout).
        """
        for qparam in _WEIGHT_QPARAM_NAMES:
            suffix = qparam.removeprefix("weight_")
            if self.has_gate:
                gate = self._stack_proj_attr("gate_proj", qparam)
                up = self._stack_proj_attr("up_proj", qparam)
                if gate is not None and up is not None:
                    packed = _combine_gate_up(gate, up)
                    _set_fused_param(fused, f"gate_up_proj_{suffix}", packed)
            else:
                up = self._stack_proj_attr("up_proj", qparam)
                if up is not None:
                    _set_fused_param(fused, f"up_proj_{suffix}", up)

            down = self._stack_proj_attr("down_proj", qparam)
            if down is not None:
                _set_fused_param(fused, f"down_proj_{suffix}", down)

    def _stack_proj_attr(self, proj_name: str, qparam: str) -> torch.Tensor | None:
        # Index by num_experts: ModuleList also stores act_fn after the experts.
        vals = [
            getattr(getattr(self[i], proj_name), qparam, None)
            for i in range(self.num_experts)
        ]
        if any(val is None for val in vals):
            return None
        return torch.stack(vals)

    def _pack_compressed_projections(self, fused: FusedExpertsProtocol) -> None:
        packed: dict[str, dict[str, torch.Tensor]] = {}
        if self.has_gate:
            packed["gate_up_proj"] = self._pack_gated_compressed_state()
        else:
            packed["up_proj"] = self._stack_compressed_proj("up_proj")
        packed["down_proj"] = self._stack_compressed_proj("down_proj")

        for name, state in packed.items():
            if name in fused._parameters:
                del fused._parameters[name]
            fused.add_module(name, CompressedFusedLinear(state))

    def _stack_compressed_proj(self, proj_name: str) -> dict[str, torch.Tensor]:
        states = [
            _linear_direct_state(getattr(self[i], proj_name))
            for i in range(self.num_experts)
        ]
        return _stack_expert_states(states, proj_name)

    def _pack_gated_compressed_state(self) -> dict[str, torch.Tensor]:
        gate_states = [
            _linear_direct_state(self[i].gate_proj) for i in range(self.num_experts)
        ]
        up_states = [
            _linear_direct_state(self[i].up_proj) for i in range(self.num_experts)
        ]
        gate_keys = _require_identical_keys(gate_states, "gate_proj")
        up_keys = _require_identical_keys(up_states, "up_proj")
        if gate_keys != up_keys:
            raise RuntimeError(
                "Cannot pack gated compressed experts: gate_proj and up_proj "
                f"have different compressed keys {gate_keys} vs {up_keys}."
            )
        packed: dict[str, torch.Tensor] = {}
        for key in gate_keys:
            packed[key] = _combine_gate_up(
                torch.stack([state[key] for state in gate_states]),
                torch.stack([state[key] for state in up_states]),
            )
        return packed

    def __init__(self, config: PreTrainedConfig, *args, **kwargs):
        moe_config = MoEConfig.from_config(config)

        # store num_experts before appending `act_fn` to module list
        self.num_experts = moe_config.num_experts
        self.intermediate_size = moe_config.intermediate_size
        act_fn: torch.nn.Module = ACT2FN[moe_config.hidden_act]

        expert_cls = (
            self.expert_cls_with_gate if self.has_gate else self.expert_cls_without_gate
        )
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
        self.swiglu_limit = moe_config.limit

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


def _set_fused_param(
    fused: FusedExpertsProtocol, name: str, tensor: torch.Tensor
) -> None:
    setattr(fused, name, torch.nn.Parameter(tensor, requires_grad=False))


def _combine_gate_up(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Combine per-expert gate/up tensors along the out-feature axis.

    Scalars (e.g. global_scale) stack to ``[E, 2]`` so Transformers
    ``Interleave(dim=1)`` still applies to the packed companion.
    """
    if gate.shape != up.shape:
        raise RuntimeError(
            "Cannot combine gate/up compressed tensors with shapes "
            f"{tuple(gate.shape)} and {tuple(up.shape)}."
        )
    if gate.ndim <= 1 or all(size == 1 for size in gate.shape[1:]):
        # Per-expert scalars, stacked [E], and [E, 1, ...] global scales
        # become [E, 2] so Interleave(dim=1) still applies.
        squeezed_gate = gate.reshape(gate.shape[0])
        squeezed_up = up.reshape(up.shape[0])
        return torch.stack([squeezed_gate, squeezed_up], dim=-1)
    # Concatenate along the out-feature axis: stacked [E, O, ...] uses dim 1.
    concat_dim = 1 if gate.ndim >= 3 else -1
    return torch.cat([gate, up], dim=concat_dim)


def _linear_direct_state(linear: torch.nn.Linear) -> dict[str, torch.Tensor]:
    state = dict(get_direct_state_dict(linear))
    if "weight" in state:
        raise RuntimeError(
            "Cannot pack compressed experts that still have a dense "
            "`weight`. Call ModelCompressor.compress_model(model) "
            "before repack_moe()."
        )
    # Native fused experts keep bias as sibling Parameters, not nested
    # ``gate_up_proj.bias`` / ``down_proj.bias``.
    state.pop("bias", None)
    if not state:
        raise RuntimeError(
            "Cannot pack compressed experts with empty projection state."
        )
    return {name: tensor.detach() for name, tensor in state.items()}


def _require_identical_keys(
    states: list[dict[str, torch.Tensor]], proj_name: str
) -> list[str]:
    keys = [frozenset(state) for state in states]
    if not keys or any(k != keys[0] for k in keys):
        raise RuntimeError(
            f"Cannot pack compressed {proj_name}: experts have mismatched "
            f"parameter names {[sorted(k) for k in keys]}."
        )
    return sorted(keys[0])


def _stack_expert_states(
    states: list[dict[str, torch.Tensor]], proj_name: str
) -> dict[str, torch.Tensor]:
    packed: dict[str, torch.Tensor] = {}
    for key in _require_identical_keys(states, proj_name):
        packed[key] = torch.stack([state[key] for state in states])
    return packed
