"""Helpers for packing linearized experts back into fused 3D modules."""

from enum import Enum, auto

import torch
from compressed_tensors.utils import get_direct_state_dict, replace_direct_state_dict

from .helpers import FusedExpertsProtocol


class ExpertPackMode(Enum):
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


def has_dense_weight(linear: torch.nn.Linear) -> bool:
    return "weight" in dict(get_direct_state_dict(linear))


def extra_param_names(linear: torch.nn.Linear) -> set[str]:
    return set(get_direct_state_dict(linear)) - {"weight", "bias"}


def fused_qparam_suffix(name: str) -> str:
    return name.removeprefix("weight_") if name.startswith("weight_") else name


def set_fused_param(
    fused: FusedExpertsProtocol, name: str, tensor: torch.Tensor
) -> None:
    setattr(fused, name, torch.nn.Parameter(tensor, requires_grad=False))


def combine_gate_up(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
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


def linear_direct_state(linear: torch.nn.Linear) -> dict[str, torch.Tensor]:
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


def require_identical_keys(
    states: list[dict[str, torch.Tensor]], proj_name: str
) -> list[str]:
    keys = [frozenset(state) for state in states]
    if not keys or any(k != keys[0] for k in keys):
        raise RuntimeError(
            f"Cannot pack compressed {proj_name}: experts have mismatched "
            f"parameter names {[sorted(k) for k in keys]}."
        )
    return sorted(keys[0])


def stack_expert_states(
    states: list[dict[str, torch.Tensor]], proj_name: str
) -> dict[str, torch.Tensor]:
    packed: dict[str, torch.Tensor] = {}
    for key in require_identical_keys(states, proj_name):
        packed[key] = torch.stack([state[key] for state in states])
    return packed
