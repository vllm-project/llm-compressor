"""REAP support for Transformers native fine-grained FP8 experts."""

import torch
import torch.nn as nn
from compressed_tensors import align_module_device
from transformers.integrations.finegrained_fp8 import FP8Experts

from llmcompressor.modeling.moe.context import get_calibrate_all_experts_flag


class FP8PrunableExperts(FP8Experts):
    """A REAP-aware view of a Transformers :class:`FP8Experts` module.

    Existing FP8 modules are changed to this class in place. This preserves
    their packed parameters, offload state, and checkpoint key names. The stock
    Transformers forward remains active except while REAP norm collection is
    enabled, when an eager path records each unweighted expert output norm.
    """

    _EXPERT_TENSOR_NAMES = (
        "gate_up_proj",
        "gate_up_proj_scale_inv",
        "up_proj",
        "up_proj_scale_inv",
        "down_proj",
        "down_proj_scale_inv",
        "gate_up_proj_activation_scale",
        "down_proj_activation_scale",
    )
    num_experts: int
    _reap_norms: dict[int, torch.Tensor] | None

    def start_reap_norm_collection(self) -> None:
        self._reap_norms: dict[int, torch.Tensor] | None = {}

    def stop_reap_norm_collection(self) -> None:
        self._reap_norms = None

    def take_reap_norms(self) -> dict[int, torch.Tensor]:
        norms = getattr(self, "_reap_norms", None)
        if norms is None:
            return {}

        self._reap_norms = {}
        return norms

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        if getattr(self, "_reap_norms", None) is None:
            return super().forward(hidden_states, top_k_index, top_k_weights)

        # Match Transformers FP8Experts eager math. Fused expert kernels cannot
        # expose the unweighted per-expert outputs needed by REAP.
        final_hidden_states = torch.zeros_like(hidden_states, dtype=torch.float32)
        calibrate_all = get_calibrate_all_experts_flag()

        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(
                top_k_index, num_classes=self.num_experts + 1
            )
            expert_mask = expert_mask.permute(2, 1, 0)
            if calibrate_all:
                expert_indices = range(self.num_experts)
            else:
                expert_indices = (
                    torch.greater(expert_mask.sum(dim=(-1, -2)), 0)
                    .nonzero(as_tuple=False)
                    .view(-1)
                )

        for expert_index in expert_indices:
            if int(expert_index) == self.num_experts:
                continue

            top_k_pos, token_indices = torch.where(expert_mask[expert_index])
            current_state = (
                hidden_states if calibrate_all else hidden_states[token_indices]
            )
            gate_up_activation_scale = (
                self.gate_up_proj_activation_scale[expert_index]
                if self.activation_scheme == "static"
                else None
            )
            projected = self.linear(
                current_state,
                self.gate_up_proj[expert_index]
                if self.has_gate
                else self.up_proj[expert_index],
                self.gate_up_proj_scale_inv[expert_index]
                if self.has_gate
                else self.up_proj_scale_inv[expert_index],
                activation_scale=gate_up_activation_scale,
            )
            projected = (
                self._apply_gate(projected) if self.has_gate else self.act_fn(projected)
            )
            down_activation_scale = (
                self.down_proj_activation_scale[expert_index]
                if self.activation_scheme == "static"
                else None
            )
            expert_output = self.linear(
                projected,
                self.down_proj[expert_index],
                self.down_proj_scale_inv[expert_index],
                activation_scale=down_activation_scale,
            )

            norms = self._reap_norms
            assert norms is not None
            with torch.no_grad():
                norms[int(expert_index)] = torch.linalg.norm(
                    expert_output.float(), dim=-1
                ).reshape(-1)

            if calibrate_all:
                expert_output = expert_output[token_indices]
            routing_weights = top_k_weights[token_indices, top_k_pos, None]
            weighted_output = expert_output * routing_weights.to(expert_output.dtype)
            final_hidden_states.index_add_(
                0,
                token_indices,
                weighted_output.to(final_hidden_states.dtype),
            )

        return final_hidden_states.to(hidden_states.dtype)

    def prune_experts_(self, retained: list[int]) -> None:
        """Slice packed FP8 weights and scales along their expert dimension."""
        weight = self.gate_up_proj if self.has_gate else self.up_proj
        if weight.dtype != torch.float8_e4m3fn:
            raise TypeError(
                "REAP native packed expert pruning only supports e4m3 FP8 "
                f"weights, got {weight.dtype}"
            )
        if not retained:
            raise ValueError("REAP must retain at least one expert")
        if len(set(retained)) != len(retained):
            raise ValueError("REAP retained expert indices must be unique")
        if min(retained) < 0 or max(retained) >= self.num_experts:
            raise IndexError(
                f"REAP retained expert indices must be in [0, {self.num_experts})"
            )

        retained_indices = torch.tensor(retained, dtype=torch.long)
        sliced: dict[str, tuple[torch.Tensor, bool | None]] = {}
        with align_module_device(self):
            for name in self._EXPERT_TENSOR_NAMES:
                tensor = getattr(self, name, None)
                if tensor is None:
                    continue
                if tensor.ndim == 0 or tensor.shape[0] != self.num_experts:
                    raise ValueError(
                        f"Cannot REAP-prune FP8 expert tensor {name}: expected "
                        f"leading dimension {self.num_experts}, got "
                        f"{tuple(tensor.shape)}"
                    )
                indices = retained_indices.to(tensor.device)
                is_parameter = name in self._parameters
                is_buffer = name in self._buffers
                if not is_parameter and not is_buffer:
                    raise TypeError(
                        f"Cannot REAP-prune FP8 expert tensor {name}: expected a "
                        "registered parameter or buffer"
                    )
                sliced[name] = (
                    tensor.detach()[indices].contiguous(),
                    tensor.requires_grad if is_parameter else None,
                )

        for name, (tensor, requires_grad) in sliced.items():
            if requires_grad is not None:
                setattr(
                    self,
                    name,
                    nn.Parameter(tensor, requires_grad=requires_grad),
                )
            else:
                setattr(self, name, tensor)

        self.num_experts = len(retained)


def make_fp8_experts_reap_prunable(module: nn.Module) -> nn.Module:
    """Adapt a loaded Transformers FP8 expert module without copying tensors."""
    if isinstance(module, FP8Experts) and not isinstance(module, FP8PrunableExperts):
        weight = module.gate_up_proj if module.has_gate else module.up_proj
        if weight.dtype != torch.float8_e4m3fn:
            raise TypeError(
                "REAP native packed expert pruning only supports e4m3 FP8 "
                f"weights, got {weight.dtype}"
            )
        module.__class__ = FP8PrunableExperts

        # Both Accelerate and compressed-tensors wrap ``forward`` by saving the
        # method that was active when offloading was installed. REAP adaptation
        # commonly happens after that installation, so changing ``__class__``
        # alone would leave the wrapper calling the stock FP8Experts forward and
        # norm collection would silently remain empty. Refresh only known
        # wrapper slots; preserve any unrelated custom forward implementation.
        old_forward = getattr(module, "_old_forward", None)
        if getattr(old_forward, "__func__", None) is FP8Experts.forward:
            module._old_forward = FP8PrunableExperts.forward.__get__(module)

        if getattr(module, "_original_forward_func", None) is FP8Experts.forward:
            module._original_forward_func = FP8PrunableExperts.forward
    return module
