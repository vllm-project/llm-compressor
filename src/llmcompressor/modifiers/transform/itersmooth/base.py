"""Fused IterSmooth preprocessing, based on msModelSlim's analytic algorithm.

The implementation uses llm-compressor's calibration lifecycle and offload API;
it does not depend on msmodelslim or replace model classes at runtime.
"""

from dataclasses import dataclass, field
from fnmatch import fnmatchcase

import torch
import torch.distributed as dist
from compressed_tensors.offload import get_execution_device, update_offload_parameter
from compressed_tensors.offload.dist_utils import is_distributed
from compressed_tensors.utils import match_modules_set
from loguru import logger
from pydantic import Field, PrivateAttr
from torch import nn
from torch.utils._pytree import tree_leaves

from llmcompressor.core import Event, State
from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.transform.itersmooth.mappings import (
    IterSmoothMapping,
    SubgraphType,
)
from llmcompressor.modifiers.transform.smoothquant.dynamic_mappings import (
    get_layer_mappings_from_model,
)

__all__ = ["IterSmoothModifier"]

_PRIORITY = {"up-down": 1, "ov": 2, "linear-linear": 3, "norm-linear": 4}


@dataclass
class _ResolvedMapping:
    config: IterSmoothMapping
    name: str
    source: nn.Module
    targets: list[nn.Module]
    source_slice: slice = field(default_factory=lambda: slice(None))
    heads: int = 1
    kv_heads: int = 1
    handles: set = field(default_factory=set)
    applied: bool = False


class IterSmoothModifier(Modifier):
    """Redistribute activation outliers into weights before quantization.

    ``s = absmax(X)**alpha / absmax(W)**(1-alpha)`` is calculated once per
    mapped subgraph, in up-down, OV, linear-linear, norm-linear order.
    Statistics are collected at the first target's *input*, including the
    gate or attention operation between source and target. All consumers of
    a source must be included in its mapping to preserve model outputs.

    ``symmetric=False`` centers norm-linear activations and compensates the
    target biases. This requires existing bias parameters on the norm and
    every target so ordinary save/load preserves the model architecture.
    It does not configure the downstream quantizer's symmetry.

    Place this modifier before QuantizationModifier or GPTQModifier and use
    ``pipeline="independent"`` so their activation observers see the transformed
    inputs. Separate oneshot calls are also supported.
    """

    requires_calibration_data: bool = True
    alpha: float = Field(default=0.9, ge=0, le=1, allow_inf_nan=False)
    scale_min: float = Field(default=1e-5, gt=0, le=1, allow_inf_nan=False)
    symmetric: bool = True
    mappings: list[IterSmoothMapping] | None = None
    enable_subgraph_type: list[SubgraphType] = Field(
        default_factory=lambda: list(_PRIORITY)
    )
    include: list[str] | None = None
    exclude: list[str] | None = None

    _resolved: list[_ResolvedMapping] = PrivateAttr(default_factory=list)
    _stats: dict[str, tuple[torch.Tensor, torch.Tensor]] = PrivateAttr(
        default_factory=dict
    )

    def on_initialize(self, state: State, **kwargs) -> bool:
        if self.start not in (None, -1) or self.end not in (None, -1):
            raise ValueError("IterSmoothModifier only supports one-shot compression")
        if state.data.calib is None:
            raise ValueError("IterSmoothModifier requires a calibration dataset")
        self._stats.clear()
        self._resolved = self._resolve_mappings(state.model)
        if not self._resolved:
            raise ValueError("No IterSmooth mappings matched the model")
        return True

    def _infer_mappings(self, model: nn.Module) -> list[IterSmoothMapping]:
        mappings = [
            IterSmoothMapping(source=source, targets=list(targets))
            for targets, source in get_layer_mappings_from_model(model)
        ]
        # Only infer separate projections with the conventional gated-MLP and
        # attention layout. Fused/custom layouts require explicit mappings.
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            parent_name, _, leaf = name.rpartition(".")
            parent = model.get_submodule(parent_name)
            prefix = f"{parent_name}." if parent_name else ""
            if leaf == "up_proj" and all(
                isinstance(getattr(parent, attr, None), nn.Linear)
                for attr in ("gate_proj", "down_proj")
            ):
                mappings.append(
                    IterSmoothMapping(
                        source=name,
                        targets=[prefix + "down_proj"],
                        subgraph_type="up-down",
                    )
                )
            elif leaf == "v_proj" and all(
                isinstance(getattr(parent, attr, None), nn.Linear)
                for attr in ("q_proj", "k_proj", "o_proj")
            ):
                if getattr(parent, "v_norm", None) is not None:
                    continue
                mappings.append(
                    IterSmoothMapping(
                        source=name, targets=[prefix + "o_proj"], subgraph_type="ov"
                    )
                )
        return mappings

    def _resolve_mappings(self, model: nn.Module) -> list[_ResolvedMapping]:
        names = {module: name for name, module in model.named_modules()}
        resolved = []
        seen = set()
        configs = (
            self.mappings if self.mappings is not None else self._infer_mappings(model)
        )
        for config in configs:
            if config.subgraph_type not in self.enable_subgraph_type:
                continue
            for *target_sets, sources in match_modules_set(
                model, [*config.targets, config.source]
            ):
                if len(sources) != 1:
                    raise ValueError("IterSmooth mappings must resolve to one source")
                source = sources[0]
                name = names[source]
                if self.include and not any(
                    fnmatchcase(name, pattern) for pattern in self.include
                ):
                    continue
                if any(fnmatchcase(name, pattern) for pattern in self.exclude or []):
                    continue
                if name in seen:
                    raise ValueError(f"Duplicate IterSmooth source: {name}")
                targets = list(dict.fromkeys(tree_leaves(target_sets)))
                mapping = _ResolvedMapping(config, name, source, targets)
                self._validate_mapping(model, mapping)
                resolved.append(mapping)
                seen.add(name)
        return sorted(
            resolved, key=lambda mapping: _PRIORITY[mapping.config.subgraph_type]
        )

    def _validate_mapping(self, model: nn.Module, mapping: _ResolvedMapping):
        source, targets, config = mapping.source, mapping.targets, mapping.config
        if not targets or any(not isinstance(layer, nn.Linear) for layer in targets):
            raise ValueError(f"{mapping.name}: targets must be torch.nn.Linear layers")
        if source in targets:
            raise ValueError(f"{mapping.name}: source cannot also be a target")
        width = targets[0].in_features
        if any(layer.in_features != width for layer in targets):
            raise ValueError(f"{mapping.name}: targets must have matching input widths")
        if config.subgraph_type == "norm-linear":
            weight = getattr(source, "weight", None)
            if weight is None or weight.ndim != 1 or weight.numel() != width:
                raise ValueError(
                    f"{mapping.name}: expected an affine normalization layer"
                )
            # Offset-gamma norms (e.g. Gemma) require (1 + weight) / s - 1.
            if "gemma" in type(source).__name__.lower():
                raise ValueError(f"{mapping.name}: offset-gamma norms are unsupported")
            if config.fused:
                raise ValueError("fused is only supported for OV and up-down")
            if not self.symmetric and any(
                getattr(layer, "bias", None) is None for layer in [source, *targets]
            ):
                raise ValueError(
                    f"{mapping.name}: asymmetric IterSmooth requires existing biases "
                    "on the norm and all targets for save/load compatibility; "
                    "use symmetric=True for bias-free RMSNorm models"
                )
            return
        if not isinstance(source, nn.Linear) or len(targets) != 1:
            raise ValueError(f"{mapping.name}: expected one linear source and target")
        if config.subgraph_type == "ov":
            parent = model.get_submodule(mapping.name.rpartition(".")[0])
            root_config = getattr(model, "config", None)
            root_config = getattr(root_config, "text_config", root_config)
            attn_config = getattr(parent, "config", root_config)
            mapping.heads = config.num_attention_heads or getattr(
                attn_config, "num_attention_heads", None
            )
            mapping.kv_heads = config.num_key_value_heads or getattr(
                attn_config, "num_key_value_heads", mapping.heads
            )
            if (
                not mapping.heads
                or not mapping.kv_heads
                or mapping.heads % mapping.kv_heads
                or width % mapping.heads
            ):
                raise ValueError(f"{mapping.name}: invalid or missing OV head counts")
            v_width = width // mapping.heads * mapping.kv_heads
            expected = width + 2 * v_width if config.fused else v_width
            if source.out_features != expected:
                raise ValueError(f"{mapping.name}: OV projection widths do not match")
            if config.fused:
                mapping.source_slice = slice(expected - v_width, expected)
        elif config.subgraph_type == "up-down" and config.fused:
            if source.out_features != 2 * width:
                raise ValueError(
                    f"{mapping.name}: expected contiguous [gate, up] weights"
                )
            mapping.source_slice = slice(width, 2 * width)
        elif config.fused or source.out_features != width:
            raise ValueError(f"{mapping.name}: incompatible linear projection widths")

    def on_calibration_start(self, state: State, event: Event, **kwargs):
        for mapping in self._resolved:

            def hook(module, args, kwargs, mapping=mapping):
                if mapping.applied:
                    return
                tensor = args[0] if args else kwargs.get("input")
                if tensor is None:
                    raise ValueError(f"{mapping.name}: missing linear input")
                tensor = tensor.detach().reshape(-1, tensor.shape[-1]).float()
                if tensor.shape[0] == 0:
                    return
                low, high = tensor.amin(dim=0), tensor.amax(dim=0)
                if mapping.name in self._stats:
                    prev_low, prev_high = self._stats[mapping.name]
                    low, high = (
                        torch.minimum(low, prev_low),
                        torch.maximum(high, prev_high),
                    )
                self._stats[mapping.name] = (low, high)

            mapping.handles.add(
                self.register_hook(
                    mapping.targets[0], hook, "forward_pre", with_kwargs=True
                )
            )

    def on_sequential_epoch_end(
        self, state: State, event: Event, modules: list[nn.Module], **kwargs
    ):
        module_set = set(modules)
        for mapping in self._resolved:
            if mapping.applied or mapping.targets[0] not in module_set:
                continue
            if any(
                layer not in module_set for layer in [mapping.source, *mapping.targets]
            ):
                raise ValueError(
                    f"{mapping.name}: mapping crosses sequential subgraphs; "
                    "choose a larger sequential_targets block or pipeline='basic'"
                )
            stats = self._global_stats(mapping)
            if stats is not None:
                self._smooth(mapping, *stats)
            else:
                logger.debug(f"Skipping unobserved IterSmooth subgraph {mapping.name}")
            self._stats.pop(mapping.name, None)
            self.remove_hooks(mapping.handles)
            mapping.handles.clear()
            mapping.applied = True

    def _global_stats(self, mapping: _ResolvedMapping):
        stats = self._stats.get(mapping.name)
        if not is_distributed():
            return stats
        # Every rank enters collectives in mapping order, even for locally
        # unobserved experts. This supports replicated data-parallel models.
        device = get_execution_device(mapping.targets[0])
        width = mapping.targets[0].in_features
        low, high = (
            stats
            if stats is not None
            else (
                torch.full((width,), torch.inf, device=device),
                torch.full((width,), -torch.inf, device=device),
            )
        )
        low, high = low.to(device), high.to(device)
        dist.all_reduce(low, op=dist.ReduceOp.MIN)
        dist.all_reduce(high, op=dist.ReduceOp.MAX)
        return (low, high) if torch.isfinite(low).any() else None

    @torch.no_grad()
    def _smooth(self, mapping: _ResolvedMapping, low: torch.Tensor, high: torch.Tensor):
        source, targets, config = mapping.source, mapping.targets, mapping.config
        if not torch.isfinite(low).all() or not torch.isfinite(high).all():
            raise ValueError(f"{mapping.name}: non-finite calibration activations")
        shifted = not self.symmetric and config.subgraph_type == "norm-linear"
        shift = (high + low) / 2 if shifted else None
        activations = (
            (high - low) / 2 if shifted else torch.maximum(low.abs(), high.abs())
        )
        device = targets[0].weight.device
        weight_scale = (
            torch.stack(
                [
                    layer.weight.detach().float().abs().amax(0).to(device)
                    for layer in targets
                ]
            )
            .amax(0)
            .clamp_min(1e-5)
        )
        scales = activations.to(device).pow(self.alpha) / weight_scale.pow(
            1 - self.alpha
        )
        scales = scales.clamp_min(self.scale_min)
        source_scales = scales
        if config.subgraph_type == "ov":
            head_dim = scales.numel() // mapping.heads
            groups = mapping.heads // mapping.kv_heads
            source_scales = scales.reshape(mapping.kv_heads, groups, head_dim).mean(1)
            scales = source_scales.repeat_interleave(groups, dim=0).reshape(-1)
            source_scales = source_scales.reshape(-1)
        if not torch.isfinite(scales).all():
            raise ValueError(f"{mapping.name}: non-finite IterSmooth scales")
        # Check representability before changing any parameters. In particular,
        # gamma / 1e-5 can overflow FP16 for unobserved/dead activation channels.
        source_rows = source.weight.detach()[mapping.source_slice]
        row_max = (
            source_rows.abs() if source_rows.ndim == 1 else source_rows.abs().amax(1)
        ).float()
        bound = row_max / source_scales.to(row_max.device)
        if getattr(source, "bias", None) is not None:
            bias = source.bias.detach().float()
            if shifted:
                bias = bias - shift.to(bias.device)
            bias_bound = bias[mapping.source_slice].abs() / source_scales.to(
                bias.device
            )
            bound = torch.maximum(bound, bias_bound.to(bound.device))
        if (
            not torch.isfinite(bound).all()
            or (bound > torch.finfo(source.weight.dtype).max).any()
        ):
            raise ValueError(
                f"{mapping.name}: inverse smoothing overflows the source dtype; "
                "increase scale_min or calibrate in float32"
            )
        for target in targets:
            weight = target.weight
            if shifted:
                delta = weight.float() @ shift.to(weight.device)
                update_offload_parameter(
                    target, "bias", target.bias + delta.to(target.bias)
                )
            update_offload_parameter(
                target,
                "weight",
                (weight.float() * scales.to(weight.device)).to(weight.dtype),
            )
        weight = source.weight
        source_scales = source_scales.to(weight.device)
        new_weight = weight.float().clone()
        if weight.ndim == 1:
            new_weight /= source_scales
        else:
            new_weight[mapping.source_slice] /= source_scales[:, None]
        update_offload_parameter(source, "weight", new_weight.to(weight.dtype))
        if getattr(source, "bias", None) is not None:
            bias = source.bias
            new_bias = bias.float().clone()
            if shifted:
                new_bias -= shift.to(bias.device)
            new_bias[mapping.source_slice] /= source_scales.to(bias.device)
            update_offload_parameter(source, "bias", new_bias.to(bias.dtype))
        logger.info(f"Applied IterSmooth to {mapping.name} ({config.subgraph_type})")

    def on_calibration_end(self, state: State, event: Event, **kwargs):
        self.remove_hooks()

    def on_finalize(self, state: State, **kwargs) -> bool:
        self.remove_hooks()
        if self._stats:
            raise ValueError(f"Unprocessed IterSmooth statistics: {list(self._stats)}")
        self._resolved.clear()
        return True
