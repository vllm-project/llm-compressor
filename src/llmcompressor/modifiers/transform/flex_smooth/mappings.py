# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""GLM MLA smoothing subgraphs, independent of the QuaRot rotation plan."""

from dataclasses import dataclass
from typing import Literal

import torch
from compressed_tensors.distributed import is_distributed, is_source_process
from compressed_tensors.offload import align_module_device, update_offload_parameter
from compressed_tensors.offload.cache import DistributedCPUCache, DistributedDiskCache
from torch import nn

from llmcompressor.modifiers.transform.utils.distributed import finish_transform_phase


@dataclass(frozen=True)
class SmoothMapping:
    kind: Literal["norm-linear", "ov"]
    source: str
    consumers: tuple[str, ...]
    heads: int = 1
    value_dim: int = 0
    key_dim: int = 0

    @property
    def targets(self):
        return (self.source, *self.consumers)


def glm_mappings(
    model: nn.Module, subgraphs: tuple[str, ...]
) -> tuple[SmoothMapping, ...]:
    """Resolve the GLM52 reference's two norm groups and segmented OV per layer.

    Supports expanded MLA V heads, not GQA-packed KV weights or MTP. The adapter
    does not infer additional norm fusions from the presence of other linears.
    """
    config = model.config
    layers = model.get_submodule("model.layers")
    dimensions = {}
    for name in (
        "hidden_size",
        "num_attention_heads",
        "q_lora_rank",
        "kv_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
    ):
        value = getattr(config, name, None)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"FlexSmooth requires positive {name}")
        dimensions[name] = value
    if not len(layers) or len(layers) != config.num_hidden_layers:
        raise ValueError("GLM decoder count does not match config")
    h, heads = dimensions["hidden_size"], dimensions["num_attention_heads"]
    q, kv = dimensions["q_lora_rank"], dimensions["kv_lora_rank"]
    key, rope, value = (
        dimensions[name]
        for name in ("qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim")
    )
    if "ov" in subgraphs and getattr(config, "num_key_value_heads", heads) != heads:
        raise ValueError(
            "GLM MLA OV requires expanded V heads; grouped KV layout is unsupported"
        )

    def linear(name, inputs, outputs=None):
        try:
            module = model.get_submodule(name)
        except AttributeError as error:
            raise ValueError(f"Missing FlexSmooth consumer: {name}") from error
        if not isinstance(module, nn.Linear) or module.weight.ndim != 2:
            raise ValueError(f"{name}: expected explicit Linear")
        if module.weight.shape[1] != inputs or (
            outputs is not None and module.weight.shape[0] != outputs
        ):
            raise ValueError(f"{name}: incompatible channel geometry")

    mappings = []
    for index, layer in enumerate(layers):
        if hasattr(layer, "eh_proj") or hasattr(layer, "shared_head"):
            raise ValueError("FlexSmooth MTP is unsupported")
        prefix = f"model.layers.{index}"
        attn = f"{prefix}.self_attn"
        if "ov" in subgraphs:
            linear(f"{attn}.kv_b_proj", kv, heads * (key + value))
            linear(f"{attn}.o_proj", heads * value, h)
            mappings.append(
                SmoothMapping(
                    "ov", f"{attn}.kv_b_proj", (f"{attn}.o_proj",), heads, value, key
                )
            )
        if "norm-linear" in subgraphs:
            linear(f"{attn}.q_a_proj", h, q)
            linear(f"{attn}.kv_a_proj_with_mqa", h, kv + rope)
            linear(f"{attn}.q_b_proj", q, heads * (key + rope))
            inputs = [f"{attn}.q_a_proj", f"{attn}.kv_a_proj_with_mqa"]
            queries = [f"{attn}.q_b_proj"]
            if getattr(layer.self_attn, "indexer", None) is not None:
                inputs += [f"{attn}.indexer.wk", f"{attn}.indexer.weights_proj"]
                queries += [f"{attn}.indexer.wq_b"]
            for norm, consumers, width in (
                (f"{prefix}.input_layernorm", inputs, h),
                (f"{attn}.q_a_layernorm", queries, q),
            ):
                module = model.get_submodule(norm)
                if (
                    not hasattr(module, "weight")
                    or module.weight.shape != (width,)
                    or not (
                        isinstance(module, nn.LayerNorm)
                        or type(module).__name__.endswith("RMSNorm")
                    )
                ):
                    raise ValueError(f"{norm}: expected affine RMSNorm or LayerNorm")
                for name in consumers:
                    linear(name, width)
                mappings.append(SmoothMapping("norm-linear", norm, tuple(consumers)))
    # Apply all OV subgraphs first, preserving reference priority and layer order.
    return tuple(sorted(mappings, key=lambda mapping: mapping.kind != "ov"))


@torch.no_grad()
def apply_scales(model: nn.Module, mapping: SmoothMapping, scale: torch.Tensor):
    """Commit one mapping after every rank has finished reading its old weights.

    Shared backing is written once. Other ranks copy their already computed
    result into CT's current resident cache; no extra clone or backing read.
    """
    error, pending = None, []
    try:
        pending = _prepare_scale_updates(model, mapping, scale)
    except Exception as caught:
        error = caught
    finish_transform_phase(error, f"FlexSmooth prepare scales: {mapping.source}")
    error = None
    try:
        for module, name, value in pending:
            cache = module._parameters
            shared = isinstance(cache, (DistributedCPUCache, DistributedDiskCache))
            if not is_distributed() or not shared or is_source_process():
                update_offload_parameter(module, name, value)
            else:
                backing = cache.offloaded_values[name]
                resident = cache.keep_onloaded_values.get(backing)
                if resident is not None and resident is not backing:
                    resident.copy_(value)
    except Exception as caught:
        error = caught
    finish_transform_phase(error, f"FlexSmooth commit scales: {mapping.source}")


def _prepare_scale_updates(model, mapping, scale):
    if not torch.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError("FlexSmooth scales must be finite and positive")
    pending = []
    for name in mapping.targets:
        module = model.get_submodule(name)
        with align_module_device(module):
            local = scale.to(device=module.weight.device, dtype=module.weight.dtype)
            if name != mapping.source:
                factor = local.reshape(1, -1)
            elif mapping.kind == "norm-linear":
                factor = local.reciprocal()
            else:
                factor = torch.ones(
                    module.weight.shape[0], device=local.device, dtype=local.dtype
                )
                factor.view(mapping.heads, mapping.key_dim + mapping.value_dim)[
                    :, mapping.key_dim :
                ] = local.reshape(mapping.heads, mapping.value_dim).reciprocal()
                factor = factor[:, None]
            updated = module.weight.detach() * factor
            if not torch.isfinite(updated).all():
                raise ValueError(f"Nonfinite smoothed weight: {name}")
            pending.append((module, "weight", updated))
            if name == mapping.source and getattr(module, "bias", None) is not None:
                updated_bias = module.bias.detach() * factor.flatten()
                if not torch.isfinite(updated_bias).all():
                    raise ValueError(f"Nonfinite smoothed bias: {name}")
                pending.append((module, "bias", updated_bias))
    return pending
