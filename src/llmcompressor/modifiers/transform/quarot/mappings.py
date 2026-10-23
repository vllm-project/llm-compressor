# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Validated topology for an explicit-expert, non-MTP GLM MLA decoder.

This is a non-mutating plan, not an automatic architecture registry fallback.
It describes rotation geometry independently of transform/quantization execution.
"""

from dataclasses import dataclass

from torch import nn

from llmcompressor.modeling.moe.linear_experts import LinearExperts2D


@dataclass(frozen=True)
class RotationSpace:
    name: str
    size: int
    shifted: bool = False


@dataclass(frozen=True)
class WeightRotation:
    target: str
    space: str
    axis: int
    stride: int
    offset: int = 0


@dataclass(frozen=True)
class NormFusion:
    norm: str
    consumers: tuple[str, ...]


@dataclass(frozen=True)
class RotationPlan:
    spaces: tuple[RotationSpace, ...]
    fusions: tuple[NormFusion, ...]
    embedding: WeightRotation
    rotations: tuple[WeightRotation, ...]


def build_glm_plan(model: nn.Module) -> RotationPlan:
    """Resolve every consumer and validate shapes before any mutation can occur.

    Supports bias-free RMSNorm, positive q_lora_rank, explicit unsharded experts
    and shared experts. Optional indexers are determined by actual module presence.
    MTP and fused/batched experts need additional topology adapters.
    """
    config = model.config
    dimensions = {}
    for name in (
        "hidden_size",
        "q_lora_rank",
        "kv_lora_rank",
        "v_head_dim",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "num_attention_heads",
        "num_hidden_layers",
    ):
        value = getattr(config, name, None)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        dimensions[name] = value
    hidden = dimensions["hidden_size"]
    q_rank = dimensions["q_lora_rank"]
    kv_rank = dimensions["kv_lora_rank"]
    value_dim = dimensions["v_head_dim"]
    nope = dimensions["qk_nope_head_dim"]
    rope = dimensions["qk_rope_head_dim"]
    heads = dimensions["num_attention_heads"]
    layers = model.get_submodule("model.layers")
    if len(layers) != dimensions["num_hidden_layers"]:
        raise ValueError("num_hidden_layers does not match the decoder topology")
    dense_layers = config.first_k_dense_replace
    if (
        not isinstance(dense_layers, int)
        or isinstance(dense_layers, bool)
        or not 0 <= dense_layers <= len(layers)
    ):
        raise ValueError("first_k_dense_replace is outside the decoder topology")
    if any(
        hasattr(layer, "eh_proj") or hasattr(layer, "shared_head") for layer in layers
    ):
        raise NotImplementedError("MTP requires a separate topology adapter")

    spaces = (
        RotationSpace("rot", hidden),
        RotationSpace("rot_b_proj", q_rank, shifted=True),
        RotationSpace("rot_uv", value_dim),
        RotationSpace("rot_kv_b_proj", kv_rank),
    )
    sizes = {space.name: space.size for space in spaces}
    rotations = {space.name: [] for space in spaces}
    fusions = []

    def weight(name, output=None, input=None):
        try:
            module = model.get_submodule(name)
        except AttributeError as error:
            raise ValueError(f"Missing required QuaRot target: {name}") from error
        param = getattr(module, "weight", None)
        if param is None or param.ndim != 2:
            raise ValueError(f"{name} must expose a 2D weight")
        if output is not None and param.shape[0] != output:
            raise ValueError(f"{name}: expected {output} output channels")
        if input is not None and param.shape[1] != input:
            raise ValueError(f"{name}: expected {input} input channels")
        return module

    def add(target, space, axis, stride=None, offset=0):
        module = weight(target)
        stride = sizes[space] if stride is None else stride
        if module.weight.shape[axis] % stride or offset + sizes[space] > stride:
            raise ValueError(f"Invalid rotation segment at {target}")
        rotations[space].append(WeightRotation(target, space, axis, stride, offset))

    def fuse(norm_name, consumers, size):
        norm = model.get_submodule(norm_name)
        if isinstance(norm, nn.LayerNorm) or getattr(norm, "bias", None) is not None:
            raise ValueError(f"{norm_name}: only bias-free RMSNorm is supported")
        if norm.weight.shape != (size,):
            raise ValueError(f"{norm_name}: expected norm gain of size {size}")
        for consumer in consumers:
            weight(consumer, input=size)
        fusions.append(NormFusion(norm_name, tuple(consumers)))

    weight("model.embed_tokens", input=hidden)
    weight("lm_head", input=hidden)
    embedding = WeightRotation("model.embed_tokens", "rot", 1, hidden)
    add("lm_head", "rot", 1)
    for index, layer in enumerate(layers):
        prefix = f"model.layers.{index}"
        attn = f"{prefix}.self_attn"
        qa, qb = f"{attn}.q_a_proj", f"{attn}.q_b_proj"
        kva, kvb = f"{attn}.kv_a_proj_with_mqa", f"{attn}.kv_b_proj"
        out = f"{attn}.o_proj"
        weight(qa, q_rank, hidden)
        weight(qb, heads * (nope + rope), q_rank)
        weight(kva, kv_rank + rope, hidden)
        weight(kvb, heads * (nope + value_dim), kv_rank)
        weight(out, hidden, heads * value_dim)
        inputs, q_consumers = [qa, kva], [qb]
        if getattr(layer.self_attn, "indexer", None) is not None:
            inputs += [f"{attn}.indexer.wk", f"{attn}.indexer.weights_proj"]
            q_consumers += [f"{attn}.indexer.wq_b"]
        fuse(f"{prefix}.input_layernorm", inputs, hidden)
        fuse(f"{attn}.q_a_layernorm", q_consumers, q_rank)
        fuse(f"{attn}.kv_a_layernorm", [kvb], kv_rank)
        for name in inputs:
            add(name, "rot", 1)
        add(out, "rot", 0)
        add(qa, "rot_b_proj", 0)
        for name in q_consumers:
            add(name, "rot_b_proj", 1)
        add(kvb, "rot_uv", 0, nope + value_dim, nope)
        add(out, "rot_uv", 1)
        add(kva, "rot_kv_b_proj", 0, kv_rank + rope)
        add(kvb, "rot_kv_b_proj", 1)

        mlp = f"{prefix}.mlp"
        consumers = []
        if index < dense_layers:
            expert_names = [mlp]
        else:
            experts = getattr(layer.mlp, "experts", None)
            # LLMC's ModuleList subclass also registers an activation module;
            # len(experts) therefore includes one non-expert child.
            count = (
                experts.num_experts
                if isinstance(experts, LinearExperts2D)
                else len(experts)
                if isinstance(experts, nn.ModuleList)
                else 0
            )
            if (
                not isinstance(experts, nn.ModuleList)
                or count != config.n_routed_experts
            ):
                raise ValueError(f"{mlp}: expected all explicit, unsharded experts")
            if getattr(layer.mlp, "shared_experts", None) is None:
                raise ValueError(f"{mlp}: shared experts are required in this adapter")
            expert_names = [f"{mlp}.experts.{i}" for i in range(count)]
            expert_names += [f"{mlp}.shared_experts"]
            weight(f"{mlp}.gate", count, hidden)
            consumers.append(f"{mlp}.gate")
        for expert in expert_names:
            up = weight(f"{expert}.up_proj", input=hidden)
            weight(f"{expert}.gate_proj", up.weight.shape[0], hidden)
            weight(f"{expert}.down_proj", hidden, up.weight.shape[0])
            consumers += [f"{expert}.gate_proj", f"{expert}.up_proj"]
            add(f"{expert}.down_proj", "rot", 0)
        fuse(f"{prefix}.post_attention_layernorm", consumers, hidden)
        for name in consumers:
            add(name, "rot", 1)
    fuse("model.norm", ["lm_head"], hidden)
    return RotationPlan(
        spaces,
        tuple(fusions),
        embedding,
        tuple(op for space in spaces for op in rotations[space.name]),
    )
