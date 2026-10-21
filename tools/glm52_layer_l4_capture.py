# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Capture one complete prefill during an existing baseline inference run."""

import inspect
import json
from contextlib import ExitStack, contextmanager
from pathlib import Path

import torch
from safetensors.torch import save_file

from tools.glm52_l4 import digest_tensor, safe_output
from tools.glm52_layer_l4 import (
    CACHE_KEY,
    layer_plan,
    one_layer_config,
    validate_values,
)


@contextmanager
def capture_layer(
    model,
    *,
    model_dir,
    layer,
    output,
    input_kind,
    input_description,
    max_tokens=128,
    max_weight_gib=2,
):
    """Observe, never load/run/transform a model. No partial context truncation.

    Use an unquantized baseline with explicit linear experts, batch=1, eager
    attention, use_cache=False and an unpadded causal prefill. The caller owns
    inference/offloading. Exceptions remove hooks and write no artifact.
    All selected live parameters must match the checkpoint used by replay.
    """
    if input_kind not in ("real", "synthetic") or not input_description.strip():
        raise ValueError("Declare input_kind and a nonempty description")
    if not 1 <= max_tokens <= 4096:
        raise ValueError("max_tokens must be 1..4096")
    root = Path(model_dir).resolve()
    destination = safe_output(Path(output), [root])
    cfg, plan = layer_plan(root, layer, max_weight_gib)
    live = one_layer_config(model.config.to_dict(), layer)
    for key in (
        "hidden_size",
        "intermediate_size",
        "moe_intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "q_lora_rank",
        "kv_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
        "rms_norm_eps",
        "rope_parameters",
        "attention_bias",
        "mlp_bias",
        "hidden_act",
        "n_routed_experts",
        "n_shared_experts",
        "num_experts_per_tok",
        "n_group",
        "topk_group",
        "norm_topk_prob",
        "routed_scaling_factor",
        "index_topk",
        "index_n_heads",
        "index_head_dim",
        "indexer_types",
        "mlp_layer_types",
    ):
        if getattr(live, key) != getattr(cfg, key):
            raise ValueError(f"Live and checkpoint configs differ: {key}")
    if any(
        getattr(model.config, key, None)
        for key in ("quarot_config", "flex_smooth_config", "quantization_config")
    ):
        raise ValueError("Capture requires an untransformed, unquantized baseline")
    decoder = model.get_submodule(f"model.layers.{layer}")
    if decoder.training or decoder.self_attn.config._attn_implementation != "eager":
        raise ValueError("Capture requires eval() and eager attention")
    if any(
        getattr(module, "quantization_scheme", None) is not None
        for module in decoder.modules()
    ):
        raise ValueError("Quantized modules are not baseline capture sources")

    def fingerprints():
        result = {}
        for name in plan["selected_weights"]:
            parent, leaf = name.rsplit(".", 1)
            value = getattr(decoder.get_submodule(parent), leaf)
            if value.is_meta or not torch.isfinite(value).all():
                raise ValueError(
                    f"Selected live tensor is unavailable/nonfinite: {name}"
                )
            result[name] = digest_tensor(value)
        return result

    initial = fingerprints()
    captured, completed = {}, False
    signature = inspect.signature(decoder.forward)

    def pre_hook(module, args, kwargs):
        if captured:
            raise ValueError("Capture exactly one selected-layer forward per artifact")
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = bound.arguments
        if arguments.get("past_key_values") is not None or arguments.get("use_cache"):
            raise ValueError("Only cache-free prefill is supported")
        extras = arguments.get("kwargs", {})
        if extras:
            raise ValueError(f"Unsupported forward kwargs: {sorted(extras)}")
        x = arguments["hidden_states"]
        if x.ndim != 3 or x.shape[1] > max_tokens:
            raise ValueError("Prefill exceeds max_tokens; do not truncate context")
        position = arguments.get("position_embeddings")
        if not isinstance(position, (tuple, list)) or len(position) != 2:
            raise ValueError("Capture explicit RoPE cos/sin tensors")
        values = {
            "hidden_states": x,
            "attention_mask": arguments.get("attention_mask"),
            "position_ids": arguments.get("position_ids"),
            "cos": position[0],
            "sin": position[1],
        }
        if arguments.get("prev_topk_indices") is not None:
            values["prev_topk_indices"] = arguments["prev_topk_indices"]
        if any(not isinstance(value, torch.Tensor) for value in values.values()):
            raise ValueError("All prefill inputs must be explicit tensors")
        captured.update(
            {
                name: value.detach().cpu().contiguous().clone()
                for name, value in values.items()
            }
        )
        validate_values(captured, cfg)

    def post_hook(module, args, kwargs, result):
        nonlocal completed
        if not isinstance(result, tuple) or len(result) != 2:
            raise ValueError("Unexpected complete decoder output")
        captured["baseline_output"] = result[0].detach().cpu().contiguous().clone()
        captured["baseline_topk"] = result[1].detach().cpu().contiguous().clone()
        validate_values(captured, cfg)
        completed = True

    hooks = [
        decoder.register_forward_pre_hook(pre_hook, with_kwargs=True),
        decoder.register_forward_hook(post_hook, with_kwargs=True),
    ]
    try:
        yield
        if not completed:
            raise ValueError("Selected layer did not finish; no cache written")
        if fingerprints() != initial:
            raise ValueError("Selected baseline weights changed during capture")
        metadata = {
            "schema_version": 1,
            "stage": "baseline_complete_layer",
            "input_kind": input_kind,
            "input_description": input_description,
            "layer": layer,
            "config_sha256": plan["config_sha256"],
            "weight_sha256": initial,
            "tokens": captured["hidden_states"].shape[1],
            "attention_implementation": "eager",
            "use_cache": False,
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Reserve exclusively before safetensors writes; never overwrite user data.
        with destination.open("xb"):
            pass
        try:
            save_file(
                captured, str(destination), metadata={CACHE_KEY: json.dumps(metadata)}
            )
        except Exception:
            destination.unlink(missing_ok=True)
            raise
    finally:
        for hook in hooks:
            hook.remove()


@torch.no_grad()
def capture_prefill(
    model,
    inputs,
    *,
    model_dir,
    output_dir,
    layers=(0, 3),
    input_kind,
    input_description,
    max_tokens=128,
    max_weight_gib=40,
):
    """Use an already-loaded baseline; stop AFTER the last selected whole layer.

    No model/tokenizer is loaded here. The caller supplies one existing tokenized
    prompt and its existing offload context. No old o_proj early-stop hook may
    remain installed. The returned files can be replayed independently.
    """
    layers = tuple(sorted(set(layers)))
    if not 1 <= len(layers) <= 4:
        raise ValueError("Select 1..4 distinct decoder layers")
    if inputs.get("past_key_values") is not None:
        raise ValueError("A fresh prefill is required")
    outputs = {
        layer: Path(output_dir) / f"glm52-layer{layer}-complete.safetensors"
        for layer in layers
    }

    class Complete(Exception):
        pass

    def stop(module, args, result):
        raise Complete()

    with ExitStack() as stack:
        for layer in layers:
            stack.enter_context(
                capture_layer(
                    model,
                    model_dir=model_dir,
                    layer=layer,
                    output=outputs[layer],
                    input_kind=input_kind,
                    input_description=input_description,
                    max_tokens=max_tokens,
                    max_weight_gib=max_weight_gib,
                )
            )
        handle = model.get_submodule(
            f"model.layers.{layers[-1]}"
        ).register_forward_hook(stop)
        try:
            try:
                model(**{**inputs, "use_cache": False})
            except Complete:
                pass
        finally:
            handle.remove()
    return outputs
