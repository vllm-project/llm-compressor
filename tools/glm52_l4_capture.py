# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Capture bounded baseline inputs during an EXISTING user-managed inference run.

This helper never loads a model or runs inference. Use it as a context manager;
only a selected attention layer is observed, with at most 128 tokens per hook.
"""

import json
from contextlib import contextmanager
from pathlib import Path

import torch
from safetensors.torch import save_file

from tools.glm52_l4 import (
    CACHE_MODULES,
    checkpoint_plan,
    digest_tensor,
    safe_output,
)


@contextmanager
def capture_inputs(
    model,
    *,
    model_dir,
    layer,
    output,
    input_kind,
    input_description,
    max_tokens=128,
):
    """Observe a baseline layer without changing inputs, outputs or parameters.

    input_kind must explicitly be 'real' or 'synthetic'. Describe the existing
    tokenization/dataset, mask, position and prefill/decode settings in
    input_description. The artifact is for isolated projection tests, not replay
    of attention, so it does not serialize attention masks or positions.
    Hooks are removed even when inference fails. Failed captures write no file.
    """
    if input_kind not in ("real", "synthetic") or not input_description.strip():
        raise ValueError("Declare input_kind and a nonempty input_description")
    if not 1 <= max_tokens <= 128:
        raise ValueError("max_tokens must be between 1 and 128")
    model_dir = Path(model_dir).resolve()
    destination = safe_output(Path(output), [model_dir])
    _, plan = checkpoint_plan(model_dir, layer)
    if any(
        getattr(model.config, key, None)
        for key in ("quarot_config", "flex_smooth_config", "quantization_config")
    ):
        raise ValueError(
            "Capture requires an untransformed, unquantized baseline model"
        )
    for name in plan["selected_weights"]:
        module = model.get_submodule(f"model.layers.{layer}.{name}")
        if getattr(module, "quantization_scheme", None) is not None:
            raise ValueError(
                "Quantization-disabled modules are insufficient; use baseline modules"
            )

    def fingerprints():
        return {
            name: digest_tensor(model.get_parameter(entry["key"]))
            for name, entry in plan["selected_weights"].items()
        }

    initial_fingerprints = fingerprints()
    cache, used, seen = {}, {}, {}
    handles = []

    def keep(key, value):
        if not isinstance(value, torch.Tensor) or value.ndim < 2:
            raise ValueError(f"Expected batched tensor for {key}")
        flat = value.detach().reshape(-1, value.shape[-1])
        seen[key] = seen.get(key, 0) + flat.shape[0]
        take = min(flat.shape[0], max_tokens - used.get(key, 0))
        if take <= 0:
            return
        sample = flat[:take].cpu().clone()
        if not sample.is_floating_point() or not torch.isfinite(sample).all():
            raise ValueError(f"Invalid activation: {key}")
        cache.setdefault(key, []).append(sample)
        used[key] = used.get(key, 0) + take

    def get_input(args, kwargs):
        if args:
            return args[0]
        return kwargs.get("hidden_states", kwargs.get("input"))

    def norm_hook(key):
        def hook(module, args, kwargs, result):
            keep(key + ".input", get_input(args, kwargs))
            keep(key + ".output", result)

        return hook

    def input_hook(key):
        def hook(module, args, kwargs):
            keep(key + ".input", get_input(args, kwargs))

        return hook

    try:
        for key, name in CACHE_MODULES.items():
            module = model.get_submodule(f"model.layers.{layer}.{name}")
            if key.endswith("norm"):
                handles.append(
                    module.register_forward_hook(norm_hook(key), with_kwargs=True)
                )
            else:
                handles.append(
                    module.register_forward_pre_hook(input_hook(key), with_kwargs=True)
                )
        yield
        expected = {
            key + "." + suffix
            for key in CACHE_MODULES
            for suffix in (("input", "output") if key.endswith("norm") else ("input",))
        }
        if set(cache) != expected:
            raise ValueError(
                f"Missing executed hooks: {sorted(expected - set(cache))}; "
                "fused attention may bypass modules"
            )
        if fingerprints() != initial_fingerprints:
            raise ValueError("Selected model weights changed during capture")
        metadata = {
            "schema_version": 1,
            "stage": "baseline",
            "layer": layer,
            "config_sha256": plan["config_sha256"],
            "weight_sha256": initial_fingerprints,
            "input_kind": input_kind,
            "input_description": input_description,
            "tokens_seen": seen,
            "tokens_saved": used,
            "scope": "isolated attention subgraphs; no mask/position replay",
        }
        # Exclusive creation protects an existing result, including concurrent runs.
        # save_file requires a path; reserve it first and remove only our file on error.
        with destination.open("xb"):
            pass
        try:
            save_file(
                {key: torch.cat(parts).contiguous() for key, parts in cache.items()},
                str(destination),
                metadata={"glm52_l4": json.dumps(metadata)},
            )
        except Exception:
            destination.unlink()
            raise
    finally:
        for handle in handles:
            handle.remove()
