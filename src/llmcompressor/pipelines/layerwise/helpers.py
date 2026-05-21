"""
Helpers for layerwise weight loading and offloading from safetensors files.

Optimizations:
- On-demand shard downloads: only fetches the safetensors shards needed for
  the current subgraph instead of downloading the full model upfront.
- Compress-as-you-go: compresses and saves each subgraph immediately after
  calibration, keeping only 1 subgraph of base weights in memory at a time.
- Background prefetch: downloads the next subgraph's shards while the current
  subgraph is being calibrated on GPU.
"""

import json
import os
import threading
from pathlib import Path

import torch
from loguru import logger
from safetensors import safe_open
from safetensors.torch import save_file
from torch.nn import Module

__all__ = [
    "build_weight_map",
    "build_key_remapping",
    "get_subgraph_weight_names",
    "load_subgraph_weights",
    "move_subgraph_buffers",
    "offload_subgraph_weights",
    "compress_and_save_subgraph",
    "copy_passthrough_weights",
    "write_safetensors_index",
    "ShardPrefetcher",
]


def _all_named_parameters(model: Module):
    """
    Yield (name, param) for ALL parameters including tied duplicates.

    Unlike ``model.named_parameters()`` which deduplicates by ``id(param)``,
    this yields every parameter path. This is needed to detect tied weights
    like ``lm_head.weight`` tied to ``model.embed_tokens.weight``.
    """
    for module_name, module in model.named_modules():
        for param_name, param in module._parameters.items():
            if param is not None:
                full_name = f"{module_name}.{param_name}" if module_name else param_name
                yield full_name, param


def _detect_tied_weights(model: Module, weight_map: dict[str, str]) -> dict[str, str]:
    """
    Detect model parameters that are tied to other parameters but missing
    from the weight map. Returns ``{tied_name: source_name}`` where
    ``source_name`` is the key in ``weight_map`` that holds the data.

    Uses two detection methods:
    1. On meta device with tie_weights() called: tied params share same ``id()``.
    2. Fallback: check model config's ``tie_word_embeddings`` flag for the
       common lm_head <-> embed_tokens tying pattern (since ``from_config()``
       with ``init_empty_weights()`` may not actually share tensor objects).
    """
    # Ensure tie_weights() is called so tied params share the same id
    if hasattr(model, "tie_weights"):
        model.tie_weights()

    # Map tensor id -> first param name found in weight_map
    id_to_source: dict[int, str] = {}
    all_params: dict[str, torch.nn.Parameter] = {}

    for name, param in _all_named_parameters(model):
        all_params[name] = param
        tid = id(param)
        if name in weight_map and tid not in id_to_source:
            id_to_source[tid] = name

    tied: dict[str, str] = {}
    for name, param in all_params.items():
        if name not in weight_map:
            source = id_to_source.get(id(param))
            if source:
                tied[name] = source

    # Fallback: if id-based detection didn't find anything, check config
    # for the common tie_word_embeddings pattern
    if not tied:
        config = getattr(model, "config", None)
        if config and getattr(config, "tie_word_embeddings", False):
            # Common pattern: lm_head.weight is tied to embed_tokens.weight
            # Find the embed_tokens weight in weight_map
            embed_key = None
            for key in weight_map:
                if "embed_tokens.weight" in key:
                    embed_key = key

            # Also find lm_head.weight in model params (even if not in wm)
            lm_head_param = None
            for name in all_params:
                if name.endswith("lm_head.weight") or name == "lm_head.weight":
                    lm_head_param = name

            if embed_key and lm_head_param and lm_head_param not in weight_map:
                tied[lm_head_param] = embed_key

    if tied:
        logger.info(
            f"Detected {len(tied)} tied weight(s): "
            + ", ".join(f"{t} -> {s}" for t, s in tied.items())
        )
    return tied


def build_key_remapping(
    weight_map: dict[str, str],
    model: Module,
) -> tuple[dict[str, str], dict[str, str], list[str], dict[str, str]]:
    """
    Auto-detect and build a bidirectional key remapping between safetensors
    weight names and model parameter names.

    This handles VL/multimodal models where safetensors keys use a different
    prefix than the CausalLM model. For example:
      - Safetensors: ``model.language_model.layers.0.*``
      - CausalLM:    ``model.layers.0.*``

    Also detects tied weights (e.g., ``lm_head.weight`` tied to
    ``model.embed_tokens.weight``) and adds them to the weight map.

    :param weight_map: raw mapping from safetensors weight name to file path
    :param model: the model (on meta device)
    :return: tuple of:
        - remapped_weight_map: weight_map with keys changed to model param names
        - model_to_safetensors: mapping from model param name back to original
          safetensors key (for saving with original naming)
        - passthrough_keys: safetensors keys that don't map to any model param
          (e.g., visual encoder, mtp weights) — to be copied as-is to output
        - tied_weights: mapping from tied param name to source param name
          (for loading tied weights from the source key in safetensors)
    """
    # Use _all_named_parameters to avoid dedup (finds tied weights)
    model_params = set(name for name, _ in _all_named_parameters(model))
    safetensors_keys = set(weight_map.keys())

    # Fast path: if keys already match substantially, no remapping needed.
    # Check overlap relative to safetensors keys (which are unique/non-dedup),
    # since model_params from _all_named_parameters can be inflated by
    # tied/shared weights appearing at multiple paths.
    overlap = model_params & safetensors_keys
    overlap_ratio = len(overlap) / len(safetensors_keys) if safetensors_keys else 0
    if overlap_ratio > 0.5:
        # Keys match directly — passthrough is anything not in model
        passthrough = sorted(safetensors_keys - model_params)
        if passthrough:
            logger.info(
                f"Key remapping: {len(passthrough)} passthrough weights "
                f"(not in model, will be copied as-is)"
            )
        result_wm = dict(weight_map)
        # Detect tied weights (e.g., lm_head.weight tied to embed_tokens.weight)
        # Add to weight_map so subgraph assignment works, but keep separate
        # from model_to_safetensors so saving uses the model key (not source).
        tied = _detect_tied_weights(model, result_wm)
        for tied_name, source_name in tied.items():
            result_wm[tied_name] = result_wm[source_name]
        return result_wm, {}, passthrough, tied

    if overlap:
        logger.info(
            f"Key remapping: partial overlap ({len(overlap)}/{len(model_params)} "
            f"= {overlap_ratio:.1%} of model params). Attempting prefix remapping."
        )

    # No/low overlap — try to detect a common prefix mismatch.
    # Strategy: pick a model param NOT in the direct overlap and find the
    # safetensors key that shares the same suffix but with a different prefix.
    # Example: model has "model.layers.0.self_attn.q_proj.weight"
    #          safetensors has "model.language_model.layers.0.self_attn.q_proj.weight"
    #          => prefix_to_strip = "model.language_model."
    #          => prefix_to_add   = "model."
    prefix_to_strip = None
    prefix_to_add = None

    # Skip params that already match directly (e.g., lm_head.weight) —
    # they would yield empty prefixes and mask the real mismatch.
    mismatched_params = sorted(model_params - overlap)

    for model_param in mismatched_params:
        # Find a safetensors key that ends with a suffix matching this param
        # after stripping the common model prefix
        for sf_key in safetensors_keys:
            if sf_key == model_param:
                continue  # exact match, not useful for prefix detection
            if sf_key.endswith(model_param):
                # e.g., sf_key = "model.language_model.layers.0.weight"
                #        model_param = "model.layers.0.weight"
                # => prefix_to_strip = "model.language_model."
                # => prefix_to_add   = "model."
                # But this is the degenerate case; more robust:
                prefix_to_strip = sf_key[: len(sf_key) - len(model_param)]
                prefix_to_add = ""
                break
            # Try matching by the portion after the first dot segment
            # e.g., model="model.layers.0.weight" vs
            # sf="model.language_model.layers.0.weight"
            # Find common suffix
            mp_parts = model_param.split(".")
            sf_parts = sf_key.split(".")
            # Align from the end
            common_suffix_len = 0
            for i in range(1, min(len(mp_parts), len(sf_parts)) + 1):
                if mp_parts[-i] == sf_parts[-i]:
                    common_suffix_len = i
                else:
                    break
            if common_suffix_len >= 3:  # need at least module.param_type.weight
                sf_prefix = ".".join(sf_parts[: len(sf_parts) - common_suffix_len])
                mp_prefix = ".".join(mp_parts[: len(mp_parts) - common_suffix_len])
                prefix_to_strip = sf_prefix + "." if sf_prefix else ""
                prefix_to_add = mp_prefix + "." if mp_prefix else ""
                break
        if prefix_to_strip is not None:
            break

    if prefix_to_strip is None:
        logger.warning(
            "Key remapping: could not detect prefix mismatch between "
            "safetensors keys and model parameters. Proceeding without remapping."
        )
        result_wm = dict(weight_map)
        tied = _detect_tied_weights(model, result_wm)
        for tied_name, source_name in tied.items():
            result_wm[tied_name] = result_wm[source_name]
        return result_wm, {}, sorted(safetensors_keys - model_params), tied

    logger.info(
        f"Key remapping: detected prefix mismatch\n"
        f"  safetensors prefix: '{prefix_to_strip}'\n"
        f"  model prefix:       '{prefix_to_add}'\n"
        f"  Remapping {len(safetensors_keys)} safetensors keys"
    )

    remapped_weight_map: dict[str, str] = {}
    model_to_safetensors: dict[str, str] = {}
    passthrough_keys: list[str] = []

    # Build set of named module paths for detecting fused weights.
    # E.g., fused MoE expert tensors (gate_up_proj, down_proj) have parent
    # modules that exist in the model but the fused weight itself is not a
    # regular nn.Parameter (it gets unfused during calibration).
    model_modules = set(name for name, _ in model.named_modules())

    for sf_key, file_path in weight_map.items():
        if sf_key.startswith(prefix_to_strip):
            model_key = prefix_to_add + sf_key[len(prefix_to_strip) :]
            if model_key in model_params:
                remapped_weight_map[model_key] = file_path
                model_to_safetensors[model_key] = sf_key
            elif "." in model_key and model_key.rsplit(".", 1)[0] in model_modules:
                # Fused weight: parent module exists but this specific weight
                # is not a regular parameter (e.g., fused MoE expert tensors).
                # Keep in remapped map so subgraph assignment works.
                remapped_weight_map[model_key] = file_path
                model_to_safetensors[model_key] = sf_key
            else:
                passthrough_keys.append(sf_key)
        elif sf_key in model_params:
            # Direct match (e.g., MTP keys that already use the model's naming)
            remapped_weight_map[sf_key] = file_path
        else:
            passthrough_keys.append(sf_key)

    # Count fused vs param matches
    fused_count = sum(1 for k in remapped_weight_map if k not in model_params)
    matched = len(remapped_weight_map)
    logger.info(
        f"Key remapping: {matched} matched ({matched - fused_count} params, "
        f"{fused_count} fused), "
        f"{len(passthrough_keys)} passthrough (visual/mtp/etc)"
    )
    if passthrough_keys[:3]:
        logger.debug(f"  Sample passthrough: {passthrough_keys[:3]}")

    # Detect tied weights in the slow path too
    tied = _detect_tied_weights(model, remapped_weight_map)
    for tied_name, source_name in tied.items():
        remapped_weight_map[tied_name] = remapped_weight_map[source_name]
        # Don't add to model_to_safetensors — that's used for save remapping.
        # Tied weights should save under their model key, not the source key.

    return remapped_weight_map, model_to_safetensors, passthrough_keys, tied


def copy_passthrough_weights(
    passthrough_keys: list[str],
    weight_map: dict[str, str],
    output_dir: str | os.PathLike,
    shard_weight_map: dict[str, str],
    model_path: str | None = None,
) -> int:
    """
    Copy passthrough weights (e.g., visual encoder, mtp) from source
    safetensors directly to the output directory without modification.

    These are weights present in the original model safetensors that are not
    part of the CausalLM model (e.g., vision encoder for VL models).

    :param passthrough_keys: list of safetensors key names to copy
    :param weight_map: raw (un-remapped) weight_map from build_weight_map()
    :param output_dir: directory to save the passthrough shard
    :param shard_weight_map: dict to update with weight_name -> shard_file
    :param model_path: HF Hub model ID for on-demand downloads
    :return: total size in bytes of copied tensors
    """
    if not passthrough_keys:
        return 0

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by source shard file
    file_to_keys: dict[str, list[str]] = {}
    for key in passthrough_keys:
        if key in weight_map:
            file_path = weight_map[key]
            file_to_keys.setdefault(file_path, []).append(key)

    # Load and save passthrough tensors
    tensors: dict[str, torch.Tensor] = {}
    for file_path, keys in file_to_keys.items():
        resolved_path = _ensure_shard_available(file_path, model_path)
        with safe_open(resolved_path, framework="pt", device="cpu") as f:
            for key in keys:
                tensors[key] = f.get_tensor(key)

    if not tensors:
        return 0

    shard_name = "model-passthrough-of-99999.safetensors"
    shard_path = output_dir / shard_name
    save_file(tensors, str(shard_path))

    total_size = sum(t.nbytes for t in tensors.values())
    for key in tensors:
        shard_weight_map[key] = shard_name

    logger.info(
        f"Copied {len(tensors)} passthrough weights "
        f"({total_size / 1e9:.2f} GB) -> {shard_name}"
    )
    return total_size


def build_weight_map(model_path: str | os.PathLike) -> dict[str, str]:
    """
    Build a mapping from weight name -> safetensors file path.

    For HF Hub models, downloads only the index file (not full model weights).
    Shard files are downloaded on-demand during load_subgraph_weights().

    :param model_path: path to the model directory or HF hub model ID
    :return: dict mapping weight name to absolute safetensors file path
    """
    model_path_str = str(model_path)
    model_path = Path(model_path_str)

    # If model_path is a local directory, use it directly
    if model_path.is_dir():
        return _build_weight_map_from_dir(model_path)

    # For HF Hub model IDs, download only the index/config files first
    from huggingface_hub import hf_hub_download

    # Try to download just the index file (sharded model)
    try:
        index_path = hf_hub_download(model_path_str, "model.safetensors.index.json")
        with open(index_path) as f:
            index = json.load(f)

        # The index file is in the snapshot dir — derive the model dir from it
        snapshot_dir = Path(index_path).parent
        weight_map = {}
        for weight_name, shard_file in index["weight_map"].items():
            weight_map[weight_name] = str(snapshot_dir / shard_file)
        return weight_map
    except Exception:
        pass

    # Try single-file model
    try:
        single_path = hf_hub_download(model_path_str, "model.safetensors")
        weight_map = {}
        with safe_open(single_path, framework="pt") as f:
            for key in f.keys():
                weight_map[key] = single_path
        return weight_map
    except Exception:
        pass

    # Fall back to full snapshot_download
    from huggingface_hub import snapshot_download

    model_path = Path(snapshot_download(model_path_str))
    return _build_weight_map_from_dir(model_path)


def _build_weight_map_from_dir(model_path: Path) -> dict[str, str]:
    """Build weight map from a local directory."""
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        weight_map = {}
        for weight_name, shard_file in index["weight_map"].items():
            weight_map[weight_name] = str(model_path / shard_file)
        return weight_map

    single_file = model_path / "model.safetensors"
    if single_file.exists():
        weight_map = {}
        with safe_open(str(single_file), framework="pt") as f:
            for key in f.keys():
                weight_map[key] = str(single_file)
        return weight_map

    raise FileNotFoundError(
        f"No safetensors files found in {model_path}. "
        "Layerwise quantization requires safetensors format."
    )


def _get_module_weight_names(
    model: Module, subgraph_modules: set[Module]
) -> dict[str, str]:
    """
    Get the full parameter names for all parameters in the given modules.

    :param model: the full model (for resolving parameter names)
    :param subgraph_modules: set of modules whose weights to load
    :return: dict mapping parameter name -> module name
    """
    # Build a mapping from module to its fully qualified name
    module_to_name = {module: name for name, module in model.named_modules()}

    param_names = {}
    for module in subgraph_modules:
        module_name = module_to_name.get(module)
        if module_name is None:
            continue
        for param_name, _ in module.named_parameters(recurse=True):
            full_name = f"{module_name}.{param_name}" if module_name else param_name
            param_names[full_name] = module_name

    return param_names


def get_subgraph_weight_names(
    model: Module,
    weight_map: dict[str, str],
    sequential_targets: list[str],
    subgraph_index: int,
    num_subgraphs: int,
) -> list[str]:
    """
    Determine which weight names belong to a given subgraph based on the
    sequential target partitioning.

    The subgraphs are laid out as:
    - subgraph 0: head (embedding, pre-layer norms, etc.) — everything before
      the first sequential target
    - subgraph 1..N: one per sequential target (decoder layer)
    - Note: the last subgraph may include post-layer modules (final norm, lm_head)

    :param model: the model (on meta device)
    :param weight_map: mapping from weight name to safetensors file path
    :param sequential_targets: list of target module name patterns
    :param subgraph_index: which subgraph (0-indexed)
    :param num_subgraphs: total number of subgraphs
    :return: list of weight names to load for this subgraph
    """
    from compressed_tensors.utils.match import match_named_modules

    # Find actual target module names (preserving model order)
    target_modules = list(match_named_modules(model, sequential_targets))
    target_names = [name for name, _ in target_modules]

    all_weight_names = list(weight_map.keys())
    target_prefixes = [f"{name}." for name in target_names]

    if subgraph_index == 0:
        # First subgraph: non-target weights that come BEFORE the first target
        # in the model's module definition order (e.g., embed_tokens)
        module_order = {
            name: idx for idx, (name, _) in enumerate(model.named_modules())
        }
        first_target_order = min(
            module_order.get(name, float("inf")) for name in target_names
        )
        return [
            w
            for w in all_weight_names
            if not any(w.startswith(p) for p in target_prefixes)
            and module_order.get(w.rsplit(".", 1)[0] if "." in w else "", float("inf"))
            < first_target_order
        ]
    elif subgraph_index == num_subgraphs - 1:
        # Last subgraph: last target's weights + post-target non-target weights
        # (e.g., model.norm, lm_head that come after all decoder layers)
        target_idx = subgraph_index - 1
        result = []
        if target_idx < len(target_names):
            prefix = f"{target_names[target_idx]}."
            result = [w for w in all_weight_names if w.startswith(prefix)]

        module_order = {
            name: idx for idx, (name, _) in enumerate(model.named_modules())
        }
        last_target_order = max(module_order.get(name, 0) for name in target_names)
        for w in all_weight_names:
            if any(w.startswith(p) for p in target_prefixes):
                continue
            module_name = w.rsplit(".", 1)[0] if "." in w else ""
            if module_order.get(module_name, -1) > last_target_order:
                result.append(w)
        return result
    else:
        # Middle subgraph: just the target's weights
        target_idx = subgraph_index - 1
        if target_idx < len(target_names):
            prefix = f"{target_names[target_idx]}."
            return [w for w in all_weight_names if w.startswith(prefix)]
        return []


def load_subgraph_weights(
    model: Module,
    weight_names: list[str],
    weight_map: dict[str, str],
    device: torch.device,
    model_path: str | None = None,
    model_to_safetensors: dict[str, str] | None = None,
    tied_weights: dict[str, str] | None = None,
) -> None:
    """
    Load weights from safetensors files for the given weight names,
    replacing meta-device parameters with real tensors on the target device.

    If a shard file does not exist locally, it will be downloaded on-demand
    from the HF Hub using model_path as the repo ID.

    :param model: the full model (meta device)
    :param weight_names: list of parameter names (model param names) to load
    :param weight_map: mapping from model param name to safetensors file path
        (after remapping by build_key_remapping)
    :param device: device to load weights onto
    :param model_path: HF Hub model ID for on-demand shard downloads
    :param model_to_safetensors: optional mapping from model param name to
        the original safetensors key (for VL models where keys differ)
    :param tied_weights: optional mapping from tied param name to source param
        name (e.g., lm_head.weight -> model.embed_tokens.weight). Used to
        resolve the safetensors key for tied weights.
    """
    if not weight_names:
        return

    # Group parameters by safetensors file for efficient loading
    file_to_params: dict[str, list[str]] = {}
    missing_params = []
    for param_name in weight_names:
        if param_name in weight_map:
            file_path = weight_map[param_name]
            file_to_params.setdefault(file_path, []).append(param_name)
        else:
            missing_params.append(param_name)

    if missing_params:
        logger.warning(
            f"Could not find {len(missing_params)} parameters in weight map: "
            f"{missing_params[:5]}{'...' if len(missing_params) > 5 else ''}"
        )

    # Load weights from each safetensors file, downloading on demand if needed
    loaded_count = 0
    for file_path, params in file_to_params.items():
        resolved_path = _ensure_shard_available(file_path, model_path)
        with safe_open(resolved_path, framework="pt", device=str(device)) as f:
            for param_name in params:
                # Use original safetensors key if remapping exists
                sf_key = (
                    model_to_safetensors.get(param_name, param_name)
                    if model_to_safetensors
                    else param_name
                )
                # For tied weights, load from the source param's key
                if tied_weights and param_name in tied_weights:
                    source = tied_weights[param_name]
                    sf_key = (
                        model_to_safetensors.get(source, source)
                        if model_to_safetensors
                        else source
                    )
                tensor = f.get_tensor(sf_key)
                # Try normal set; fall back to fused MoE splitting
                if not _try_set_fused_moe(model, param_name, tensor):
                    _set_parameter(model, param_name, tensor)
                loaded_count += 1

    logger.debug(f"Loaded {loaded_count} parameters for subgraph onto {device}")

    # Also move any quantization buffers (observers, scales, zero_points)
    # that were initialized on meta device to the target device
    _move_quantization_buffers(model, weight_names, device)


def move_subgraph_buffers(
    model: Module,
    subgraph_modules: set,
    device: torch.device,
) -> None:
    """
    Move all buffers in a subgraph's modules from meta/CPU to the target device.

    This is needed because some modules (e.g., RoPE rotary_emb) have buffers
    like ``inv_freq`` that are not stored in safetensors and thus don't get
    loaded by :func:`load_subgraph_weights`. In layerwise mode these buffers
    stay on their init device (meta or CPU), causing device mismatches during
    the forward pass.

    :param model: the model (not used directly, but kept for API consistency)
    :param subgraph_modules: set of modules from
        ``subgraph.submodules(model, recurse=True)``
    :param device: target device to move buffers to
    """
    moved = 0
    for module in subgraph_modules:
        for attr_name, buf in list(module._buffers.items()):
            if buf is not None and buf.device != device:
                module._buffers[attr_name] = buf.to(device)
                moved += 1
    if moved:
        logger.debug(f"Moved {moved} subgraph buffers to {device}")


def offload_subgraph_weights(
    model: Module,
    weight_names: list[str],
    device: str = "meta",
) -> None:
    """
    Offload subgraph weights to the specified device to free GPU memory.

    :param model: the full model
    :param weight_names: list of parameter names to offload
    :param device: target device ("meta" to free all memory, "cpu" to keep on CPU)
    """
    target_device = torch.device(device)
    freed_count = 0

    # Extract unique module prefixes from weight names
    module_prefixes = set()
    for name in weight_names:
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            module_prefixes.add(parts[0])

    for prefix in module_prefixes:
        try:
            parts = prefix.split(".")
            module = model
            for part in parts:
                module = getattr(module, part)
        except AttributeError:
            continue

        # For unfused MoE experts, process all descendant modules
        modules_to_process = [module]
        if isinstance(module, torch.nn.ModuleList):
            modules_to_process.extend(m for m in module.modules() if m is not module)

        for mod in modules_to_process:
            for attr_name in list(mod._parameters.keys()):
                param = mod._parameters[attr_name]
                if param is not None and param.device != target_device:
                    if target_device.type == "meta":
                        new_param = torch.nn.Parameter(
                            torch.empty_like(param, device="meta"),
                            requires_grad=param.requires_grad,
                        )
                    else:
                        new_param = torch.nn.Parameter(
                            param.data.to(target_device),
                            requires_grad=param.requires_grad,
                        )
                    mod._parameters[attr_name] = new_param
                    freed_count += 1

            for attr_name in list(mod._buffers.keys()):
                buf = mod._buffers[attr_name]
                if buf is not None and buf.device != target_device:
                    if target_device.type == "meta":
                        mod._buffers[attr_name] = torch.empty_like(buf, device="meta")
                    else:
                        mod._buffers[attr_name] = buf.to(target_device)
                    freed_count += 1

    if freed_count > 0:
        logger.debug(f"Offloaded {freed_count} parameters/buffers to {device}")
        if device != "cpu":
            torch.cuda.empty_cache()


def save_subgraph_weights(
    model: Module,
    subgraph_modules: set[Module],
    output_dir: str | os.PathLike,
    shard_index: int,
    weight_map_output: dict[str, str],
) -> int:
    """
    Save the current weights of subgraph modules to a safetensors shard.

    :param model: the full model
    :param subgraph_modules: set of modules whose weights to save
    :param output_dir: directory to save safetensors shards
    :param shard_index: index for naming the shard file
    :param weight_map_output: dict to update with weight_name -> shard_file mappings
    :return: total size in bytes of saved tensors
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    param_names = _get_module_weight_names(model, subgraph_modules)
    if not param_names:
        return 0

    # Collect tensors to save
    tensors = {}
    for param_name in param_names:
        parts = param_name.split(".")
        obj = model
        for part in parts:
            obj = getattr(obj, part)
        if isinstance(obj, torch.Tensor) and obj.device.type != "meta":
            tensors[param_name] = obj.contiguous().cpu()

    if not tensors:
        return 0

    shard_name = f"model-{shard_index:05d}-of-99999.safetensors"
    shard_path = output_dir / shard_name
    save_file(tensors, str(shard_path))

    total_size = sum(t.nbytes for t in tensors.values())
    for name in tensors:
        weight_map_output[name] = shard_name

    logger.info(
        f"Saved {len(tensors)} tensors ({total_size / 1e9:.2f} GB) to {shard_name}"
    )
    return total_size


def _set_parameter(model: Module, param_name: str, tensor: torch.Tensor) -> None:
    """
    Set a parameter on the model by its fully qualified name,
    replacing a meta-device parameter with a real tensor.

    :param model: the model to set the parameter on
    :param param_name: fully qualified parameter name (e.g., "model.layers.0.weight")
    :param tensor: the real tensor to set
    """
    from llmcompressor.modeling.offset_norm import NormCalibrationModule

    parts = param_name.split(".")
    module = model
    for part in parts[:-1]:
        module = getattr(module, part)

    param_attr = parts[-1]

    # Apply weight transformation for norm calibration modules (e.g.,
    # CalibrationOffsetNorm needs 1 + raw_weight when loaded from safetensors)
    if isinstance(module, NormCalibrationModule):
        tensor = module.transform_loaded_weight(param_attr, tensor)

    old_param = getattr(module, param_attr, None)

    if isinstance(old_param, torch.nn.Parameter):
        new_param = torch.nn.Parameter(tensor, requires_grad=old_param.requires_grad)
        module._parameters[param_attr] = new_param
    else:
        setattr(module, param_attr, tensor)


def _try_set_fused_moe(model: Module, param_name: str, tensor: torch.Tensor) -> bool:
    """
    Handle fused MoE expert tensors when the model has been unfused by
    MoE calibration modules (e.g., SequentialQwen3_5MoeExperts).

    When ``moe_calibration_context`` replaces a fused ``Qwen3_5MoeExperts``
    module with a ``ModuleList`` of individual MLPs, the fused parameter names
    (e.g., ``experts.gate_up_proj``) no longer exist on the model. This function
    detects such cases and splits the 3D tensor across individual experts.

    :param model: the full model
    :param param_name: weight name from the weight map (fused naming)
    :param tensor: the fused 3D tensor loaded from safetensors
    :return: True if handled (fused MoE split), False if not applicable
    """
    if tensor.dim() != 3:
        return False

    parts = param_name.split(".")
    attr = parts[-1]  # e.g., "gate_up_proj" or "down_proj"

    # Navigate to parent module
    module = model
    try:
        for part in parts[:-1]:
            module = getattr(module, part)
    except AttributeError:
        return False

    # Check if parent is a ModuleList (unfused experts)
    if not isinstance(module, torch.nn.ModuleList):
        return False

    num_experts = len(module)
    if tensor.shape[0] != num_experts:
        return False

    if attr == "gate_up_proj":
        intermediate_size = tensor.shape[1] // 2
        for i in range(num_experts):
            gate_up = tensor[i]  # [2*intermediate, hidden]
            module[i].gate_proj.weight = torch.nn.Parameter(
                gate_up[:intermediate_size, :].clone().contiguous()
            )
            module[i].up_proj.weight = torch.nn.Parameter(
                gate_up[intermediate_size:, :].clone().contiguous()
            )
        logger.debug(
            f"Split fused gate_up_proj into {num_experts} experts "
            f"for {'.'.join(parts[:-1])}"
        )
        return True
    elif attr == "down_proj":
        for i in range(num_experts):
            module[i].down_proj.weight = torch.nn.Parameter(
                tensor[i].clone().contiguous()
            )
        logger.debug(
            f"Split fused down_proj into {num_experts} experts "
            f"for {'.'.join(parts[:-1])}"
        )
        return True

    return False


def _move_quantization_buffers(
    model: Module, weight_names: list[str], device: torch.device
) -> None:
    """
    Move quantization-related buffers and parameters (observers, scales, zero_points)
    from meta device to the target device. These are created by QuantizationModifier
    during initialization and may be on meta device for layerwise models.

    :param model: the model
    :param weight_names: list of weight names that were just loaded (used to
        identify which modules to process)
    :param device: device to move buffers to
    """
    # Extract unique module prefixes from weight names
    module_prefixes = set()
    for name in weight_names:
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            module_prefixes.add(parts[0])

    moved_count = 0
    for prefix in module_prefixes:
        try:
            parts = prefix.split(".")
            module = model
            for part in parts:
                module = getattr(module, part)
        except AttributeError:
            continue

        # For unfused MoE experts (ModuleList), we need to process all
        # descendant modules too, not just the container. The weight names
        # use fused keys (e.g., "experts.gate_up_proj") but after MoE
        # calibration the model has individual expert modules with their
        # own quantization buffers (e.g., "experts.0.gate_proj.weight_scale").
        modules_to_process = [module]
        if isinstance(module, torch.nn.ModuleList):
            modules_to_process.extend(m for m in module.modules() if m is not module)

        for mod in modules_to_process:
            # Move all meta-device parameters and buffers on this module
            for attr_name in list(mod._parameters.keys()):
                param = mod._parameters[attr_name]
                if param is not None and param.device.type == "meta":
                    new_param = torch.nn.Parameter(
                        torch.zeros(param.shape, dtype=param.dtype, device=device),
                        requires_grad=param.requires_grad,
                    )
                    mod._parameters[attr_name] = new_param
                    moved_count += 1

            for attr_name in list(mod._buffers.keys()):
                buf = mod._buffers[attr_name]
                if buf is not None and buf.device.type == "meta":
                    mod._buffers[attr_name] = torch.zeros(
                        buf.shape, dtype=buf.dtype, device=device
                    )
                    moved_count += 1

            # Move observer modules if they exist
            for attr_name in dir(mod):
                if attr_name.endswith("_observer"):
                    observer = getattr(mod, attr_name, None)
                    if observer is not None and isinstance(observer, Module):
                        for pname, param in observer.named_parameters():
                            if param.device.type == "meta":
                                parts_p = pname.split(".")
                                target = observer
                                for p in parts_p[:-1]:
                                    target = getattr(target, p)
                                target._parameters[parts_p[-1]] = torch.nn.Parameter(
                                    torch.zeros(
                                        param.shape, dtype=param.dtype, device=device
                                    ),
                                    requires_grad=param.requires_grad,
                                )
                                moved_count += 1
                        for bname, buf in observer.named_buffers():
                            if buf.device.type == "meta":
                                parts_b = bname.split(".")
                                target = observer
                                for p in parts_b[:-1]:
                                    target = getattr(target, p)
                                target._buffers[parts_b[-1]] = torch.zeros(
                                    buf.shape, dtype=buf.dtype, device=device
                                )
                                moved_count += 1

    if moved_count > 0:
        logger.debug(f"Moved {moved_count} quantization buffers/params to {device}")


def _ensure_shard_available(file_path: str, model_path: str | None = None) -> str:
    """
    Ensure a safetensors shard file exists locally. If the file is missing
    (because we only downloaded the index, not all shards), download it
    on-demand from the HF Hub.

    :param file_path: expected local path to the shard file
    :param model_path: HF Hub model ID for downloading
    :return: resolved local path to the shard file
    """
    if os.path.isfile(file_path):
        return file_path

    if model_path is None:
        raise FileNotFoundError(
            f"Shard file not found: {file_path}. "
            "Provide model_path for on-demand downloading."
        )

    # Extract shard filename from the path
    shard_filename = os.path.basename(file_path)
    logger.info(f"Downloading shard on-demand: {shard_filename}")

    from huggingface_hub import hf_hub_download

    downloaded_path = hf_hub_download(model_path, shard_filename)
    return downloaded_path


def compress_and_save_subgraph(
    model: Module,
    weight_names: list[str],
    output_dir: str | os.PathLike,
    shard_index: int,
    shard_weight_map: dict[str, str],
    model_to_safetensors: dict[str, str] | None = None,
    tied_weights: dict[str, str] | None = None,
) -> int:
    """
    Compress quantized modules for the given weight names in-place, then save
    all parameters (compressed weights + quantization params) to a safetensors
    shard. After saving, offloads weights to meta device to free memory.

    This enables "compress-as-you-go": each subgraph is compressed and saved
    immediately after calibration, so only 1 subgraph's base weights are ever
    in CPU/GPU memory at a time.

    :param model: the full model
    :param weight_names: list of parameter names belonging to this subgraph
    :param output_dir: directory to save safetensors shards
    :param shard_index: index for naming the shard file
    :param shard_weight_map: dict to update with weight_name -> shard_file mappings
    :param model_to_safetensors: optional mapping from model param name to
        original safetensors key (for saving with original naming convention)
    :return: total size in bytes of saved tensors
    """
    from compressed_tensors.compressors.base import compress_module
    from compressed_tensors.quantization.utils.helpers import is_module_quantized

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Identify modules that contain the subgraph's weights
    module_prefixes = set()
    for name in weight_names:
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            module_prefixes.add(parts[0])

    # Compress quantized modules in this subgraph
    compressed_count = 0
    for prefix in module_prefixes:
        try:
            parts = prefix.split(".")
            module = model
            for part in parts:
                module = getattr(module, part)
        except AttributeError:
            continue

        # For unfused MoE experts (ModuleList), compress each child's
        # quantized submodules instead of the ModuleList itself
        modules_to_compress = []
        if isinstance(module, torch.nn.ModuleList):
            for child in module:
                for _, submod in child.named_modules():
                    modules_to_compress.append(submod)
        else:
            modules_to_compress.append(module)

        for mod in modules_to_compress:
            if is_module_quantized(mod):
                # Ensure all params/buffers are on the same device before compression
                devices = set()
                for p in mod.parameters():
                    if p.device.type != "meta":
                        devices.add(p.device)
                for b in mod.buffers():
                    if b.device.type != "meta":
                        devices.add(b.device)
                if len(devices) > 1:
                    # Move everything to CPU for consistent compression
                    for pname, param in mod.named_parameters(recurse=False):
                        if param.device.type != "cpu" and param.device.type != "meta":
                            mod._parameters[pname] = torch.nn.Parameter(
                                param.data.cpu(), requires_grad=param.requires_grad
                            )
                    for bname in list(mod._buffers.keys()):
                        buf = mod._buffers[bname]
                        if (
                            buf is not None
                            and buf.device.type != "cpu"
                            and buf.device.type != "meta"
                        ):
                            mod._buffers[bname] = buf.cpu()
                compress_module(mod)
                compressed_count += 1

    if compressed_count > 0:
        logger.debug(f"Compressed {compressed_count} quantized modules in subgraph")

    # Collect all non-meta tensors (parameters + buffers) from these modules.
    # For unfused MoE (ModuleList), recurse into children to capture all
    # individual expert parameters.
    from llmcompressor.modeling.offset_norm import NormCalibrationModule

    tensors = {}
    for prefix in module_prefixes:
        try:
            parts = prefix.split(".")
            module = model
            for part in parts:
                module = getattr(module, part)
        except AttributeError:
            continue

        use_recurse = isinstance(module, torch.nn.ModuleList)

        for pname, param in module.named_parameters(recurse=use_recurse):
            full_name = f"{prefix}.{pname}"
            if param.device.type != "meta":
                tensor = param.data.contiguous().cpu()
                # Reverse calibration transformation for norm modules so that
                # saved weights are in the original (offset) convention
                if isinstance(module, NormCalibrationModule):
                    tensor = module.transform_save_weight(pname, tensor)
                tensors[full_name] = tensor

        for bname, buf in module.named_buffers(recurse=use_recurse):
            full_name = f"{prefix}.{bname}"
            if buf.device.type != "meta":
                tensors[full_name] = buf.contiguous().cpu()

    if not tensors:
        return 0

    # Skip tied weight aliases (e.g., lm_head.weight when tied to
    # embed_tokens.weight). The canonical copy is saved under its own name;
    # the framework restores the tie at load time via config flags.
    if tied_weights:
        for alias in tied_weights:
            tensors.pop(alias, None)

    if not tensors:
        return 0

    # Remap keys back to original safetensors naming for VL model compatibility.
    # This handles both original weights (in model_to_safetensors) and new
    # quantization params (weight_scale, weight_zero_point, etc.) that share
    # the same module prefix but weren't in the original safetensors.
    if model_to_safetensors:
        # Build a prefix mapping from model prefix -> safetensors prefix
        # e.g., "model.layers.0.self_attn.q_proj" ->
        # "model.language_model.layers.0.self_attn.q_proj"
        prefix_map: dict[str, str] = {}
        for model_key, sf_key in model_to_safetensors.items():
            model_prefix = model_key.rsplit(".", 1)[0] if "." in model_key else ""
            sf_prefix = sf_key.rsplit(".", 1)[0] if "." in sf_key else ""
            if model_prefix and sf_prefix and model_prefix not in prefix_map:
                prefix_map[model_prefix] = sf_prefix

        def _remap_prefix(name: str) -> str:
            """Find the best prefix mapping for a tensor name.

            First tries an exact prefix match, then walks up parent prefixes.
            This handles unfused MoE expert names like
            ``model.layers.0.mlp.experts.0.gate_proj.weight`` which have
            prefix ``model.layers.0.mlp.experts`` in the map but the full
            module prefix ``model.layers.0.mlp.experts.0.gate_proj`` is not.
            """
            module_prefix = name.rsplit(".", 1)[0] if "." in name else ""
            param_suffix = name[len(module_prefix) + 1 :] if module_prefix else name

            # Exact match
            if module_prefix in prefix_map:
                return f"{prefix_map[module_prefix]}.{param_suffix}"

            # Walk up parent prefixes
            parts = module_prefix.split(".")
            for i in range(len(parts) - 1, 0, -1):
                parent = ".".join(parts[:i])
                if parent in prefix_map:
                    child_suffix = ".".join(parts[i:])
                    return f"{prefix_map[parent]}.{child_suffix}.{param_suffix}"

            return name

        remapped_tensors = {}
        for name, tensor in tensors.items():
            if name in model_to_safetensors:
                # Exact match: use the known mapping
                remapped_tensors[model_to_safetensors[name]] = tensor
            else:
                remapped_tensors[_remap_prefix(name)] = tensor
        tensors = remapped_tensors

    # Save to shard file
    shard_name = f"model-{shard_index + 1:05d}-of-99999.safetensors"
    shard_path = output_dir / shard_name
    save_file(tensors, str(shard_path))

    total_size = sum(t.nbytes for t in tensors.values())
    for name in tensors:
        shard_weight_map[name] = shard_name

    logger.info(
        f"Saved subgraph {shard_index + 1}: "
        f"{len(tensors)} tensors ({total_size / 1e9:.2f} GB) -> {shard_name}"
    )

    # Offload to meta to free memory
    offload_subgraph_weights(model, weight_names, device="meta")

    return total_size


def write_safetensors_index(
    output_dir: str | os.PathLike,
    shard_weight_map: dict[str, str],
    total_size: int,
) -> None:
    """
    Write the model.safetensors.index.json file and rename shard files to
    reflect the actual total number of shards.

    :param output_dir: directory containing the shard files
    :param shard_weight_map: mapping of weight name -> shard filename
    :param total_size: total size in bytes of all saved tensors
    """
    output_dir = Path(output_dir)

    # Determine actual shard files used
    shard_files = sorted(set(shard_weight_map.values()))
    num_shards = len(shard_files)

    # Rename shards to have correct total count
    rename_map = {}
    for i, old_name in enumerate(shard_files):
        new_name = f"model-{i + 1:05d}-of-{num_shards:05d}.safetensors"
        if old_name != new_name:
            old_path = output_dir / old_name
            new_path = output_dir / new_name
            if old_path.exists():
                old_path.rename(new_path)
            rename_map[old_name] = new_name

    # Update weight map with renamed shard files
    final_weight_map = {}
    for weight_name, shard_file in shard_weight_map.items():
        final_weight_map[weight_name] = rename_map.get(shard_file, shard_file)

    # Write index file
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": final_weight_map,
    }
    index_path = output_dir / "model.safetensors.index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    logger.info(
        f"Written safetensors index: {num_shards} shards, "
        f"{total_size / 1e9:.2f} GB total"
    )


class ShardPrefetcher:
    """
    Background prefetcher that downloads the next subgraph's shard files
    while the current subgraph is being calibrated on GPU.

    Usage:
        prefetcher = ShardPrefetcher(model_path)
        prefetcher.prefetch(next_weight_names, weight_map)
        # ... calibrate current subgraph ...
        prefetcher.wait()  # ensure next shards are ready
    """

    def __init__(self, model_path: str | None = None):
        self._model_path = model_path
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None

    def prefetch(
        self,
        weight_names: list[str],
        weight_map: dict[str, str],
    ) -> None:
        """Start background download of shard files for the given weight names."""
        self.wait()  # ensure previous prefetch is complete

        # Determine unique shard files needed
        shard_files = set()
        for name in weight_names:
            if name in weight_map:
                shard_files.add(weight_map[name])

        # Filter to only files that need downloading
        files_to_download = [f for f in shard_files if not os.path.isfile(f)]
        if not files_to_download:
            return

        self._error = None
        self._thread = threading.Thread(
            target=self._download_shards,
            args=(files_to_download,),
            daemon=True,
        )
        self._thread.start()

    def _download_shards(self, file_paths: list[str]) -> None:
        """Download shard files in background thread."""
        try:
            for file_path in file_paths:
                _ensure_shard_available(file_path, self._model_path)
        except Exception as e:
            self._error = e

    def wait(self) -> None:
        """Wait for background prefetch to complete. Raises if download failed."""
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._error is not None:
            error = self._error
            self._error = None
            raise error
