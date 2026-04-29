"""
Utilities for loading lm_head and final norm weights from model checkpoints
without loading the full model into memory.

Uses accelerate's init_empty_weights to create a zero-memory meta model,
then discovers the correct parameter names via get_output_embeddings().
Only the lm_head and norm tensors are loaded from the safetensors checkpoint.

Note: vLLM's hidden state extraction at layer index num_hidden_layers captures
the output BEFORE the final norm (pre-norm). The norm must be applied before
passing hidden states to lm_head.
"""

import json
import os
import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM


def _discover_weight_names(model_id: str) -> dict:
    """
    Use a meta (zero-memory) model to discover the lm_head and final norm
    parameter names for any architecture. This replaces manual architecture-
    to-weight-name mappings with a dynamic approach.

    :param model_id: HuggingFace model ID or local path
    :return: dict with lm_head_weight, lm_head_bias, norm_weight,
             embed_weight names and tie_word_embeddings flag
    """
    from accelerate import init_empty_weights

    config = AutoConfig.from_pretrained(model_id)
    text_config = getattr(config, "text_config", config)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)

    lm_head = model.get_output_embeddings()
    if lm_head is None:
        raise ValueError(
            f"Model {model_id} does not have output embeddings. "
            "Cannot determine lm_head weight name."
        )

    # Find the parameter names by identity matching
    lm_head_weight_name = None
    lm_head_bias_name = None
    for name, param in model.named_parameters():
        if param is lm_head.weight:
            lm_head_weight_name = name
        if hasattr(lm_head, "bias") and lm_head.bias is not None:
            if param is lm_head.bias:
                lm_head_bias_name = name

    if lm_head_weight_name is None:
        raise ValueError(
            f"Could not find lm_head weight parameter in model {model_id}. "
            "get_output_embeddings() returned a module but its weight was not "
            "found in named_parameters()."
        )

    # Find the final norm — it's the norm module on the inner model
    # (e.g., model.model.norm for LlamaForCausalLM)
    norm_weight_name = None
    inner_model = getattr(model, "model", None) or getattr(model, "transformer", None)
    if inner_model is not None:
        norm_module = getattr(inner_model, "norm", None) or getattr(
            inner_model, "final_layernorm", None
        ) or getattr(inner_model, "ln_f", None)
        if norm_module is not None and hasattr(norm_module, "weight"):
            for name, param in model.named_parameters():
                if param is norm_module.weight:
                    norm_weight_name = name
                    break

    if norm_weight_name is None:
        raise ValueError(
            f"Could not find final norm weight in model {model_id}. "
            "The norm is required to transform pre-norm hidden states "
            "before applying lm_head."
        )

    # Determine norm epsilon from config
    norm_eps = getattr(text_config, "rms_norm_eps", None) or getattr(
        text_config, "layer_norm_epsilon", None
    ) or getattr(text_config, "layer_norm_eps", None) or 1e-6

    # Determine norm type from config
    # RMSNorm if rms_norm_eps is present, otherwise LayerNorm
    norm_type = "rms_norm" if hasattr(text_config, "rms_norm_eps") else "layer_norm"

    # Check for tied embeddings — if tied, the weight may be stored under
    # the input embedding name in the checkpoint
    tie_word_embeddings = getattr(text_config, "tie_word_embeddings", False)
    input_embed_name = None
    if tie_word_embeddings:
        input_embed = model.get_input_embeddings()
        if input_embed is not None:
            for name, param in model.named_parameters():
                if param is input_embed.weight:
                    input_embed_name = name
                    break

    del model
    return {
        "lm_head_weight": lm_head_weight_name,
        "lm_head_bias": lm_head_bias_name,
        "norm_weight": norm_weight_name,
        "norm_eps": norm_eps,
        "norm_type": norm_type,
        "embed_weight": input_embed_name,
        "tie_word_embeddings": tie_word_embeddings,
    }


def detect_last_layer_index(model_id: str) -> int:
    """
    Return the layer index for extracting hidden states from vLLM.
    This is num_hidden_layers, which gives the output after the last
    transformer block but BEFORE the final layer norm (pre-norm).
    The norm must be applied separately before passing to lm_head.

    :param model_id: HuggingFace model ID or local path
    :return: layer index for vLLM extraction
    """
    config = AutoConfig.from_pretrained(model_id)
    config = getattr(config, "text_config", config)
    return config.num_hidden_layers


def _get_weight_map(model_path: str) -> dict:
    """
    Get tensor name -> shard file mapping for a model checkpoint.
    Handles both sharded and single-file models, local and hub paths.
    """
    if os.path.isdir(model_path):
        index_path = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                index = json.load(f)
            return {
                name: os.path.join(model_path, shard)
                for name, shard in index["weight_map"].items()
            }
        single_path = os.path.join(model_path, "model.safetensors")
        if os.path.exists(single_path):
            with safe_open(single_path, framework="pt") as f:
                return {name: single_path for name in f.keys()}
        raise FileNotFoundError(
            f"No safetensors files found in {model_path}. "
            "Expected model.safetensors or model.safetensors.index.json"
        )
    else:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError

        try:
            index_path = hf_hub_download(model_path, "model.safetensors.index.json")
            with open(index_path) as f:
                index = json.load(f)
            return {
                name: (model_path, shard) for name, shard in index["weight_map"].items()
            }
        except (EntryNotFoundError, FileNotFoundError):
            # No sharded index — try single safetensors file
            single_path = hf_hub_download(model_path, "model.safetensors")
            with safe_open(single_path, framework="pt") as f:
                return {name: single_path for name in f.keys()}


def _load_tensor(file_ref, tensor_name: str) -> torch.Tensor:
    """Load a single tensor from either a local path or (repo_id, filename) tuple."""
    if isinstance(file_ref, tuple):
        repo_id, filename = file_ref
        from huggingface_hub import hf_hub_download

        local_path = hf_hub_download(repo_id, filename)
    else:
        local_path = file_ref

    with safe_open(local_path, framework="pt") as f:
        return f.get_tensor(tensor_name)


def load_lm_head_weights(
    model_id: str,
    device: str = "cpu",
) -> dict:
    """
    Load the lm_head weight, final norm weight, and optional bias from a
    model checkpoint without loading the full model.

    Uses get_output_embeddings() on a meta model to dynamically discover
    the correct parameter names for any architecture.

    :param model_id: HuggingFace model ID or local path
    :param device: device to load tensors to
    :return: dict with keys: lm_head_weight, lm_head_bias (or None),
             norm_weight, norm_eps, norm_type
    """
    names = _discover_weight_names(model_id)
    weight_map = _get_weight_map(model_id)

    # Load lm_head weight (handle tied embeddings)
    lm_head_key = names["lm_head_weight"]
    if lm_head_key not in weight_map and names["tie_word_embeddings"]:
        lm_head_key = names["embed_weight"]

    if lm_head_key not in weight_map:
        raise KeyError(
            f"Cannot find lm_head weight '{names['lm_head_weight']}' "
            f"(or embed weight '{names['embed_weight']}' for tied embeddings) "
            f"in checkpoint for {model_id}"
        )

    lm_head_weight = _load_tensor(weight_map[lm_head_key], lm_head_key).to(device)

    # Load optional lm_head bias
    lm_head_bias = None
    if names["lm_head_bias"] and names["lm_head_bias"] in weight_map:
        lm_head_bias = _load_tensor(
            weight_map[names["lm_head_bias"]], names["lm_head_bias"]
        ).to(device)

    # Load final norm weight
    norm_key = names["norm_weight"]
    if norm_key not in weight_map:
        raise KeyError(
            f"Cannot find norm weight '{norm_key}' in checkpoint for {model_id}"
        )
    norm_weight = _load_tensor(weight_map[norm_key], norm_key).to(device)

    return {
        "lm_head_weight": lm_head_weight,
        "lm_head_bias": lm_head_bias,
        "norm_weight": norm_weight,
        "norm_eps": names["norm_eps"],
        "norm_type": names["norm_type"],
    }
