"""
Utilities for loading lm_head weights from model checkpoints
without loading the full model into memory.

Uses accelerate's init_empty_weights to create a zero-memory meta model,
then discovers the correct lm_head parameter name via get_output_embeddings().
Only the lm_head tensor is loaded from the safetensors checkpoint.
"""

import json
import os
import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM


def _discover_lm_head_names(model_id: str) -> dict:
    """
    Use a meta (zero-memory) model to discover the lm_head parameter names
    for any architecture. This replaces the manual architecture-to-weight-name
    mapping with a dynamic approach that works for all HuggingFace causal LMs.

    :param model_id: HuggingFace model ID or local path
    :return: dict with lm_head_weight name, lm_head_bias name (or None),
             and tie_word_embeddings flag
    """
    from accelerate import init_empty_weights

    config = AutoConfig.from_pretrained(model_id)
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

    # Check for tied embeddings — if tied, the weight may be stored under
    # the input embedding name in the checkpoint
    tie_word_embeddings = getattr(
        getattr(config, "text_config", config), "tie_word_embeddings", False
    )
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
        "embed_weight": input_embed_name,
        "tie_word_embeddings": tie_word_embeddings,
    }


def detect_last_layer_index(model_id: str) -> int:
    """
    Return the layer index for extracting post-norm hidden states from vLLM.
    This is num_hidden_layers (NOT num_hidden_layers - 1), which gives the
    output after the final layer norm — ready to pass directly to lm_head.

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
    Load only the lm_head weight (and optional bias) from a model checkpoint
    without loading the full model.

    Uses get_output_embeddings() on a meta model to dynamically discover
    the correct parameter names for any architecture.

    :param model_id: HuggingFace model ID or local path
    :param device: device to load tensors to
    :return: dict with keys: lm_head_weight, lm_head_bias (or None)
    """
    names = _discover_lm_head_names(model_id)
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

    return {
        "lm_head_weight": lm_head_weight,
        "lm_head_bias": lm_head_bias,
    }
