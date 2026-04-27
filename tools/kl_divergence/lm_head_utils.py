"""
Utilities for loading lm_head and final norm weights from model checkpoints
without loading the full model into memory.
"""

import json
import os
from typing import Optional

import torch
from safetensors import safe_open
from transformers import AutoConfig


# Architecture -> weight name mappings
# Most modern LLMs follow the Llama pattern
_DEFAULT_WEIGHT_NAMES = {
    "norm_weight": "model.norm.weight",
    "lm_head_weight": "lm_head.weight",
    "lm_head_bias": None,
    "embed_weight": "model.embed_tokens.weight",
}

_GPTNEOX_WEIGHT_NAMES = {
    "norm_weight": "gpt_neox.final_layer_norm.weight",
    "lm_head_weight": "embed_out.weight",
    "lm_head_bias": None,
    "embed_weight": "gpt_neox.embed_in.weight",
}

_GPT2_WEIGHT_NAMES = {
    "norm_weight": "transformer.ln_f.weight",
    "lm_head_weight": "lm_head.weight",
    "lm_head_bias": None,
    "embed_weight": "transformer.wte.weight",
}

ARCHITECTURE_WEIGHT_NAMES = {
    # Llama family
    "LlamaForCausalLM": _DEFAULT_WEIGHT_NAMES,
    "MistralForCausalLM": _DEFAULT_WEIGHT_NAMES,
    "Qwen2ForCausalLM": _DEFAULT_WEIGHT_NAMES,
    "Qwen3ForCausalLM": _DEFAULT_WEIGHT_NAMES,
    "GemmaForCausalLM": _DEFAULT_WEIGHT_NAMES,
    "Gemma2ForCausalLM": _DEFAULT_WEIGHT_NAMES,
    "Gemma3ForCausalLM": _DEFAULT_WEIGHT_NAMES,
    "PhiForCausalLM": _DEFAULT_WEIGHT_NAMES,
    "Phi3ForCausalLM": _DEFAULT_WEIGHT_NAMES,
    "GraniteForCausalLM": _DEFAULT_WEIGHT_NAMES,
    "InternLM2ForCausalLM": _DEFAULT_WEIGHT_NAMES,
    "CohereForCausalLM": _DEFAULT_WEIGHT_NAMES,
    "DeepseekV3ForCausalLM": _DEFAULT_WEIGHT_NAMES,
    # GPT-NeoX family
    "GPTNeoXForCausalLM": _GPTNEOX_WEIGHT_NAMES,
    # GPT-2 family
    "GPT2LMHeadModel": _GPT2_WEIGHT_NAMES,
}


def infer_weight_names(
    model_id: str,
    norm_weight_name: Optional[str] = None,
    lm_head_weight_name: Optional[str] = None,
    lm_head_bias_name: Optional[str] = None,
    embed_weight_name: Optional[str] = None,
) -> dict:
    """
    Infer the weight tensor names for lm_head and final norm based on model
    architecture. Manual overrides take precedence.

    :param model_id: HuggingFace model ID or local path
    :param norm_weight_name: override for final norm weight name
    :param lm_head_weight_name: override for lm_head weight name
    :param lm_head_bias_name: override for lm_head bias name
    :param embed_weight_name: override for embed weight name (for tied embeddings)
    :return: dict with keys: norm_weight, lm_head_weight, lm_head_bias,
             embed_weight, tie_word_embeddings, norm_type, norm_eps
    """
    config = AutoConfig.from_pretrained(model_id)
    config = getattr(config, "text_config", config)

    # Detect architecture
    architectures = getattr(config, "architectures", [])
    arch = architectures[0] if architectures else None

    # Get defaults from registry
    defaults = ARCHITECTURE_WEIGHT_NAMES.get(arch, _DEFAULT_WEIGHT_NAMES)

    # Detect norm type and eps
    norm_type = "rms_norm"
    norm_eps = 1e-6
    if hasattr(config, "rms_norm_eps"):
        norm_type = "rms_norm"
        norm_eps = config.rms_norm_eps
    elif hasattr(config, "layer_norm_epsilon"):
        norm_type = "layer_norm"
        norm_eps = config.layer_norm_epsilon
    elif hasattr(config, "layer_norm_eps"):
        norm_type = "layer_norm"
        norm_eps = config.layer_norm_eps

    # Check for tied embeddings
    tie_word_embeddings = getattr(config, "tie_word_embeddings", False)

    return {
        "norm_weight": norm_weight_name or defaults["norm_weight"],
        "lm_head_weight": lm_head_weight_name or defaults["lm_head_weight"],
        "lm_head_bias": lm_head_bias_name or defaults["lm_head_bias"],
        "embed_weight": embed_weight_name or defaults["embed_weight"],
        "tie_word_embeddings": tie_word_embeddings,
        "norm_type": norm_type,
        "norm_eps": norm_eps,
    }


def detect_last_layer_index(model_id: str) -> int:
    """
    Read model config and return the layer index for extracting post-norm
    hidden states from vLLM. This is num_hidden_layers (NOT num_hidden_layers - 1),
    which gives us the output after the final layer norm — ready to be passed
    directly to lm_head without additional normalization.

    :param model_id: HuggingFace model ID or local path
    :return: layer index for vLLM extraction (num_hidden_layers)
    """
    config = AutoConfig.from_pretrained(model_id)
    config = getattr(config, "text_config", config)
    return config.num_hidden_layers


def _get_weight_map(model_path: str) -> dict:
    """
    Get the tensor name -> shard file mapping for a model checkpoint.
    Handles both sharded and single-file models, local and hub paths.

    :param model_path: local directory or HuggingFace model ID
    :return: dict mapping tensor names to file paths
    """
    if os.path.isdir(model_path):
        # Local directory
        index_path = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                index = json.load(f)
            return {
                name: os.path.join(model_path, shard)
                for name, shard in index["weight_map"].items()
            }
        else:
            # Single safetensors file
            single_path = os.path.join(model_path, "model.safetensors")
            if os.path.exists(single_path):
                with safe_open(single_path, framework="pt") as f:
                    return {name: single_path for name in f.keys()}
            raise FileNotFoundError(
                f"No safetensors files found in {model_path}. "
                "Expected model.safetensors or model.safetensors.index.json"
            )
    else:
        # HuggingFace hub - download required files
        from huggingface_hub import hf_hub_download

        try:
            index_path = hf_hub_download(
                model_path, "model.safetensors.index.json"
            )
            with open(index_path) as f:
                index = json.load(f)
            return {
                name: (model_path, shard)
                for name, shard in index["weight_map"].items()
            }
        except Exception:
            # Try single file
            single_path = hf_hub_download(model_path, "model.safetensors")
            with safe_open(single_path, framework="pt") as f:
                return {name: single_path for name in f.keys()}


def _load_tensor_from_path(file_ref, tensor_name: str) -> torch.Tensor:
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
    norm_weight_name: Optional[str] = None,
    lm_head_weight_name: Optional[str] = None,
    lm_head_bias_name: Optional[str] = None,
    embed_weight_name: Optional[str] = None,
    device: str = "cpu",
) -> dict:
    """
    Load only the lm_head weight, optional bias, and final norm weight
    from a model checkpoint without loading the full model.

    :param model_id: HuggingFace model ID or local path
    :param norm_weight_name: override for final norm weight tensor name
    :param lm_head_weight_name: override for lm_head weight tensor name
    :param lm_head_bias_name: override for lm_head bias tensor name
    :param embed_weight_name: override for embedding weight name (tied case)
    :param device: device to load tensors to
    :return: dict with keys: lm_head_weight, lm_head_bias (or None),
             norm_weight, norm_type, norm_eps
    """
    names = infer_weight_names(
        model_id,
        norm_weight_name=norm_weight_name,
        lm_head_weight_name=lm_head_weight_name,
        lm_head_bias_name=lm_head_bias_name,
        embed_weight_name=embed_weight_name,
    )

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

    lm_head_weight = _load_tensor_from_path(
        weight_map[lm_head_key], lm_head_key
    ).to(device)

    # Load optional lm_head bias
    lm_head_bias = None
    if names["lm_head_bias"] and names["lm_head_bias"] in weight_map:
        lm_head_bias = _load_tensor_from_path(
            weight_map[names["lm_head_bias"]], names["lm_head_bias"]
        ).to(device)

    # Load norm weight
    norm_key = names["norm_weight"]
    if norm_key not in weight_map:
        raise KeyError(
            f"Cannot find norm weight '{norm_key}' in checkpoint for {model_id}"
        )
    norm_weight = _load_tensor_from_path(
        weight_map[norm_key], norm_key
    ).to(device)

    return {
        "lm_head_weight": lm_head_weight,
        "lm_head_bias": lm_head_bias,
        "norm_weight": norm_weight,
        "norm_type": names["norm_type"],
        "norm_eps": names["norm_eps"],
    }


