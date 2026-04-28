"""
Utilities for loading lm_head weights from model checkpoints
without loading the full model into memory.
"""

import json
import os
from typing import Optional

import torch
from safetensors import safe_open
from transformers import AutoConfig

# Architecture -> weight name mappings
_DEFAULT_WEIGHT_NAMES = {
    "lm_head_weight": "lm_head.weight",
    "lm_head_bias": None,
    "embed_weight": "model.embed_tokens.weight",
}

_GPTNEOX_WEIGHT_NAMES = {
    "lm_head_weight": "embed_out.weight",
    "lm_head_bias": None,
    "embed_weight": "gpt_neox.embed_in.weight",
}

_GPT2_WEIGHT_NAMES = {
    "lm_head_weight": "lm_head.weight",
    "lm_head_bias": None,
    "embed_weight": "transformer.wte.weight",
}

ARCHITECTURE_WEIGHT_NAMES = {
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
    "GPTNeoXForCausalLM": _GPTNEOX_WEIGHT_NAMES,
    "GPT2LMHeadModel": _GPT2_WEIGHT_NAMES,
}


def infer_weight_names(
    model_id: str,
    lm_head_weight_name: Optional[str] = None,
    lm_head_bias_name: Optional[str] = None,
    embed_weight_name: Optional[str] = None,
) -> dict:
    """
    Infer the weight tensor names for lm_head based on model architecture.
    Manual overrides take precedence.

    :param model_id: HuggingFace model ID or local path
    :param lm_head_weight_name: override for lm_head weight name
    :param lm_head_bias_name: override for lm_head bias name
    :param embed_weight_name: override for embed weight name (for tied embeddings)
    :return: dict with keys: lm_head_weight, lm_head_bias, embed_weight,
             tie_word_embeddings
    """
    config = AutoConfig.from_pretrained(model_id)
    config = getattr(config, "text_config", config)

    architectures = getattr(config, "architectures", [])
    arch = architectures[0] if architectures else None
    defaults = ARCHITECTURE_WEIGHT_NAMES.get(arch, _DEFAULT_WEIGHT_NAMES)
    tie_word_embeddings = getattr(config, "tie_word_embeddings", False)

    return {
        "lm_head_weight": lm_head_weight_name or defaults["lm_head_weight"],
        "lm_head_bias": lm_head_bias_name or defaults["lm_head_bias"],
        "embed_weight": embed_weight_name or defaults["embed_weight"],
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
    lm_head_weight_name: Optional[str] = None,
    lm_head_bias_name: Optional[str] = None,
    embed_weight_name: Optional[str] = None,
    device: str = "cpu",
) -> dict:
    """
    Load only the lm_head weight (and optional bias) from a model checkpoint
    without loading the full model.

    :param model_id: HuggingFace model ID or local path
    :param lm_head_weight_name: override for lm_head weight tensor name
    :param lm_head_bias_name: override for lm_head bias tensor name
    :param embed_weight_name: override for embedding weight name (tied case)
    :param device: device to load tensors to
    :return: dict with keys: lm_head_weight, lm_head_bias (or None)
    """
    names = infer_weight_names(
        model_id,
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
