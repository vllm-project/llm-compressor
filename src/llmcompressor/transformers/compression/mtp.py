"""Checkpoint handling for MTP layers supported by Transformers."""

import json
import os
import re
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps

from compressed_tensors.quantization import QuantizationMetadata
from compressed_tensors.utils import patch_attr
from compressed_tensors.utils.mtp import save_mtp_tensors_to_checkpoint
from compressed_tensors.utils.safetensors_load import (
    get_checkpoint_files,
    get_safetensors_header,
    get_weight_map,
)
from loguru import logger
from safetensors import safe_open
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, cached_file

FALLBACK_EXAMPLE = "examples/model_free_ptq/mtp_fp8_fallback.py"


def _mtp_patterns(model: PreTrainedModel) -> list[str]:
    """Use the same checkpoint patterns and layer-count filter as MtpModel."""
    num_layers = getattr(model.config.get_text_config(), "num_hidden_layers", None)
    patterns = getattr(model, "_keys_to_ignore_on_load_unexpected", None) or []
    return [
        pattern
        for pattern in patterns
        if (match := re.search(r"\.(\d+)", pattern)) is None
        or num_layers is None
        or int(match.group(1)) >= num_layers
    ]


def _checkpoint_weights(
    source: str, revision: str | None = None
) -> dict[str, tuple[str, str | None]]:
    """Map tensor names to shards without downloading every Hub weight shard."""
    if os.path.isdir(source):
        files = get_checkpoint_files(source)
        return {
            name: (files[shard], None) for name, shard in get_weight_map(files).items()
        }

    index = cached_file(
        source,
        SAFE_WEIGHTS_INDEX_NAME,
        revision=revision,
        _raise_exceptions_for_missing_entries=False,
    )
    if index is None:
        path = cached_file(source, SAFE_WEIGHTS_NAME, revision=revision)
        with safe_open(path, framework="pt") as handle:
            return {name: (path, None) for name in handle.keys()}
    with open(index, encoding="utf-8") as handle:
        weight_map = json.load(handle)["weight_map"]
    return {
        name: (os.path.join(os.path.dirname(index), shard), None)
        for name, shard in weight_map.items()
    }


def _mtp_weights(
    model: PreTrainedModel,
) -> tuple[dict[str, tuple[str, str | None]], list[str]]:
    source = model.name_or_path
    if not source:
        return {}, []
    weights = _checkpoint_weights(source, getattr(model.config, "_commit_hash", None))
    patterns = _mtp_patterns(model)
    try:
        decoder = model.get_decoder()
    except (AttributeError, NotImplementedError):
        decoder = None
    decoder_prefix = next(
        (name for name, module in model.named_modules() if module is decoder), None
    )
    trailing_prefix = (
        f"{decoder_prefix + '.' if decoder_prefix else ''}layers."
        if decoder_prefix is not None
        else None
    )
    num_layers = getattr(model.config.get_text_config(), "num_hidden_layers", None)

    def is_mtp(name: str) -> bool:
        if any(re.search(pattern, name) for pattern in patterns):
            return True
        if re.search(r"(?:^|\.)mtp(?:\.|_)", name):
            return True
        if trailing_prefix and name.startswith(trailing_prefix):
            index = name[len(trailing_prefix) :].split(".", 1)[0]
            return (
                num_layers is not None and index.isdigit() and int(index) >= num_layers
            )
        return False

    mtp_weights = {name: info for name, info in weights.items() if is_mtp(name)}
    if mtp_weights:
        # Only resolve shards containing MTP; cached_file supports offline mode.
        shards = {
            shard: shard
            if os.path.isfile(shard)
            else cached_file(
                source,
                os.path.basename(shard),
                revision=getattr(model.config, "_commit_hash", None),
            )
            for shard, _ in mtp_weights.values()
        }
        headers = {
            shard: get_safetensors_header(path) for shard, path in shards.items()
        }
        mtp_weights = {
            name: (shards[shard], headers[shard][name]["dtype"])
            for name, (shard, _) in mtp_weights.items()
        }
    return mtp_weights, patterns


def has_mtp(model: PreTrainedModel) -> bool:
    """Whether the model config declares MTP layers."""
    text_config = model.config.get_text_config()
    return any(
        getattr(text_config, name, 0)
        for name in (
            "num_mtp_layers",
            "mtp_num_hidden_layers",
            "num_nextn_predict_layers",
        )
    )


def validate_mtp_copy_source(
    model: PreTrainedModel,
) -> tuple[dict[str, tuple[str, str | None]], list[str]] | None:
    if not has_mtp(model):
        return None

    weights, patterns = _mtp_weights(model)
    qparams = set(QuantizationMetadata.all_qparam_names()) | {
        "weight_packed",
        "weight_scale_inv",
    }
    quantized = any(
        name.rpartition(".")[-1] in qparams
        or (dtype is not None and dtype.startswith("F8"))
        or (
            name.endswith(".weight")
            and dtype in ("I8", "U8")
            and f"{name.removesuffix('.weight')}.scale" in weights
        )
        for name, (_, dtype) in weights.items()
    )
    if quantized:
        raise ValueError(
            "Cannot copy source-quantized MTP weights without their serving "
            f"scheme. Convert them first; see {FALLBACK_EXAMPLE}."
        )
    return weights, patterns


@contextmanager
def load_with_mtp_model(
    model_cls: type[PreTrainedModel] = AutoModelForCausalLM,
):
    """Attach MTP before MoE conversion and distributed offloading."""
    original_from_pretrained = model_cls.from_pretrained

    @classmethod
    @wraps(original_from_pretrained)
    def from_pretrained(cls, *args, **kwargs):
        model = original_from_pretrained(*args, **kwargs)
        if hasattr(model, "mtp"):
            return model
        if not _mtp_patterns(model):
            raise ValueError(
                f"{type(model).__name__} has no registered MTP checkpoint patterns. "
                f"For unsupported layouts, see {FALLBACK_EXAMPLE}."
            )

        from transformers.modeling_layers import MtpModel

        # Upstream MtpModel does not dequantize source FP8/packed tensors.
        validate_mtp_copy_source(model)
        device_map = {"": "meta"} if kwargs.get("device_map") == "meta" else None
        backbone_modules = set(model.modules())
        model.mtp = MtpModel.from_pretrained(model, device_map=device_map)

        # Keep only MTP-specific mappings here. The surrounding MoE loader owns
        # the backbone mappings and finishes configuring them after this returns.
        model._mtp_weight_conversions = [
            conv
            for conv in model.mtp._weight_conversions
            if conv not in model._weight_conversions
        ]
        # The backbone owns these modules. Keep MTP's references without
        # registering the same modules twice for offloading and serialization.
        for name in ("embed_tokens", "shared_head", "rotary_emb"):
            shared = getattr(model.mtp, name, None)
            if shared is not None and shared in backbone_modules:
                model.mtp._modules.pop(name)
                object.__setattr__(model.mtp, name, shared)
        return model

    with patch_attr(model_cls, "from_pretrained", from_pretrained):
        yield


def extend_mtp_conversions(model: PreTrainedModel) -> None:
    """Combine MTP and backbone conversions after the MoE loader finishes."""
    from transformers.core_model_loading import PrefixChange

    conversions = deepcopy(model.__dict__.pop("_mtp_weight_conversions", []))
    if not conversions:
        return
    # MtpModel's local names now live under the parent's `mtp` subtree.
    for conv in conversions:
        conv.scope_prefix = "mtp"
        conv.base_model_prefix = ""
    model._weight_conversions = [
        PrefixChange(prefix_to_add="mtp"),
        *conversions,
        *model._weight_conversions,
    ]


def save_mtp_tensors(
    model: PreTrainedModel,
    destination: str,
) -> None:
    """Preserve source MTP when it was not loaded into the model."""
    source_mtp = validate_mtp_copy_source(model)
    if source_mtp is None:
        return
    weights, patterns = source_mtp
    if not weights:
        message = "Model config indicates MTP, but no MTP checkpoint weights were found"
        if not patterns:
            message += f". For unsupported FP8 layouts, see {FALLBACK_EXAMPLE}."
        logger.warning(message)
        return
    save_mtp_tensors_to_checkpoint(
        model.name_or_path,
        destination,
        source_weight_map={name: shard for name, (shard, _) in weights.items()},
    )
