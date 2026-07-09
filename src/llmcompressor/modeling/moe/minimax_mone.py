"""
MiniMax-M2 helpers for MoNE-pruned checkpoints.

MiniMax-M2 stores experts as fused tensors in the stock Transformers model, while
MoNE checkpoints may contain a mix of full logical experts and ``approx_value``
novice experts. This module provides an opt-in loading context that constructs a
MiniMax model with that mixed expert representation.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import tempfile
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from compressed_tensors.utils import patch_attr
from loguru import logger
from torch import nn
from transformers import AutoModelForCausalLM
from transformers.activations import ACT2FN

from llmcompressor.modeling.moe.linear_experts import LinearExperts2D, NoviceExpertMLP
from llmcompressor.modeling.moe.mone import (
    MoNEModelSupport,
    register_mone_model_support,
)

__all__ = [
    "MiniMaxMoNELayout",
    "configure_minimax_m2_stock_fp8_kernels",
    "is_minimax_mone_checkpoint",
    "load_minimax_mone_model",
    "linearize_minimax_mone_experts",
    "minimax_m2_native_fp8_conversion_context",
    "minimax_m2_rotary_device_context",
    "minimax_mone_modeling_context",
    "normalize_minimax_m2_rope_parameters",
    "postprocess_minimax_mone_export",
    "prepare_minimax_m2_for_mone",
    "prepare_minimax_mone_for_save",
    "prepare_minimax_mone_checkpoint_view",
    "scan_minimax_mone_layout",
]


REAL_EXPERT_RE = (
    r"^(?:model\.)?layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(w[123])\.weight$"
)
APPROX_EXPERT_RE = (
    r"^(?:model\.)?layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.approx_value$"
)
EXPORT_MLP_EXPERT_RE = r"^(?:model\.)?layers\.(\d+)\.mlp\.experts\.(\d+)\.(.+)$"
SAVED_LINEAR_EXPERT_RE = (
    r"^(model\.layers\.(\d+)\.)"
    r"(?:mlp|block_sparse_moe)\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.(weight|weight_scale_inv)$"
)
SAVED_APPROX_EXPERT_RE = (
    r"^(model\.layers\.(\d+)\.)"
    r"(?:mlp|block_sparse_moe)\.experts\.(\d+)\.approx_value$"
)
SAVED_FUSED_EXPERT_RE = (
    r"^(?:model\.)?layers\.(\d+)\.mlp\.experts\."
    r"(gate_up_proj|gate_up_proj_scale_inv|down_proj|down_proj_scale_inv)$"
)
SAVED_SOURCE_DENSE_EXPERT_RE = (
    r"^(?:model\.)?layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\."
    r"w[123]\.(?:weight|weight_scale_inv)$"
)
MINIMAX_MONE_MODEL_TYPE = "minimax_m2_compressed"
MINIMAX_MONE_LAYOUT_ATTR = "_llmcompressor_minimax_mone_layout"
MINIMAX_MONE_SOURCE_ATTR = "_llmcompressor_minimax_mone_source_dir"
MINIMAX_MONE_VIEW_ATTR = "_llmcompressor_minimax_mone_view_dir"
MINIMAX_M2_STOCK_FP8_TRITON_PATCH = "stock_fp8_triton_fallback"
_MINIMAX_M2_STOCK_FP8_TRITON_LOGGED = False


@dataclass(frozen=True)
class MiniMaxMoNELayout:
    num_layers: int
    num_logical_experts: int
    real_experts_by_layer: dict[int, tuple[int, ...]]
    constant_experts_by_layer: dict[int, tuple[int, ...]]

    @classmethod
    def from_config(cls, config: Any) -> "MiniMaxMoNELayout":
        approximate_experts = _normalize_layer_map(
            getattr(config, "approximate_experts", None) or {}
        )
        num_layers = int(config.num_hidden_layers)
        num_logical_experts = int(config.num_local_experts)

        real_by_layer = {}
        const_by_layer = {}
        all_experts = set(range(num_logical_experts))
        for layer_idx in range(num_layers):
            constants = tuple(sorted(approximate_experts.get(layer_idx, ())))
            real = tuple(sorted(all_experts - set(constants)))
            real_by_layer[layer_idx] = real
            const_by_layer[layer_idx] = constants

        return cls(
            num_layers=num_layers,
            num_logical_experts=num_logical_experts,
            real_experts_by_layer=real_by_layer,
            constant_experts_by_layer=const_by_layer,
        )

    def as_approximate_experts_config(self) -> dict[str, list[int]]:
        return {
            str(layer_idx): list(experts)
            for layer_idx, experts in self.constant_experts_by_layer.items()
        }


def is_minimax_mone_checkpoint(model_dir: str | Path) -> bool:
    """
    Return True for local MiniMax-M2 MoNE checkpoints.
    """

    config_path = Path(model_dir) / "config.json"
    if not config_path.exists():
        return False
    try:
        config = _load_json(config_path)
    except json.JSONDecodeError:
        return False
    return config.get("model_type") == MINIMAX_MONE_MODEL_TYPE


def normalize_minimax_m2_rope_parameters(config: dict[str, Any]) -> dict[str, Any]:
    """
    Preserve MiniMax-M2 partial RoPE when loading through stock Transformers.

    MiniMax-M2 checkpoints expose ``rotary_dim`` as a top-level field. Stock
    Transformers reads MiniMax rotary width from ``rope_parameters`` instead,
    so a plain checkpoint view otherwise rotates the full attention head.
    """

    raw_rope_parameters = config.get("rope_parameters")
    rope_parameters = (
        dict(raw_rope_parameters) if isinstance(raw_rope_parameters, dict) else {}
    )
    raw_rope_scaling = config.get("rope_scaling")
    rope_scaling = dict(raw_rope_scaling) if isinstance(raw_rope_scaling, dict) else {}

    rope_type = (
        rope_parameters.get("rope_type")
        or rope_parameters.get("type")
        or rope_scaling.get("rope_type")
        or rope_scaling.get("type")
        or "default"
    )
    rope_parameters.setdefault("rope_type", rope_type)

    rope_theta = (
        rope_parameters.get("rope_theta")
        or rope_scaling.get("rope_theta")
        or config.get("rope_theta")
        or 5000000
    )
    rope_parameters["rope_theta"] = rope_theta

    if rope_parameters.get("partial_rotary_factor") is None:
        partial_rotary_factor = (
            config.get("partial_rotary_factor")
            or rope_scaling.get("partial_rotary_factor")
            or _partial_rotary_factor_from_rotary_dim(config)
            or 1.0
        )
        rope_parameters["partial_rotary_factor"] = partial_rotary_factor

    config["rope_parameters"] = rope_parameters
    return config


def configure_minimax_m2_stock_fp8_kernels() -> bool:
    """
    Select a stable stock Transformers FP8 kernel path.

    Transformers 5.9 maps ``deep-gemm`` to a stale kernel-hub version. Newer
    DeepGEMM builds can load, but have shown CUDA illegal memory accesses during
    sharded MiniMax-M2 calibration. Removing only the DeepGEMM hub entry keeps
    execution on Transformers' documented Triton fine-grained-FP8 fallback.
    """

    global _MINIMAX_M2_STOCK_FP8_TRITON_LOGGED

    try:
        from transformers.integrations import hub_kernels
    except Exception:
        return False

    kernel_mapping = getattr(hub_kernels, "_HUB_KERNEL_MAPPING", None)
    if not isinstance(kernel_mapping, dict):
        return False

    kernel_mapping.pop("deep-gemm", None)

    module_mapping = getattr(hub_kernels, "_KERNEL_MODULE_MAPPING", None)
    if isinstance(module_mapping, dict):
        module_mapping["deep-gemm"] = None

    if not _MINIMAX_M2_STOCK_FP8_TRITON_LOGGED:
        logger.info("Configured stock MiniMax-M2 FP8 Triton fallback")
        _MINIMAX_M2_STOCK_FP8_TRITON_LOGGED = True

    return True


def prepare_minimax_m2_for_mone(model: nn.Module) -> list[str]:
    """
    Apply stock MiniMax-M2 runtime preparation needed by MoNE.

    The model architecture and forward behavior stay stock Transformers. Today
    this only selects the stable stock Triton FP8 fallback when MiniMax-M2 FP8
    expert kernels are available.
    """
    patches = []
    if _is_minimax_m2_model(model) and configure_minimax_m2_stock_fp8_kernels():
        patches.append(MINIMAX_M2_STOCK_FP8_TRITON_PATCH)
    return patches


def scan_minimax_mone_layout(model_dir: str | Path) -> MiniMaxMoNELayout:
    """
    Read a MiniMax MoNE checkpoint layout from its config/index.
    """

    model_dir = Path(model_dir)
    config = _load_json(model_dir / "config.json")
    num_layers = int(config["num_hidden_layers"])
    num_logical_experts = int(config.get("num_local_experts") or config["num_experts"])
    weight_map = _load_weight_map(model_dir)

    import re

    real_re = re.compile(REAL_EXPERT_RE)
    approx_re = re.compile(APPROX_EXPERT_RE)
    real_parts: dict[tuple[int, int], set[str]] = defaultdict(set)
    constants: dict[int, set[int]] = defaultdict(set)

    for key in weight_map:
        if match := real_re.match(key):
            real_parts[(int(match.group(1)), int(match.group(2)))].add(match.group(3))
        elif match := approx_re.match(key):
            constants[int(match.group(1))].add(int(match.group(2)))

    expected_parts = {"w1", "w2", "w3"}
    real_by_layer = {}
    const_by_layer = {}
    for layer_idx in range(num_layers):
        incomplete = {
            expert_idx: parts
            for (layer, expert_idx), parts in real_parts.items()
            if layer == layer_idx and parts != expected_parts
        }
        if incomplete:
            raise ValueError(
                f"Layer {layer_idx} has incomplete MiniMax MoNE experts: "
                f"{dict(sorted(incomplete.items())[:8])}"
            )

        real = tuple(
            sorted(
                expert_idx
                for (layer, expert_idx), parts in real_parts.items()
                if layer == layer_idx and parts == expected_parts
            )
        )
        const = tuple(sorted(constants.get(layer_idx, set())))
        invalid = (set(real) | set(const)) - set(range(num_logical_experts))
        if invalid:
            raise ValueError(
                f"Layer {layer_idx} references MiniMax MoNE expert ids outside "
                f"0..{num_logical_experts - 1}: {sorted(invalid)[:16]}"
            )

        overlap = set(real) & set(const)
        if overlap:
            raise ValueError(
                f"Layer {layer_idx} marks MiniMax MoNE experts as both real "
                f"and novice: {sorted(overlap)[:16]}"
            )

        missing = set(range(num_logical_experts)) - set(real) - set(const)
        if missing:
            raise ValueError(
                f"Layer {layer_idx} does not classify all logical experts; "
                f"missing first ids: {sorted(missing)[:16]}"
            )

        real_by_layer[layer_idx] = real
        const_by_layer[layer_idx] = const

    config_approx = config.get("approximate_experts")
    if isinstance(config_approx, dict):
        config_approx = _normalize_layer_map(config_approx)
        for layer_idx, const in const_by_layer.items():
            if (
                layer_idx in config_approx
                and tuple(sorted(config_approx[layer_idx])) != const
            ):
                raise ValueError(
                    "config.approximate_experts disagrees with safetensors "
                    f"index at layer {layer_idx}"
                )

    return MiniMaxMoNELayout(
        num_layers=num_layers,
        num_logical_experts=num_logical_experts,
        real_experts_by_layer=real_by_layer,
        constant_experts_by_layer=const_by_layer,
    )


@contextlib.contextmanager
def minimax_mone_modeling_context(layout: MiniMaxMoNELayout) -> Iterator[None]:
    """
    Temporarily patch Transformers' MiniMax-M2 MoE block for MoNE loading.
    """

    from transformers.models.minimax_m2 import modeling_minimax_m2 as modeling

    block_cls = _build_mone_sparse_moe_block(modeling, layout)
    model_classes = [
        getattr(modeling, name)
        for name in (
            "MiniMaxM2PreTrainedModel",
            "MiniMaxM2Model",
            "MiniMaxM2ForCausalLM",
        )
        if hasattr(modeling, name)
    ]

    with contextlib.ExitStack() as stack:
        stack.enter_context(patch_attr(modeling, "MiniMaxM2SparseMoeBlock", block_cls))
        for cls in model_classes:
            _patch_ignore_patterns(stack, cls)
        yield


@contextlib.contextmanager
def minimax_m2_native_fp8_conversion_context() -> Iterator[None]:
    """
    Register a MiniMax-M2 native-FP8 load mapping that fuses expert scales.

    Transformers' base MiniMax conversion fuses source-format expert weights
    ``w1/w3`` into ``gate_up_proj`` and ``w2`` into ``down_proj``. In native FP8
    mode the matching ``weight_scale_inv`` tensors must be fused too; otherwise
    ``FP8Experts`` runs with missing or uninitialized expert scales.
    """

    from transformers.conversion_mapping import (
        get_checkpoint_conversion_mapping,
        register_checkpoint_conversion_mapping,
    )
    from transformers.core_model_loading import (
        Concatenate,
        MergeModulelist,
        WeightConverter,
        WeightRenaming,
    )

    original = get_checkpoint_conversion_mapping("minimax_m2")
    corrected = [
        WeightRenaming(".block_sparse_moe.", ".mlp."),
        WeightConverter(
            source_patterns=[
                ".experts.*.w1.weight",
                ".experts.*.w3.weight",
            ],
            target_patterns=".experts.gate_up_proj",
            operations=[MergeModulelist(dim=0), Concatenate(dim=1)],
        ),
        WeightConverter(
            source_patterns=[
                ".experts.*.w1.weight_scale_inv",
                ".experts.*.w3.weight_scale_inv",
            ],
            target_patterns=".experts.gate_up_proj_scale_inv",
            operations=[MergeModulelist(dim=0), Concatenate(dim=1)],
        ),
        WeightConverter(
            source_patterns=[".experts.*.w2.weight"],
            target_patterns=".experts.down_proj",
            operations=[MergeModulelist(dim=0)],
        ),
        WeightConverter(
            source_patterns=[".experts.*.w2.weight_scale_inv"],
            target_patterns=".experts.down_proj_scale_inv",
            operations=[MergeModulelist(dim=0)],
        ),
        WeightRenaming(
            ".block_sparse_moe.e_score_correction_bias",
            ".mlp.e_score_correction_bias",
        ),
    ]

    register_checkpoint_conversion_mapping(
        "minimax_m2",
        corrected,
        overwrite=True,
    )
    try:
        yield
    finally:
        if original is not None:
            register_checkpoint_conversion_mapping(
                "minimax_m2",
                original,
                overwrite=True,
            )

@contextlib.contextmanager
def minimax_m2_rotary_device_context() -> Iterator[None]:
    """
    Make Transformers MiniMax-M2 forwards safe under device_map sharding.

    The stock forward computes shared rotary ``cos``/``sin`` tensors once, then
    passes them to every decoder layer. With ``device_map="auto"``, those tensors
    can live on a different GPU than the current attention layer.

    Some sharded MiniMax runs also enter a decoder layer with ``hidden_states``
    on the previous layer's device while inner submodules move their outputs to
    the current layer's device. Align residual tensors before residual adds.
    """

    try:
        from transformers.models.minimax_m2 import modeling_minimax_m2 as modeling
    except Exception:
        yield
        return

    original_apply_rotary = modeling.apply_rotary_pos_emb

    def patched_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
        if cos.device != q.device:
            cos = cos.to(q.device)
        if sin.device != q.device:
            sin = sin.to(q.device)
        return original_apply_rotary(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)

    def patched_decoder_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if attention_mask is not None and attention_mask.device != hidden_states.device:
            attention_mask = attention_mask.to(hidden_states.device)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
        if residual.device != hidden_states.device:
            residual = residual.to(hidden_states.device)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if residual.device != hidden_states.device:
            residual = residual.to(hidden_states.device)
        hidden_states = residual + hidden_states
        return hidden_states

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            patch_attr(modeling, "apply_rotary_pos_emb", patched_apply_rotary_pos_emb)
        )
        stack.enter_context(
            patch_attr(
                modeling.MiniMaxM2DecoderLayer,
                "forward",
                patched_decoder_forward,
            )
        )
        yield


def prepare_minimax_mone_checkpoint_view(
    model_dir: str | Path,
    view_dir: str | Path,
    layout: MiniMaxMoNELayout | None = None,
) -> Path:
    """
    Create a lightweight checkpoint view loadable by stock ``minimax_m2`` config.
    """

    model_dir = Path(model_dir).resolve()
    view_dir = Path(view_dir).resolve()
    if model_dir == view_dir:
        raise ValueError("view_dir must be different from model_dir")

    layout = layout or scan_minimax_mone_layout(model_dir)
    config = _load_json(model_dir / "config.json")
    config["_llmcompressor_mone_original_model_type"] = config.get("model_type")
    config["model_type"] = "minimax_m2"
    config["architectures"] = ["MiniMaxM2ForCausalLM"]
    config["approximate_experts"] = layout.as_approximate_experts_config()
    config.pop("auto_map", None)

    normalize_minimax_m2_rope_parameters(config)

    quant_config = config.get("quantization_config")
    if isinstance(quant_config, dict) and quant_config.get("quant_method") == "fp8":
        quant_config = dict(quant_config)
        quant_config["dequantize"] = True
        config["quantization_config"] = quant_config

    dtype = config.get("torch_dtype") or config.get("dtype") or "bfloat16"
    config.setdefault("torch_dtype", dtype)
    config.setdefault("dtype", dtype)

    view_dir.mkdir(parents=True, exist_ok=True)
    (view_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    for src in model_dir.iterdir():
        if src.name == "config.json":
            continue
        dst = view_dir / src.name
        if dst.exists() or dst.is_symlink():
            if dst.is_symlink() and Path(os.readlink(dst)) == src:
                continue
            if dst.is_dir():
                raise IsADirectoryError(
                    f"{dst} already exists as a directory; choose another view_dir"
                )
            dst.unlink()
        dst.symlink_to(src)

    return view_dir


def load_minimax_mone_model(
    model_dir: str | Path,
    *,
    view_dir: str | Path | None = None,
    model_cls=AutoModelForCausalLM,
    load_constants: bool = True,
    linearize: bool = False,
    keep_checkpoint_view: bool = False,
    **from_pretrained_kwargs,
):
    """
    Load a MiniMax MoNE checkpoint with Transformers.
    """

    model_dir = Path(model_dir).resolve()
    layout = scan_minimax_mone_layout(model_dir)

    retained_tmp_dir = None
    with contextlib.ExitStack() as stack:
        if view_dir is None:
            if keep_checkpoint_view:
                retained_tmp_dir = tempfile.TemporaryDirectory()
                view_path = Path(retained_tmp_dir.name) / "minimax-mone-view"
            else:
                tmp = stack.enter_context(tempfile.TemporaryDirectory())
                view_path = Path(tmp) / "minimax-mone-view"
        else:
            view_path = Path(view_dir)

        prepare_minimax_mone_checkpoint_view(model_dir, view_path, layout)

        with minimax_mone_modeling_context(layout):
            model = model_cls.from_pretrained(view_path, **from_pretrained_kwargs)
        patches = prepare_minimax_m2_for_mone(model)

        if load_constants:
            load_minimax_mone_constants(model, model_dir, layout)

        prepare_minimax_mone_for_save(
            model,
            source_dir=model_dir,
            layout=layout,
            linearize=linearize,
        )
        _set_minimax_mone_runtime_metadata(
            model,
            patches_enabled=patches,
        )
        if retained_tmp_dir is not None:
            setattr(model, "_llmcompressor_minimax_mone_tmp_dir", retained_tmp_dir)
            setattr(model, MINIMAX_MONE_VIEW_ATTR, str(view_path))
            model.config._name_or_path = str(view_path)

    return model


def _set_minimax_mone_runtime_metadata(
    model: nn.Module,
    *,
    patches_enabled: list[str],
) -> None:
    config = getattr(model, "config", None)
    if config is None:
        return
    config.llmcompressor_mone_implementation = {
        "algorithm": "mone",
        "patches_enabled": patches_enabled,
    }


def load_minimax_mone_constants(
    model: nn.Module,
    model_dir: str | Path,
    layout: MiniMaxMoNELayout,
) -> int:
    """
    Load ``approx_value`` tensors into non-persistent MiniMax MoNE buffers.
    """

    from safetensors import safe_open

    model_dir = Path(model_dir)
    weight_map = _load_weight_map(model_dir)
    keys_by_file: dict[str, list[str]] = defaultdict(list)
    for layer_idx, expert_ids in layout.constant_experts_by_layer.items():
        for expert_id in expert_ids:
            key = (
                f"model.layers.{layer_idx}.block_sparse_moe.experts."
                f"{expert_id}.approx_value"
            )
            try:
                keys_by_file[weight_map[key]].append(key)
            except KeyError as exc:
                raise KeyError(
                    f"Missing MiniMax MoNE novice tensor in checkpoint: {key}"
                ) from exc

    modules_by_layer = {
        int(module.layer_idx): module
        for module in model.modules()
        if hasattr(module, "constant_expert_values")
        and hasattr(module, "constant_expert_to_row")
        and hasattr(module, "layer_idx")
    }
    if len(modules_by_layer) != layout.num_layers:
        raise RuntimeError(
            f"Expected {layout.num_layers} MiniMax MoNE expert modules, "
            f"found {len(modules_by_layer)}"
        )

    loaded = 0
    for filename, keys in sorted(keys_by_file.items()):
        with safe_open(model_dir / filename, framework="pt", device="cpu") as handle:
            for key in keys:
                layer_idx, expert_id = _parse_approx_key(key)
                module = modules_by_layer[layer_idx]
                if module.constant_expert_values.is_meta:
                    reference = next(module.parameters())
                    module._buffers["constant_expert_values"] = torch.empty(
                        module.constant_expert_values.shape,
                        device=reference.device,
                        dtype=reference.dtype,
                    )
                row = module.constant_expert_to_row[expert_id]
                target = module.constant_expert_values.data[row]
                tensor = handle.get_tensor(key).reshape(target.shape)
                target.copy_(tensor.to(device=target.device, dtype=target.dtype))
                loaded += 1

    logger.info(f"Loaded {loaded} MiniMax MoNE novice constants")
    return loaded


def prepare_minimax_mone_for_save(
    model: nn.Module,
    *,
    source_dir: str | Path | None = None,
    layout: MiniMaxMoNELayout | None = None,
    linearize: bool = True,
) -> nn.Module:
    """
    Attach MiniMax MoNE export metadata and optionally expose real experts as Linear.
    """

    layout = layout or MiniMaxMoNELayout.from_config(model.config)
    setattr(model, MINIMAX_MONE_LAYOUT_ATTR, layout)
    if source_dir is not None:
        setattr(model, MINIMAX_MONE_SOURCE_ATTR, str(Path(source_dir).resolve()))

    if linearize:
        linearize_minimax_mone_experts(model, layout)

    return model


def linearize_minimax_mone_experts(
    model: nn.Module,
    layout: MiniMaxMoNELayout | None = None,
) -> int:
    """
    Convert patched MiniMax MoNE fused experts into logical-id Linear modules.
    """

    layout = layout or getattr(model, MINIMAX_MONE_LAYOUT_ATTR, None)
    if layout is None:
        layout = MiniMaxMoNELayout.from_config(model.config)

    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise ValueError("Expected MiniMax model to expose `model.layers`")

    converted = 0
    for layer_idx, layer in enumerate(layers):
        fused = layer.mlp.experts
        if isinstance(fused, MiniMaxM2MoNELinearExperts):
            continue
        if not _is_minimax_mone_fused_experts(fused):
            raise ValueError(
                f"Layer {layer_idx} is not a MiniMax MoNE fused experts module"
            )

        layer.mlp.experts = MiniMaxM2MoNELinearExperts.from_fused(
            fused,
            model.config,
            layout,
        )
        converted += 1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if converted and hasattr(model, "_weight_conversions"):
        model._weight_conversions = []
    elif converted:
        setattr(model, "_weight_conversions", [])

    logger.info(
        f"Linearized {converted} MiniMax MoNE expert layers into logical-id "
        "Linear modules"
    )
    return converted


def postprocess_minimax_mone_export(
    model: nn.Module,
    output_dir: str | Path,
) -> None:
    """
    Restore MiniMax MoNE config and copy novice constants into a saved checkpoint.
    """

    layout = _export_layout_for_model(model)
    if layout is None:
        return

    _rewrite_fused_expert_keys_to_source_format(output_dir, layout)
    output_dir = Path(output_dir)
    _rewrite_dense_expert_keys_to_source_format(output_dir, layout)
    _remove_dense_constant_expert_keys(output_dir, layout)
    _add_constants_to_saved_safetensors(output_dir, model, layout)
    _restore_source_precision_tensors(output_dir, model)
    _restore_minimax_mone_config(output_dir, model, layout)
    _validate_minimax_mone_export(output_dir, layout)


def _build_mone_sparse_moe_block(modeling, layout: MiniMaxMoNELayout):
    class MiniMaxM2MoNEExperts(modeling.MiniMaxM2Experts):
        _keys_to_ignore_on_load_missing = {r"constant_expert_values"}
        _keys_to_ignore_on_load_unexpected = {r"approx_value"}
        _keys_to_ignore_on_save = {r"constant_expert_values"}

        def __init__(self, config, layer_idx: int):
            nn.Module.__init__(self)
            self.config = config
            self.layer_idx = layer_idx
            self.logical_expert_ids = tuple(layout.real_experts_by_layer[layer_idx])
            self.constant_expert_ids = tuple(
                layout.constant_experts_by_layer[layer_idx]
            )
            self.num_experts = len(self.logical_expert_ids)
            self.num_logical_experts = config.num_local_experts
            self.hidden_dim = config.hidden_size
            self.intermediate_dim = config.intermediate_size
            self.gate_up_proj = nn.Parameter(
                torch.empty(
                    self.num_experts,
                    2 * self.intermediate_dim,
                    self.hidden_dim,
                )
            )
            self.down_proj = nn.Parameter(
                torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim)
            )
            self.register_buffer(
                "constant_expert_values",
                torch.empty(len(self.constant_expert_ids), self.hidden_dim),
                persistent=False,
            )
            self.act_fn = ACT2FN[config.hidden_act]
            self.logical_to_physical = {
                logical_id: row
                for row, logical_id in enumerate(self.logical_expert_ids)
            }
            self.constant_expert_to_row = {
                logical_id: row
                for row, logical_id in enumerate(self.constant_expert_ids)
            }

        def forward(self, hidden_states, top_k_index, top_k_weights):
            final_hidden_states = torch.zeros_like(hidden_states)
            with torch.no_grad():
                expert_mask = F.one_hot(
                    top_k_index,
                    num_classes=self.num_logical_experts,
                ).permute(2, 1, 0)
                expert_hit = torch.greater(
                    expert_mask.sum(dim=(-1, -2)),
                    0,
                ).nonzero()

            for expert_idx in expert_hit:
                logical_id = int(expert_idx[0].item())
                top_k_pos, token_idx = torch.where(expert_mask[logical_id])
                if token_idx.numel() == 0:
                    continue

                physical_id = self.logical_to_physical.get(logical_id)
                if physical_id is not None:
                    current_state = hidden_states[token_idx]
                    gate, up = F.linear(
                        current_state,
                        self.gate_up_proj[physical_id],
                    ).chunk(2, dim=-1)
                    current_hidden_states = self.act_fn(gate) * up
                    current_hidden_states = F.linear(
                        current_hidden_states,
                        self.down_proj[physical_id],
                    )
                else:
                    row = self.constant_expert_to_row.get(logical_id)
                    if row is None:
                        continue
                    current_hidden_states = self.constant_expert_values[row].to(
                        dtype=hidden_states.dtype
                    )
                    current_hidden_states = current_hidden_states.expand(
                        token_idx.numel(),
                        -1,
                    )

                current_hidden_states = (
                    current_hidden_states
                    * top_k_weights[
                        token_idx,
                        top_k_pos,
                        None,
                    ]
                )
                final_hidden_states.index_add_(
                    0,
                    token_idx,
                    current_hidden_states.to(final_hidden_states.dtype),
                )

            return final_hidden_states

    class MiniMaxM2MoNESparseMoeBlock(modeling.MiniMaxM2SparseMoeBlock):
        _layer_counter = 0

        def __init__(self, config):
            nn.Module.__init__(self)
            layer_idx = MiniMaxM2MoNESparseMoeBlock._layer_counter
            MiniMaxM2MoNESparseMoeBlock._layer_counter += 1
            if layer_idx >= layout.num_layers:
                layer_idx %= layout.num_layers

            self.layer_idx = layer_idx
            self.top_k = config.num_experts_per_tok
            self.jitter_noise = config.router_jitter_noise
            self.gate = modeling.MiniMaxM2TopKRouter(config)
            self.experts = MiniMaxM2MoNEExperts(config, layer_idx)
            self.register_buffer(
                "e_score_correction_bias",
                torch.zeros(config.num_local_experts),
            )

        def forward(self, hidden_states):
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            if self.training and self.jitter_noise > 0:
                hidden_states *= torch.empty_like(hidden_states).uniform_(
                    1.0 - self.jitter_noise,
                    1.0 + self.jitter_noise,
                )
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
            _, top_k_weights, top_k_index = self.gate(
                hidden_states,
                self.e_score_correction_bias,
            )
            hidden_states = self.experts(hidden_states, top_k_index, top_k_weights)
            return hidden_states.reshape(batch_size, sequence_length, hidden_dim)

    return MiniMaxM2MoNESparseMoeBlock


class MiniMaxM2MoNEExpertMLP(nn.Module):
    def __init__(self, config, *, dtype: torch.dtype, device: torch.device):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.up_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.act_fn = ACT2FN[config.hidden_act]

    @torch.no_grad()
    def copy_from_fused(self, fused: nn.Module, physical_idx: int) -> None:
        intermediate = fused.intermediate_dim
        self.gate_proj.weight.copy_(fused.gate_up_proj[physical_idx, :intermediate])
        self.up_proj.weight.copy_(fused.gate_up_proj[physical_idx, intermediate:])
        self.down_proj.weight.copy_(fused.down_proj[physical_idx])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class MiniMaxM2MoNELinearExperts(nn.ModuleDict):
    """
    Logical-id ModuleDict of real experts, with MoNE constants handled separately.
    """

    _keys_to_ignore_on_save = {r"constant_expert_values"}

    def __init__(
        self,
        config,
        layout: MiniMaxMoNELayout,
        layer_idx: int,
        constant_expert_values: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ):
        modules = {
            str(logical_id): MiniMaxM2MoNEExpertMLP(
                config,
                dtype=dtype,
                device=device,
            )
            for logical_id in layout.real_experts_by_layer[layer_idx]
        }
        super().__init__(modules)
        self.layer_idx = layer_idx
        self.num_logical_experts = config.num_local_experts
        self.constant_expert_ids = tuple(layout.constant_experts_by_layer[layer_idx])
        self.constant_expert_to_row = {
            logical_id: row for row, logical_id in enumerate(self.constant_expert_ids)
        }
        self.register_buffer(
            "constant_expert_values",
            constant_expert_values.to(device=device, dtype=dtype),
            persistent=False,
        )

    @classmethod
    @torch.no_grad()
    def from_fused(cls, fused: nn.Module, config, layout: MiniMaxMoNELayout):
        dtype = fused.gate_up_proj.dtype
        device = fused.gate_up_proj.device
        constants = fused.constant_expert_values.detach().clone()
        self = cls(
            config,
            layout,
            fused.layer_idx,
            constants,
            dtype=dtype,
            device=device,
        )
        for logical_id in layout.real_experts_by_layer[fused.layer_idx]:
            physical_idx = fused.logical_to_physical[logical_id]
            self[str(logical_id)].copy_from_fused(fused, physical_idx)
        return self

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        from llmcompressor.modeling.moe.context import get_calibrate_all_experts_flag

        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(
                top_k_index,
                num_classes=self.num_logical_experts,
            ).permute(2, 1, 0)

        calibrate_all = get_calibrate_all_experts_flag()
        for logical_id_str, expert in self.items():
            logical_id = int(logical_id_str)
            top_k_pos, token_idx = torch.where(expert_mask[logical_id])
            if calibrate_all:
                expert_output = expert(hidden_states)[token_idx]
            elif token_idx.numel() == 0:
                continue
            else:
                expert_output = expert(hidden_states[token_idx])

            if token_idx.numel() == 0:
                continue
            expert_output = expert_output * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(
                0,
                token_idx,
                expert_output.to(final_hidden_states.dtype),
            )

        for logical_id in self.constant_expert_ids:
            top_k_pos, token_idx = torch.where(expert_mask[logical_id])
            if token_idx.numel() == 0:
                continue
            row = self.constant_expert_to_row[logical_id]
            output = self.constant_expert_values[row].to(dtype=hidden_states.dtype)
            output = output.expand(token_idx.numel(), -1)
            output = output * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, output)

        return final_hidden_states


def _is_minimax_mone_fused_experts(module: nn.Module) -> bool:
    return (
        hasattr(module, "logical_to_physical")
        and hasattr(module, "constant_expert_values")
        and hasattr(module, "constant_expert_to_row")
        and hasattr(module, "layer_idx")
    )


def _export_layout_for_model(model: nn.Module) -> MiniMaxMoNELayout | None:
    layout = getattr(model, MINIMAX_MONE_LAYOUT_ATTR, None)
    if layout is not None:
        return layout

    config = getattr(model, "config", None)
    if config is None:
        return None
    if getattr(config, "model_type", None) not in (
        "minimax_m2",
        MINIMAX_MONE_MODEL_TYPE,
    ):
        return None
    if not getattr(config, "approximate_experts", None):
        return None

    return MiniMaxMoNELayout.from_config(config)


def _rewrite_fused_expert_keys_to_source_format(
    output_dir: str | Path,
    layout: MiniMaxMoNELayout,
) -> None:
    """
    Split stock MiniMax ``FP8Experts`` tensors into logical source-format experts.

    Native Transformers MiniMax saves expert weights as fused tensors:
    ``mlp.experts.gate_up_proj`` and ``mlp.experts.down_proj``. MoNE exports
    need one tensor set per preserved logical expert, while novice experts are
    represented only by ``approx_value``.
    """

    from safetensors import safe_open
    from safetensors.torch import save_file

    output_dir = Path(output_dir)
    index, _ = _load_safetensors_index(output_dir)
    weight_map = dict(index["weight_map"])

    fused_re = re.compile(SAVED_FUSED_EXPERT_RE)
    fused_by_layer: dict[int, dict[str, str]] = defaultdict(dict)
    for key in weight_map:
        if match := fused_re.match(key):
            fused_by_layer[int(match.group(1))][match.group(2)] = key

    if not fused_by_layer:
        return

    new_tensors: dict[str, torch.Tensor] = {}
    removed_keys: set[str] = set()

    for layer_idx, part_keys in sorted(fused_by_layer.items()):
        if "gate_up_proj" not in part_keys or "down_proj" not in part_keys:
            raise ValueError(
                f"Layer {layer_idx} has incomplete fused MiniMax expert tensors: "
                f"{sorted(part_keys)}"
            )

        gate_up = _read_safetensor_key(
            output_dir,
            weight_map,
            part_keys["gate_up_proj"],
        )
        down = _read_safetensor_key(output_dir, weight_map, part_keys["down_proj"])
        gate_up_scale = (
            _read_safetensor_key(
                output_dir,
                weight_map,
                part_keys["gate_up_proj_scale_inv"],
            )
            if "gate_up_proj_scale_inv" in part_keys
            else None
        )
        down_scale = (
            _read_safetensor_key(
                output_dir,
                weight_map,
                part_keys["down_proj_scale_inv"],
            )
            if "down_proj_scale_inv" in part_keys
            else None
        )

        if gate_up.ndim != 3 or down.ndim != 3:
            raise ValueError(
                f"Layer {layer_idx} fused MiniMax expert tensors must be 3D, "
                f"got gate_up={tuple(gate_up.shape)}, down={tuple(down.shape)}"
            )
        if gate_up.shape[1] % 2 != 0:
            raise ValueError(
                f"Layer {layer_idx} gate_up_proj output dimension is not even: "
                f"{tuple(gate_up.shape)}"
            )

        intermediate = gate_up.shape[1] // 2
        gate_scale_split = (
            _split_gate_up_scale(gate_up_scale, layer_idx)
            if gate_up_scale is not None
            else None
        )

        for expert_id in layout.real_experts_by_layer.get(layer_idx, ()):
            prefix = f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_id}"
            new_tensors[f"{prefix}.w1.weight"] = gate_up[
                expert_id,
                :intermediate,
            ].contiguous()
            new_tensors[f"{prefix}.w3.weight"] = gate_up[
                expert_id,
                intermediate:,
            ].contiguous()
            new_tensors[f"{prefix}.w2.weight"] = down[expert_id].contiguous()

            if gate_scale_split is not None:
                w1_scale, w3_scale = gate_scale_split
                new_tensors[f"{prefix}.w1.weight_scale_inv"] = w1_scale[
                    expert_id
                ].contiguous()
                new_tensors[f"{prefix}.w3.weight_scale_inv"] = w3_scale[
                    expert_id
                ].contiguous()
            if down_scale is not None:
                new_tensors[f"{prefix}.w2.weight_scale_inv"] = down_scale[
                    expert_id
                ].contiguous()

        removed_keys.update(part_keys.values())

    keys_by_file: dict[str, list[str]] = defaultdict(list)
    for key, filename in weight_map.items():
        keys_by_file[filename].append(key)

    for filename, keys in sorted(keys_by_file.items()):
        if not any(key in removed_keys for key in keys):
            continue

        path = output_dir / filename
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {"format": "pt"}
            tensors = {
                key: handle.get_tensor(key)
                for key in handle.keys()
                if key not in removed_keys
            }

        if tensors:
            save_file(tensors, path, metadata=metadata)
        else:
            path.unlink()

    experts_filename = "model-mone-experts.safetensors"
    if new_tensors:
        save_file(
            new_tensors,
            output_dir / experts_filename,
            metadata={"format": "pt"},
        )

    new_weight_map = {
        key: filename for key, filename in weight_map.items() if key not in removed_keys
    }
    for key in new_tensors:
        new_weight_map[key] = experts_filename

    index["weight_map"] = dict(sorted(new_weight_map.items()))
    index["metadata"] = {
        **dict(index.get("metadata") or {}),
        "total_size": _indexed_safetensors_total_size(output_dir, new_weight_map),
    }
    _write_safetensors_index(output_dir, index)
    logger.info(
        f"Split {len(removed_keys)} fused MiniMax expert tensors into "
        f"{len(new_tensors)} source-format expert tensors"
    )


def _rewrite_dense_expert_keys_to_source_format(
    output_dir: Path,
    layout: MiniMaxMoNELayout,
) -> None:
    from safetensors import safe_open
    from safetensors.torch import save_file

    index, _ = _load_safetensors_index(output_dir)
    weight_map = dict(index["weight_map"])
    renames = {
        key: renamed
        for key in weight_map
        if (renamed := _rename_saved_key_to_source_format(key, layout)) != key
    }
    if not renames:
        return

    collision = set(renames.values()) & (set(weight_map) - set(renames))
    if collision:
        raise ValueError(
            "Cannot rewrite MiniMax MoNE expert keys due to collisions: "
            f"{sorted(collision)[:8]}"
        )

    keys_by_file: dict[str, list[str]] = defaultdict(list)
    for key, filename in weight_map.items():
        keys_by_file[filename].append(key)

    for filename, keys in sorted(keys_by_file.items()):
        if not any(key in renames for key in keys):
            continue

        path = output_dir / filename
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {"format": "pt"}
            tensors = {
                renames.get(key, key): handle.get_tensor(key) for key in handle.keys()
            }
        save_file(tensors, path, metadata=metadata)

    index["weight_map"] = {
        renames.get(key, key): filename for key, filename in weight_map.items()
    }
    _write_safetensors_index(output_dir, index)
    logger.info(f"Rewrote {len(renames)} MiniMax MoNE dense expert keys")


def _remove_dense_constant_expert_keys(
    output_dir: Path,
    layout: MiniMaxMoNELayout,
) -> None:
    """
    Remove full expert weights for MoNE novice experts from saved shards.

    Native MiniMax saves may be source-converted by Transformers into
    ``block_sparse_moe.experts.<id>.w*`` keys for every logical expert. For MoNE
    novices those dense weights must be omitted; the export adds ``approx_value``
    tensors separately.
    """

    index, _ = _load_safetensors_index(output_dir)
    weight_map = dict(index["weight_map"])
    remove_keys = {
        key for key in weight_map if _is_dense_constant_expert_key(key, layout)
    }
    if not remove_keys:
        return

    _remove_safetensor_keys(output_dir, weight_map, remove_keys)
    new_weight_map = {
        key: filename for key, filename in weight_map.items() if key not in remove_keys
    }
    index["weight_map"] = dict(sorted(new_weight_map.items()))
    index["metadata"] = {
        **dict(index.get("metadata") or {}),
        "total_size": _indexed_safetensors_total_size(output_dir, new_weight_map),
    }
    _write_safetensors_index(output_dir, index)
    logger.info(f"Removed {len(remove_keys)} dense tensors for MiniMax MoNE novices")


def _rename_saved_key_to_source_format(
    key: str,
    layout: MiniMaxMoNELayout,
) -> str:
    if match := re.match(SAVED_LINEAR_EXPERT_RE, key):
        layer_idx = int(match.group(2))
        expert_id = int(match.group(3))
        proj = match.group(4)
        param = match.group(5)
        if expert_id not in set(layout.real_experts_by_layer.get(layer_idx, ())):
            return key
        source_proj = {
            "gate_proj": "w1",
            "up_proj": "w3",
            "down_proj": "w2",
        }[proj]
        return (
            f"model.layers.{layer_idx}.block_sparse_moe.experts."
            f"{expert_id}.{source_proj}.{param}"
        )

    if match := re.match(SAVED_APPROX_EXPERT_RE, key):
        layer_idx = int(match.group(2))
        expert_id = int(match.group(3))
        if expert_id not in set(layout.constant_experts_by_layer.get(layer_idx, ())):
            return key
        return (
            f"model.layers.{layer_idx}.block_sparse_moe.experts."
            f"{expert_id}.approx_value"
        )

    if ".mlp.gate." in key or key.endswith(".mlp.e_score_correction_bias"):
        return key.replace(".mlp.", ".block_sparse_moe.")

    return key


def _is_dense_constant_expert_key(
    key: str,
    layout: MiniMaxMoNELayout,
) -> bool:
    if match := re.match(SAVED_LINEAR_EXPERT_RE, key):
        layer_idx = int(match.group(2))
        expert_id = int(match.group(3))
        return expert_id in set(layout.constant_experts_by_layer.get(layer_idx, ()))

    if match := re.match(SAVED_SOURCE_DENSE_EXPERT_RE, key):
        layer_idx = int(match.group(1))
        expert_id = int(match.group(2))
        return expert_id in set(layout.constant_experts_by_layer.get(layer_idx, ()))

    return False


def _remove_safetensor_keys(
    output_dir: Path,
    weight_map: dict[str, str],
    remove_keys: set[str],
) -> None:
    from safetensors import safe_open
    from safetensors.torch import save_file

    keys_by_file: dict[str, list[str]] = defaultdict(list)
    for key, filename in weight_map.items():
        keys_by_file[filename].append(key)

    for filename, keys in sorted(keys_by_file.items()):
        if not any(key in remove_keys for key in keys):
            continue

        path = output_dir / filename
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {"format": "pt"}
            tensors = {
                key: handle.get_tensor(key)
                for key in handle.keys()
                if key not in remove_keys
            }

        if tensors:
            save_file(tensors, path, metadata=metadata)
        else:
            path.unlink()


def _patch_ignore_patterns(stack: contextlib.ExitStack, cls: type):
    missing = set(getattr(cls, "_keys_to_ignore_on_load_missing", None) or [])
    unexpected = set(getattr(cls, "_keys_to_ignore_on_load_unexpected", None) or [])
    save = set(getattr(cls, "_keys_to_ignore_on_save", None) or [])
    missing.add(r"constant_expert_values")
    unexpected.add(r"\.experts\.\d+\.approx_value$")
    save.add(r"constant_expert_values")
    stack.enter_context(patch_attr(cls, "_keys_to_ignore_on_load_missing", missing))
    stack.enter_context(
        patch_attr(cls, "_keys_to_ignore_on_load_unexpected", unexpected)
    )
    stack.enter_context(patch_attr(cls, "_keys_to_ignore_on_save", save))


def _load_weight_map(model_dir: Path) -> dict[str, str]:
    index_path = model_dir / "model.safetensors.index.json"
    single_file = model_dir / "model.safetensors"
    if index_path.exists():
        return _load_json(index_path)["weight_map"]
    if single_file.exists():
        from safetensors import safe_open

        with safe_open(single_file, framework="pt", device="cpu") as handle:
            return {key: single_file.name for key in handle.keys()}
    raise FileNotFoundError(f"No safetensors checkpoint found in {model_dir}")


def _collect_constant_tensors(
    model: nn.Module,
    layout: MiniMaxMoNELayout,
) -> dict[str, torch.Tensor]:
    constants = {}
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is not None:
        for layer_idx, layer in enumerate(layers):
            experts = getattr(getattr(layer, "mlp", None), "experts", None)
            constants.update(
                _collect_constants_from_experts(
                    experts,
                    layout,
                    layer_idx,
                )
            )

    if len(constants) == sum(len(v) for v in layout.constant_experts_by_layer.values()):
        return constants

    for module in model.modules():
        if not (
            hasattr(module, "constant_expert_values")
            and hasattr(module, "constant_expert_to_row")
            and hasattr(module, "layer_idx")
        ):
            continue
        constants.update(
            _collect_constants_from_experts(
                module,
                layout,
                int(module.layer_idx),
            )
        )

    expected = sum(len(v) for v in layout.constant_experts_by_layer.values())
    if len(constants) != expected:
        raise RuntimeError(
            f"Expected {expected} MiniMax MoNE novice constants, found {len(constants)}"
        )
    return constants


def _collect_constants_from_experts(
    experts: nn.Module | None,
    layout: MiniMaxMoNELayout,
    layer_idx: int,
) -> dict[str, torch.Tensor]:
    if experts is None:
        return {}

    constants = {}
    for expert_id in layout.constant_experts_by_layer.get(layer_idx, ()):
        key = (
            f"model.layers.{layer_idx}.block_sparse_moe.experts."
            f"{expert_id}.approx_value"
        )
        if hasattr(experts, "constant_expert_values") and hasattr(
            experts,
            "constant_expert_to_row",
        ):
            row = experts.constant_expert_to_row[expert_id]
            constants[key] = experts.constant_expert_values[row].detach().cpu()
            continue

        if isinstance(experts, LinearExperts2D):
            expert = experts[expert_id]
            if not isinstance(expert, NoviceExpertMLP):
                continue
            constants[key] = expert.approx_value.detach().cpu()

    return constants


def _add_constants_to_saved_safetensors(
    output_dir: Path,
    model: nn.Module,
    layout: MiniMaxMoNELayout,
) -> None:
    from safetensors.torch import save_file

    index, _ = _load_safetensors_index(output_dir)
    constants = _collect_constant_tensors(model, layout)
    constants_filename = "model-mone-constants.safetensors"
    new_weight_map = dict(index["weight_map"])
    missing_constants = {
        key: value for key, value in constants.items() if key not in new_weight_map
    }

    if missing_constants:
        save_file(
            missing_constants,
            output_dir / constants_filename,
            metadata={"format": "pt"},
        )
    for key in missing_constants:
        new_weight_map[key] = constants_filename

    metadata = dict(index.get("metadata") or {})
    total_size = int(metadata.get("total_size") or 0)
    if total_size == 0:
        total_size = _safetensors_total_size(output_dir, index["weight_map"])
    total_size += sum(
        tensor.numel() * tensor.element_size() for tensor in missing_constants.values()
    )

    metadata["total_size"] = total_size
    index["metadata"] = metadata
    index["weight_map"] = dict(sorted(new_weight_map.items()))
    _write_safetensors_index(output_dir, index)
    if missing_constants:
        logger.info(
            f"Copied {len(missing_constants)} MiniMax MoNE constants into export"
        )


def _restore_minimax_mone_config(
    output_dir: Path,
    model: nn.Module,
    layout: MiniMaxMoNELayout,
) -> None:
    config_path = output_dir / "config.json"
    export_config = _load_json(config_path) if config_path.exists() else {}
    source_config = _load_source_config(model) or export_config

    restored_config = dict(source_config)
    restored_config["model_type"] = MINIMAX_MONE_MODEL_TYPE
    restored_config["architectures"] = ["MiniMaxM2ForCausalLM"]
    restored_config["approximate_experts"] = layout.as_approximate_experts_config()
    restored_config["approximate_expert_init_tokens"] = (
        _approximate_expert_init_tokens_config(model, layout)
    )
    restored_config.pop("auto_map", None)

    for key in ("dtype", "torch_dtype"):
        if key in export_config:
            restored_config[key] = export_config[key]

    if "quantization_config" in export_config:
        restored_config["quantization_config"] = export_config["quantization_config"]
    elif (
        _checkpoint_has_float8_weights(output_dir)
        and "quantization_config" in source_config
    ):
        quant_config = dict(source_config["quantization_config"])
        quant_config.pop("dequantize", None)
        quant_config.pop("fmt", None)
        restored_config["quantization_config"] = quant_config
    else:
        restored_config.pop("quantization_config", None)

    implementation_metadata = getattr(
        getattr(model, "config", None),
        "llmcompressor_mone_implementation",
        None,
    )
    if implementation_metadata is None:
        implementation_metadata = export_config.get("llmcompressor_mone_implementation")
    if implementation_metadata is not None:
        restored_config["llmcompressor_mone_implementation"] = implementation_metadata

    for key in list(restored_config):
        if key.startswith("_llmcompressor_mone_"):
            restored_config.pop(key, None)

    config_path.write_text(json.dumps(restored_config, indent=2) + "\n")


def _approximate_expert_init_tokens_config(
    model: nn.Module,
    layout: MiniMaxMoNELayout,
) -> dict[str, list[int]]:
    config = getattr(model, "config", None)
    raw_tokens = getattr(config, "approximate_expert_init_tokens", None)
    tokens_by_layer = (
        _normalize_layer_map(raw_tokens) if isinstance(raw_tokens, dict) else {}
    )

    tokens_config = {}
    for layer_idx, experts in layout.constant_experts_by_layer.items():
        tokens = list(tokens_by_layer.get(layer_idx, ()))
        if len(tokens) != len(experts):
            tokens = [0 for _ in experts]
        tokens_config[str(layer_idx)] = tokens
    return tokens_config


def _restore_source_precision_tensors(
    output_dir: Path,
    model: nn.Module,
) -> None:
    """
    Restore tensors that were not FP8 in the source checkpoint but saved as FP8.

    MiniMax-M2 FP8 checkpoints leave some tensors unquantized, such as
    embeddings, layer norms, router gates, and the language-model head. The
    generic save wrapper can emit those as FP8 with ``weight_scale_inv`` sidecars.
    Mode-pd/vLLM-style MiniMax loaders expect the original source precision for
    those tensors, so copy them back from the source checkpoint when possible.
    """

    source_dir = _source_dir_for_model(model)
    if source_dir is None:
        return

    try:
        source_weight_map = _load_weight_map(source_dir)
    except FileNotFoundError:
        return

    from safetensors import safe_open
    from safetensors.torch import save_file

    index, _ = _load_safetensors_index(output_dir)
    output_weight_map = dict(index["weight_map"])

    output_dtypes = _read_tensor_dtypes(output_dir, output_weight_map)
    source_dtypes = _read_tensor_dtypes(
        source_dir,
        {
            key: filename
            for key, filename in source_weight_map.items()
            if key in output_weight_map
        },
    )

    replacements = {
        key: source_weight_map[key]
        for key, output_dtype in output_dtypes.items()
        if key.endswith(".weight")
        and key in source_weight_map
        and _is_float8_dtype(output_dtype)
        and not _is_float8_dtype(source_dtypes.get(key))
    }
    if not replacements:
        return

    removals = {
        key.replace(".weight", ".weight_scale_inv")
        for key in replacements
        if key.replace(".weight", ".weight_scale_inv") in output_weight_map
        and key.replace(".weight", ".weight_scale_inv") not in source_weight_map
    }

    keys_by_file: dict[str, list[str]] = defaultdict(list)
    for key, filename in output_weight_map.items():
        keys_by_file[filename].append(key)

    def source_tensor(key: str) -> torch.Tensor:
        filename = replacements[key]
        with safe_open(source_dir / filename, framework="pt", device="cpu") as h:
            return h.get_tensor(key)

    touched_files = set()
    for filename, keys in sorted(keys_by_file.items()):
        if not any(key in replacements or key in removals for key in keys):
            continue

        path = output_dir / filename
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {"format": "pt"}
            tensors = {}
            for key in handle.keys():
                if key in removals:
                    continue
                tensors[key] = (
                    source_tensor(key)
                    if key in replacements
                    else handle.get_tensor(key)
                )
        save_file(tensors, path, metadata=metadata)
        touched_files.add(filename)

    new_weight_map = {
        key: filename
        for key, filename in output_weight_map.items()
        if key not in removals
    }
    index["weight_map"] = dict(sorted(new_weight_map.items()))
    _write_safetensors_index(output_dir, index)
    logger.info(
        f"Restored {len(replacements)} MiniMax source-precision tensors and "
        f"removed {len(removals)} stale scale tensors across "
        f"{len(touched_files)} shards"
    )


def _source_dir_for_model(model: nn.Module) -> Path | None:
    candidates = [
        getattr(model, MINIMAX_MONE_SOURCE_ATTR, None),
        getattr(getattr(model, "config", None), "_name_or_path", None),
        getattr(model, "name_or_path", None),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.is_dir() and (
            (path / "model.safetensors.index.json").exists()
            or (path / "model.safetensors").exists()
        ):
            return path
    return None


def _read_tensor_dtypes(
    model_dir: Path,
    weight_map: dict[str, str],
) -> dict[str, torch.dtype]:
    from safetensors import safe_open

    keys_by_file: dict[str, list[str]] = defaultdict(list)
    for key, filename in weight_map.items():
        keys_by_file[filename].append(key)

    dtypes = {}
    for filename, keys in sorted(keys_by_file.items()):
        with safe_open(model_dir / filename, framework="pt", device="cpu") as handle:
            for key in keys:
                dtypes[key] = handle.get_tensor(key).dtype
    return dtypes


def _is_float8_dtype(dtype: torch.dtype | None) -> bool:
    return dtype is not None and str(dtype).startswith("torch.float8")


def _load_source_config(model: nn.Module) -> dict[str, Any] | None:
    source_dir = getattr(model, MINIMAX_MONE_SOURCE_ATTR, None)
    if source_dir is not None:
        config_path = Path(source_dir) / "config.json"
        if config_path.exists():
            return _load_json(config_path)

    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "to_dict"):
        return config.to_dict()
    return None


def _checkpoint_has_float8_weights(output_dir: Path) -> bool:
    from safetensors import safe_open

    weight_map = _load_weight_map(output_dir)
    for key, filename in weight_map.items():
        if not key.endswith(".weight"):
            continue
        with safe_open(output_dir / filename, framework="pt", device="cpu") as handle:
            if str(handle.get_tensor(key).dtype).startswith("torch.float8"):
                return True
    return False


def _validate_minimax_mone_export(
    output_dir: Path,
    layout: MiniMaxMoNELayout,
) -> None:
    index_path = output_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing export index: {index_path}")

    keys = set(_load_json(index_path)["weight_map"])
    constant_keys = set(_iter_approx_value_keys(layout))
    missing_constants = constant_keys - keys
    if missing_constants:
        raise ValueError(
            f"Export is missing {len(missing_constants)} approx_value tensors; "
            f"first ids: {sorted(missing_constants)[:8]}"
        )

    exported_linear = defaultdict(set)
    export_re = re.compile(EXPORT_MLP_EXPERT_RE)
    for key in keys:
        if match := export_re.match(key):
            exported_linear[int(match.group(1))].add(int(match.group(2)))

    bad_constants = []
    for layer_idx, constants in layout.constant_experts_by_layer.items():
        bad_ids = exported_linear.get(layer_idx, set()) & set(constants)
        bad_ids.update(
            expert_id
            for expert_id in constants
            if _has_dense_weights_for_constant_expert(keys, layer_idx, expert_id)
        )
        if bad_ids:
            bad_constants.append(f"model.layers.{layer_idx}.mlp.experts.{min(bad_ids)}")
    if bad_constants:
        raise ValueError(
            "Export contains Linear expert tensors for MiniMax MoNE novices: "
            f"{bad_constants[:8]}"
        )

    missing_real = []
    for layer_idx, real_ids in layout.real_experts_by_layer.items():
        for expert_id in real_ids:
            if not _has_exported_real_expert(keys, layer_idx, expert_id):
                missing_real.append(
                    f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_id}"
                )
    if missing_real:
        raise ValueError(
            "Export is missing Linear expert tensors for MiniMax MoNE real "
            f"experts: {missing_real[:8]}"
        )

    if any("constant_expert_values" in key for key in keys):
        raise ValueError("Export still contains adapter-only constant_expert_values")

    logger.info(f"Validated MiniMax MoNE export with {len(constant_keys)} constants")


def _has_exported_real_expert(
    keys: set[str],
    layer_idx: int,
    expert_id: int,
) -> bool:
    source_prefix = (
        f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_id}."
    )
    if all(f"{source_prefix}{part}.weight" in keys for part in ("w1", "w2", "w3")):
        return True

    linear_prefix = f"model.layers.{layer_idx}.mlp.experts.{expert_id}.gate_proj."
    return any(key.startswith(linear_prefix) for key in keys)


def _has_dense_weights_for_constant_expert(
    keys: set[str],
    layer_idx: int,
    expert_id: int,
) -> bool:
    prefixes = (
        f"model.layers.{layer_idx}.mlp.experts.{expert_id}.",
        f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_id}.",
    )
    dense_parts = (
        "gate_proj.",
        "up_proj.",
        "down_proj.",
        "w1.",
        "w2.",
        "w3.",
    )
    return any(
        key.startswith(prefix) and any(part in key for part in dense_parts)
        for prefix in prefixes
        for key in keys
    )


def _iter_approx_value_keys(layout: MiniMaxMoNELayout) -> list[str]:
    keys = []
    for layer_idx, experts in layout.constant_experts_by_layer.items():
        for expert_id in experts:
            keys.append(
                f"model.layers.{layer_idx}.block_sparse_moe.experts."
                f"{expert_id}.approx_value"
            )
    return keys


def _load_safetensors_index(output_dir: Path) -> tuple[dict[str, Any], bool]:
    from safetensors import safe_open

    index_path = output_dir / "model.safetensors.index.json"
    single_file = output_dir / "model.safetensors"
    if index_path.exists():
        return _load_json(index_path), True
    if single_file.exists():
        with safe_open(single_file, framework="pt", device="cpu") as handle:
            weight_map = {key: single_file.name for key in handle.keys()}
        return {"metadata": {}, "weight_map": weight_map}, False
    raise FileNotFoundError(f"No safetensors checkpoint found in {output_dir}")


def _write_safetensors_index(output_dir: Path, index: dict[str, Any]) -> None:
    index_path = output_dir / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps(
            {
                "metadata": dict(index.get("metadata") or {}),
                "weight_map": dict(sorted(index["weight_map"].items())),
            },
            indent=2,
        )
        + "\n"
    )


def _read_safetensor_key(
    output_dir: Path,
    weight_map: dict[str, str],
    key: str,
) -> torch.Tensor:
    from safetensors import safe_open

    with safe_open(
        output_dir / weight_map[key],
        framework="pt",
        device="cpu",
    ) as handle:
        return handle.get_tensor(key)


def _split_gate_up_scale(
    gate_up_scale: torch.Tensor,
    layer_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if gate_up_scale.ndim < 2:
        return gate_up_scale, gate_up_scale

    scale_out_dim = gate_up_scale.shape[1]
    if scale_out_dim == 1:
        return gate_up_scale, gate_up_scale
    if scale_out_dim % 2 != 0:
        raise ValueError(
            f"Layer {layer_idx} gate_up_proj_scale_inv output dimension is not "
            f"even: {tuple(gate_up_scale.shape)}"
        )

    return gate_up_scale.chunk(2, dim=1)


def _indexed_safetensors_total_size(
    output_dir: Path,
    weight_map: dict[str, str],
) -> int:
    from safetensors import safe_open

    total_size = 0
    keys_by_file: dict[str, list[str]] = defaultdict(list)
    for key, filename in weight_map.items():
        keys_by_file[filename].append(key)

    for filename, keys in sorted(keys_by_file.items()):
        path = output_dir / filename
        if not path.exists():
            continue
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in keys:
                tensor = handle.get_tensor(key)
                total_size += tensor.numel() * tensor.element_size()
    return total_size


def _safetensors_total_size(
    output_dir: Path,
    weight_map: dict[str, str],
) -> int:
    from safetensors import safe_open

    total_size = 0
    for filename in sorted(set(weight_map.values())):
        with safe_open(output_dir / filename, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                tensor = handle.get_tensor(key)
                total_size += tensor.numel() * tensor.element_size()
    return total_size


def _partial_rotary_factor_from_rotary_dim(config: dict[str, Any]) -> float | None:
    rotary_dim = config.get("rotary_dim")
    head_dim = config.get("head_dim")
    if (
        head_dim is None
        and config.get("hidden_size")
        and config.get("num_attention_heads")
    ):
        head_dim = config["hidden_size"] // config["num_attention_heads"]
    if rotary_dim is None or head_dim in (None, 0):
        return None
    return float(rotary_dim) / float(head_dim)


def _is_minimax_m2_model(model: nn.Module) -> bool:
    config = getattr(model, "config", None)
    return getattr(config, "model_type", None) in {
        "minimax_m2",
        MINIMAX_MONE_MODEL_TYPE,
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _normalize_layer_map(values: dict[Any, Any]) -> dict[int, tuple[int, ...]]:
    normalized = {}
    for key, layer_values in values.items():
        normalized[int(key)] = tuple(int(value) for value in layer_values)
    return normalized


def _parse_approx_key(key: str) -> tuple[int, int]:
    import re

    match = re.match(APPROX_EXPERT_RE, key)
    if match is None:
        raise ValueError(f"Unexpected MiniMax MoNE approx_value key: {key}")
    return int(match.group(1)), int(match.group(2))


def _prepare_minimax_model_for_save(model: nn.Module) -> None:
    config = getattr(model, "config", None)
    if config is None or not _is_minimax_m2_model(model):
        return
    if not getattr(config, "approximate_experts", None):
        return

    prepare_minimax_mone_for_save(model, linearize=False)


def _get_minimax_mone_processor_source(model: nn.Module) -> str | None:
    return getattr(model, MINIMAX_MONE_VIEW_ATTR, None)


def _register_minimax_mone_support() -> None:
    register_mone_model_support(
        MoNEModelSupport(
            name="minimax_m2",
            prepare_model=prepare_minimax_m2_for_mone,
            prepare_for_save=_prepare_minimax_model_for_save,
            postprocess_export=postprocess_minimax_mone_export,
            is_checkpoint=is_minimax_mone_checkpoint,
            load_checkpoint=load_minimax_mone_model,
            processor_source=_get_minimax_mone_processor_source,
        )
    )


_register_minimax_mone_support()
