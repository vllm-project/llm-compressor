#!/usr/bin/env python3
"""Quantize one NVFP4 imatrix-observer ablation variant."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import struct
import sys
import time
from pathlib import Path

import torch
import transformers
from compressed_tensors.quantization import preset_name_to_scheme

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context

MODEL_SPECS = {
    "qwen38_27b": {
        "model_id": "Qwen/Qwen3.8-27B",
        "class_name": "Qwen3_5ForConditionalGeneration",
        "processor": True,
        "targets": "Linear",
        "ignore": [
            "lm_head",
            r"re:.*mlp\.gate$",
            r"re:.*shared_expert_gate.*",
            r"re:.*embed_tokens$",
            r"re:visual.*",
            r"re:model\.visual.*",
            r"re:.*linear_attn.*",
        ],
        "moe_calibrate_all_experts": True,
    },
    "qwen3_30b_a3b": {
        "model_id": "Qwen/Qwen3-30B-A3B",
        "class_name": "AutoModelForCausalLM",
        "processor": False,
        "targets": "Linear",
        "ignore": ["lm_head", r"re:.*mlp\.gate$"],
        "moe_calibrate_all_experts": True,
    },
    "nemotron35_lightning_30b_a3b": {
        "model_id": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
        "class_name": "AutoModelForCausalLM",
        "processor": False,
        "targets": "Linear",
        # NVIDIA's released Lightning recipe uses W4A16 NVFP4 for linears;
        # W4A4 activations are not the supported path for this hybrid model.
        "scheme": "NVFP4A16",
        "ignore": [
            r"re:.*conv1d.*",
            r"re:.*backbone\.embeddings.*",
            r"re:.*backbone\.embedding$",
            r"re:.*_latent_proj.*",
            r"re:.*mixer\.gate$",
            r"re:.*mixer\.gate\..*",
            r"re:.*mtp\.layers.*",
            # Nemotron 3.5's released NVFP4 recipes keep attention projections
            # in BF16; quantizing these alongside the MoE and Mamba linears
            # produced unusable generations in this ablation.
            # Transformers exposes these Linear modules under model.layers;
            # the saved checkpoint uses the `backbone.layers` tensor prefix.
            r"re:.*model\.layers\.\d+\.mixer\.(q_proj|k_proj|v_proj|o_proj)$",
            r"re:.*router.*",
            r"re:.*mlp\.gate$",
            "backbone.norm_f",
            "lm_head",
        ],
        "moe_calibrate_all_experts": True,
    },
    "qwen25_7b_instruct": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "class_name": "AutoModelForCausalLM",
        "processor": False,
        "targets": "Linear",
        "ignore": ["lm_head"],
        "moe_calibrate_all_experts": False,
    },
    "gemma4_26b_a4b_it": {
        "model_id": "google/gemma-4-26B-A4B-it",
        "class_name": "Gemma4ForConditionalGeneration",
        "processor": True,
        "targets": "Linear",
        "ignore": [
            "lm_head",
            r"re:.*embed.*",
            r"re:.*router.*",
            r"re:.*vision_tower.*",
            r"re:.*audio.*",
        ],
        "moe_calibrate_all_experts": True,
    },
}

CALIBRATION_DATASET = "perfectblend"
CALIBRATION_SPLIT = "train[:512]"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 4096
ATTENTION_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
ATTENTION_ONLY_PATTERNS = {
    "qwen3_30b_a3b": re.compile(
        r"^model\.layers\.\d+\.self_attn\.(?:q_proj|k_proj|v_proj|o_proj)$"
    ),
    "gemma4_26b_a4b_it": re.compile(
        r"^model\.language_model\.layers\.\d+\.self_attn\.(?:q_proj|k_proj|v_proj|o_proj)$"
    ),
    "nemotron35_lightning_30b_a3b": re.compile(
        r"^model\.layers\.\d+\.mixer\.(?:q_proj|k_proj|v_proj|o_proj)$"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_SPECS, required=True)
    parser.add_argument("--power", type=float, required=True)
    parser.add_argument(
        "--objective",
        choices=("basic", "groupmax", "groupmax_div", "groupmaxinv"),
        required=True,
    )
    parser.add_argument("--variant-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help=(
            "Optional checkpoint destination on the same filesystem as its "
            "temporary directory"
        ),
    )
    parser.add_argument(
        "--module-profile",
        choices=("all", "attention_only"),
        default="all",
        help="Restrict Qwen3-30B-A3B NVFP4 to self-attention Q/K/V/O projections",
    )
    return parser.parse_args()


def _load_model_and_processor(spec: dict):
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    model_cls = (
        AutoModelForCausalLM
        if spec["class_name"] == "AutoModelForCausalLM"
        else getattr(transformers, spec["class_name"])
    )
    load_kwargs = {
        "dtype": "auto",
        "device_map": "auto",
        "trust_remote_code": True,
    }
    with load_context(model_cls):
        model = model_cls.from_pretrained(spec["model_id"], **load_kwargs)

    if spec["processor"]:
        preprocessor = AutoProcessor.from_pretrained(
            spec["model_id"], trust_remote_code=True
        )
    else:
        preprocessor = AutoTokenizer.from_pretrained(
            spec["model_id"], trust_remote_code=True
        )
    return model, preprocessor


def _make_nvfp4_scheme(
    targets: str | list[str], power: float, objective: str, preset: str = "NVFP4"
):
    scheme = preset_name_to_scheme(
        preset, [targets] if isinstance(targets, str) else targets
    )
    scheme.weights.observer = "imatrix_mse"
    scheme.weights.observer_kwargs = {"norm": power, "objective": objective}
    return scheme


def _make_recipe(spec: dict, power: float, objective: str):
    scheme = _make_nvfp4_scheme(
        spec["targets"], power, objective, preset=spec.get("scheme", "NVFP4")
    )
    ignores = list(spec["ignore"])
    return [
        QuantizationModifier(
            config_groups={"group_0": scheme},
            ignore=ignores,
        )
    ]


def _validate_nemotron_attention_ignores(model: torch.nn.Module, spec: dict) -> None:
    """Fail before calibration if any attention Linear misses the BF16 ignore."""
    from compressed_tensors.utils.match import is_match

    attention = [
        (name, module)
        for name, module in model.named_modules()
        if re.fullmatch(
            r"model\.layers\.\d+\.mixer\.(?:q_proj|k_proj|v_proj|o_proj)",
            name,
        )
    ]
    if not attention:
        raise RuntimeError("Could not find Nemotron attention projections in model")

    unmatched = [
        name
        for name, module in attention
        if not any(is_match(name, module, pattern) for pattern in spec["ignore"])
    ]
    if unmatched:
        raise RuntimeError(
            "Nemotron attention projections must stay BF16, but these modules do "
            f"not match any ignore pattern: {unmatched}"
        )
    print(
        "Validated BF16 exclusions for "
        f"{len(attention)} Nemotron attention projections",
        flush=True,
    )


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def _quantized_nemotron_attention_modules(checkpoint_dir: Path) -> list[str]:
    """Return Nemotron attention projections packed as NVFP4, if any."""
    packed_pattern = re.compile(
        r"^backbone\.layers\.\d+\.mixer\.(?:q_proj|k_proj|v_proj|o_proj)\.weight_packed$"
    )
    packed_modules: set[str] = set()
    for tensor_file in checkpoint_dir.glob("*.safetensors"):
        with tensor_file.open("rb") as stream:
            header_length = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(header_length))
        for key in header:
            if packed_pattern.fullmatch(key):
                packed_modules.add(key.removesuffix(".weight_packed"))
    return sorted(packed_modules)


def _quantized_router_modules(checkpoint_dir: Path) -> list[str]:
    """Return router module weights packed as NVFP4, if any."""
    router_path = re.compile(
        r"(?:^|\.)router(?:\.|$)|(?:^|\.)mlp\.gate(?:\.|$)|"
        r"(?:^|\.)mixer\.gate(?:\.|$)"
    )
    packed_modules: set[str] = set()
    for tensor_file in checkpoint_dir.glob("*.safetensors"):
        with tensor_file.open("rb") as stream:
            header_length = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(header_length))
        for key in header:
            if not key.endswith(".weight_packed"):
                continue
            module = key.removesuffix(".weight_packed")
            if router_path.search(module):
                packed_modules.add(module)
    return sorted(packed_modules)


def _quantized_lm_head_modules(checkpoint_dir: Path) -> list[str]:
    """Return language-model head weights packed as NVFP4, if any."""
    packed_modules: set[str] = set()
    for tensor_file in checkpoint_dir.glob("*.safetensors"):
        with tensor_file.open("rb") as stream:
            header_length = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(header_length))
        for key in header:
            if key.endswith("lm_head.weight_packed"):
                packed_modules.add(key.removesuffix(".weight_packed"))
    return sorted(packed_modules)


def _packed_attention_modules(checkpoint_dir: Path) -> list[str]:
    pattern = re.compile(
        r"^model\.layers\.\d+\.self_attn\.(?:q_proj|k_proj|v_proj|o_proj)\.weight_packed$"
    )
    modules: set[str] = set()
    for tensor_file in checkpoint_dir.glob("*.safetensors"):
        with tensor_file.open("rb") as stream:
            header_length = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(header_length))
        modules.update(
            key.removesuffix(".weight_packed")
            for key in header
            if pattern.fullmatch(key)
        )
    return sorted(modules)


def _all_packed_weight_modules(checkpoint_dir: Path) -> list[str]:
    modules: set[str] = set()
    for tensor_file in checkpoint_dir.glob("*.safetensors"):
        with tensor_file.open("rb") as stream:
            header_length = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(header_length))
        modules.update(
            key.removesuffix(".weight_packed")
            for key in header
            if key.endswith(".weight_packed")
        )
    return sorted(modules)


def _expected_attention_only_targets(
    model: torch.nn.Module, model_slug: str
) -> list[str]:
    """Return and validate this model's text-attention projection allowlist."""
    pattern = ATTENTION_ONLY_PATTERNS.get(model_slug)
    if pattern is None:
        raise ValueError(f"attention_only is unsupported for {model_slug}")

    targets = {
        name
        for name, module in model.named_modules()
        if pattern.fullmatch(name) and isinstance(module, torch.nn.Linear)
    }
    if model_slug == "qwen3_30b_a3b":
        expected = {
            f"model.layers.{layer}.self_attn.{projection}"
            for layer in range(48)
            for projection in ATTENTION_PROJECTIONS
        }
    elif model_slug == "gemma4_26b_a4b_it":
        text_config = getattr(model.config, "text_config", None)
        layer_count = getattr(text_config, "num_hidden_layers", None)
        if not isinstance(layer_count, int) or layer_count <= 0:
            raise RuntimeError("Gemma4 text config has no valid decoder layer count")
        required_query_output = {
            f"model.language_model.layers.{layer}.self_attn.{projection}"
            for layer in range(layer_count)
            for projection in ("q_proj", "o_proj")
        }
        if not required_query_output.issubset(targets):
            raise RuntimeError(
                "Gemma4 text attention allowlist is incomplete; missing Q/O "
                f"projections: {sorted(required_query_output - targets)[:12]}"
            )
        expected = targets
    else:
        config = model.config
        layer_types = getattr(config, "layers_block_type", None)
        if layer_types is None:
            layer_types = getattr(config, "layer_types", None)
        if not isinstance(layer_types, (list, tuple)):
            raise RuntimeError("Nemotron config has no hybrid layer type list")
        attention_layers = {
            index
            for index, layer_type in enumerate(layer_types)
            if layer_type in ("attention", "full_attention")
        }
        expected = {
            f"model.layers.{layer}.mixer.{projection}"
            for layer in attention_layers
            for projection in ATTENTION_PROJECTIONS
        }
        if not expected:
            raise RuntimeError("Nemotron config contains no attention mixer layers")

    if targets != expected:
        raise RuntimeError(
            f"{model_slug} attention-only allowlist mismatch: found {len(targets)} "
            f"Linear projections, expected {len(expected)}; missing="
            f"{sorted(expected - targets)[:12]}, "
            f"extra={sorted(targets - expected)[:12]}"
        )
    print(
        f"Validated {model_slug} attention-only allowlist: "
        f"{len(targets)} Q/K/V/O projections",
        flush=True,
    )
    return sorted(targets)


def _checkpoint_violations(
    checkpoint_dir: Path,
    model: str,
    module_profile: str = "all",
    expected_attention_targets: list[str] | None = None,
) -> list[str]:
    violations = []
    routers = _quantized_router_modules(checkpoint_dir)
    lm_heads = _quantized_lm_head_modules(checkpoint_dir)
    if routers:
        violations.append(f"packed routers: {routers[:8]}")
    if lm_heads:
        violations.append(f"packed lm_head: {lm_heads[:8]}")
    if module_profile == "attention_only":
        packed = set(_all_packed_weight_modules(checkpoint_dir))
        if expected_attention_targets is None:
            try:
                metadata = json.loads(
                    (checkpoint_dir / "imatrix_ablation_config.json").read_text()
                )
                expected_attention_targets = metadata.get("target_module_names")
            except (OSError, json.JSONDecodeError):
                expected_attention_targets = None
        if expected_attention_targets is None and model == "qwen3_30b_a3b":
            expected_attention_targets = sorted(
                {
                    f"model.layers.{layer}.self_attn.{projection}"
                    for layer in range(48)
                    for projection in ATTENTION_PROJECTIONS
                }
            )
        if not expected_attention_targets:
            violations.append(
                "attention-only checkpoint lacks its target-module manifest"
            )
            expected = set()
        else:
            expected = set(expected_attention_targets)
            if model == "nemotron35_lightning_30b_a3b":
                expected = {
                    name.replace("model.layers.", "backbone.layers.")
                    for name in expected
                }
        unexpected = sorted(packed - expected)
        missing = sorted(expected - packed)
        if unexpected:
            violations.append(
                f"weights outside the attention-only allowlist packed: {unexpected[:8]}"
            )
        if missing:
            violations.append(
                f"attention-only checkpoint missing {len(missing)} "
                f"allowlisted projections: {missing[:8]}"
            )
    if model == "nemotron35_lightning_30b_a3b":
        attention = _quantized_nemotron_attention_modules(checkpoint_dir)
        if module_profile == "all" and attention:
            violations.append(f"packed attention projections: {attention[:8]}")
        try:
            config = json.loads((checkpoint_dir / "config.json").read_text())
            groups = config["quantization_config"]["config_groups"]
            activations = [group.get("input_activations") for group in groups.values()]
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
            violations.append(f"missing/unreadable quantization config: {error}")
        else:
            if any(activation is not None for activation in activations):
                violations.append("Nemotron requires NVFP4A16 with unquantized inputs")
    return violations


def _validate_router_linear_ignores(model: torch.nn.Module, spec: dict) -> None:
    """Ensure any router implemented as a Linear is covered by an ignore rule."""
    from compressed_tensors.utils.match import is_match

    candidates = [
        (name, module)
        for name, module in model.named_modules()
        if torch.nn.Linear in type(module).__mro__
        and re.search(
            r"(?:^|\.)router(?:\.|$)|(?:^|\.)mlp\.gate(?:\.|$)|"
            r"(?:^|\.)mixer\.gate(?:\.|$)",
            name,
        )
    ]
    unmatched = [
        name
        for name, module in candidates
        if not any(is_match(name, module, pattern) for pattern in spec["ignore"])
    ]
    if unmatched:
        raise RuntimeError(
            f"Router Linear modules must stay BF16 but miss ignore rules: {unmatched}"
        )


def _attention_only_spec(spec: dict) -> dict:
    from copy import deepcopy

    result = deepcopy(spec)
    slug = next(
        key
        for key, value in MODEL_SPECS.items()
        if value["model_id"] == spec["model_id"]
    )
    try:
        result["targets"] = "re:" + ATTENTION_ONLY_PATTERNS[slug].pattern
    except KeyError as error:
        raise ValueError(f"attention_only is unsupported for {slug}") from error
    if slug == "nemotron35_lightning_30b_a3b":
        result["ignore"] = [
            pattern
            for pattern in result["ignore"]
            if pattern
            != r"re:.*model\.layers\.\d+\.mixer\.(q_proj|k_proj|v_proj|o_proj)$"
        ]
    return result


def main() -> int:
    args = parse_args()
    if args.objective == "groupmaxinv":
        args.objective = "groupmax_div"
    if args.power not in (2.0, 2.2, 2.4, 2.6, 2.8, 3.0):
        raise ValueError(f"unsupported power: {args.power}")

    spec = MODEL_SPECS[args.model]
    if args.module_profile == "attention_only":
        spec = _attention_only_spec(spec)
    variant_dir = args.variant_dir.resolve()
    variant_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_link = variant_dir / "checkpoint"
    if args.checkpoint_dir is not None:
        checkpoint_dir = args.checkpoint_dir.resolve()
        checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
        if checkpoint_link.is_symlink():
            if checkpoint_link.resolve() != checkpoint_dir:
                raise ValueError(
                    f"checkpoint link {checkpoint_link} points to "
                    f"{checkpoint_link.resolve()}, expected {checkpoint_dir}"
                )
        elif checkpoint_link.exists():
            raise ValueError(
                f"checkpoint destination was provided but {checkpoint_link} "
                "already exists and is not a symlink"
            )
        else:
            checkpoint_link.symlink_to(checkpoint_dir, target_is_directory=True)
    else:
        checkpoint_dir = checkpoint_link
    temp_dir = checkpoint_dir.with_name(f".{checkpoint_dir.name}.incomplete")

    if (checkpoint_dir / "config.json").is_file():
        violations = _checkpoint_violations(
            checkpoint_dir, args.model, args.module_profile
        )
        try:
            existing_metadata = json.loads(
                (checkpoint_dir / "imatrix_ablation_config.json").read_text()
            )
        except (OSError, json.JSONDecodeError):
            existing_metadata = {}
        for key, expected_value in (
            ("model", args.model),
            ("objective", args.objective),
            ("power", args.power),
            ("module_profile", args.module_profile),
        ):
            if existing_metadata.get(key) != expected_value:
                violations.append(
                    f"existing checkpoint metadata "
                    f"{key}={existing_metadata.get(key)!r}, "
                    f"expected {expected_value!r}"
                )
        if violations:
            archive_dir = checkpoint_dir.with_name(
                f"{checkpoint_dir.name}_invalid_module_quantization_"
                + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            )
            print(
                "Existing checkpoint packed excluded modules; "
                f"archiving at {archive_dir}: {violations}",
                flush=True,
            )
            checkpoint_dir.rename(archive_dir)
            if checkpoint_link.is_symlink() and not checkpoint_link.exists():
                checkpoint_link.unlink()
                checkpoint_link.symlink_to(checkpoint_dir, target_is_directory=True)
        else:
            print(f"Checkpoint already exists: {checkpoint_dir}", flush=True)
            return 0
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    variant_start = time.time()
    print(
        f"Loading {spec['model_id']} power={args.power:g} objective={args.objective}",
        flush=True,
    )
    model, preprocessor = _load_model_and_processor(spec)
    _validate_router_linear_ignores(model, spec)
    expected_attention_targets = None
    if args.module_profile == "attention_only":
        expected_attention_targets = _expected_attention_only_targets(model, args.model)
        if args.model == "gemma4_26b_a4b_it":
            # vLLM rewrites Gemma4 module paths from
            # `model.language_model.*` to `language_model.model.*`, but leaves
            # regex targets unchanged. Store exact target paths so vLLM can
            # map them before matching the fused qkv projection.
            spec["targets"] = expected_attention_targets
    if args.model == "nemotron35_lightning_30b_a3b" and args.module_profile == "all":
        _validate_nemotron_attention_ignores(model, spec)
    recipe = _make_recipe(spec, args.power, args.objective)

    oneshot_kwargs = {
        "model": model,
        "recipe": recipe,
        "dataset": CALIBRATION_DATASET,
        "splits": CALIBRATION_SPLIT,
        "max_seq_length": MAX_SEQUENCE_LENGTH,
        "num_calibration_samples": NUM_CALIBRATION_SAMPLES,
        "moe_calibrate_all_experts": spec["moe_calibrate_all_experts"],
        "preprocessing_num_workers": 8,
        "dataloader_num_workers": 0,
    }
    oneshot_kwargs["processor" if spec["processor"] else "tokenizer"] = preprocessor
    oneshot(**oneshot_kwargs)

    temp_dir.mkdir(parents=True)
    model.save_pretrained(temp_dir, save_compressed=True)
    preprocessor.save_pretrained(temp_dir)
    torch.accelerator.synchronize()

    violations = _checkpoint_violations(
        temp_dir,
        args.model,
        args.module_profile,
        expected_attention_targets=expected_attention_targets,
    )
    if violations:
        raise RuntimeError(
            "Checkpoint violates model-specific NVFP4 exclusions: " f"{violations}"
        )

    recipe_metadata = {
        "model": args.model,
        "model_id": spec["model_id"],
        "scheme": spec.get("scheme", "NVFP4"),
        "weight_observer": "imatrix_mse",
        "objective": args.objective,
        "power": args.power,
        "calibration_dataset": CALIBRATION_DATASET,
        "calibration_split": CALIBRATION_SPLIT,
        "num_calibration_samples": NUM_CALIBRATION_SAMPLES,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "targets": spec["targets"],
        "module_profile": args.module_profile,
        "target_module_names": expected_attention_targets,
        "ignore": spec["ignore"],
        "moe_calibrate_all_experts": spec["moe_calibrate_all_experts"],
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "llmcompressor_version": __import__("llmcompressor").__version__,
        "elapsed_seconds": round(time.time() - variant_start, 2),
    }
    _write_json(temp_dir / "imatrix_ablation_config.json", recipe_metadata)
    temp_dir.rename(checkpoint_dir)
    print(f"Saved compressed checkpoint: {checkpoint_dir}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"QUANTIZATION FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        raise
