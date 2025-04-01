from typing import TYPE_CHECKING

from compressed_tensors.quantization import (
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
    is_preset_scheme,
    preset_name_to_scheme,
)

if TYPE_CHECKING:
    from llmcompressor.modifiers import Modifier

__all__ = ["resolve_modifier_quantization_config"]


def resolve_modifier_quantization_config(modifier: "Modifier") -> QuantizationConfig:
    scheme = getattr(modifier, "scheme", None)
    targets = getattr(modifier, "targets", [])
    config_groups = getattr(modifier, "config_groups", None)
    kv_cache_scheme = getattr(modifier, "kv_cache_scheme", None)
    ignore = getattr(modifier, "ignore", None)

    if isinstance(targets, str):
        targets = [targets]

    if scheme is not None:
        # takes precedence over config_groups

        if isinstance(scheme, str) and is_preset_scheme(scheme):
            # attach targets to scheme
            scheme = {scheme: targets}

        config_groups = {}
        for idx, key in enumerate(scheme.keys()):
            if is_preset_scheme(key):
                scheme = preset_name_to_scheme(key, scheme[key])
            else:
                scheme = QuantizationScheme.model_validate(
                    {"targets": scheme[key], **scheme}
                )

            group_name = f"group_{idx}"
            config_groups[group_name] = scheme

    if config_groups is None or len(config_groups) == 0:
        default_quant_scheme = QuantizationScheme(targets=targets)
        config_groups = {"group_0": default_quant_scheme}

    return QuantizationConfig(
        config_groups=config_groups,
        kv_cache_scheme=kv_cache_scheme,
        quantization_status=QuantizationStatus.INITIALIZED,
        ignore=ignore,
    )
