"""Default search parameters for MSE and iMatrix observers."""

from typing import Literal

from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from compressed_tensors.quantization.utils.mxfp_utils import (
    should_generate_mx_scales,
)

ObserverKind = Literal["imatrix", "mse"]
ObserverDefaults = dict[str, dict[str, dict[str, int | float | bool | str | None]]]


def get_defaults(
    args: QuantizationArgs,
    observer: ObserverKind,
    scheme: QuantizationScheme | None = None,
    *,
    observer_defaults: ObserverDefaults | None = None,
) -> dict[str, int | float | bool | str | None]:
    """Return search defaults with format and modifier-specific adjustments."""
    is_mx = (
        args.num_bits == 4 and args.type == "float" and should_generate_mx_scales(args)
    )

    is_nvfp4 = args.num_bits == 4 and args.type == "float" and args.group_size == 16
    is_mxfp4a8 = (
        is_mx
        and scheme is not None
        and scheme.input_activations is not None
        and scheme.input_activations.num_bits == 8
    )
    format_profile = (
        "nvfp4"
        if is_nvfp4
        else "mxfp4a8"
        if is_mxfp4a8
        else "mx"
        if is_mx
        else "default"
    )

    # Expanded observer ranges become the regular defaults for microscaled
    # formats. NVFP4 expands to 1.8x and searches down to 0.8x; MX formats
    # search the 1x to 1/64x range on an exponential grid.
    search_defaults: dict[str, int | float | bool | str | None] = {
        "maxshrink": 0.95 if observer == "imatrix" else 0.20,
        "patience": 5,
        "grid": 100.0 if observer == "mse" else 100,
        "expand": 1.0,
        "grid_type": "exponential" if is_mx else "linear",
    }
    if is_nvfp4:
        search_defaults.update(
            maxshrink=1 - 0.8 / 1.8,
            patience=1000,
            grid=200,
            expand=1.8,
            grid_type="linear",
        )
    elif is_mx:
        search_defaults.update(
            maxshrink=1 - 1.0 / 64.0,
            patience=1000,
            grid=200,
            expand=1.0,
            grid_type="exponential",
        )

    if observer == "mse":
        defaults = {
            **search_defaults,
            "norm": 2.4,
            "magnitude_normalization": True,
            "averaging_constant": 0.01,
        }
    elif observer == "imatrix":
        defaults = {
            **search_defaults,
            "norm": 3.0,
            "strict": False,
            "magnitude_normalization": True,
        }
    else:
        raise ValueError(f"Unknown observer kind: {observer}")

    profile = (observer_defaults or {}).get(observer, {})
    defaults.update(profile.get("default", {}))
    defaults.update(profile.get(format_profile, {}))

    return defaults
