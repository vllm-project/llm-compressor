"""Default search parameters for MSE and iMatrix observers."""

from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from compressed_tensors.quantization.utils.mxfp_utils import (
    should_generate_mx_scales,
)

ObserverKwargs = dict[str, int | float | bool | str | None]


def is_nvfp4(args: QuantizationArgs) -> bool:
    """Return whether quantization args match the packed NVFP4 weight format."""
    return args.num_bits == 4 and args.type == "float" and args.group_size == 16


def is_mxfp4(args: QuantizationArgs) -> bool:
    """Return whether quantization args use the MXFP4 weight format."""
    return (
        args.num_bits == 4 and args.type == "float" and should_generate_mx_scales(args)
    )


def is_mxfp4a8(scheme: QuantizationScheme) -> bool:
    """Return whether the scheme quantizes MXFP4 weights with 8-bit activations."""
    weights = scheme.weights
    activations = scheme.input_activations
    return (
        weights is not None
        and is_mxfp4(weights)
        and activations is not None
        and activations.num_bits == 8
    )


def get_defaults(
    args: QuantizationArgs,
) -> dict[str, int | float | bool | str | None]:
    """Return observer defaults for its quantization format."""
    if args.observer in ("mse", "memoryless_mse", "nvfp4_expanded_mse"):
        observer = "mse"
    elif args.observer == "imatrix_mse":
        observer = "imatrix"
    else:
        raise ValueError(f"Unknown observer: {args.observer}")

    is_mx = is_mxfp4(args)

    is_nvfp4_format = is_nvfp4(args)
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
    if is_nvfp4_format:
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

    return defaults
