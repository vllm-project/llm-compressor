import contextlib

import torch
from compressed_tensors.compressors import compress_module, decompress_module
from compressed_tensors.quantization import QuantizationStatus

# Compression qparams that decompression needs and that quant init overwrites.
_CT_COMPRESSED_QPARAMS = ("weight_scale", "weight_zero_point", "weight_shape")

__all__ = [
    "stash_input_formats",
    "decompressed_modules",
    "ensure_dense_for_nonsequential",
]


def stash_input_formats(model: torch.nn.Module, recompress: bool) -> None:
    """Record each compressed module's input format before any modifier runs.

    Sets ``module._ct_input_format`` on every ``COMPRESSED`` module so per-subgraph
    decompression can target the *input* format even after a modifier overwrites
    ``quantization_scheme``. Also records model-level flags read by the sequential
    pipeline and the non-sequential-pipeline guard.

    Args:
        model: model loaded with ``run_compressed=True``.
        recompress: whether processed subgraphs should be recompressed during
            calibration (derived from ``save_compressed``).
    """
    # Loading with run_compressed=True registers a forward pre-hook that decompresses
    # the whole model on the first forward. We manage decompression per subgraph, so
    # remove it to avoid a wholesale decompress (and hook errors once modules change).
    if hasattr(model, "ct_decompress_hook"):
        model.ct_decompress_hook.remove()
        delattr(model, "ct_decompress_hook")

    active = False
    for module in model.modules():
        status = getattr(module, "quantization_status", None)
        if status == QuantizationStatus.COMPRESSED:
            scheme = getattr(module, "quantization_scheme", None)
            module._ct_input_format = getattr(scheme, "format", None)
            # Snapshot the compression qparams: quant init (at session.initialize)
            # overwrites weight_scale/zero_point with new-scheme observers before the
            # pipeline decompresses, which would corrupt decompression.
            module._ct_compressed_qparams = {
                name: getattr(module, name).data.clone()
                for name in _CT_COMPRESSED_QPARAMS
                if hasattr(module, name)
            }
            active = True
    model._sequential_decompression_active = active
    model._recompress_on_calibration = bool(recompress) and active


def ensure_dense_for_nonsequential(model: torch.nn.Module) -> None:
    """Fully decompress a compressed-loaded model for non-sequential pipelines.

    ``data_free``/``basic`` pipelines do not decompress per subgraph, so a model
    loaded with ``run_compressed=True`` must be decompressed wholesale before they run.
    No-op unless the model was loaded compressed for sequential decompression.
    """
    if not getattr(model, "_sequential_decompression_active", False):
        return

    from compressed_tensors.utils.offload import align_modules

    targets = [
        m
        for m in model.modules()
        if getattr(m, "quantization_status", None) == QuantizationStatus.COMPRESSED
    ]
    with align_modules(targets):
        for module in targets:
            decompress_module(module, format=getattr(module, "_ct_input_format", None))
    model._sequential_decompression_active = False


def _is_fp8_linear(module: torch.nn.Module) -> bool:
    return type(module).__name__ == "FP8Linear"


def _is_compressed(module: torch.nn.Module) -> bool:
    """A module is physically compressed if it has no dense ``weight`` but carries
    packed weights. Status alone is unreliable: a modifier may flip
    ``quantization_status`` away from COMPRESSED at init while the tensors stay packed.
    """
    if hasattr(module, "weight"):
        return False
    return hasattr(module, "weight_packed") or (
        getattr(module, "quantization_status", None) == QuantizationStatus.COMPRESSED
    )


def _resolve_handlers(module: torch.nn.Module):
    """Return ``(decompress_fn, recompress_fn)`` for a module, or None if unhandled."""
    if _is_fp8_linear(module):
        raise NotImplementedError(
            "native-FP8 (FP8Linear) JIT decompression is Path A; not yet implemented"
        )
    if not _is_compressed(module):
        return None
    fmt = getattr(module, "_ct_input_format", None)
    return (_decompress_ct, lambda m: compress_module(m, format=fmt))


def _set_weight_param(module: torch.nn.Module, name: str, value) -> None:
    existing = getattr(module, name, None)
    if isinstance(value, torch.nn.Parameter):
        module.register_parameter(name, value)
    elif isinstance(existing, torch.nn.Parameter) or existing is None:
        module.register_parameter(name, torch.nn.Parameter(value, requires_grad=False))
    else:
        setattr(module, name, value)


def _decompress_ct(module: torch.nn.Module) -> None:
    """Decompress a CT module while preserving the calibration scheme's qparams.

    Quant init (at session.initialize) overwrote the input compression scale with the
    new scheme's observer params. Temporarily restore the input compression qparams so
    decompression uses the correct scale, then hand the scheme qparams back to the
    (now dense) module so the calibrating modifier keeps its observer state.
    """
    scheme_qparams = {
        name: getattr(module, name)
        for name in ("weight_scale", "weight_zero_point")
        if hasattr(module, name)
    }
    saved = getattr(module, "_ct_compressed_qparams", None)
    if saved:
        for name, tensor in saved.items():
            _set_weight_param(module, name, tensor.clone())

    decompress_module(module, format=getattr(module, "_ct_input_format", None))

    # decompression drops the compression qparams; restore the scheme's qparams so the
    # modifier's observers still have their targets on the dense weight.
    for name, param in scheme_qparams.items():
        if not hasattr(module, name):
            _set_weight_param(module, name, param)


@contextlib.contextmanager
def decompressed_modules(modules, recompress: bool):
    """Decompress supported modules for the block, then optionally recompress them.

    The single place that mutates compressed<->dense module state.

    Args:
        modules: iterable of modules to (potentially) decompress.
        recompress: whether to recompress each touched module on exit.
    """
    touched = []
    for module in modules:
        handlers = _resolve_handlers(module)
        if handlers is None:
            continue
        decompress_fn, recompress_fn = handlers
        decompress_fn(module)
        touched.append((module, recompress_fn))
    try:
        yield
    finally:
        if recompress:
            for module, recompress_fn in touched:
                recompress_fn(module)
