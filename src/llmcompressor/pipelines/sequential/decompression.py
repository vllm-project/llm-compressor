import contextlib

import torch
from compressed_tensors.compressors import compress_module, decompress_module
from compressed_tensors.quantization import QuantizationStatus

__all__ = ["stash_input_formats", "decompressed_modules"]


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
    active = False
    for module in model.modules():
        if getattr(module, "quantization_status", None) == QuantizationStatus.COMPRESSED:
            scheme = getattr(module, "quantization_scheme", None)
            module._ct_input_format = getattr(scheme, "format", None)
            active = True
    model._sequential_decompression_active = active
    model._recompress_on_calibration = bool(recompress) and active


def _is_fp8_linear(module: torch.nn.Module) -> bool:
    return type(module).__name__ == "FP8Linear"


def _resolve_handlers(module: torch.nn.Module):
    """Return ``(decompress_fn, recompress_fn)`` for a module, or None if unhandled."""
    if getattr(module, "quantization_status", None) == QuantizationStatus.COMPRESSED:
        if _is_fp8_linear(module):
            raise NotImplementedError(
                "native-FP8 (FP8Linear) JIT decompression is Path A; not yet implemented"
            )
        fmt = getattr(module, "_ct_input_format", None)
        return (
            lambda m: decompress_module(m, format=fmt),
            lambda m: compress_module(m),
        )
    return None


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
