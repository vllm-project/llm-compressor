"""
Support for loading pre-quantized compressed-tensors checkpoints which also
require checkpoint conversion mappings (``WeightConverter``s).

When a checkpoint carries a ``quantization_config``, Transformers asks the
quantizer to rewrite the model's conversion mapping via
``HfQuantizer.update_weight_conversions``: every ``WeightConverter`` whose
source patterns end in a serialized parameter name (e.g. ``.weight``) is
rewritten to instead collect the serialized companion tensors (e.g.
``weight_packed`` + ``weight_scale`` + ``weight_shape`` for NVFP4) and to
dequantize them before applying the converter's own operations.

That rewrite is only correct when the checkpoint actually stores those
companions for the converter's sources, i.e. when the modules backing the
sources are quantized. For converters which target *unquantized* modules, the
rewrite silently detaches the converter from the checkpoint: source patterns
are name-boundary anchored, so ``self_attn.q_conv1d.weight`` can no longer
match once the pattern becomes ``self_attn.q_conv1d.weight_packed``. The
checkpoint's plain ``.weight`` keys are then reported as UNEXPECTED and the
target parameter is randomly initialized.

This is visible when loading ``RedHatAI/GLM-5.3-Flash-NVFP4`` (an NVFP4
expert-only checkpoint) under :func:`llmcompressor.utils.load_context`: the
``{q,k,v}_conv1d.weight -> conv1d.weight`` fusion registered for ``glm5_next``
targets the dense causal-conv modules, which the checkpoint does not quantize.
The rewrite leaves every ``self_attn.conv1d.weight`` randomly initialized while
reporting the checkpoint's ``q/k/v_conv1d.weight`` keys as UNEXPECTED.

``checkpoint_aware_weight_conversions`` wraps
``HfQuantizer.update_weight_conversions`` so that the dequantize rewrite is
only applied to converters whose serialized companions are actually present in
the checkpoint being loaded. Converters for unquantized modules are left as
registered, allowing them to consume the plain checkpoint keys.
"""

import contextlib
import re
from typing import Iterable

from compressed_tensors.utils import patch_attr
from loguru import logger

__all__ = ["checkpoint_aware_weight_conversions"]


def _companions_in_checkpoint(
    sources: Iterable[str],
    param: str,
    serialized: list[str],
    checkpoint_keys: Iterable[str],
) -> bool:
    """
    Whether the checkpoint stores serialized companion tensors (e.g.
    ``weight_packed``/``weight_scale``) for any of a converter's source
    patterns, i.e. whether the modules backing the sources are quantized.

    Companion source patterns are built exactly as
    ``HfQuantizer.update_weight_conversions`` builds them, so that presence
    checks agree with the patterns the rewrite would produce.
    """
    companions = [name for name in serialized if name != param]
    if not companions:
        return False

    checkpoint_keys = list(checkpoint_keys)
    for source in sources:
        for companion in companions:
            pattern = re.sub(rf"{param}$", companion, source)
            # mirror the wildcard handling of WeightTransform.__init__
            pattern = pattern.replace("*.", r".*\.")
            regex = re.compile(pattern)
            if any(regex.search(key) for key in checkpoint_keys):
                return True
    return False


def _guarded_update_weight_conversions(
    quantizer,
    weight_conversions,
    model,
    checkpoint_keys: set[str],
):
    """
    Reproduction of ``HfQuantizer.update_weight_conversions`` (as of
    transformers v5.22.1) with one guard: the dequantize rewrite is skipped for
    converters whose serialized companions are absent from the checkpoint,
    since those converters target modules which the checkpoint does not
    quantize. Revisit this once the behavior is fixed upstream.
    """
    from transformers.core_model_loading import WeightConverter

    param_names = "\n".join(name for name, _ in model.named_parameters())
    dequantize = (
        quantizer.get_load_operations().get("dequantize")
        if quantizer.pre_quantized
        else None
    )
    updated = []
    for converter in weight_conversions:
        updated.append(converter)
        if not isinstance(converter, WeightConverter):
            continue
        sources, targets = (
            converter._original_source_patterns,
            converter._original_target_patterns,
        )
        param = sources[0].rpartition(".")[2]
        serialized = quantizer.serialized_names.get(param)
        if not serialized or not all(
            source.endswith(f".{param}") for source in sources
        ):
            continue
        # `weight_scale_inv` -> `_scale_inv`
        suffixes = [
            name.removeprefix(param)
            for name in serialized
            if name.startswith(param) and name != param
        ]
        if suffixes and re.search(
            f"{targets[0]}{suffixes[0]}$", param_names, re.MULTILINE
        ):
            # The module keeps the companions as params: each one gets its own
            # copy of the converter
            for suffix in suffixes:
                updated.append(
                    converter.copy(
                        [source + suffix for source in sources],
                        [target + suffix for target in targets],
                        operations=list(converter.operations),
                    )
                )
        elif dequantize is not None and _companions_in_checkpoint(
            sources, param, serialized, checkpoint_keys
        ):
            # The module only has `param` and the checkpoint serializes its
            # companions: the converter also collects the companions and
            # dequantizes first
            all_sources = [
                re.sub(rf"{param}$", name, source)
                for source in sources
                for name in serialized
            ]
            updated[-1] = converter.copy(
                all_sources,
                targets,
                operations=[dequantize(quantizer)] + converter.operations,
            )
        # else: the checkpoint does not serialize companions for this
        # converter's sources, meaning the modules it targets are not
        # quantized. Keep the converter as registered so that it can consume
        # the plain checkpoint keys
    return updated + quantizer.get_weight_conversions()


@contextlib.contextmanager
def checkpoint_aware_weight_conversions(
    model_stub: str | None = None,
    revision: str | None = None,
    checkpoint_keys: set[str] | None = None,
):
    """
    Context manager under which ``HfQuantizer.update_weight_conversions`` does
    not rewrite ``WeightConverter``s whose serialized companions are missing
    from the checkpoint being loaded (see module docstring).

    Compressed-tensors checkpoints only: other quantizers are left untouched.
    The checkpoint keys are resolved lazily from ``model_stub`` the first time
    a quantizer rewrites a conversion mapping, and loading falls back to
    upstream behavior if they cannot be resolved. Pass ``checkpoint_keys``
    directly to override the resolution (useful for testing).

    :param model_stub: model id or local path of the checkpoint being loaded
    :param revision: hub revision of the checkpoint being loaded
    :param checkpoint_keys: checkpoint tensor names to use instead of resolving
        them from ``model_stub``
    """
    from transformers.quantizers.base import HfQuantizer
    from transformers.quantizers.quantizer_compressed_tensors import (
        CompressedTensorsHfQuantizer,
    )

    original = HfQuantizer.update_weight_conversions
    resolved_keys: set[str] | None = checkpoint_keys

    def update_weight_conversions(self, weight_conversions, model):
        nonlocal resolved_keys
        if not isinstance(self, CompressedTensorsHfQuantizer):
            return original(self, weight_conversions, model)

        if resolved_keys is None:
            resolved_keys = _resolve_checkpoint_keys(model_stub, revision)
        if resolved_keys is None:
            # could not inspect the checkpoint: keep upstream behavior
            return original(self, weight_conversions, model)

        return _guarded_update_weight_conversions(
            self, weight_conversions, model, resolved_keys
        )

    with patch_attr(
        HfQuantizer, "update_weight_conversions", update_weight_conversions
    ):
        yield


def _resolve_checkpoint_keys(
    model_stub: str | None, revision: str | None
) -> set[str] | None:
    """Tensor names of the checkpoint at ``model_stub``, or None on failure."""
    if model_stub is None:
        return None

    try:
        from llmcompressor.transformers.compression.mtp import _checkpoint_weights

        return set(_checkpoint_weights(model_stub, revision=revision))
    except Exception as error:
        logger.warning(
            f"Could not resolve checkpoint keys of {model_stub} at revision "
            f"{revision}: {error}. Falling back to upstream quantizer weight "
            "conversion behavior."
        )
        return None
