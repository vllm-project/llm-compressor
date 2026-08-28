"""
Checkpoint-referenced weight materialization for the streaming pipeline.

Models loaded with ``device_map="meta"`` carry no weight data. This module
builds a mapping from model parameter names to keys in the original
safetensors checkpoint (inverting the load-time weight conversions), and
materializes parameters on demand, per subgraph, directly from the original
shards. Unquantized weights are never copied to a secondary disk location.
"""

import json
import os
from dataclasses import dataclass

import torch
from loguru import logger
from safetensors import safe_open
from transformers import PreTrainedModel
from transformers.conversion_mapping import extract_weight_conversions_for_model
from transformers.core_model_loading import (
    WeightConverter,
    WeightRenaming,
    rename_source_key,
)

__all__ = [
    "CheckpointReferenceError",
    "CheckpointEntry",
    "CheckpointMap",
    "stage_modules",
    "commit_staged",
    "materialize_modules",
    "materialize_buffers",
    "release_modules",
]


class CheckpointReferenceError(RuntimeError):
    """
    Raised when a model's parameters cannot be referenced directly from the
    original checkpoint (e.g. the load-time conversions include tensor
    transforms, or the checkpoint is not safetensors). The caller should
    fall back to a pipeline which performs regular weight loading.
    """


@dataclass
class CheckpointEntry:
    """Location of one tensor in the original checkpoint."""

    file: str  # absolute path to the safetensors shard
    key: str  # tensor key within the shard
    dtype: torch.dtype
    shape: torch.Size
    # for parameters linearized out of a fused expert tensor (e.g.
    # `experts.gate_up_proj[i, :I]` -> `experts.i.gate_proj.weight`):
    # slices into the fused checkpoint tensor, and whether to transpose after
    slices: tuple | None = None
    transpose: bool = False


class CheckpointMap:
    """
    Maps model parameter names to their tensors in the original checkpoint.

    Built by reversing the load-time weight conversions (the same reversion
    transformers applies when saving), so each parameter name resolves to the
    checkpoint key its bytes were (or would have been) loaded from.
    """

    def __init__(
        self,
        entries: dict[str, CheckpointEntry],
        unresolved: list[str] | None = None,
        unresolved_buffers: list[str] | None = None,
    ):
        self.entries = entries
        # params with no checkpoint source, e.g. quantization scales attached
        # by modifiers; they are given zeroed storage at materialize time and
        # are expected to be written during calibration
        self.unresolved = unresolved or []
        # buffers with no checkpoint source, e.g. non-persistent rotary
        # `inv_freq`; recomputed by `materialize_buffers` instead of staging
        self.unresolved_buffers = unresolved_buffers or []

    @classmethod
    def from_model(cls, model: PreTrainedModel) -> "CheckpointMap":
        model_dir = model.name_or_path
        if not os.path.isdir(model_dir):
            raise CheckpointReferenceError(
                f"Model path '{model_dir}' is not a local directory; the "
                "streaming pipeline requires a local safetensors checkpoint"
            )

        ckpt_index = _read_checkpoint_index(model_dir)
        conversions = _get_load_conversions(model)
        renamings, converters = _split_reverse_conversions(conversions)

        entries: dict[str, CheckpointEntry] = {}
        tied_sources = _tied_parameter_sources(model)
        unresolved: list[str] = []
        unresolved_buffers: list[str] = []
        converter_matched: set[str] = set()
        named_tensors = list(
            model.named_parameters(remove_duplicate=False)
        ) + list(model.named_buffers(remove_duplicate=False))
        buffer_names = {name for name, _ in model.named_buffers(remove_duplicate=False)}
        for name, tensor in named_tensors:
            ckpt_key, converter_pattern = rename_source_key(
                name, renamings, converters, reverse=True
            )
            if converter_pattern is not None:
                # a converter matched; a linearized-expert param may still be
                # resolvable as a slice of the fused checkpoint tensor (second
                # pass below). Anything else cannot be referenced directly.
                converter_matched.add(name)
                unresolved.append(name)
                continue
            file_entry = ckpt_index.get(ckpt_key)
            if file_entry is None and name in tied_sources:
                # tied parameters (e.g. lm_head) are not stored separately;
                # reference the source tensor they are tied to
                source_entry = entries.get(tied_sources[name])
                if source_entry is not None:
                    entries[name] = source_entry
                    continue
            if file_entry is None:
                # not a checkpoint tensor: either a quantization scale attached
                # by a modifier (given zeroed storage at materialize time) or a
                # non-persistent buffer (recomputed by `materialize_buffers`)
                if name in buffer_names:
                    unresolved_buffers.append(name)
                else:
                    unresolved.append(name)
                continue
            file, dtype, shape = file_entry
            if tuple(shape) != tuple(tensor.shape):
                raise CheckpointReferenceError(
                    f"Shape mismatch for '{name}': checkpoint {tuple(shape)} vs "
                    f"model {tuple(tensor.shape)}. Use the 'sequential' pipeline "
                    "instead."
                )
            entries[name] = CheckpointEntry(file=file, key=ckpt_key, dtype=dtype,
                                            shape=shape)

        if unresolved:
            logger.debug(
                f"{len(unresolved)} parameters have no checkpoint source and "
                f"will be calibrated from scratch (first: {unresolved[0]!r})"
            )
        if unresolved_buffers:
            logger.debug(
                f"{len(unresolved_buffers)} buffers have no checkpoint source "
                f"and will be recomputed (first: {unresolved_buffers[0]!r})"
            )

        # second pass: params of linearized experts (LinearExperts2D) which
        # are slices of fused checkpoint tensors, e.g. `experts.gate_up_proj[i]`
        unresolved = _resolve_expert_slices(
            model, ckpt_index, renamings, entries, unresolved
        )
        still_converted = converter_matched & set(unresolved)
        if still_converted:
            name = sorted(still_converted)[0]
            raise CheckpointReferenceError(
                f"Parameter '{name}' passes through a weight converter during "
                "loading, so its bytes cannot be referenced directly from the "
                "checkpoint. Use the 'sequential' pipeline instead."
            )
        return cls(entries, unresolved, unresolved_buffers)

    def entry(self, param_name: str) -> CheckpointEntry | None:
        return self.entries.get(param_name)


def _read_checkpoint_index(
    model_dir: str,
) -> dict[str, tuple[str, torch.dtype, torch.Size]]:
    """key -> (absolute file path, dtype, shape), read from headers only."""
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        files = sorted(set(weight_map.values()))
        key_to_file = weight_map
    else:
        single = os.path.join(model_dir, "model.safetensors")
        if not os.path.exists(single):
            raise CheckpointReferenceError(
                f"No safetensors checkpoint found in {model_dir}; the streaming "
                "pipeline requires a safetensors checkpoint"
            )
        files = ["model.safetensors"]
        key_to_file = None

    index: dict[str, tuple[str, torch.dtype, torch.Size]] = {}
    for file_name in files:
        file_path = os.path.join(model_dir, file_name)
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key_to_file is not None and key_to_file.get(key) != file_name:
                    continue
                slice_ = f.get_slice(key)
                index[key] = (file_path, slice_.get_dtype(), slice_.get_shape())
    return index


def _get_load_conversions(model: PreTrainedModel):
    """
    Load-time weight conversions for the model. Populated on
    ``model._load_weight_conversions`` by `load_quantizable_moe` when
    linearized load mappings were registered; otherwise derived from the
    model's registered conversion mapping.
    """
    conversions = getattr(model, "_load_weight_conversions", None)
    if conversions is None:
        conversions = extract_weight_conversions_for_model(model) or []
    return conversions


def _split_reverse_conversions(
    conversions: list,
) -> tuple[list[WeightRenaming], list[WeightConverter]]:
    """
    Reverse load-time conversions so they map model names back to checkpoint
    keys (the same reversion applied at save time), split by kind.
    """
    reversed_conversions = []
    for conversion in conversions:
        try:
            reversed_conversions.append(conversion.reverse_transform())
        except (NotImplementedError, AttributeError) as e:
            raise CheckpointReferenceError(
                f"Weight conversion {conversion} is not reversible: {e}. "
                "Use the 'sequential' pipeline instead."
            )
    renamings = [c for c in reversed_conversions if isinstance(c, WeightRenaming)]
    converters = [c for c in reversed_conversions if isinstance(c, WeightConverter)]
    return renamings, converters


def _tied_parameter_sources(model: PreTrainedModel) -> dict[str, str]:
    """tied param name -> param name it shares an object with."""
    sources: dict[str, str] = {}
    seen: dict[int, str] = {}
    for name, param in model.named_parameters(remove_duplicate=False):
        key = id(param)
        if key in seen:
            sources[name] = seen[key]
        else:
            seen[key] = name
    return sources


def _expert_slice_spec(
    experts_module, index: int, proj: str, kind: str
) -> tuple[str, tuple, bool] | None:
    """
    Map a linearized expert param (`{i}.{proj}.{kind}`) to its location in the
    fused experts tensor, mirroring `ExpertMLP.copy_from_experts_module`.
    Returns (fused param name suffix, slices, transpose).
    """
    intermediate = experts_module.intermediate_size
    transposed = experts_module.is_transposed
    has_gate = experts_module.has_gate
    bias = kind == "bias"
    if bias and not experts_module.has_bias:
        return None
    suffix = "_bias" if bias else ""

    if has_gate and proj in ("gate_proj", "up_proj"):
        fused = f"gate_up_proj{suffix}"
        lo = 0 if proj == "gate_proj" else intermediate
        hi = intermediate if proj == "gate_proj" else 2 * intermediate
        if transposed:
            return fused, (index, slice(None), slice(lo, hi)), True
        return fused, (index, slice(lo, hi)), False
    if proj == "down_proj" or (not has_gate and proj == "up_proj"):
        fused = f"{proj}{suffix}"
        return fused, (index,), transposed
    return None


def _resolve_expert_slices(
    model: PreTrainedModel,
    ckpt_index: dict,
    renamings: list,
    entries: dict[str, CheckpointEntry],
    unresolved: list[str],
) -> list[str]:
    """
    Resolve params of LinearExperts2D modules which have no exact checkpoint
    key to slices of the fused expert tensors they were linearized from.
    Raises `CheckpointReferenceError` for unresolvable expert weights (rather
    than letting them be materialized as zeros).
    """
    from llmcompressor.modeling.moe.linear_experts import LinearExperts2D

    unresolved_set = set(unresolved)
    resolved: set[str] = set()
    for module_name, module in model.named_modules():
        if not isinstance(module, LinearExperts2D):
            continue
        for index, expert in enumerate(module):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                linear = getattr(expert, proj, None)
                if linear is None:
                    continue
                for kind, param in (("weight", linear.weight), ("bias", linear.bias)):
                    if param is None:
                        continue
                    fqn = f"{module_name}.{index}.{proj}.{kind}"
                    if fqn not in unresolved_set:
                        continue
                    spec = _expert_slice_spec(module, index, proj, kind)
                    if spec is None:
                        raise CheckpointReferenceError(
                            f"Cannot resolve '{fqn}' to a checkpoint slice "
                            f"(layout of {type(module).__name__} unsupported). Use "
                            "the 'sequential' pipeline instead."
                        )
                    fused_suffix, slices, transpose = spec
                    fused_name = f"{module_name}.{fused_suffix}"
                    # reverse renamings only: a converter would mean the fused
                    # tensor itself is transformed, which we cannot reference
                    ckpt_key, conv = rename_source_key(
                        fused_name, renamings, [], reverse=True
                    )
                    file_entry = ckpt_index.get(ckpt_key) or ckpt_index.get(fused_name)
                    if file_entry is None:
                        raise CheckpointReferenceError(
                            f"Fused checkpoint tensor for '{fqn}' (expected key "
                            f"'{ckpt_key}') not found in checkpoint. Use the "
                            "'sequential' pipeline instead."
                        )
                    file, dtype, shape = file_entry
                    _validate_slice(fqn, shape, slices, transpose, param.shape)
                    entries[fqn] = CheckpointEntry(
                        file=file,
                        key=ckpt_key if ckpt_key in ckpt_index else fused_name,
                        dtype=dtype,
                        shape=torch.Size(param.shape),
                        slices=slices,
                        transpose=transpose,
                    )
                    resolved.add(fqn)
    return [name for name in unresolved if name not in resolved]


def _validate_slice(fqn, fused_shape, slices, transpose, param_shape) -> None:
    sliced = []
    for dim, sel in enumerate(slices):
        if isinstance(sel, slice):
            sliced.append(len(range(*sel.indices(fused_shape[dim]))))
    # dims beyond the slice spec are kept whole
    for dim in range(len(slices), len(fused_shape)):
        sliced.append(fused_shape[dim])
    if transpose:
        sliced = sliced[::-1]
    if tuple(sliced) != tuple(param_shape):
        raise CheckpointReferenceError(
            f"Slice of fused tensor for '{fqn}' has shape {tuple(sliced)}, "
            f"expected {tuple(param_shape)}. Use the 'sequential' pipeline instead."
        )


@torch.no_grad()
def stage_modules(
    model: torch.nn.Module,
    modules: list[torch.nn.Module],
    ckpt_map: CheckpointMap,
) -> dict[tuple[torch.nn.Module, str, str], torch.Tensor]:
    """
    Read all meta parameters and buffers directly owned by `modules` from the
    original checkpoint shards into CPU tensors, grouping reads by shard file.

    This performs no model mutation and releases the GIL during reads, so it
    can run in a background thread to prefetch the next subgraph while the
    current one is being calibrated.

    :return: staged tensors keyed by (owning module, "param"|"buffer", name)
    """
    module_names = {module: name for name, module in model.named_modules()}
    grouped: dict[str, list[tuple[tuple, CheckpointEntry, torch.dtype]]] = {}
    staged: dict[tuple[torch.nn.Module, str, str], torch.Tensor] = {}
    for module in modules:
        for kind, mapping in (("param", module._parameters), ("buffer", module._buffers)):
            for local_name, tensor in mapping.items():
                if tensor is None or tensor.device.type != "meta":
                    continue
                module_name = module_names[module]
                fqn = f"{module_name}.{local_name}" if module_name else local_name
                entry = ckpt_map.entry(fqn)
                if entry is None:
                    if fqn in ckpt_map.unresolved_buffers:
                        # recomputed by `materialize_buffers`, not staged
                        continue
                    if fqn not in ckpt_map.unresolved:
                        raise CheckpointReferenceError(
                            f"No checkpoint entry for '{fqn}'"
                        )
                    # not a checkpoint tensor (e.g. a quantization scale
                    # attached by a modifier); give it real zeroed storage so
                    # it can be written during calibration (zeros match
                    # observer defaults for never-calibrated params)
                    staged[(module, kind, local_name)] = torch.zeros_like(
                        tensor, device="cpu"
                    )
                    continue
                key = (module, kind, local_name)
                grouped.setdefault(entry.file, []).append((key, entry, tensor.dtype))

    # norm calibration replacements (see offset_norm.py) hold
    # standard-convention weights (1 + checkpoint value)
    from llmcompressor.modeling.offset_norm import NormCalibrationModule

    for file_path, items in grouped.items():
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for key, entry, dtype in items:
                module, kind, local_name = key
                if entry.slices is not None:
                    tensor = f.get_slice(entry.key)[entry.slices]
                    if entry.transpose:
                        tensor = tensor.T.contiguous()
                else:
                    tensor = f.get_tensor(entry.key)
                if (
                    kind == "param"
                    and local_name == "weight"
                    and isinstance(module, NormCalibrationModule)
                ):
                    tensor = 1.0 + tensor.float()
                if tensor.dtype != dtype:
                    tensor = tensor.to(dtype)
                staged[key] = tensor
    logger.debug(
        f"staged {len(staged)} tensors from {len(grouped)} checkpoint shard files"
    )
    return staged


@torch.no_grad()
def commit_staged(
    staged: dict[tuple[torch.nn.Module, str, str], torch.Tensor],
    device: torch.device,
) -> int:
    """
    Install staged tensors as parameters/buffers on their owning modules,
    moving them to `device`. Objects are replaced (meta tensors cannot be
    assigned real data); hooks hold module references and are unaffected.

    :return: number of tensors materialized
    """
    for (module, kind, local_name), tensor in staged.items():
        tensor = tensor.to(device)
        if kind == "param":
            param = module._parameters[local_name]
            module._parameters[local_name] = torch.nn.Parameter(
                tensor, requires_grad=param.requires_grad
            )
        else:
            old = module._buffers[local_name]
            if isinstance(old, torch.nn.Buffer):
                tensor = torch.nn.Buffer(
                    tensor, persistent=getattr(old, "persistent", True)
                )
            module._buffers[local_name] = tensor
    return len(staged)


@torch.no_grad()
def materialize_modules(
    model: torch.nn.Module,
    modules: list[torch.nn.Module],
    ckpt_map: CheckpointMap,
    device: torch.device,
) -> int:
    """Synchronous convenience wrapper: `stage_modules` + `commit_staged`."""
    return commit_staged(stage_modules(model, modules, ckpt_map), device)


@torch.no_grad()
def materialize_buffers(
    model: torch.nn.Module, device: torch.device, ckpt_map: "CheckpointMap"
) -> int:
    """
    Recompute buffers which remain on the meta device after a meta-device load
    and have no checkpoint source (e.g. non-persistent rotary `inv_freq`), by
    re-instantiating their owning module and copying its buffers. Buffers
    which do exist in the checkpoint are staged from it by `stage_modules`
    instead. Raises `CheckpointReferenceError` for buffers whose owning module
    cannot be re-instantiated from its config.
    """
    module_names = {module: name for name, module in model.named_modules()}
    count = 0
    for module in model.modules():
        module_name = module_names[module]
        meta_buffers = []
        for name, buf in module._buffers.items():
            if buf is None or getattr(buf, "device", None) is None:
                continue
            if buf.device.type != "meta":
                continue
            fqn = f"{module_name}.{name}" if module_name else name
            if ckpt_map.entry(fqn) is not None:
                continue  # checkpoint-backed, staged by `stage_modules`
            meta_buffers.append(name)
        if not meta_buffers:
            continue
        try:
            with torch.device("cpu"):
                fresh = type(module)(module.config)
        except Exception as e:
            raise CheckpointReferenceError(
                f"Cannot recompute meta buffers {meta_buffers} of "
                f"{type(module).__name__} by re-instantiation: {e}"
            )
        for name in meta_buffers:
            module._buffers[name] = fresh._buffers[name].to(device)
        for attr in ("attention_scaling",):
            if hasattr(fresh, attr):
                setattr(module, attr, getattr(fresh, attr))
        count += len(meta_buffers)
    return count


@torch.no_grad()
def release_modules(
    modules: list[torch.nn.Module],
    device: torch.device = torch.device("cpu"),
    ckpt_map: "CheckpointMap | None" = None,
    model: torch.nn.Module | None = None,
) -> None:
    """
    Move all parameters and buffers of the given modules to `device`.

    With ``device="meta"``, checkpoint-backed parameters are dropped back to
    the meta device instead (they can be re-materialized from the original
    checkpoint at any time), keeping host memory flat across layers. Only
    valid when no modifier mutates weights (e.g. RTN-style
    `QuantizationModifier`); weight-mutating modifiers (GPTQ, AWQ, ...) must
    use the default cpu offload. Norm-calibration replacement weights and
    quantization params (no checkpoint source) are always kept.
    """
    from llmcompressor.modeling.offset_norm import NormCalibrationModule

    to_meta = device.type == "meta"
    move_to = torch.device("cpu") if to_meta else device
    module_names = (
        {module: name for name, module in model.named_modules()}
        if to_meta and model is not None
        else {}
    )
    for module in modules:
        for name, param in module._parameters.items():
            if param is None:
                continue
            if (
                to_meta
                and ckpt_map is not None
                and not isinstance(module, NormCalibrationModule)
            ):
                # drop checkpoint-backed weights; they are re-readable from
                # the original shards. Quantization params (not in the
                # checkpoint) are kept on cpu
                module_name = module_names.get(module, "")
                fqn = f"{module_name}.{name}" if module_name else name
                if ckpt_map.entry(fqn) is not None:
                    module._parameters[name] = torch.nn.Parameter(
                        torch.empty_like(param, device="meta"),
                        requires_grad=param.requires_grad,
                    )
                    continue
            if param.device != move_to:
                param.data = param.data.to(move_to)
        for buf in module._buffers.values():
            if buf is not None and getattr(buf, "device", move_to) != move_to:
                buf.data = buf.data.to(move_to)
