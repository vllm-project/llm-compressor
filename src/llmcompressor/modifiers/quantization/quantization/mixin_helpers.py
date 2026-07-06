from collections.abc import Iterable

from compressed_tensors.quantization import QuantizationStatus
from torch.nn import Module

__all__ = ["CALIBRATION_OBSERVER_BASE_NAMES", "validate_module_calibration"]

CALIBRATION_OBSERVER_BASE_NAMES = ("input", "weight", "output", "q", "k", "v")


def validate_module_calibration(
    model: Module,
    modules: Module | Iterable[Module],
    base_names: str | Iterable[str] = CALIBRATION_OBSERVER_BASE_NAMES,
) -> None:
    module_names = {id(module): name for name, module in model.named_modules()}
    observer_base_names = _normalize_base_names(base_names)
    failures = list(
        _iter_unobserved_observers(modules, observer_base_names, module_names)
    )
    if failures:
        raise ValueError(_format_calibration_error(failures))


def _normalize_base_names(base_names: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(base_names, str):
        return (base_names,)

    return tuple(base_names)


def _iter_unobserved_observers(
    modules: Module | Iterable[Module],
    base_names: tuple[str, ...],
    module_names: dict[int, str],
) -> Iterable[str]:
    for module in _iter_unique_modules(modules):
        if _is_frozen(module):
            continue
        for base_name in base_names:
            observer_name = f"{base_name}_observer"
            observer = getattr(module, observer_name, None)
            if observer is None:
                continue
            if getattr(observer, "num_observations", 0) == 0:
                yield _format_module_attr(module, observer_name, module_names)


def _iter_unique_modules(modules: Module | Iterable[Module]) -> Iterable[Module]:
    modules = (modules,) if isinstance(modules, Module) else modules
    seen = set()
    for module in modules:
        for submodule in module.modules():
            if id(submodule) in seen:
                continue
            seen.add(id(submodule))
            yield submodule


def _is_frozen(module: Module) -> bool:
    return getattr(module, "quantization_status", None) == QuantizationStatus.FROZEN


def _format_module_attr(
    module: Module,
    attr_name: str,
    module_names: dict[int, str],
) -> str:
    module_name = module_names.get(id(module), module.__class__.__name__)
    return f"{module_name}.{attr_name}" if module_name else attr_name


def _format_calibration_error(failures: list[str]) -> str:
    return (
        "Quantization calibration failed. The following observers were never "
        f"called: {', '.join(failures)}. This usually means the calibration data "
        "did not execute these quantized modules, or the model did not route "
        "tensors through their calibration hooks."
    )
