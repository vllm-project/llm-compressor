import contextlib
from functools import wraps
from typing import Type

import torch
import tqdm
from compressed_tensors.offload import (
    OffloadCache,
    get_cache_init_kwargs,
    offload_module,
)
from compressed_tensors.utils import patch_attr
from loguru import logger
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PreTrainedConfig,
    PreTrainedModel,
)
from transformers.conversion_mapping import (
    register_checkpoint_conversion_mapping,
)
from transformers.monkey_patching import clear_patch_mapping, register_patch_mapping

from llmcompressor.modeling.moe.helpers import FusedExpertsProtocol

from .conversion_mappings import (
    get_linearize_load_mappings,
    has_linearize_load_mappings,
    set_save_conversion_mapping,
)
from .linear_experts import LinearExperts2D


@contextlib.contextmanager
def load_quantizable_moe(model_cls: Type[PreTrainedModel] = AutoModelForCausalLM):
    """
    Context manager for loading MoE models with linearized experts for
    efficient calibration and quantization.

    This context manager patches the `from_pretrained` method of the given model class
    to automatically linearize MoE (Mixture-of-Experts) layers during model loading.
    Linearization converts 3D expert weight tensors into 2D format, enabling more
    efficient calibration and quantization of individual experts.

    Two loading pathways are supported:
    1. Direct loading: If the model checkpoint contains 2D weights and conversion
        mappings areregistered for the model type, weights are loaded directly in
        linearized format.
    2. Post-load conversion: If no conversion mappings exist, a resident model is
        linearized via `linearize_moe`. An offloaded model stays fused. The
        sequential pipeline linearizes each experts module after that subgraph is
        onloaded, so the weights are not loaded once to linearize and again to
        calibrate.

    :param model_cls: The model class to patch, defaults to AutoModelForCausalLM
    """
    original_from_pretrained = model_cls.from_pretrained
    patched_fn_called = False

    @classmethod
    @wraps(original_from_pretrained)
    def patched(cls, *args, **kwargs):
        nonlocal patched_fn_called
        patched_fn_called = True

        config = AutoConfig.from_pretrained(*args, **kwargs)
        model_type = config.model_type

        # model is 3d (or otherwise doesn't have mappings)
        # fall back to post-load conversion
        if not has_linearize_load_mappings(model_type):
            model = original_from_pretrained(*args, **kwargs)
            if _is_offloaded(model):
                return model
            linearize_moe(model)
            return model

        # prepare to load linearized weights
        experts_cls, load_map, save_map = get_linearize_load_mappings(model_type)
        linear_experts_2d_cls = LinearExperts2D.get_linear_experts_cls(experts_cls)
        register_patch_mapping({experts_cls.__name__: linear_experts_2d_cls})
        register_checkpoint_conversion_mapping(model_type, load_map, overwrite=True)

        # load model
        model: PreTrainedModel = original_from_pretrained(*args, **kwargs)

        # prepare for saving to be called later
        clear_patch_mapping()
        set_save_conversion_mapping(model, save_map)
        register_checkpoint_conversion_mapping(model_type, save_map, overwrite=True)

        for _, module in get_linearized_moes(model):
            module._source_experts_cls = experts_cls
            module._source_config = _experts_config(module, model)

        return model

    with patch_attr(model_cls, "from_pretrained", patched):
        try:
            yield
        finally:
            if not patched_fn_called:
                logger.warning(
                    f"`{model_cls.__name__}.from_pretrained` was never called. If you "
                    f"are loading with a model class other than {model_cls.__name__}, "
                    "please pass as argument to `load_quantizable_moe`"
                )


def linearize_moe(model: PreTrainedModel):
    """
    Linearize a mixture-of-experts model after it has been loaded. For more
    runtime-efficient loading, please see `load_quantizable_moe`.

    Experts modules will be replaced by either two pathways:
    1. The expert module has a registered replacement. This is required for
    2. The expert module conforms to the standard transformers MoE format
    (as designated by the `use_experts_implementation` decorator)

    :param model: model containing MoE layers to linearize
    """
    non_linearized_moes = get_non_linearized_moes(model)

    if len(non_linearized_moes) <= 0:
        return model

    logger.warning(
        "MoE is being linearized after loading in order to support efficient "
        "calibration of experts. However, this may be inefficient if the model "
        "checkpoint is already linearized (2D -> 3D -> 2D). Consider registering "
        "a load converter for faster load times. See "
        "https://docs.vllm.ai/projects/llm-compressor/en/latest/developer-tutorials/add-moe-support"  # noqa: E501
    )

    for name, module in tqdm.tqdm(non_linearized_moes, desc="Linearizing experts"):
        linear_moe, offload_kwargs = _install_linear_experts(model, name, module)
        if offload_kwargs is not None:
            for child in linear_moe.modules():
                offload_module(child, **offload_kwargs)

    return model


def _is_offloaded(model: torch.nn.Module) -> bool:
    return any(
        isinstance(getattr(module, "_parameters", None), OffloadCache)
        for module in model.modules()
    )


def _is_fused_experts(module: torch.nn.Module) -> bool:
    return isinstance(module, FusedExpertsProtocol) or (
        LinearExperts2D.get_registration(module.__class__) is not None
    )


def _install_linear_experts(
    model: torch.nn.Module,
    name: str,
    module: torch.nn.Module,
) -> tuple[LinearExperts2D, dict | None]:
    """Replace one fused module with resident 2D experts.

    The returned kwargs are the fused module's offload settings, when it had
    any. Callers that linearize inside an onload window ignore them and offload
    later with the kwargs captured before that window.
    """
    offload_kwargs = (
        get_cache_init_kwargs(module)
        if isinstance(getattr(module, "_parameters", None), OffloadCache)
        else None
    )
    config = _experts_config(module, model)
    linear_experts_cls = LinearExperts2D.get_linear_experts_cls(module.__class__)
    linear_moe = linear_experts_cls.from_experts_module(module, config)
    model.set_submodule(name, linear_moe)
    return linear_moe, offload_kwargs


def linearize_onload_experts(
    model: torch.nn.Module,
    subgraph,
    subgraph_modules: dict[str, torch.nn.Module],
    offload_kwargs: dict[str, dict],
) -> list[torch.nn.Module]:
    """Linearize fused experts that are already onloaded, and keep them resident.

    Refreshes ``subgraph_modules`` and copies the pre-onload offload kwargs onto
    the new child names. ``subgraph_offload_modules`` then returns those linears
    to the fused module's original device.
    """
    replaced_names = []
    roots = []
    for name, module in list(subgraph_modules.items()):
        if not _is_fused_experts(module):
            continue
        linear_moe, _ = _install_linear_experts(model, name, module)
        replaced_names.append(name)
        roots.append(linear_moe)
    if not roots:
        return []
    subgraph_modules.clear()
    subgraph_modules.update(subgraph.submodule_dict(model))
    assign_child_offload_kwargs(replaced_names, list(subgraph_modules), offload_kwargs)
    return roots


def assign_child_offload_kwargs(
    replaced_names: list[str],
    module_names: list[str],
    offload_kwargs: dict[str, dict],
) -> None:
    """Give new child modules the offload settings of the fused module they replaced."""
    for name in module_names:
        if name in offload_kwargs:
            continue
        parent = None
        for candidate in replaced_names:
            if candidate not in offload_kwargs:
                continue
            if name.startswith(candidate + "."):
                if parent is None or len(candidate) > len(parent):
                    parent = candidate
        if parent is not None:
            offload_kwargs[name] = offload_kwargs[parent]


def linearize_remaining_moe(model: PreTrainedModel) -> list[LinearExperts2D]:
    """Model-wide linearize for pipelines that have no per-subgraph onload window."""
    if len(get_non_linearized_moes(model)) <= 0:
        return []
    logger.warning(
        "Detected an MoE model which has not been linearized. First load "
        "model `with llmcompressor.modeling.moe.linearize.load_quantizable_moe`"
        " before passing to `oneshot`. Falling back to post-load linearization."
    )
    linearize_moe(model)
    return [module for _, module in get_linearized_moes(model)]


def _experts_config(
    module: torch.nn.Module, model: PreTrainedModel
) -> PreTrainedConfig:
    """Prefer a nested text config so fused experts construct with the right
    constructor (e.g. InklingExperts(InklingTextConfig)).
    """
    if (config := getattr(module, "config", None)) is not None:
        return config
    config = model.config
    get_text_config = getattr(config, "get_text_config", None)
    if callable(get_text_config):
        return get_text_config()
    return getattr(config, "text_config", config)


def get_linearized_moes(
    model: torch.nn.Module,
) -> list[tuple[str, LinearExperts2D]]:
    """
    Return all :class:`LinearExperts2D` modules in ``model`` (post-linearization).
    """
    return [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, LinearExperts2D)
    ]


def repack_moe(model: PreTrainedModel) -> PreTrainedModel:
    """
    Explicitly pack linearized :class:`LinearExperts2D` modules back into native
    fused 3D expert modules.

    Call this after linearization. Unquantized recipes (for example
    ``REAPPruningModifier`` only) and frozen quantized experts restore
    native fused Parameters, fusing extra qparams alongside ``weight``.
    If experts are already compressed (no dense ``weight``), nested packed
    projections are written instead. For compressed 3D checkpoints:

    ```python
    compressor.compress_model(model)
    repack_moe(model)
    model.save_pretrained(output_dir)
    ```

    Dense experts restore native fused Parameters. Compressed experts replace
    ``gate_up_proj`` / ``down_proj`` with serialization-only nested modules
    that own packed tensors and qparams so Transformers conversion mappings
    can emit native 3D keys. Compressed fused modules are not intended for
    fused expert forward.

    :param model: model containing linearized MoE layers to repack
    :return: the same model with fused expert modules restored
    """
    linearized = get_linearized_moes(model)
    if len(linearized) <= 0:
        return model

    for name, module in tqdm.tqdm(linearized, desc="Repacking experts"):
        fused = module.to_experts_module()
        model.set_submodule(name, fused)

    return model


def get_non_linearized_moes(
    model: torch.nn.Module,
) -> list[tuple[str, torch.nn.Module]]:
    """
    Return all modules which are recognized to be experts layers. A module is recognized
    as an experts layer if it conforms to the `FusedExpertsProtocol` or is registered by
    `LinearExperts2D`.

    :param model: model with modules to check for experts
    :return: list of named modules which are recognized as experts layers
    """
    return [
        (name, module)
        for name, module in model.named_modules()
        if _is_fused_experts(module)
    ]
