import inspect

import torch
from loguru import logger
from torch.nn import Parameter
from transformers import PreTrainedModel

from llmcompressor.typing import NamedModules

__all__ = [
    "untie_word_embeddings",
    "targets_embeddings",
    "get_embeddings",
    "warn_inference_mode_forwards",
]


def untie_word_embeddings(model: PreTrainedModel):
    """
    Untie word embeddings, if possible. This function raises a warning if
    embeddings cannot be found in the model definition.

    The model config will be updated to reflect that embeddings are now untied

    :param model: transformers model containing word embeddings
    """
    input_embed, output_embed = get_embeddings(model)
    if input_embed is None or output_embed is None:
        logger.warning(
            "Cannot untie embeddings. If this model has word embeddings, please "
            "implement `get_input_embeddings` and `get_output_embeddings`"
        )
        return

    # clone data to untie
    for module in (input_embed, output_embed):
        weight = module.weight
        param = Parameter(weight.data.clone(), requires_grad=weight.requires_grad)

        # Delete before re-registering: when offloaded, the OffloadCache updates the
        # existing (tied, shared) storage in place instead of allocating new storage,
        # so the embeddings would stay tied and save_pretrained would drop one of them.
        # In the non-offloaded case this is a harmless no-op.
        delattr(module, "weight")
        module.register_parameter("weight", param)

    # modify model config
    if hasattr(model.config, "tie_word_embeddings"):
        model.config.tie_word_embeddings = False


def targets_embeddings(
    model: PreTrainedModel,
    targets: NamedModules,
    check_input: bool = True,
    check_output: bool = True,
) -> bool:
    """
    Returns True if the given targets target the word embeddings of the model

    :param model: containing word embeddings
    :param targets: named modules to check
    :param check_input: whether to check if input embeddings are targeted
    :param check_output: whether to check if output embeddings are targeted
    :return: True if embeddings are targeted, False otherwise
    """
    input_embed, output_embed = get_embeddings(model)
    if (check_input and input_embed) is None or (check_output and output_embed is None):
        logger.warning(
            "Cannot check embeddings. If this model has word embeddings, please "
            "implement `get_input_embeddings` and `get_output_embeddings`"
        )
        return False

    targets = set(module for _, module in targets)
    return (check_input and input_embed in targets) or (
        check_output and output_embed in targets
    )


def get_embeddings(
    model: PreTrainedModel,
) -> tuple[torch.nn.Module | None, torch.nn.Module | None]:
    """
    Returns input and output embeddings of a model. If `get_input_embeddings`/
    `get_output_embeddings` is not implemented on the model, then None will be returned
    instead.

    :param model: model to get embeddings from
    :return: tuple of containing embedding modules or none
    """
    try:
        input_embed = model.get_input_embeddings()

    except (AttributeError, NotImplementedError, TypeError):
        input_embed = None

    try:
        output_embed = model.get_output_embeddings()
    except (AttributeError, NotImplementedError):
        output_embed = None

    return input_embed, output_embed


def warn_inference_mode_forwards(model: torch.nn.Module):
    """
    Warn if any submodule's forward method is decorated with
    @torch.inference_mode(). Any tensor produced during such a call stays
    tagged as an inference tensor even after the call returns. If a later
    stage of calibration, for example GPTQ's weight writeback or the
    offload cache, tries to update a tensor derived from that call in
    place, the update crashes with "Inplace update to inference tensor
    outside InferenceMode is not allowed", potentially after a long
    compression run has already made it through most of the model.

    :param model: model to scan for inference_mode decorated forwards
    """
    found = []
    for name, module in model.named_modules():
        forward = inspect.getattr_static(module, "forward", None)
        closure = getattr(forward, "__closure__", None)
        if closure is None:
            continue

        for cell in closure:
            # torch's decorator context managers capture a bound method
            # (e.g. self.clone) as the free variable, not the context
            # manager instance itself, so unwrap __self__ before checking
            content = cell.cell_contents
            bound_to = getattr(content, "__self__", content)
            if isinstance(bound_to, torch.inference_mode):
                found.append(name or type(module).__name__)
                break

    if found:
        logger.warning(
            "Found forward methods decorated with @torch.inference_mode() on: "
            f"{found}. Tensors produced during these calls remain inference "
            "tensors even after the call returns, which can cause in place "
            "weight updates to crash later in calibration. Consider removing "
            "the decorator or using torch.no_grad() instead."
        )
