import torch
from loguru import logger
from torch.nn import Parameter
from transformers import PreTrainedModel

from llmcompressor.typing import NamedModules

__all__ = ["untie_word_embeddings", "targets_embeddings", "get_embeddings"]


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

    except (AttributeError, NotImplementedError):
        input_embed = None

    try:
        output_embed = model.get_output_embeddings()
    except (AttributeError, NotImplementedError):
        output_embed = None

    return input_embed, output_embed
