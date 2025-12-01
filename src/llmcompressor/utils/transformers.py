import torch
from compressed_tensors import has_offloaded_params
from loguru import logger
from transformers import PreTrainedModel

from llmcompressor.typing import NamedModules

__all__ = ["untie_word_embeddings", "targets_embeddings", "get_embeddings"]


def untie_word_embeddings(model: PreTrainedModel):
    """Untie word embeddings, if possible."""
    input_embed, output_embed = get_embeddings(model)
    if input_embed is None or output_embed is None:
        logger.warning(
            "Cannot untie embeddings. If this model has word embeddings, please "
            "implement `get_input_embeddings` and `get_output_embeddings`"
        )
        return

    # clone data to untie
    for module in (input_embed, output_embed):
        if not has_offloaded_params(module):
            module.weight.data = module.weight.data.clone()
        else:
            weights_map = module._hf_hook.weights_map
            weights_map["weight"] = weights_map["weight"].clone()

    # modify model config
    if hasattr(model.config, "tie_word_embeddings"):
        model.config.tie_word_embeddings = False


def targets_embeddings(
    model: PreTrainedModel,
    targets: NamedModules,
    check_input: bool = True,
    check_output: bool = True,
) -> bool:
    input_embed, output_embed = get_embeddings(model)
    if check_input and input_embed is None or check_output and output_embed is None:
        logger.warning(
            "Cannot check embeddings. If this model has word embeddings, please "
            "implement `get_input_embeddings` and `get_output_embeddings`"
        )
        return False

    targets = set(module for _, module in targets)
    return (not check_input or input_embed in targets) and (
        not check_output or output_embed in targets
    )


def get_embeddings(
    model: PreTrainedModel,
) -> tuple[torch.nn.Module | None, torch.nn.Module | None]:
    try:
        input_embed = model.get_input_embeddings()

    except (AttributeError, NotImplementedError):
        input_embed = None

    try:
        output_embed = model.get_output_embeddings()
    except (AttributeError, NotImplementedError):
        output_embed = None

    return input_embed, output_embed
