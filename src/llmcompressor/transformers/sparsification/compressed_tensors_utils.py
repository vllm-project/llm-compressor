import os
import re
import weakref
from functools import wraps
from typing import Dict, Optional

import torch
import transformers
from accelerate.accelerator import get_state_dict_offloaded_model
from compressed_tensors import (
    CompressionFormat,
    ModelCompressor,
    SparsityCompressionConfig,
    is_module_offloaded,
    update_offload_parameter,
)
from loguru import logger
from safetensors.torch import storage_ptr
from transformers import PreTrainedModel

from llmcompressor.core import active_session
from llmcompressor.pytorch.model_load.helpers import copy_python_files_from_model_cache
from llmcompressor.recipe.recipe import Recipe
from llmcompressor.transformers.compression.quantization_format import (
    infer_quantization_format,
)
from llmcompressor.transformers.compression.sparsity_metadata_config import (
    SparsityConfigMetadata,
)
from llmcompressor.transformers.utils import RECIPE_FILE_NAME
from llmcompressor.transformers.utils.helpers import infer_recipe_from_model_path

__all__ = ["modify_save_pretrained"]


def modify_save_pretrained(model: PreTrainedModel):
    """
    Overrides a PreTrainedModel's save_pretrained() method with a wrapped version that
    supports compression
    """

    def save_pretrained_compressed(save_pretrained_method):
        if getattr(save_pretrained_method, "_overridden", False):
            # `model.save_pretrained` has already been replaced, return.
            return save_pretrained_method

        # Keep a weak reference to the model class and unbound save_pretrained
        # method so we can call the original
        model_ref = weakref.ref(save_pretrained_method.__self__)
        original_save_pretrained = save_pretrained_method.__func__
        model_class = model_ref().__class__
        del save_pretrained_method

        @wraps(original_save_pretrained)
        def save_pretrained_wrapper(
            save_directory: str,
            sparsity_config: Optional[SparsityCompressionConfig] = None,
            quantization_format: Optional[str] = None,
            save_compressed: bool = True,
            safe_serialization: bool = True,
            skip_sparsity_compression_stats: bool = True,
            disable_sparse_compression: bool = False,
            **kwargs,
        ):
            """
            Wrapper around PreTrainedModel.save_pretrained(), adds functionality for
            saving models in a compressed format on disk. The compression format is
            saved to the model's config file

            :param save_directory: output directory to save model to
            :param sparsity_config: optional sparsity config to compress model with,
                if no config is provided it will be inferred from the model
            :param quantization_format: optional compression format for quantized
                models. If none is provided it will be inferred from the model
            :param save_compressed: whether or not to compress the model on disk
            :param skip_sparsity_compression_stats: whether to skip the calculation of
                sparsity statistics (such as global sparsity and sparsity structure)
                when saving a model in dense format
            :param disable_sparse_compression: whether to skip sparse compression
                during save, default is False
            :param kwargs: additional kwargs to pass on to model.save_pretrained
            """

            # HACK: Override the dtype_byte_size function in transformers to
            # support float8 types. Fix is posted upstream
            # https://github.com/huggingface/transformers/pull/30488
            transformers.modeling_utils.dtype_byte_size = new_dtype_byte_size

            # state_dict gets passed in as a kwarg for FSDP models
            state_dict = kwargs.pop("state_dict", None)
            if state_dict is None:
                logger.info("Fetching state_dict - this may take some time")
                state_dict = get_state_dict_offloaded_model(model)

            logger.info("Fetching compressor")
            compressor = get_model_compressor(
                model=model,
                sparsity_config=sparsity_config,
                quantization_format=quantization_format,
                save_compressed=save_compressed,
                skip_sparsity_compression_stats=skip_sparsity_compression_stats,
                state_dict=state_dict,
                disable_sparse_compression=disable_sparse_compression,
            )

            if compressor is None:
                # model is not compressed or quantized, save as normal
                original_save_pretrained_func = original_save_pretrained.__get__(
                    model, model_class
                )
                original_save_pretrained_func(
                    save_directory, state_dict=state_dict, **kwargs
                )
                return

            # make sure we're on the main process when saving
            if state_dict is not None and len(state_dict) > 0:
                compressed_state_dict = compressor.compress(model, state_dict)
                logger.info("Saving compressed model to disk")
                original_save_pretrained.__get__(model, model_class)(
                    save_directory,
                    state_dict=compressed_state_dict,
                    safe_serialization=safe_serialization,
                    **kwargs,
                )
                compressor.update_config(save_directory)

            # update existing recipe
            update_and_save_recipe(model.name_or_path, save_directory)

            # copy python files from cache dir to save_path if any
            copy_python_files_from_model_cache(model, save_directory)

        save_pretrained_wrapper._overriden = True
        return save_pretrained_wrapper

    # wrap save_pretrained if not already
    if not getattr(model.save_pretrained, "_overriden", False):
        model.save_pretrained = save_pretrained_compressed(model.save_pretrained)


# HACK: Override the dtype_byte_size function in transformers to support float8 types
# Fix is posted upstream https://github.com/huggingface/transformers/pull/30488
def new_dtype_byte_size(dtype):
    if dtype == torch.bool:
        return 1 / 8
    bit_search = re.search(r"[^\d](\d+)_?", str(dtype))
    if bit_search is None:
        raise ValueError(f"`dtype` is not a valid dtype: {dtype}.")
    bit_size = int(bit_search.groups()[0])
    return bit_size // 8


def patch_tied_tensors_bug(model: torch.nn.Module):
    """
    Patches bug where HF transformers will fail to untie weights under specific
    circumstances (https://github.com/huggingface/transformers/issues/33689).

    This function detects those cases and unties the tensors if applicable

    :param model: model to fix
    """
    if (
        hasattr(model.config, "tie_word_embeddings")
        and not model.config.tie_word_embeddings
    ):
        input_embed = model.get_input_embeddings()
        output_embed = model.get_output_embeddings()

        if input_embed is None or output_embed is None:
            # some models fail to properly override the abstract methods
            return

        if storage_ptr(input_embed.weight) == storage_ptr(output_embed.weight):
            for module in (input_embed, output_embed):
                if not is_module_offloaded(module):
                    # create new storage ptr for onloaded weight
                    untied_data = module.weight.data.clone()
                    module.weight.data = untied_data
                else:
                    # create new storage ptr for offloaded weight
                    # note `update_offload_parameter` does not create a new storage ptr
                    untied_data = module._hf_hook.weights_map["weight"].clone()
                    update_offload_parameter(module, "weight", untied_data)


def get_model_compressor(
    model: torch.nn.Module,
    sparsity_config: Optional[SparsityCompressionConfig] = None,
    quantization_format: Optional[str] = None,
    save_compressed: bool = True,
    skip_sparsity_compression_stats: bool = True,
    state_dict: Optional[Dict] = None,
    disable_sparse_compression: bool = False,
):
    """
    Obtain the compressor based on the config and the
        quantization_format

    :param model: torch model
    :param sparsify_config: Sparsity Compression config
    :param quantization_format: Format that the model was quantized to.
        if not provivided, will be extrapolated from `infer_quantization_format`
    :param save_compressed: boolean representing to save in a compressed
        format
    :param skip_sparsity_compression_stats: bool allowing compression stats on std out
    :param state_dict: state_dict of the model
    :param disable_sparse_compression: bool to skip sparse compression
    """

    # find offloaded state dict if none is provided
    if state_dict is None:
        state_dict = get_state_dict_offloaded_model(model)

    if sparsity_config is None:
        """
        Case 1: No sparsity config is provided
            1. Will either skip sparsity compression
            2. Or we will infer sparsity from the model directly

        Check recipe for applied sparsity:
            - Set skip_sparsity_compression_stats to False if don't find a
                sparsity structure from the recipe
            - If we identify sparsity based on the recipe or the user
                set skip_sparsity_compression_stats to False, generate config
        """
        sparsity_structure = SparsityConfigMetadata.infer_sparsity_structure(
            model, check_only_modifiers=True
        )
        if sparsity_structure is not None:
            skip_sparsity_compression_stats = False

        if skip_sparsity_compression_stats:
            logger.info(
                "skip_sparsity_compression_stats set to True. Skipping sparsity "
                "compression statistic calculations. No sparsity compressor will "
                "be applied."
            )
            sparsity_config = None
        else:
            sparsity_config = SparsityConfigMetadata.from_pretrained(
                model,
                state_dict=state_dict,
                compress=save_compressed,
                quantization_format=quantization_format,
                disable_sparse_compression=disable_sparse_compression,
                sparsity_structure=sparsity_structure,
            )
    else:
        """
        # Case 2: User provides a Sparsity Config
            - This is the case when there is existing sparsity in the
                model that we'd like to account for while compressing
            - Users should provide a SparsityConfig, conveying the model's
                sparsity structure when saving the model
        """
        if sparsity_config.sparsity_structure is None:
            logger.info(
                "SparsityConfigMetadata provided without indicating ",
                "the sparsity structure. Sparisty will be inferred from the model. "
                "Consider providing the structure to skip this step ",
            )
            sparsity_config.sparsity_structure = (
                SparsityConfigMetadata.infer_sparsity_structure(model)
            )

    quantization_format: Optional[CompressionFormat] = infer_quantization_format(
        model=model,
        quantization_format=quantization_format,
        save_compressed=save_compressed,
        sparsity_structure=None
        if sparsity_config is None
        else sparsity_config.sparsity_structure,
    )

    return ModelCompressor.from_pretrained_model(
        model,
        sparsity_config=sparsity_config,
        quantization_format=quantization_format,
    )


def update_and_save_recipe(model_path: str, save_directory: str):
    recipes_to_save = []
    existing_recipe = infer_recipe_from_model_path(model_path)
    if existing_recipe is not None:
        recipes_to_save.append(existing_recipe)

    new_recipe = active_session().lifecycle.recipe_container.compiled_recipe
    if new_recipe is not None:
        recipes_to_save.append(new_recipe)

    recipe = Recipe.simplify_combine_recipes(recipes_to_save)

    # save recipe
    recipe_path = os.path.join(save_directory, RECIPE_FILE_NAME)
    recipe.yaml(recipe_path)
