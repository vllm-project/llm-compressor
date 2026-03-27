import os
import weakref
from functools import wraps

import torch
from compressed_tensors import ModelCompressor, SparsityCompressionConfig
from compressed_tensors.config import CompressionFormat
from compressed_tensors.offload import from_accelerate, is_rank0, to_accelerate
from compressed_tensors.utils import deprecated
from loguru import logger
from transformers import PreTrainedModel

from llmcompressor.core import active_session
from llmcompressor.pytorch.model_load.helpers import copy_python_files_from_model_cache
from llmcompressor.transformers.utils import RECIPE_FILE_NAME
from llmcompressor.transformers.utils.helpers import infer_recipe_from_model_path

__all__ = ["modify_save_pretrained"]


def modify_save_pretrained(model: PreTrainedModel):
    """
    Overrides a PreTrainedModel's save_pretrained() method with a wrapped version that
    supports compression. The new save_pretrained function performs the following saving
    operations:

    1. Saves the model state, potentially in a compressed format
    2. Saves the recipe, appending any current recipes to existing recipe files
    3. Copies any necessary python files from the model cache
    """

    def save_pretrained_compressed(save_pretrained_method):
        if getattr(save_pretrained_method, "_overridden", False):
            # `model.save_pretrained` has already been replaced, return.
            return save_pretrained_method

        # Keep a weak reference to the model class and unbound save_pretrained
        # method so we can call the original
        model_ref = weakref.ref(save_pretrained_method.__self__)
        original_save_fn = save_pretrained_method.__func__
        model_class = model_ref().__class__
        del save_pretrained_method

        @wraps(original_save_fn)
        def save_pretrained_wrapper(
            save_directory: str,
            quantization_format: str | None = None,
            save_compressed: bool = True,
            **kwargs,
        ):
            """
            Wrapper around PreTrainedModel.save_pretrained(), adds functionality for
            saving models in a compressed format on disk. The compression format is
            saved to the model's config file

            :param save_directory: output directory to save model to
            :param quantization_format: optional compression format override. If none
                is provided, the compression format will be inferred from the model
            :param save_compressed: whether or not to compress the model. If true,
                weights will be compressed. Otherwise, weights will remain in full
                precision in the "FROZEN" state.
            :param kwargs: additional kwargs to pass on to model.save_pretrained
            """

            # compress model using compressor
            compressor = ModelCompressor.from_pretrained_model(
                model, quantization_format=quantization_format
            )
            if save_compressed:
                compressor.compress_model(model)

            # convert to accelerate offloaded for optimal saving with transformers
            to_accelerate(model)

            if is_rank0():
                # save model structure
                original_save_fn.__get__(model, model_class)(save_directory, **kwargs)

                # update config to reflect quantization
                compressor.update_config(save_directory)

                # update existing recipe
                update_and_save_recipe(model.name_or_path, save_directory)

                # copy python files from cache dir to save_path if any
                copy_python_files_from_model_cache(model, save_directory)

            # convert back from accelerate to restore model to original form
            from_accelerate(model)

        save_pretrained_wrapper._overridden = True
        return save_pretrained_wrapper

    # wrap save_pretrained if not already
    if not getattr(model.save_pretrained, "_overridden", False):
        model.save_pretrained = save_pretrained_compressed(model.save_pretrained)


@deprecated("ModelCompressor.from_pretrained_model")
def get_model_compressor(
    model: torch.nn.Module,
    sparsity_config: SparsityCompressionConfig | None = None,
    quantization_format: str | None = None,
    save_compressed: bool = True,
    skip_sparsity_compression_stats: bool = True,
    disable_sparse_compression: bool = False,
):
    """
    Obtain the compressor based on the config and the quantization_format

    :param model: torch model
    :param sparsify_config: Sparsity Compression config
    :param quantization_format: Format that the model was quantized to.
        if not provivided, will be extrapolated from `infer_quantization_format`
    :param save_compressed: boolean representing to save in a compressed
        format
    :param skip_sparsity_compression_stats: bool allowing compression stats on std out
    :param disable_sparse_compression: bool to skip sparse compression
    """

    if (
        sparsity_config is not None
        or not skip_sparsity_compression_stats
        or disable_sparse_compression
    ):
        logger.warning(
            "Sparse compression is no longer supported by compressed-tensors"
        )

    if not save_compressed:
        if quantization_format not in (None, CompressionFormat.dense.value):
            raise ValueError(
                "A quantizatiom format was provided but "
                "save_compressed is set to False. "
                "A compression format can only be applied when "
                "saving the model compressed"
            )
        quantization_format = CompressionFormat.dense.value

    return ModelCompressor.from_pretrained_model(
        model,
        quantization_format=quantization_format,
    )


def update_and_save_recipe(model_stub: str, save_directory: str):
    """
    Save a recipe ontop of any existing recipe files located at model_stub

    :param model_stub: path to existing model or model stub which may contain an
        existing recipe
    :param save_directory: path to save combined existing recipe and current recipe
    """

    existing_recipe = infer_recipe_from_model_path(model_stub)

    recipe = active_session().lifecycle.recipe

    recipe_path = os.path.join(save_directory, RECIPE_FILE_NAME)
    recipe.yaml(file_path=recipe_path, existing_recipe_path=existing_recipe)
