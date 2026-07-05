import datetime
import os
import weakref
from contextlib import contextmanager
from functools import wraps

import torch
import torch.distributed as dist
from compressed_tensors import ModelCompressor, SparsityCompressionConfig
from compressed_tensors.config import CompressionFormat
from compressed_tensors.distributed import is_source_process
from compressed_tensors.offload import OffloadCache, from_accelerate, to_accelerate
from compressed_tensors.utils import deprecated
from loguru import logger
from transformers import PreTrainedModel

from llmcompressor.core import active_session
from llmcompressor.pytorch.model_load.helpers import copy_python_files_from_model_cache
from llmcompressor.transformers.utils import RECIPE_FILE_NAME
from llmcompressor.transformers.utils.helpers import infer_recipe_from_model_path
from llmcompressor.utils.transformers import get_embeddings

__all__ = ["modify_save_pretrained"]


def _named_tensors(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return a module's own (non-None) parameters and buffers keyed by name.

    ``None`` placeholders are skipped so e.g. a biasless ``lm_head`` (which still
    registers ``bias=None``) compares equal to an ``Embedding`` that has none.
    """
    names = list(module._parameters.keys()) + list(module._buffers.keys())
    tensors = {name: getattr(module, name) for name in names}
    return {name: tensor for name, tensor in tensors.items() if tensor is not None}


def _retie_quantized_embeddings(model: PreTrainedModel):
    """Re-tie identically-quantized input and output embeddings before saving.

    Embeddings are untied during calibration so the input and output embeddings
    can be quantized independently (each with its own qparams). At save time, if
    the two ended up with identical tensors -- i.e. the same matrix quantized
    twice -- share the input's tensors with the output so transformers' save-time
    de-duplication writes a single shared table, and restore
    ``tie_word_embeddings`` so the tie is reconstructed at load. If they differ
    (quantized differently, or only one was quantized) they are left untied and
    both are kept, preserving the model's integrity.

    Args:
        model: The model about to be saved.
    """
    input_embed, output_embed = get_embeddings(model)
    if input_embed is None or output_embed is None or input_embed is output_embed:
        return

    input_tensors = _named_tensors(input_embed)
    output_tensors = _named_tensors(output_embed)
    # Only the compressed case; dense (re-)tying is handled by transformers and
    # ``_retie_offloaded_weights``. A dense untie clones the weight, leaving the
    # two value-identical, so acting here would wrongly re-tie an untied model.
    if "weight_packed" not in input_tensors:
        return
    if input_tensors.keys() != output_tensors.keys() or not all(
        torch.equal(input_tensors[name], output_tensors[name])
        for name in input_tensors
    ):
        return

    # Share the input's storage so transformers' save-time de-duplication keeps a
    # single copy. ``disable_onloading`` makes the read and assignment go by
    # reference (a plain assignment to an offloaded module copies); it can be
    # dropped once compressed-tensors makes ``__setitem__`` a non-copying
    # replacement (vllm-project/compressed-tensors#709).
    with OffloadCache.disable_onloading():
        for name in input_tensors:
            setattr(output_embed, name, getattr(input_embed, name))

    config = getattr(model, "config", None)
    get_text_config = getattr(config, "get_text_config", None)
    text_config = get_text_config(decoder=True) if callable(get_text_config) else config
    if text_config is not None:
        text_config.tie_word_embeddings = True
    logger.info(
        "Re-tied identically-quantized input/output embeddings; saving a single "
        "shared table."
    )


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

            # Re-tie weights before offload conversion. Offloading splits tied
            # weights (e.g. lm_head and embed_tokens) into separate parameters,
            # which defeats transformers' save-time de-duplication and writes a
            # redundant copy. Doing this before `to_accelerate` keeps accelerate's
            # tied-parameter bookkeeping consistent.
            _retie_offloaded_weights(model)

            # Embeddings are untied during calibration so input and output can be
            # quantized independently; if they ended up identical, re-tie the
            # packed weights so save de-duplicates them into one shared table.
            _retie_quantized_embeddings(model)

            # convert to accelerate offloaded for optimal saving with transformers
            to_accelerate(model)

            with suspend_distributed_timeout():
                if is_source_process():
                    # save model structure
                    original_save_fn.__get__(model, model_class)(
                        save_directory, **kwargs
                    )

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


def _retie_offloaded_weights(model: PreTrainedModel):
    """Re-tie weights split by offload conversion so the tied head isn't saved twice.

    Offloading gives the input embeddings and a tied head (e.g. ``lm_head``)
    separate parameters, defeating transformers' save-time de-duplication and
    writing a redundant copy. Re-tying restores the shared parameter.

    Only runs when ``config.tie_word_embeddings`` is set, so models that were
    explicitly untied (e.g. by SpinQuant via ``untie_word_embeddings``) are left
    untouched. Failures in a model's ``tie_weights`` are logged rather than raised.
    """
    get_text_config = getattr(model.config, "get_text_config", None)
    text_config = (
        get_text_config(decoder=True) if callable(get_text_config) else model.config
    )
    if not getattr(text_config, "tie_word_embeddings", False):
        return

    tie_weights = getattr(model, "tie_weights", None)
    if not callable(tie_weights):
        return

    logger.info("Re-tying word embeddings before save to avoid duplicate weights")
    try:
        tie_weights()
    except Exception as e:
        logger.warning(
            f"Failed to re-tie word embeddings before save ({e}); a tied head "
            "such as lm_head may be written as a redundant duplicate."
        )


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


@contextmanager
def suspend_distributed_timeout(
    timeout: datetime.timedelta = datetime.timedelta(hours=3),
    current_group: dist.ProcessGroup | None = None,
):
    """
    Context manager that extends the timeout for distributed operations.

    Creates a temporary process group with an extended timeout to prevent
    timeout errors during long-running operations (e.g., model saving) in
    distributed training environments. The context manager synchronizes all
    processes before and after the operation using barriers.

    :param timeout: The extended timeout for the temporary process group.
        Defaults to 3 hours
    :param current_group: The current process group to synchronize. If None,
        defaults to dist.group.WORLD
    """
    if not dist.is_initialized():
        yield
        return

    if current_group is None:
        current_group = dist.group.WORLD
    suspend_group = dist.new_group(backend="gloo", timeout=timeout)

    try:
        dist.barrier(group=current_group)
        yield
    finally:
        dist.barrier(group=suspend_group)
        dist.barrier(group=current_group)
        dist.destroy_process_group(group=suspend_group)
