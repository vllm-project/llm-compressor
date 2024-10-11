import re
import weakref
from functools import reduce, wraps
from typing import Optional

import torch
import transformers
from accelerate.accelerator import get_state_dict_offloaded_model
from compressed_tensors import ModelCompressor, SparsityCompressionConfig
from loguru import logger
from safetensors.torch import _find_shared_tensors
from transformers import PreTrainedModel

from llmcompressor.transformers.compression.quantization_format import (
    infer_quantization_format,
)
from llmcompressor.transformers.compression.sparsity_config import (
    SparsityConfigMetadata,
)
# from llmcompressor.transformers.finetune.trainer import Trainer
from llmcompressor.utils.fsdp.helpers import (
    find_and_move_state_dicts_to_cpu,
    is_fsdp_model,
    unwrap_and_export_model,
)
import os
from llmcompressor.core import active_session


__all__ = ["modify_save_pretrained"]


def modify_save_pretrained(trainer, tokenizer):
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
            skip_compression_stats: bool = False,
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
            :param skip_compression_stats: whether to skip the calculation of
            compression statistics (such as global sparsity and sparsity structure) when
            saving a model in dense format
            :param kwargs: additional kwargs to pass on to model.save_pretrained
            """
            if is_fsdp_model(trainer.model):
                try:
                    trainer.save_model(output_dir=save_directory, _is_oneshot=True)
                except AssertionError:
                    # fallback to this in the case of quantization
                    unwrap_and_export_model(
                        model=trainer.model,
                        accelerator=trainer.accelerator,
                        output_dir=save_directory,
                        tokenizer=tokenizer,
                    )
                    # only allow the main process move the state
                    # dicts to cpu
                    if trainer.accelerator.is_main_process:
                        # assuming quantization is the last step
                        # we no longer need the original model
                        # and can safely delete it to save memory
                        del trainer.model
                        find_and_move_state_dicts_to_cpu(save_directory)

            else:
                # HACK: Override the dtype_byte_size function in transformers to
                # support float8 types. Fix is posted upstream
                # https://github.com/huggingface/transformers/pull/30488
                transformers.modeling_utils.dtype_byte_size = new_dtype_byte_size

                model_ = model_ref()
                model = trainer.model
                breakpoint()
                # state_dict gets passed in as a kwarg for FSDP models
                state_dict = kwargs.pop("state_dict", None)

                # find offloaded state dict if none is provided
                if state_dict is None:
                    state_dict = get_state_dict_offloaded_model(model)

                if sparsity_config is not None:
                    sparsity_config.global_sparsity = (
                        SparsityConfigMetadata.infer_global_sparsity(
                            model, state_dict=state_dict
                        )
                    )
                    sparsity_config.sparsity_structure = (
                        SparsityConfigMetadata.infer_sparsity_structure()
                    )
                elif not skip_compression_stats:
                    # try to infer a sparsity config from the model if none is provided
                    logger.info(
                        "Inferring a sparsity configuration requires a global sparsity "
                        "calculation. This can be costly for large models. To skip the "
                        "calculation of compression statistics set "
                        "skip_compression_stats=True"
                    )
                    sparsity_config = SparsityConfigMetadata.from_pretrained(
                        model, state_dict=state_dict, compress=save_compressed
                    )

                quantization_format = infer_quantization_format(
                    model=model,
                    quantization_format=quantization_format,
                    save_compressed=save_compressed,
                    sparsity_config=sparsity_config,
                )
                compressor = ModelCompressor.from_pretrained_model(
                    model,
                    sparsity_config=sparsity_config,
                    quantization_format=quantization_format,
                )

                if compressor is None:
                    # model is not compressed or quantized, save as normal
                    original_save_pretrained.__get__(model, model_class)(
                        save_directory, state_dict=state_dict, **kwargs
                    )
                    return

                # make sure we're on the main process when saving
                if state_dict is not None and len(state_dict) > 0:
                    compressed_state_dict = compressor.compress(model, state_dict)

                    kwargs["safe_serialization"] = kwargs.get("safe_serialization", True)
                    original_save_pretrained.__get__(model, model_class)(
                        save_directory, state_dict=compressed_state_dict, **kwargs
                    )
                    compressor.update_config(save_directory)

                recipe_path = os.path.join(save_directory, "recipe.yaml")
                session = active_session()
                recipe_yaml_str = session.get_serialized_recipe()
                with open(recipe_path, "w") as fp:
                    fp.write(recipe_yaml_str)

                # copy python files from cache dir to save_path if any
                _copy_python_files_from_model_cache(model, save_directory)

        save_pretrained_wrapper._overriden = True
        return save_pretrained_wrapper

    # wrap save_pretrained
    trainer.model.save_pretrained = save_pretrained_compressed(trainer.model.save_pretrained)


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


def patch_tied_tensors_bug(model: torch.nn.Module) -> torch.nn.Module:
    """
    Patches bug where HF transformers will fail to untie weights under specific
    circumstances (https://github.com/huggingface/transformers/issues/33689).

    This function detects those cases and unties the tensors if applicable

    :param model: model to fix
    :return: model with fixed parameters
    """
    if not model.config.tie_word_embeddings:
        tensor_groups = _find_shared_tensors(get_state_dict_offloaded_model(model))
        for tensor_group in tensor_groups:
            if len(tensor_group) > 1:
                if not set(model._tied_weights_keys).intersection(tensor_group):
                    raise ValueError(
                        "Model contains unexpected shared tensors. Expected "
                        f"{model._tied_weights_keys}, found {tensor_group}"
                    )

                for tensor_path in tensor_group:
                    tensor_parts = tensor_path.split(".")
                    parameter = reduce(getattr, tensor_parts, model)
                    parameter.data = parameter.data.clone()

    return model



def _copy_python_files_from_model_cache(model, save_path: str):
    config = model.config
    cache_path = None
    if hasattr(config, "_name_or_path"):
        import os
        import shutil

        from huggingface_hub import hf_hub_download
        from transformers import TRANSFORMERS_CACHE
        from transformers.utils import http_user_agent

        cache_path = config._name_or_path
        if not os.path.exists(cache_path):
            user_agent = http_user_agent()
            config_file_path = hf_hub_download(
                repo_id=cache_path,
                filename="config.json",
                cache_dir=TRANSFORMERS_CACHE,
                force_download=False,
                user_agent=user_agent,
            )
            cache_path = os.path.sep.join(config_file_path.split(os.path.sep)[:-1])

        for file in os.listdir(cache_path):
            full_file_name = os.path.join(cache_path, file)
            if file.endswith(".py") and os.path.isfile(full_file_name):
                logger.debug(f"Transferring {full_file_name} to {save_path}")
                shutil.copy(full_file_name, save_path)