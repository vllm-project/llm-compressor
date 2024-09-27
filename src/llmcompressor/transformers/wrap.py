import logging
from pathlib import Path
from typing import Optional, Union

import torch
from accelerate import load_checkpoint_and_dispatch
from compressed_tensors.compressors import ModelCompressor
from compressed_tensors.quantization import (
    QuantizationStatus,
    apply_quantization_config,
)
from loguru import logger
from transformers import PreTrainedModel

from llmcompressor.pytorch.model_load.helpers import initialize_recipe
from llmcompressor.transformers.sparsification.compressed_tensors_utils import (
    modify_save_pretrained,
)
from llmcompressor.transformers.utils.helpers import (
    download_model_directory,
    resolve_recipe,
)

__all__ = ["wrap_hf_model_class"]


def wrap_hf_model_class(hf_model_class: PreTrainedModel) -> PreTrainedModel:
    """
    Wrap a HF PreTrainedModel class to
    1. Decompress a compressed model
    2. Initialize any saved recipes
    3. Wrap the `save_pretrained` method to allow saving as a compressed model

    :param hf_model_class: Model class to wrap
    :return: Wrapped model class
    """

    # Add the from_pretrained class method
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        run_compressed: bool = False,
        recipe: Optional[Union[str, Path]] = None,
        *model_args,
        **kwargs,
    ) -> PreTrainedModel:
        """
        A wrapper around the PreTrainedModel.from_pretrained method

        :param pretrained_model_name_or_path: the name of or path to the model to load
        :param recipe: the path to the recipe file to apply to the model. Can be a
            string or Path object. If None, a recipe will be searched for in the
            pretrained_model_name_or_path directory and applied if found
        :return the created model for causal language modeling
        """

        def skip(*args, **kwargs):
            pass

        # Skip the initializer step. This accelerates the loading
        # of the models, especially for the quantized models
        torch.nn.init.kaiming_uniform_ = skip
        torch.nn.init.uniform_ = skip
        torch.nn.init.normal_ = skip

        pretrained_model_name_or_path = (
            pretrained_model_name_or_path.as_posix()
            if isinstance(pretrained_model_name_or_path, Path)
            else pretrained_model_name_or_path
        )

        pretrained_model_name_or_path = download_model_directory(
            pretrained_model_name_or_path, **kwargs
        )

        # instantiate compressor from model config
        compressor = ModelCompressor.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )

        # temporarily set the log level to error, to ignore printing out long missing
        # and unexpected key error messages (these are EXPECTED for quantized models)
        transformers_logger = logging.getLogger("transformers.modeling_utils")
        restore_log_level = transformers_logger.getEffectiveLevel()
        transformers_logger.setLevel(level=logging.ERROR)

        if kwargs.get("trust_remote_code"):
            # By artifically aliasing the
            # class name to the
            # hf_model_class we can "trick" the
            # `from_pretrained` method into properly
            # resolving the logic when
            # (has_remote_code and trust_remote_code) == True
            cls.__name__ = hf_model_class.__name__

        model = super(hf_model_class, cls).from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )

        if model.dtype != model.config.torch_dtype:
            logger.warning(
                f"The dtype of the loaded model: {model.dtype} is different "
                "from from the dtype specified in the model config: "
                f"{model.config.torch_dtype}."
                "To load the model in the format that it was previously saved in, "
                "set torch_dtype=`auto` in the SparseAutoModel creation call."
            )

        # restore transformers logging level now that model shell is loaded
        transformers_logger.setLevel(level=restore_log_level)

        # HfQuantizer Quantization
        if hasattr(model.config, "quantization_config"):
            return model

        # override the PreTrainedModel instance with compression save function
        modify_save_pretrained(model)

        # If model is quantized or compressed on disk, initialize quantization
        # structure and run decompression
        if compressor is not None:
            quantization_config = compressor.quantization_config
            is_compressed = (
                quantization_config is not None
                and quantization_config.quantization_status
                == QuantizationStatus.COMPRESSED
            )
            if run_compressed and is_compressed:
                # initialize quantization, don't decompress
                apply_quantization_config(
                    model, quantization_config, run_compressed=True
                )
                model = load_checkpoint_and_dispatch(
                    model, pretrained_model_name_or_path
                )
            else:
                # initialize quantization and decompress weights
                if quantization_config is not None:
                    quantization_config.quantization_status = QuantizationStatus.FROZEN
                compressor.decompress(
                    model_path=pretrained_model_name_or_path, model=model
                )
        recipe = resolve_recipe(recipe=recipe, model_path=pretrained_model_name_or_path)

        if recipe:
            initialize_recipe(model=model, recipe_path=recipe)

        return model

    # Add the wrapped methods to the new class
    wrapped_model_class = type(
        hf_model_class.__name__, (hf_model_class,), {"from_pretrained": from_pretrained}
    )

    return wrapped_model_class
