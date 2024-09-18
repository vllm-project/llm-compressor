import inspect
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
from torch.nn import Module
from transformers import AutoModelForCausalLM, PreTrainedModel

from llmcompressor.pytorch.model_load.helpers import initialize_recipe
from llmcompressor.transformers.sparsification.compressed_tensors_utils import (
    modify_save_pretrained,
)
from llmcompressor.transformers.utils.helpers import (
    download_model_directory,
    resolve_recipe,
)

__all__ = ["create_sparse_auto_model_class", "SparseAutoModel", "SparseAutoModelForCausalLM", "get_shared_tokenizer_src"]


def create_sparse_auto_model_class(auto_model_class_name):
    # Dynamically import the AutoModelForX class
    exec(f"from transformers import {auto_model_class_name}")
    auto_model_class = eval(auto_model_class_name)
    
    class_name = f"Sparse{auto_model_class_name}"
    
    # Define the new class
    new_class = type(class_name, (auto_model_class,), {})
    
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
            # By artifically aliasing
            # class name SparseCLASSNAME to
            # CLASSNAME we can "trick" the
            # `from_pretrained` method into properly
            # resolving the logic when
            # (has_remote_code and trust_remote_code) == True
            cls.__name__ = auto_model_class.__name__

        model = super(auto_model_class, cls).from_pretrained(
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

    # Add the from_pretrained method to the new class
    setattr(new_class, 'from_pretrained', from_pretrained)
    
    return new_class


SparseAutoModelForCausalLM = create_sparse_auto_model_class("AutoModelForCausalLM")


class SparseAutoModel:
    """
    Factory class for creating sparse models using transformers AutoModel classes
    """

    @staticmethod
    def text_generation_from_pretrained(
        model_name_or_path: str,
        sequence_length: Optional[int] = None,
        recipe: Optional[Union[str, Path]] = None,
        trust_remote_code: bool = False,
        torch_dtype: Union[str, torch.dtype] = "auto",
        **kwargs,
    ) -> Module:
        """
        :param model_name_or_path: the name of or path to the model to load
        :param sequence_length: the maximum length of the sequence to generate.
            If None, will use the default sequence length for the model.
            Defaults to None.
        :param recipe: the recipe to apply to the model. If None, no recipe is applied
        :param trust_remote_code: related to trust_remote_code in HF transformers.
            If True, will execute the modelling code from the model directory
            (if present). Defaults to False.
        :param torch_dtype: the torch dtype to use for the model. If "auto", will
            use the default dtype for the model. Defaults to "auto".
        :return: the created model for text generation
        """

        model = SparseAutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            recipe=recipe,
            **kwargs,
        )
        if sequence_length is not None:
            model.seqlen = sequence_length

        return model


def get_shared_tokenizer_src(student: Module, teacher: Optional[Module]) -> str:
    """
    Get a tokenizer source used for both student and teacher, assuming
    that they could be shared

    :param student: the student model
    :param teacher: the teacher model
    :return: the source for the tokenizer shared between teacher and model
    """

    if teacher is not None and teacher not in ("disable", "self"):
        student_forward_params = list(
            inspect.signature(student.forward).parameters.keys()
        )
        teacher_forward_params = list(
            inspect.signature(teacher.forward).parameters.keys()
        )
        diff = [p for p in student_forward_params if p not in teacher_forward_params]
        if diff:
            raise RuntimeError(
                "Teacher tokenizer cannot be used for student "
                f"due to missing args: {diff}"
            )
        src_model = teacher
    else:
        src_model = student
    return src_model.config._name_or_path
