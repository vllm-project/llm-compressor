# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch.nn import Module
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    PreTrainedModel,
)
from transformers.file_utils import WEIGHTS_NAME

from compressed_tensors.compressors import ModelCompressor
from sparseml.pytorch.model_load.helpers import (
    initialize_recipe,
    log_model_load,
)
from sparseml.transformers.sparsification.compressed_tensors_utils import (
    modify_save_pretrained,
)
from sparseml.transformers.utils.helpers import download_model_directory, resolve_recipe


__all__ = ["SparseAutoModel", "SparseAutoModelForCausalLM", "get_shared_tokenizer_src"]


_LOGGER = logging.getLogger(__name__)


class SparseAutoModelForCausalLM(AutoModelForCausalLM):
    """
    SparseML wrapper for the AutoModelForCausalLM class
    Its lifecycle is defined as follows:
    1. If pretrained_model_name_or_path is a SparseZoo stub
       the appropriate SparseZoo model will be downloaded
       (if required) and the path to the deployment directory
       of the model will be retrieved
    2. The original model definition will be loaded, without
        the model weights
    3. The appropriate recipy will be applied to the model
       if requested or required
    4. The appropriate set of weights will be loaded into the model
    """

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        recipe: Optional[Union[str, Path]] = None,
        *model_args,
        **kwargs,
    ) -> PreTrainedModel:
        """
         A wrapper around the AutoModelForCausalLM.from_pretrained method

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
        compressor = ModelCompressor.from_pretrained(pretrained_model_name_or_path)

        # temporarily set the log level to error, to ignore printing out long missing
        # and unexpected key error messages (these are EXPECTED for quantized models)
        logger = logging.getLogger("transformers.modeling_utils")
        restore_log_level = logger.getEffectiveLevel()
        logger.setLevel(level=logging.ERROR)
        model = super(AutoModelForCausalLM, cls).from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )
        if model.dtype != model.config.torch_dtype:
            _LOGGER.warning(
                f"The dtype of the loaded model: {model.dtype} is different "
                "from from the dtype specified in the model config: "
                f"{model.config.torch_dtype}."
                "To load the model in the format that it was previously saved in, "
                "set torch_dtype=`auto` in the SparseAutoModel creation call."
            )
        logger.setLevel(level=restore_log_level)
        # override the PreTrainedModel instance with compression save function
        modify_save_pretrained(model)

        # If model is quantized or compressed on disk, initialize quantization
        # structure and run decompression
        if compressor is not None:
            # initialize quantization and decompress weights
            compressor.decompress(model_path=pretrained_model_name_or_path, model=model)
            
        recipe = resolve_recipe(
            recipe=recipe, model_path=pretrained_model_name_or_path
        )
        if recipe:
            initialize_recipe(model=model, recipe_path=recipe)

        return model


class SparseAutoModel:
    """
    Factory class for creating sparse models using transformers AutoModel classes
    """

    @staticmethod
    def masked_language_modeling_from_pretrained(
        model_name_or_path: str,
        model_type: str,
        **kwargs,
    ) -> Module:
        """
        :param model_name_or_path: the name of or path to the model to load
        :param model_type: specify the type of model loaded for logging;
            ex one of [model, student, teacher]
        :param kwargs: keyword arguments to pass through to the AutoModel call
        :return: the created model for masked language modeling
        """
        delayed = False
        if not model_name_or_path:
            _LOGGER.info("Training new model from scratch")
            config = kwargs["config"]
            model = AutoModelForMaskedLM.from_config(config)
        else:
            SparseAutoModel._check_tf(model_name_or_path)
            if not kwargs:
                kwargs = {}
            kwargs["from_tf"] = False
            if "state_dict" not in kwargs:
                kwargs["state_dict"], delayed = SparseAutoModel._loadable_state_dict(
                    model_name_or_path
                )
            model = AutoModelForMaskedLM.from_pretrained(
                model_name_or_path,
                **kwargs,
            )
        model = modify_model(model)
        log_model_load(model, model_name_or_path, model_type, delayed)

        return model

    @staticmethod
    def masked_language_modeling_from_pretrained_distil(
        model_name_or_path: str,
        teacher_name_or_path: Optional[str],
        model_kwargs: Dict[str, Any],
        teacher_kwargs: Dict[str, Any],
    ) -> Tuple[Module, Optional[Union[Module, str]]]:
        """
        :param model_name_or_path: the name of or path to the model to load
        :param teacher_name_or_path: the name of or path to the teacher to load,
            None or one of ['self', 'disable'] will not create a teacher and
            instead return the value passed in
        :param model_kwargs: the keyword args to pass into the AutoModel for model
        :param teacher_kwargs: the keyword args to pass into the AutoModel for teacher
        :return: a tuple containing the model and distillation teacher (optional)
            for masked language modeling
        """
        model = SparseAutoModel.masked_language_modeling_from_pretrained(
            model_name_or_path,
            model_type="student" if teacher_name_or_path else "model",
            **model_kwargs,
        )
        teacher = (
            SparseAutoModel.masked_language_modeling_from_pretrained(
                teacher_name_or_path,
                model_type="teacher",
                **teacher_kwargs,
            )
            if teacher_name_or_path and teacher_name_or_path not in ["self", "disable"]
            else teacher_name_or_path
        )

        return model, teacher

    @staticmethod
    def question_answering_from_pretrained(
        model_name_or_path: str,
        model_type: str,
        **kwargs,
    ) -> Module:
        """
        :param model_name_or_path: the name of or path to the model to load
        :param model_type: specify the type of model loaded for logging;
            ex one of [model, student, teacher]
        :param kwargs: keyword arguments to pass through to the AutoModel call
        :return: the created model for question answering
        """
        SparseAutoModel._check_tf(model_name_or_path)
        if not kwargs:
            kwargs = {}
        kwargs["from_tf"] = False
        delayed = False
        if "state_dict" not in kwargs:
            kwargs["state_dict"], delayed = SparseAutoModel._loadable_state_dict(
                model_name_or_path
            )
        model = AutoModelForQuestionAnswering.from_pretrained(
            model_name_or_path,
            **kwargs,
        )
        model = modify_model(model)
        log_model_load(model, model_name_or_path, model_type, delayed)

        return model

    @staticmethod
    def question_answering_from_pretrained_distil(
        model_name_or_path: str,
        teacher_name_or_path: Optional[str],
        model_kwargs: Dict[str, Any],
        teacher_kwargs: Dict[str, Any],
    ) -> Tuple[Module, Optional[Union[Module, str]]]:
        """
        :param model_name_or_path: the name of or path to the model to load
        :param teacher_name_or_path: the name of or path to the teacher to load,
            None or one of ['self', 'disable'] will not create a teacher and
            instead return the value passed in
        :param model_kwargs: the keyword args to pass into the AutoModel for model
        :param teacher_kwargs: the keyword args to pass into the AutoModel for teacher
        :return: a tuple containing the model and distillation teacher (optional)
            for question answering
        """
        model = SparseAutoModel.question_answering_from_pretrained(
            model_name_or_path,
            model_type="student" if teacher_name_or_path else "model",
            **model_kwargs,
        )
        teacher = (
            SparseAutoModel.question_answering_from_pretrained(
                teacher_name_or_path, model_type="teacher", **teacher_kwargs
            )
            if teacher_name_or_path and teacher_name_or_path not in ["self", "disable"]
            else teacher_name_or_path
        )

        return model, teacher

    @staticmethod
    def text_classification_from_pretrained(
        model_name_or_path: str,
        model_type: str = "model",
        **kwargs,
    ) -> Module:
        """
        :param model_name_or_path: the name of or path to the model to load
        :param model_type: specify the type of model loaded for logging;
            ex one of [model, student, teacher]
        :param kwargs: keyword arguments to pass through to the AutoModel call
        :return: the created model for text classification
        """
        SparseAutoModel._check_tf(model_name_or_path)
        if not kwargs:
            kwargs = {}
        kwargs["from_tf"] = False
        delayed = False
        if "state_dict" not in kwargs:
            kwargs["state_dict"], delayed = SparseAutoModel._loadable_state_dict(
                model_name_or_path
            )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            **kwargs,
        )
        model = modify_model(model)
        log_model_load(model, model_name_or_path, model_type, delayed)

        return model

    @staticmethod
    def text_classification_from_pretrained_distil(
        model_name_or_path: str,
        teacher_name_or_path: Optional[str],
        model_kwargs: Dict[str, Any],
        teacher_kwargs: Dict[str, Any],
    ) -> Tuple[Module, Optional[Module]]:
        """
        :param model_name_or_path: the name of or path to the model to load
        :param teacher_name_or_path: the name of or path to the teacher to load,
            None or one of ['self', 'disable'] will not create a teacher and
            instead return the value passed in
        :param model_kwargs: the keyword args to pass into the AutoModel for model
        :param teacher_kwargs: the keyword args to pass into the AutoModel for teacher
        :return: a tuple containing the model and distillation teacher (optional)
            for sequence/text classification
        """
        model = SparseAutoModel.text_classification_from_pretrained(
            model_name_or_path,
            model_type="student" if teacher_name_or_path else "model",
            **model_kwargs,
        )
        teacher = (
            SparseAutoModel.text_classification_from_pretrained(
                teacher_name_or_path, model_type="teacher", **teacher_kwargs
            )
            if teacher_name_or_path and teacher_name_or_path not in ["self", "disable"]
            else teacher_name_or_path
        )

        return model, teacher

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

    @staticmethod
    def token_classification_from_pretrained(
        model_name_or_path: str,
        model_type: str,
        **kwargs,
    ) -> Module:
        """
        :param model_name_or_path: the name of or path to the model to load
        :param model_type: specify the type of model loaded for logging;
            ex one of [model, student, teacher]
        :param kwargs: keyword arguments to pass through to the AutoModel call
        :return: the created model for token classification
        """
        SparseAutoModel._check_tf(model_name_or_path)
        if not kwargs:
            kwargs = {}
        kwargs["from_tf"] = False
        delayed = False
        if "state_dict" not in kwargs:
            kwargs["state_dict"], delayed = SparseAutoModel._loadable_state_dict(
                model_name_or_path
            )
        model = AutoModelForTokenClassification.from_pretrained(
            model_name_or_path,
            **kwargs,
        )
        model = modify_model(model)
        log_model_load(model, model_name_or_path, model_type, delayed)

        return model

    @staticmethod
    def token_classification_from_pretrained_distil(
        model_name_or_path: str,
        teacher_name_or_path: Optional[str],
        model_kwargs: Dict[str, Any],
        teacher_kwargs: Dict[str, Any],
    ) -> Tuple[Module, Optional[Module]]:
        """
        :param model_name_or_path: the name of or path to the model to load
        :param teacher_name_or_path: the name of or path to the teacher to load,
            None or one of ['self', 'disable'] will not create a teacher and
            instead return the value passed in
        :param model_kwargs: the keyword args to pass into the AutoModel for model
        :param teacher_kwargs: the keyword args to pass into the AutoModel for teacher
        :return: a tuple containing the model and distillation teacher (optional)
            for token classification
        """
        model = SparseAutoModel.token_classification_from_pretrained(
            model_name_or_path,
            model_type="student" if teacher_name_or_path else "model",
            **model_kwargs,
        )
        teacher = (
            SparseAutoModel.token_classification_from_pretrained(
                teacher_name_or_path, model_type="teacher", **teacher_kwargs
            )
            if teacher_name_or_path and teacher_name_or_path not in ["self", "disable"]
            else teacher_name_or_path
        )

        return model, teacher

    @staticmethod
    def _loadable_state_dict(
        model_name_or_path: str,
    ) -> Tuple[Optional[Dict[str, Any]], bool]:
        """
        :param model_name_or_path: name of or path to model
        :return: (loaded state dict, True if overriding state dict for delayed load)
            delayed load happens when a QAT graph is detected since a recipe
            must be applied first
        """
        if not model_name_or_path or not os.path.isfile(
            os.path.join(model_name_or_path, WEIGHTS_NAME)
        ):
            return None, False

        state_dict = torch.load(
            os.path.join(model_name_or_path, WEIGHTS_NAME), map_location="cpu"
        )
        is_qat_state = any(
            [
                key.endswith(".zero_point") or key.endswith(".observer_enabled")
                for key in state_dict.keys()
            ]
        )

        if not is_qat_state:
            return None, False

        _LOGGER.warning(
            "QAT state detected, ignore any loading errors, weights will reload "
            f"after SparseML recipes have been applied {model_name_or_path}"
        )

        return None, True

    @staticmethod
    def _check_tf(model_name_or_path: str):
        if ".ckpt" in model_name_or_path:
            raise ValueError(
                "PyTorch is the only supported model type currently for SparseML "
                "HuggingFace Transformers integration. "
                "Detected a TensorFlow model from model_name_or_path: "
                f"{model_name_or_path}"
            )


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
