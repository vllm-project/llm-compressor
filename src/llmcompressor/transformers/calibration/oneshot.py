import os
from pathlib import PosixPath
from typing import Optional

from loguru import logger
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    HfArgumentParser,
    PreTrainedModel,
)

from llmcompressor.pytorch.model_load.helpers import fallback_to_cpu, parse_dtype
from llmcompressor.transformers.finetune.data.data_args import DataTrainingArguments
from llmcompressor.transformers.finetune.data.data_helpers import (
    get_calibration_dataloader,
)
from llmcompressor.transformers.finetune.model_args import (  # different file
    OneshotModelArguments,
)
from llmcompressor.transformers.sparsification.compressed_tensors_utils import (
    modify_save_pretrained,
    patch_tied_tensors_bug,
)
from llmcompressor.transformers.sparsification.sparse_model import (
    get_processor_from_model,
)
from llmcompressor.transformers.utils.recipe_args import RecipeArguments
from llmcompressor.typing import Processor


class Oneshot:
    """
    Class responsisble for carrying out oneshot calibration

    Lifecycle:
        - Instantiate CompressionLifecycle that is responsible for applying the recipe
        - Carry out preprocessing - model, tokenizer/processor instantiation, untie shared tensors,
            wrap model.save_pretrained to save models in compressed_tensors format for vllm inference
        - Get calibration dataloader for dataset to calibrate the scales and zero points
        - Applying recipe modifiers using the calibration dataloader
        - Save the model in compressed_tensors format if model was provided as a string or custom output_dir was set

    Usage:

    ```python
    oneshot_calibrator = Oneshot(model=model, recipe=recipe, dataset=dateset)
    oneshot_calibrator.run()

    ```
    """

    def __init__(self, **kwargs):
        from llmcompressor.core.lifecycle import CompressionLifecycle

        self.model_args, self.data_args, self.recipe_args = parse_oneshot_args(**kwargs)
        self.lifecycle = CompressionLifecycle()  # [TODO] singleton for

        self._preprocess()

        self.model = self.model_args.model
        self.tokenizer_or_processor = self.model_args.processor

    def run(self):
        calibration_dataloader = get_calibration_dataloader(
            self.data_args, self.tokenizer_or_processor
        )

        self.apply_recipe_modifiers(calibration_dataloader=calibration_dataloader)

        # save if model was provided as a string or custom output_dir was set
        if isinstance(self.model_args.model, str) or (
            self.model_args.output_dir
            != OneshotModelArguments.__dataclass_fields__["output_dir"].default
        ):
            self.model_args.model.save_pretrained(
                self.model_args.output_dir,
                save_compressed=self.model_args.save_compressed,
            )
            if self.tokenizer_or_processor is not None:
                self.tokenizer_or_processor.save_pretrained(self.model_args.output_dir)

        # Clean up the CompressionSession before exit if requested
        if self.recipe_args.clear_sparse_session:
            self.lifecycle.reset()

    def apply_recipe_modifiers(self, calibration_dataloader: Optional[DataLoader]):
        self.lifecycle.initialize(
            model=self.model,
            recipe=self.recipe_args.recipe,
            recipe_args=self.recipe_args.recipe_args,
            calib_data=calibration_dataloader,
            start=-1,  # oneshot specific arg
            copy_data=False,
            min_tokens_per_module=self.min_tokens_per_module,
        )

    def _preprocess(self):
        if self.model_args.tie_word_embeddings is True:
            logger.debug(
                "The tie_word_embeddings flag is by default set to False. "
                "This guarantees that the one-shot algorithm saves the final "
                "weights without errors. Detected tie_word_embeddings=True. "
                "This may cause issues with the one-shot algorithm on save. "
            )

        model = self.model_args.model
        if isinstance(model, str) or isinstance(model, PosixPath):
            model = initialize_oneshot_model(self.model_args)

        # patch a shared tensor bug in HF transformers
        # https://github.com/huggingface/transformers/issues/33689
        patch_tied_tensors_bug(model)

        # wrap model.save_pretrained in compressed_tensors format for vllm
        modify_save_pretrained(model)

        self.model_args.model = model

        processor = self.model_args.processor
        if isinstance(processor, str) or processor is None:
            self.model_args.processor = initialize_processor_from_path(
                self.model_args, model
            )

        if self.data_args is not None:
            self.min_tokens_per_module = self.data_args.min_tokens_per_module


def parse_oneshot_args(**kwargs):
    parser = HfArgumentParser(
        (OneshotModelArguments, DataTrainingArguments, RecipeArguments)
    )
    if not kwargs:
        model_args, data_args, recipe_args = parser.parse_args_into_dataclasses()
    else:
        model_args, data_args, recipe_args = parser.parse_dict(kwargs)

    if recipe_args.recipe_args is not None:
        if not isinstance(recipe_args.recipe_args, dict):
            arg_dict = {}
            for recipe_arg in recipe_args.recipe_args:
                key, value = recipe_arg.split("=")
                arg_dict[key] = value
            recipe_args.recipe_args = arg_dict

    if model_args.tokenizer:
        if model_args.processor:
            raise ValueError("Cannot use both a tokenizer and processor")

        logger.debug("Overwriting processor with tokenizer")
        model_args.processor = model_args.tokenizer

    return model_args, data_args, recipe_args


def initialize_oneshot_model(
    model_args,
):
    # Load pretrained model
    # The .from_pretrained methods guarantee that only one local process can
    # concurrently download model & vocab.
    model_path = model_args.model
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        tie_word_embeddings=model_args.tie_word_embeddings,
        trust_remote_code=model_args.trust_remote_code_model,
    )

    model_path = (
        model_args.model
        if hasattr(model_args, "model")
        else model_args.model_name_or_path
    )

    # Fallback to CPU if GPU requested and not available
    model_args.oneshot_device = fallback_to_cpu(model_args.oneshot_device)

    # Trainer handles device assignment for FSDP and training, don't do mapping here
    # if running oneshot outside of FSDP, apply user device settings
    device_map = None
    fsdp_enabled = os.environ.get("ACCELERATE_USE_FSDP", "false") == "true"
    if not fsdp_enabled:
        device_map = model_args.oneshot_device
        logger.warning(f"Moving {model_path} to device {device_map} for One-Shot")
    elif not fsdp_enabled:
        device_map = "auto"

    model_kwargs = {
        "config": config,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "torch_dtype": parse_dtype(model_args.precision),
        "device_map": device_map,
        "trust_remote_code": model_args.trust_remote_code_model,
    }

    # this calls from_pretrained under the hood so should be FSDP safe
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **model_kwargs,
    )
    if "sequence_length" in model_kwargs:
        model.seqlen = model_kwargs["sequence_length"]

    return model


def initialize_processor_from_path(
    model_args: OneshotModelArguments,
    model: PreTrainedModel,
    teacher: Optional[PreTrainedModel] = None,
) -> Processor:
    processor_src = model_args.processor
    processor_src = model_args.processor or get_processor_from_model(model, teacher)
    # The use_fast=True option is not currently supported safely in Transformers
    # See: https://github.com/huggingface/transformers/pull/34836#issuecomment-2491809727  # noqa: E501
    try:
        processor = AutoProcessor.from_pretrained(
            processor_src,
            cache_dir=model_args.cache_dir,
            use_fast=True,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            trust_remote_code=model_args.trust_remote_code_model,
        )
    except Exception:
        logger.debug("Could not load fast processor, loading slow processor instead")
        processor = AutoProcessor.from_pretrained(
            processor_src,
            cache_dir=model_args.cache_dir,
            use_fast=False,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            trust_remote_code=model_args.trust_remote_code_model,
        )

    return processor
