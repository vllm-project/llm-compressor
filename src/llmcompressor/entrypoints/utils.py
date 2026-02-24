"""
Utility functions for entrypoint pre and post-processing operations.

Provides common utility functions used by the one-shot
entrypoint. Includes model loading, configuration setup,
preprocessing steps, and post-processing operations for compression
workflows.
"""

import os
from pathlib import PosixPath

from compressed_tensors.offload import from_accelerate
from loguru import logger
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    PreTrainedModel,
)
from transformers.utils.quantization_config import CompressedTensorsConfig

from llmcompressor.args import (
    DatasetArguments,
    ModelArguments,
    RecipeArguments,
)
from llmcompressor.core import reset_session
from llmcompressor.pytorch.model_load.helpers import parse_dtype
from llmcompressor.transformers.compression.compressed_tensors_utils import (
    modify_save_pretrained,
)
from llmcompressor.transformers.utils.helpers import (
    is_model_ct_quantized_from_path,
)
from llmcompressor.typing import Processor
from llmcompressor.utils import untie_word_embeddings


def pre_process(
    model_args: ModelArguments,
    dataset_args: DatasetArguments,
    output_dir: str | None,
):
    """
    Prepares the model and tokenizer/processor for calibration.
    - Initializes the model if it's specified as a path or string.
    - Applies patches to fix tied tensor issues and modifies `save_pretrained`
        behavior.
    - Initializes the processor if specified as a path or `None`.
    - Sets the minimum tokens per module if `dataset_args` are provided.
    Raises:
        FileNotFoundError: If the model or processor path is invalid.
    """

    # Initialize model
    if isinstance(model_args.model, (str, PosixPath)):
        model = initialize_model_from_path(model_args)
        model_args.model = model

    # Initialize processor if dataset provided
    if isinstance(model_args.processor, (str, type(None))):
        try:
            model_args.processor = initialize_processor_from_path(
                model_args, model_args.model
            )
        except Exception as e:
            if dataset_args.is_dataset_provided():
                raise RuntimeError(
                    "An error occurred when attempting to initialize "
                    "model processor, which is required when a dataset "
                    "is provided. To resolve, create and pass in a "
                    "processor directly to `oneshot`/`train`."
                ) from e
            elif output_dir:
                logger.warning(
                    "Model processor could not be auto-initialized and "
                    "will not be saved along with the model. To resolve, "
                    "create and pass in a processor directly to "
                    f"`oneshot`/`train`.\nInitialization Error: {e}"
                )

    # untie tie_word_embeddings weights
    if not model_args.tie_word_embeddings:
        untie_word_embeddings(model_args.model)

    # if the model was loaded with accelerate offloading, convert to CT offloading
    if hasattr(model_args.model, "hf_device_map"):
        from_accelerate(model_args.model)

    # wrap model.save_pretrained
    modify_save_pretrained(model_args.model)


def post_process(
    model_args: ModelArguments | None = None,
    recipe_args: RecipeArguments | None = None,
    output_dir: str | None = None,
):
    """
    Saves the model and tokenizer/processor to the output directory if model_args,
    output_dir is provided.

    If the `output_dir` is not the default directory, the method resets lifecycle
    actions. The model is saved in a compressed format if specified in `model_args`.
    Additionally, the tokenizer or processor, if available, is also saved.

    Raises:
        ValueError: If saving fails due to an invalid `output_dir` or other issues.
    """
    if model_args is not None and output_dir is not None:
        if recipe_args is not None and getattr(recipe_args, "stage", None) is not None:
            output_dir = os.path.join(output_dir, recipe_args.stage)
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"[Save] Stage detected. Updating output_dir to {output_dir}")

        # TODO: support general saving parameters, beyond save_compressed
        model_args.model.save_pretrained(
            output_dir, save_compressed=model_args.save_compressed
        )

        if model_args.processor is not None:
            model_args.processor.save_pretrained(output_dir)

    else:
        logger.warning(
            "Optimized model is not saved. To save, please provide"
            "`output_dir` as input arg."
            "Ex. `oneshot(..., output_dir=...)`"
        )

    # Reset the one-time-use session upon completion
    if recipe_args is not None and recipe_args.clear_sparse_session:
        reset_session()


def initialize_model_from_path(
    model_args: ModelArguments,
) -> PreTrainedModel:
    # Load pretrained model
    # The .from_pretrained methods guarantee that only one local process can
    # concurrently download model & vocab.
    model_path = model_args.model
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_path,
        cache_dir=None,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        trust_remote_code=model_args.trust_remote_code_model,
    )

    last_checkpoint = None

    model_path = (
        last_checkpoint or model_args.model
        if hasattr(model_args, "model")
        else model_args.model_name_or_path
    )

    model_kwargs = {
        "config": config,
        "cache_dir": None,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "dtype": parse_dtype(model_args.precision),
        "trust_remote_code": model_args.trust_remote_code_model,
    }

    # optimized models must be decompressed to carry out oneshot/train/etc
    if is_model_ct_quantized_from_path(model_path):
        model_kwargs["quantization_config"] = CompressedTensorsConfig(
            run_compressed=False
        )

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    if "sequence_length" in model_kwargs:
        model.seqlen = model_kwargs["sequence_length"]

    return model


def initialize_processor_from_path(
    model_args: ModelArguments, model: PreTrainedModel
) -> Processor:
    processor_src = model_args.processor or model.config._name_or_path
    # The use_fast=True option is not currently supported safely in Transformers
    # See: https://github.com/huggingface/transformers/pull/34836#issuecomment-2491809727  # noqa: E501
    try:
        processor = AutoProcessor.from_pretrained(
            processor_src,
            cache_dir=None,
            use_fast=True,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            trust_remote_code=model_args.trust_remote_code_model,
        )

    except ValueError as exception:
        if any("trust_remote_code=True" in arg for arg in exception.args):
            raise ValueError(
                f"The repository for {processor_src} contains custom code which must "
                "be executed to correctly load the tokenizer/processor. You can "
                f"inspect the repository content at https://hf.co/{processor_src}.\n"
                "Please pass the argument `trust_remote_code_model=True`."
            )

        logger.debug("Could not load fast processor, loading slow processor instead")
        processor = AutoProcessor.from_pretrained(
            processor_src,
            cache_dir=None,
            use_fast=False,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            trust_remote_code=model_args.trust_remote_code_model,
        )

    return processor
