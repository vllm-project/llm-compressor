import inspect
import os
from pathlib import PosixPath
from typing import Optional, Tuple

from compressed_tensors.utils import remove_dispatch
from loguru import logger
from torch.nn import Module
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    PreTrainedModel,
    set_seed,
)
from transformers.utils.quantization_config import CompressedTensorsConfig

from llmcompressor.args import ModelArguments, RecipeArguments, TrainingArguments
from llmcompressor.core import reset_session
from llmcompressor.pytorch.model_load.helpers import parse_dtype
from llmcompressor.transformers.sparsification.compressed_tensors_utils import (
    modify_save_pretrained,
    patch_tied_tensors_bug,
)
from llmcompressor.transformers.utils.helpers import (
    detect_last_checkpoint,
    is_model_ct_quantized_from_path,
)
from llmcompressor.typing import Processor
from llmcompressor.utils.fsdp.helpers import is_fsdp_model


def pre_process(model_args: "ModelArguments"):
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
    _warn_tied_embeddings(model_args.tie_word_embeddings)

    # Initialize model
    if isinstance(model_args.model, (str, PosixPath)):
        model, distill_teacher = initialize_model_from_path(model_args)
        if is_fsdp_model(model):
            raise NotImplementedError(
                "FSDP models are not supported in the current release but will be "
                "suported in future releases of LLM Compressor."
            )
        model_args.model = model
        model_args.distill_teacher = distill_teacher

    # Initialize processor
    if isinstance(model_args.processor, (str, type(None))):
        model_args.processor = initialize_processor_from_path(
            model_args, model_args.model
        )

    # untie tie_word_embeddings weights
    patch_tied_tensors_bug(model_args.model)

    # wrap model.save_pretrained
    modify_save_pretrained(model_args.model)


def post_process(
    model_args: Optional["ModelArguments"] = None,
    recipe_args: Optional["RecipeArguments"] = None,
    output_dir: Optional[str] = None,
):
    """
    Saves the model and tokenizer/processor to the output directory if model_args,
    output_dir is provided.

    Save is skipped for stage runs for `train` - saves using the trainer.save_model()

    If the `output_dir` is not the default directory, the method resets lifecycle
    actions. The model is saved in a compressed format if specified in `model_args`.
    Additionally, the tokenizer or processor, if available, is also saved.

    Raises:
        ValueError: If saving fails due to an invalid `output_dir` or other issues.
    """
    # remove any existing dispatches
    if model_args is not None and model_args.model is not None:
        remove_dispatch(model_args.model)

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


def _warn_tied_embeddings(tie_word_embeddings: bool = False):
    """
    Logs a warning if the model has tied word embeddings.
    The `tie_word_embeddings` flag may cause issues during saving in the one-shot
    calibration workflow due to shared tensor addresses.
    """
    if tie_word_embeddings:
        logger.debug(
            "The tie_word_embeddings flag is by default set to False. "
            "This guarantees that the one-shot algorithm saves the final "
            "weights without errors. Detected tie_word_embeddings=True. "
            "This may cause issues with the one-shot algorithm on save."
        )


def initialize_model_from_path(
    model_args: ModelArguments,
    training_args: Optional[TrainingArguments] = None,
) -> Tuple[PreTrainedModel, Optional[PreTrainedModel]]:
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

    last_checkpoint = None
    teacher = None

    if training_args is not None:
        # Load teacher configuration if applicable
        teacher_config = (
            AutoConfig.from_pretrained(
                model_args.distill_teacher,
                use_auth_token=True if model_args.use_auth_token else None,
                tie_word_embeddings=model_args.tie_word_embeddings,
                trust_remote_code=model_args.trust_remote_code_model,
            )
            if model_args.distill_teacher
            else None
        )

        # Detect last checkpoint
        last_checkpoint = detect_last_checkpoint(training_args, model_args=model_args)

        # Set seed before initializing model
        set_seed(training_args.seed)

        # Initialize teacher model if teacher path is provided
        if model_args.distill_teacher is not None:
            teacher_device_map = (
                None
                if os.environ.get("ACCELERATE_USE_FSDP", "false") == "true"
                else "auto"
            )
            teacher_kwargs = {
                "config": teacher_config,
                "cache_dir": model_args.cache_dir,
                "use_auth_token": True if model_args.use_auth_token else None,
                "torch_dtype": parse_dtype(model_args.precision),
                "device_map": teacher_device_map,
                "trust_remote_code": model_args.trust_remote_code_model,
            }

            teacher = AutoModelForCausalLM.from_pretrained(
                model_args.distill_teacher,
                **teacher_kwargs,
            )
            if "sequence_length" in teacher_kwargs:
                teacher.seqlen = teacher_kwargs["sequence_length"]

    model_path = (
        last_checkpoint or model_args.model
        if hasattr(model_args, "model")
        else model_args.model_name_or_path
    )

    model_kwargs = {
        "config": config,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "torch_dtype": parse_dtype(model_args.precision),
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

    return model, teacher


def initialize_processor_from_path(
    model_args: ModelArguments,
    model: PreTrainedModel,
    teacher: Optional[PreTrainedModel] = None,
) -> Processor:
    processor_src = model_args.processor or get_processor_name_from_model(
        model, teacher
    )
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
            cache_dir=model_args.cache_dir,
            use_fast=False,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            trust_remote_code=model_args.trust_remote_code_model,
        )

    return processor


def get_processor_name_from_model(student: Module, teacher: Optional[Module]) -> str:
    """
    Get a processor/tokenizer source used for both student and teacher, assuming
    that they could be shared

    :param student: the student model
    :param teacher: the teacher model
    :return: the source for the processor/tokenizer shared between teacher and model
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
