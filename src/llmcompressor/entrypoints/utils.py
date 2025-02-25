import os
import warnings
from pathlib import PosixPath
from typing import Optional

from compressed_tensors.utils.helpers import deprecated
from loguru import logger
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    HfArgumentParser,
    PreTrainedModel,
    set_seed,
)
from transformers.utils.quantization_config import CompressedTensorsConfig

from llmcompressor.args import (
    DatasetArguments,
    ModelArguments,
    RecipeArguments,
    TrainingArguments,
)
from llmcompressor.core import pre_initialize_structure, reset_session
from llmcompressor.pytorch.model_load.helpers import (
    fallback_to_cpu,
    get_session_model,
    initialize_recipe,
    parse_dtype,
)
from llmcompressor.recipe import Recipe, StageRunType
from llmcompressor.transformers.finetune.runner import StageRunner
from llmcompressor.transformers.finetune.trainer import Trainer
from llmcompressor.transformers.sparsification.compressed_tensors_utils import (
    modify_fsdp_model_save_pretrained,
    modify_save_pretrained,
    patch_tied_tensors_bug,
)
from llmcompressor.transformers.sparsification.sparse_model import (
    get_processor_name_from_model,
)
from llmcompressor.transformers.utils.helpers import (
    detect_last_checkpoint,
    is_model_ct_quantized_from_path,
)
from llmcompressor.typing import Processor
from llmcompressor.utils.fsdp.helpers import is_fsdp_model

# from llmcompressor.transformers.finetune.text_generation import (
#     initialize_model_from_path,
#     initialize_processor_from_path,
# )





def preprocess(model_args: "ModelArguments"):
    """
    Prepares the model and tokenizer/processor for calibration.
    - Initializes the model if it's specified as a path or string.
    - Applies patches to fix tied tensor issues and modifies `save_pretrained`
        behavior.
    - Initializes the processor if specified as a path or `None`.
    - Sets the minimum tokens per module if `data_args` are provided.
    Raises:
        FileNotFoundError: If the model or processor path is invalid.
    """
    _warn_tied_embeddings(model_args.tie_word_embeddings)

    # Initialize model
    if isinstance(model_args.model, str):
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
    model_args: "ModelArguments",
    output_dir: Optional[str] = None,
):
    """
    Saves the model and tokenizer/processor to the output directory.

    If the `output_dir` is not the default directory, the method resets lifecycle
    actions. The model is saved in a compressed format if specified in `model_args`.
    Additionally, the tokenizer or processor, if available, is also saved.

    Raises:
        ValueError: If saving fails due to an invalid `output_dir` or other issues.
    """
    if output_dir is not None:
        model_args.model.save_pretrained(
            output_dir,
            save_compressed=model_args.save_compressed,
        )
        if model_args.processor:
            model_args.processor.save_pretrained(output_dir)

    logger.warning(
        "Optimized model is not saved. To save, please provide",
        "`output_dir` as input arg.",
        "Ex. `oneshot(..., output_dir=...)`",
    )


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

    # Fallback to CPU if GPU requested and not available
    model_args.oneshot_device = fallback_to_cpu(model_args.oneshot_device)

    # Trainer handles device assignment for FSDP and training, don't do mapping here
    # if running oneshot outside of FSDP, apply user device settings

    fsdp_enabled = os.environ.get("ACCELERATE_USE_FSDP", "false") == "true"

    device_map = model_args.oneshot_device
    if not fsdp_enabled and training_args is not None and training_args.do_train:
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

    # optimized models must be decompressed to carry out oneshot/train/etc
    if is_model_ct_quantized_from_path(model_path):
        model_kwargs["quantization_config"] = CompressedTensorsConfig(
            run_compressed=False
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **model_kwargs,
    )
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
