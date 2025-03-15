from typing import Dict, Any, Tuple, Type, Optional, List, Union
from dataclasses import dataclass, field
from loguru import logger

from llmcompressor.typing import Processor
from llmcompressor.modifiers import Modifier
from llmcompressor.recipe import Recipe, RecipeInput

from transformers import HfArgumentParser, AutoModelForCausalLM, PreTrainedModel
from transformers.utils.quantization_config import CompressedTensorsConfig

from llmcompressor.args.model_arguments import ModelArguments
from llmcompressor.args.dataset_arguments import DatasetArguments
from llmcompressor.transformers.utils.helpers import is_model_ct_quantized_from_path
from llmcompressor.entrypoints.utils import _warn_tied_embeddings, initialize_processor_from_path
from llmcompressor.pytorch.model_load.helpers import parse_dtype
from llmcompressor.transformers.sparsification.compressed_tensors_utils import patch_tied_tensors_bug, untie_weights
from llmcompressor.modifiers.quantization.gptq.base import GPTQModifier
from transformers import AutoConfig, AutoModelForCausalLM
from llmcompressor.modifiers.factory import ModifierFactory


@dataclass
class LCModelArguments(ModelArguments):
    recipe: "RecipeInput" = field(default="")


@dataclass
class LCDatasetArguments(DatasetArguments):
    split: Optional[str] = field(default=None)


def parse_args(dataclass: Type, **kwargs) -> Tuple[Any]:  # TODO: replace with custom typed arguments type
    parser = HfArgumentParser(dataclass)
    return parser.parse_dict(kwargs)[0]


def get_modifiers_from_recipe(recipe: Union[str, List[Modifier], Modifier]) -> List[Modifier]:
    ModifierFactory.refresh()
    
    if isinstance(recipe, str):
        raise ValueError()

    if isinstance(recipe, Modifier):
        recipe = [recipe]

    return recipe


def prepare_models(model_args: LCModelArguments):
    # Initialize model
    if isinstance(model_args.model, str):
        model_args.model = initialize_model_from_path(model_args.model, model_args)

    # Initialize teacher
    if isinstance(model_args.distill_teacher, str):
        model_args.distill_teacher = initialize_model_from_path(model_args.distill_teacher, model_args)

    # Initialize processor
    if isinstance(model_args.processor, (str, type(None))):
        model_args.processor = initialize_processor_from_path(
            model_args, model_args.model
        )

    # warnings and patches
    _warn_tied_embeddings(model_args.tie_word_embeddings)
    patch_tied_tensors_bug(model_args.model)  # untie tie_word_embeddings weights
    if model_args.tie_word_embeddings:
        untie_weights(model_args.model)

    # potentially attach this compressor to the model?

    return model_args.model, model_args.distill_teacher, model_args.processor


def initialize_model_from_path(model_path: str, model_args: LCModelArguments) -> PreTrainedModel:
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        trust_remote_code=model_args.trust_remote_code_model,
    )

    # TODO: seems to be redundancy between config and model kwargs
    model_kwargs = {
        "config": config,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "torch_dtype": parse_dtype(model_args.precision),
        "device_map": model_args.oneshot_device or "auto",
        "trust_remote_code": model_args.trust_remote_code_model,
    }

    # for convenience, decompress any CT compressed models
    if is_model_ct_quantized_from_path(model_path):
        logger.warning("Decompressing model")
        model_kwargs["quantization_config"] = CompressedTensorsConfig(
            run_compressed=False
        )

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    if "sequence_length" in model_kwargs:
        model.seqlen = model_kwargs["sequence_length"]  # TODO: Pretty sure the seqlen attribute is never used/ doesn't exist

    return model





def infer_calibration_pipeline(user_selection: Optional[str], modifiers: List[Modifier]):
    from llmcompressor.pipelines import get_pipeline
    
    if GPTQModifier in modifiers:
        inferred_pipeline = "sequential"
    else:
        inferred_pipeline = "basic"

    if user_selection is not None and user_selection != inferred_pipeline:
        # raise some warning
        pass
    
    return get_pipeline(inferred_pipeline)