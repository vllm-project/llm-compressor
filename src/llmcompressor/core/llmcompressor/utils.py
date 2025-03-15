from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

from loguru import logger
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    HfArgumentParser,
    PreTrainedModel,
)
from transformers.utils.quantization_config import CompressedTensorsConfig

from llmcompressor.args.dataset_arguments import DatasetArguments
from llmcompressor.args.model_arguments import ModelArguments
from llmcompressor.core.llmcompressor.globals import get_model
from llmcompressor.entrypoints.utils import (
    _warn_tied_embeddings,
    initialize_processor_from_path,
)
from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.factory import ModifierFactory
from llmcompressor.modifiers.obcq.sgpt_mixin import SparsityModifierMixin
from llmcompressor.modifiers.quantization.gptq import GPTQModifier
from llmcompressor.pipelines import get_pipeline
from llmcompressor.pytorch.model_load.helpers import parse_dtype
from llmcompressor.transformers.sparsification.compressed_tensors_utils import (
    patch_tied_tensors_bug,
    untie_weights,
)
from llmcompressor.transformers.utils.helpers import is_model_ct_quantized_from_path
from llmcompressor.utils.pytorch.module import get_no_split_params

""" llmcompressor.Args """


@dataclass
class LCDatasetArguments(DatasetArguments):
    split: Optional[str] = field(default=None)


T = TypeVar("T")


def parse_args(dataclass: Type[T], **kwargs) -> T:
    parser = HfArgumentParser(dataclass)
    return parser.parse_dict(kwargs)[0]


""" llmcompressor.recipe """


def get_modifiers_from_recipe(
    recipe: Union[str, List[Modifier], Modifier],
) -> List[Modifier]:
    ModifierFactory.refresh()

    if isinstance(recipe, str):
        raise ValueError()

    if isinstance(recipe, Modifier):
        recipe = [recipe]

    return recipe


""" llmcompressor.pytorch.model_load """


def prepare_models(model_args: ModelArguments):
    # Initialize model
    if isinstance(model_args.model, str):
        model_args.model = initialize_model_from_path(model_args.model, model_args)

    # Initialize teacher
    if isinstance(model_args.distill_teacher, str):
        model_args.distill_teacher = initialize_model_from_path(
            model_args.distill_teacher, model_args
        )

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


def initialize_model_from_path(
    model_path: str, model_args: ModelArguments
) -> PreTrainedModel:
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
        model.seqlen = model_kwargs[
            "sequence_length"
        ]  # TODO: Pretty sure the seqlen attribute is never used/ doesn't exist

    return model


""" llmcompressor.pipelines """


def resolve_calibration_pipeline(
    user_selection: Optional[str], modifiers: List[Modifier]
) -> Tuple[Callable[[Any], Any], Dict[str, Any]]:
    # infer pipeline from modifiers
    inferred_selection = infer_pipeline_from_modifiers(modifiers)

    # resolve with user selection
    pipeline = resolve_pipeline(user_selection, inferred_selection)

    # resolve pipeline kwargs
    pipeline_kwargs = infer_pipeline_kwargs(pipeline, modifiers)

    return get_pipeline(pipeline), pipeline_kwargs


def infer_pipeline_from_modifiers(modifiers: List[Modifier]) -> str:
    has_sequential_modifier = any(
        isinstance(mod, (GPTQModifier, SparsityModifierMixin)) for mod in modifiers
    )

    if has_sequential_modifier:
        return "sequential"

    return "basic"


def resolve_pipeline(user_selection: str, inferred_selection: str):
    if user_selection is not None and user_selection != inferred_selection:
        # raise some warning
        return user_selection
    else:
        return inferred_selection


def infer_pipeline_kwargs(pipeline: str, modifiers: List[Modifier]) -> Dict[str, Any]:
    if pipeline == "sequential":
        # naive resolution: use first modifier found

        for modifier in modifiers:
            if isinstance(modifier, (GPTQModifier, SparsityModifierMixin)):
                model = get_model()

                # infer sequential targets
                if modifier.sequential_targets is None:
                    sequential_targets = get_no_split_params(model)
                if isinstance(modifier.sequential_targets, str):
                    sequential_targets = [modifier.sequential_targets]

                return {
                    "sequential_targets": sequential_targets,
                    "ignore": modifier.ignore,
                    "callback_modifier": modifier,
                }
