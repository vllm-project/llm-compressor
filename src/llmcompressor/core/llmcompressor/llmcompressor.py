from typing import List, Optional, Union, Tuple, Type, Dict, Any

from transformers import PreTrainedModel

# core
from llmcompressor.utils.singleton import SingletonMixin
from llmcompressor.core import State
from llmcompressor.modifiers import Modifier
from llmcompressor.recipe import RecipeInput

# mixins
from llmcompressor.core.llmcompressor.events_mixin import EventsMixin
from llmcompressor.core.llmcompressor.train import HFSFTMixin

# todo: move
from llmcompressor.args.dataset_arguments import DatasetArguments
from llmcompressor.datasets.utils import get_calibration_dataloader

from llmcompressor.core.llmcompressor.utils import parse_args, LCModelArguments, get_modifiers_from_recipe, prepare_models, infer_calibration_pipeline


class LLMCompressor(SingletonMixin, EventsMixin, HFSFTMixin):
    state: State
    modifiers: List[Modifier]
    calibration_loader: Optional["DatasetType"] = None
    
    def __init__(
        self,
        model: Union[PreTrainedModel, str],
        recipe: RecipeInput,
        **model_kwargs
    ):
        model_args = parse_args(LCModelArguments, model=model, recipe=recipe, **model_kwargs)
        self.modifiers = get_modifiers_from_recipe(recipe)

        model, teacher_model, processor = prepare_models(model_args)

        self.state = State(
            model=model,
            teacher_model=teacher_model,
            processor=processor
        )

    def set_calibration_dataset(self, dataset: "DatasetInput", **dataset_kwargs):
        dataset_args = parse_args(DatasetArguments, dataset=dataset, **dataset_kwargs)

        # preprocess calibration dataset
        self.calibration_loader = get_calibration_dataloader(dataset_args, self.state.processor)
        
    def post_train(self, calibration_pipeline: Optional[str] = None, **pipeline_kwargs):
        self.initialize()

        # run calibration
        pipeline_fn = infer_calibration_pipeline(calibration_pipeline, self.modifiers)
        pipeline_fn(self.state.model, self.calibration_loader, **pipeline_kwargs)

        self.finalize()

    def update_state(self, **kwargs):
        self.state.update(**kwargs)

        # if future modifiers require update, do that update here
