from typing import List, Optional, Union, Tuple, Type, Dict, Any

from transformers import PreTrainedModel

# core
from llmcompressor.utils.singleton import SingletonMixin
from llmcompressor.core import State
from llmcompressor.modifiers import Modifier
from llmcompressor.recipe import RecipeInput

# mixins
from llmcompressor.core.events.events_mixin import EventsMixin
from llmcompressor.core.train import HFSFTMixin

# todo: move
from llmcompressor.datasets.utils import get_calibration_dataloader

from llmcompressor.core.utils import parse_args, LCDatasetArguments, LCModelArguments, get_modifiers_from_recipe, prepare_models, infer_calibration_pipeline


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
        model_args: LCModelArguments = parse_args(LCModelArguments, model=model, recipe=recipe, **model_kwargs)
        self.modifiers = get_modifiers_from_recipe(recipe)

        model, teacher_model, processor = prepare_models(model_args)

        self.state = State(
            model=model,
            teacher_model=teacher_model,
            processor=processor
        )

    def set_calibration_dataset(self, dataset: "DatasetInput", **dataset_kwargs):
        dataset_args: LCDatasetArguments = parse_args(LCDatasetArguments, dataset=dataset, **dataset_kwargs)

        # hack
        dataset_args.splits = {"calibration": dataset_args.split}

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
