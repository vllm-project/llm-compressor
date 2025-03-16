from typing import List, Optional, Union

from torch.utils.data import DataLoader

from llmcompressor.args.model_arguments import ModelArguments
from llmcompressor.core import State
from llmcompressor.core.llmcompressor.events_mixin import EventsMixin
from llmcompressor.core.llmcompressor.train import HFSFTMixin
from llmcompressor.core.llmcompressor.utils import (
    LCDatasetArguments,
    check_for_calibration_data,
    get_modifiers_from_recipe,
    parse_args,
    prepare_models,
    resolve_calibration_pipeline,
)
from llmcompressor.datasets.utils import get_calibration_dataloader
from llmcompressor.modifiers import Modifier
from llmcompressor.recipe import RecipeInput
from llmcompressor.typing import DatasetType, ModelInput
from llmcompressor.utils.singleton import SingletonMixin


class LLMCompressor(SingletonMixin, EventsMixin, HFSFTMixin):
    state: State
    modifiers: List[Modifier]
    calibration_loader: Optional[DataLoader] = None

    def __init__(self, model: ModelInput, recipe: RecipeInput, **kwargs):
        model_args = parse_args(ModelArguments, model=model, **kwargs)

        self.modifiers = get_modifiers_from_recipe(recipe)

        model, teacher, processor = prepare_models(model_args)
        self.state = State(model=model, teacher_model=teacher, processor=processor)

    def set_calibration_dataset(self, dataset: Union[str, DatasetType], **kwargs):
        dataset_args = parse_args(LCDatasetArguments, dataset=dataset, **kwargs)

        # temporary hack to support better interface
        if dataset_args.split is not None:
            dataset_args.splits = {"calibration": dataset_args.split}

        self.calibration_loader = get_calibration_dataloader(
            dataset_args, self.state.processor
        )

    def post_train(self, calibration_pipeline: Optional[str] = None):
        check_for_calibration_data(self.modifiers, self.calibration_loader)
        pipeline_fn, pipeline_kwargs = resolve_calibration_pipeline(
            calibration_pipeline, self.modifiers
        )

        self.initialize()
        pipeline_fn(self.state.model, self.calibration_loader, **pipeline_kwargs)
        self.finalize()
