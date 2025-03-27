from typing import List, Optional, Union

from torch.utils.data import DataLoader

from llmcompressor.args.post_train_arguments import PostTrainArguments
from llmcompressor.core import State
from llmcompressor.core.llmcompressor.events_mixin import EventsMixin
from llmcompressor.core.llmcompressor.train import HFSFTMixin
from llmcompressor.core.llmcompressor.utils import (
    LCDatasetArguments,
    LCModelArguments,
    get_modifiers_from_recipe,
    prepare_models,
)
from llmcompressor.datasets.utils import get_calibration_dataloader
from llmcompressor.modifiers import Modifier
from llmcompressor.pipelines.registry import get_pipeline_fn
from llmcompressor.pytorch.model_load.helpers import save_checkpoint
from llmcompressor.typing import DatasetType, ModelInput, RecipeInput
from llmcompressor.utils.singleton import SingletonMixin


class LLMCompressor(SingletonMixin, EventsMixin, HFSFTMixin):
    state: State
    modifiers: List[Modifier]
    calibration_loader: Optional[DataLoader] = None

    def __init__(self, model: ModelInput, recipe: RecipeInput, **kwargs):
        args = LCModelArguments(model=model, recipe=recipe, **kwargs)

        self.modifiers = get_modifiers_from_recipe(args.recipe)

        model, teacher, processor = prepare_models(args)
        self.state = State(model=model, teacher_model=teacher, processor=processor)

    def set_calibration_dataset(self, dataset: Union[str, DatasetType], **kwargs):
        args = LCDatasetArguments(dataset=dataset, **kwargs)

        # temporary hack to support better interface
        if args.split is not None:
            args.splits = {"calibration": args.split}

        self.calibration_loader = get_calibration_dataloader(args, self.state.processor)

    def post_train(
        self, pipeline: Optional[str] = "independent", save_path: Optional[str] = None
    ):
        args = PostTrainArguments(pipeline=pipeline, save_path=save_path)

        _, pipeline_fn = get_pipeline_fn(args.pipeline, self.modifiers)
        pipeline_fn(self.state.model, self.calibration_loader, args)

        self.finalize()

        if args.save_path is not None:
            save_checkpoint(args.save_path, self.state.model, self.state.processor)
