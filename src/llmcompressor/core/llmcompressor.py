from typing import TYPE_CHECKING, List, Optional

from torch.utils.data import DataLoader

from llmcompressor.args import DatasetArguments, ModelArguments, PostTrainArguments
from llmcompressor.datasets.utils import get_calibration_dataloader
from llmcompressor.pipelines.registry import get_pipeline_fn
from llmcompressor.pytorch.model_load.helpers import save_checkpoint
from llmcompressor.utils.singleton import SingletonMixin

from .events_mixin import EventsMixin
from .state import State
from .train import HFSFTMixin
from .utils import (
    add_dataclass_annotations,
    error_if_requires_calibration_data,
    get_modifiers_from_recipe,
    prepare_models,
)

if TYPE_CHECKING:
    from llmcompressor.modifiers import Modifier


class LLMCompressor(SingletonMixin, EventsMixin, HFSFTMixin):
    state: State
    modifiers: List["Modifier"]
    calibration_loader: Optional[DataLoader] = None

    @add_dataclass_annotations(ModelArguments)
    def __init__(self, *args, **kwargs):
        args = ModelArguments(*args, **kwargs)

        self.modifiers = get_modifiers_from_recipe(args.recipe)
        model, teacher, processor = prepare_models(args)
        self.state = State(model=model, teacher_model=teacher, processor=processor)

    @add_dataclass_annotations(DatasetArguments)
    def set_calibration_dataset(self, *args, **kwargs):
        args = DatasetArguments(*args, **kwargs)

        self.calibration_loader = get_calibration_dataloader(args, self.state.processor)

    @add_dataclass_annotations(PostTrainArguments)
    def post_train(self, *args, **kwargs):
        args = PostTrainArguments(*args, **kwargs)
        error_if_requires_calibration_data(self.modifiers, self.calibration_loader)

        _, pipeline_fn = get_pipeline_fn(args.pipeline, self.modifiers)
        pipeline_fn(self.state.model, self.calibration_loader, args)

        self.finalize()

        if args.output_dir is not None:
            save_checkpoint(args.output_dir, self.state.model, self.state.processor)
