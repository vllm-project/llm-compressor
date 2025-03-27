from typing import List, Optional

from torch.utils.data import DataLoader

from llmcompressor.args.dataset_arguments import DatasetArguments
from llmcompressor.args.model_arguments import ModelArguments
from llmcompressor.args.post_train_arguments import PostTrainArguments
from llmcompressor.core import State
from llmcompressor.core.llmcompressor.events_mixin import EventsMixin
from llmcompressor.core.llmcompressor.train import HFSFTMixin
from llmcompressor.core.llmcompressor.utils import (
    add_dataclass_annotations,
    get_modifiers_from_recipe,
    prepare_models,
)
from llmcompressor.datasets.utils import get_calibration_dataloader
from llmcompressor.modifiers import Modifier
from llmcompressor.pipelines.registry import get_pipeline_fn
from llmcompressor.pytorch.model_load.helpers import save_checkpoint
from llmcompressor.utils.singleton import SingletonMixin


class LLMCompressor(SingletonMixin, EventsMixin, HFSFTMixin):
    state: State
    modifiers: List[Modifier]
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
        # TODO: check requires calibration data

        _, pipeline_fn = get_pipeline_fn(args.pipeline, self.modifiers)
        pipeline_fn(self.state.model, self.calibration_loader, args)

        self.finalize()

        if args.save_path is not None:
            save_checkpoint(args.save_path, self.state.model, self.state.processor)
