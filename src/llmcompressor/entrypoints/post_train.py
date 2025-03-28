from transformers import HfArgumentParser

from llmcompressor.args import DatasetArguments, ModelArguments, PostTrainArguments
from llmcompressor.core.llmcompressor.llmcompressor import LLMCompressor

__all__ = ["oneshot", "post_train"]


def oneshot(**kwargs):
    post_train(**kwargs)


def post_train(**kwargs):
    parser = HfArgumentParser((ModelArguments, DatasetArguments, PostTrainArguments))
    model_args, dataset_args, training_args = parser.parse_dict(kwargs)

    compressor = LLMCompressor(**model_args)
    compressor.set_calibration_dataset(**dataset_args)
    compressor.train(**training_args)
