from dataclasses import fields

from llmcompressor.args import DatasetArguments, ModelArguments, PostTrainArguments
from llmcompressor.core import LLMCompressor
from llmcompressor.utils import ArgumentParser

__all__ = ["oneshot", "post_train"]


def oneshot(**kwargs):
    post_train(**kwargs)


def post_train(**kwargs):
    parser = ArgumentParser((ModelArguments, DatasetArguments, PostTrainArguments))
    model_args, dataset_args, training_args = parser.parse_dict(kwargs)

    # dataclasses.asdict will run recurisvely, accidentally affecting some dataclass values like DefaultDataCollator
    def to_dict(dataclass):
        return {
            field.name: getattr(dataclass, field.name) for field in fields(dataclass)
        }

    compressor = LLMCompressor(**to_dict(model_args))
    compressor.set_calibration_dataset(**to_dict(dataset_args))
    compressor.post_train(**to_dict(training_args))
