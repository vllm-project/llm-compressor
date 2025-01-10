from typing import List, Type, Union, Optional

import argparse

import torch
from transformers import AutoProcessor
import transformers

from llmcompressor.pipelines.sequential.helpers import trace_subgraphs
from llmcompressor.transformers import DataTrainingArguments, TextGenerationDataset
from llmcompressor.utils.pytorch.module import get_no_split_params
from llmcompressor.transformers import tracing


def attempt_trace(
    model_id: str,
    model_class: Type,
    multimodal_data: bool,
    sequential_targets: Optional[Union[List[str], str]] = None,
    ignore: Union[List[str], str] = [],
):
    # Load model
    model = model_class.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    print("Loaded model")

    # Prepare sample data
    data_args = DataTrainingArguments(
        dataset="flickr" if multimodal_data else "ultrachat-200k",
        splits={"calibration": "test[:1]" if multimodal_data else "test_sft[:1]"}
    )
    dataset = TextGenerationDataset.load_from_registry(
        data_args.dataset,
        data_args=data_args,
        split=data_args.splits["calibration"],
        processor=processor,
    )(add_labels=False)
    sample_input = next(iter(dataset))
    sample_input = {k: torch.tensor(v) for k, v in sample_input.items()}
    print("Loaded sample data")

    # infer sequential targets
    if sequential_targets is None:
        sequential_targets = get_no_split_params(model)
    if isinstance(sequential_targets, str):
        sequential_targets = [sequential_targets]

    # infer ignore
    if isinstance(ignore, str):
        ignore = [ignore]

    # Attempt trace
    print(
        "\nAttempting trace\n"
        f"    model_id={model_id}\n"
        f"    dataset={data_args.dataset}\n"
        f"    split={dataset.split}\n"
        f"    inputs={sample_input.keys()}\n"
        f"    sequential_targets={sequential_targets}\n"
        f"    ignore={ignore}\n"
    )
    subgraphs = trace_subgraphs(model, sample_input, sequential_targets, ignore)
    print(f"Successfully traced model into {len(subgraphs)} subgraphs!\n")


def get_model_class(model_class: str):
    model_cls = getattr(tracing, model_class, getattr(transformers, model_class, None))
    if model_cls is None:
        raise ValueError(f"Could not import model class {model_class}")

    return model_cls


def parse_args():
    parser = argparse.ArgumentParser(description="Trace a model into subgraphs.")
    parser.add_argument("--model_id", type=str, required=True, help="The model ID to load.")  # noqa: E501
    parser.add_argument("--model_class", type=str, required=True, help="The class name of the model.")  # noqa: E501
    parser.add_argument("--multimodal_data", action="store_true", help="Use multimodal data if set.")  # noqa: E501
    parser.add_argument("--sequential_targets", type=str, nargs="*", default=None, metavar="TARGET", help="List of targets for sequential tracing.")  # noqa: E501
    parser.add_argument("--ignore", type=str, nargs="*", default=[], metavar="PATTERN", help="List of patterns to ignore during tracing.")  # noqa: E501
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    attempt_trace(
        model_id=args.model_id,
        model_class=get_model_class(args.model_class),
        multimodal_data=args.multimodal_data,
        sequential_targets=args.sequential_targets,
        ignore=args.ignore,
    )
