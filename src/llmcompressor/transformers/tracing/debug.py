from typing import List, Type, Union, Optional, Dict

import argparse

import torch
import transformers
from transformers import AutoProcessor, PreTrainedModel

from llmcompressor.transformers import tracing
from llmcompressor.utils.pytorch.module import get_no_split_params
from llmcompressor.pipelines.sequential.helpers import trace_subgraphs
from llmcompressor.transformers import DataTrainingArguments, TextGenerationDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Trace a model into subgraphs")
    parser.add_argument("--model_id", type=str, required=True, help="The stub of the model to load")  # noqa: E501
    parser.add_argument("--model_class", type=str, required=True, help="The class name of the model")  # noqa: E501
    parser.add_argument("--sequential_targets", type=str, nargs="*", default=None, metavar="TARGET", help="List of targets for sequential tracing")  # noqa: E501
    parser.add_argument("--ignore", type=str, nargs="*", default=[], metavar="PATTERN", help="List of patterns to ignore during tracing")  # noqa: E501
    parser.add_argument("--modality", type=str, default="text", help="Modality of calibration dataset, defaults to text")  # noqa: E501
    return parser.parse_args()


def trace(
    model_id: str,
    model_class: Type[PreTrainedModel],
    sequential_targets: Optional[Union[List[str], str]] = None,
    ignore: Union[List[str], str] = [],
    modality: str = "text",
):
    """
    Debug traceability by tracing a pre-trained model into subgraphs

    :param model_id: stub of the model to load
    :param model_class: class constructor of the pre-trained model. Can use either
        HF transformers classes or `Traceable` classes defined by LLM Compressor
    :param sequential_targets: targets for sequential tracing, defaults to automatic
        inference
    :param ignore: patterns to ignore during tracing
    :param modality: data modality for dummy tracing data, defaults to 'text'

    Example usage from CLI
    llmcompressor.trace \
        --model_id Qwen/Qwen2-VL-2B-Instruct \
        --model_class Qwen2VLForConditionalGeneration \
        --sequential_targets Qwen2VLDecoderLayer \
        --ignore "lm_head" "re:visual.*" \
        --modality text
    """
    # Load model
    model = model_class.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    print("Loaded model")

    # Prepare sample data
    data_args = DataTrainingArguments(**get_dataset_kwargs(modality))
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
        f"    model_class={model_class.__name__}\n"
        f"    dataset={data_args.dataset}\n"
        f"    split={dataset.split}\n"
        f"    inputs={sample_input.keys()}\n"
        f"    sequential_targets={sequential_targets}\n"
        f"    ignore={ignore}\n"
    )
    subgraphs = trace_subgraphs(model, sample_input, sequential_targets, ignore)
    print(f"Successfully traced model into {len(subgraphs)} subgraphs!\n")


def get_model_class(model_class: str) -> Type[PreTrainedModel]:
    model_cls = getattr(tracing, model_class, getattr(transformers, model_class, None))
    if model_cls is None:
        raise ValueError(f"Could not import model class {model_class}")

    return model_cls


def get_dataset_kwargs(modality: str) -> Dict[str, str]:
    dataset_kwargs = {
        "text": {
            "dataset": "ultrachat-200k",
            "splits": {"calibration": "test_sft[:1]"},
        },
        "vision": {
            "dataset": "flickr",
            "splits": {"calibration": "test[:1]"},
        },
        "audio": {
            "dataset": "peoples_speech",
            "splits": {"calibration": "test[:1]"},
        },
    }

    if modality not in dataset_kwargs:
        raise ValueError(f"Modality must be one of {list(dataset_kwargs.keys())}")

    return dataset_kwargs[modality]


def main():
    args = parse_args()

    trace(
        model_id=args.model_id,
        model_class=get_model_class(args.model_class),
        sequential_targets=args.sequential_targets,
        ignore=args.ignore,
        modality=args.modality,
    )


if __name__ == "__main__":
    main()
