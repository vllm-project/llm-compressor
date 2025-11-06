from typing import List, Type, Union, Optional, Dict, Tuple, Any

import argparse
from contextlib import nullcontext

import torch
import transformers
from transformers import AutoProcessor, PreTrainedModel

from llmcompressor.utils.pytorch.module import get_no_split_params
from llmcompressor.pipelines.sequential.helpers import trace_subgraphs, Subgraph
from llmcompressor.transformers import TextGenerationDataset
from llmcompressor.args import DatasetArguments

from llmcompressor.utils.dev import skip_weights_download

__all__ = ["trace"]


def parse_args():
    parser = argparse.ArgumentParser(description="Trace a model into subgraphs")
    parser.add_argument("--model_id", type=str, required=True, help="The stub of the model to load")  # noqa: E501
    parser.add_argument("--model_class", type=str, required=True, help="The class name of the model")  # noqa: E501
    parser.add_argument("--sequential_targets", type=str, nargs="*", default=None, metavar="TARGET", help="List of targets for sequential tracing")  # noqa: E501
    parser.add_argument("--ignore", type=str, nargs="*", default=DatasetArguments().tracing_ignore, metavar="PATTERN", help="List of patterns to ignore during tracing")  # noqa: E501
    parser.add_argument("--modality", type=str, default="text", help="Modality of calibration dataset, defaults to text")  # noqa: E501
    parser.add_argument("--trust_remote_code", type=bool, default=False, help="Whether to trust model remote code")  # noqa: E501
    parser.add_argument("--skip_weights", type=bool, default=True, help="Whether to load the model with dummy weights")  # noqa: E501
    parser.add_argument("--device_map", type=str, default="cpu", help="Device to load model and inputs onto")  # noqa: E501
    return parser.parse_args()


def trace(
    model_id: str,
    model_class: Type[PreTrainedModel],
    sequential_targets: Optional[Union[List[str], str]] = None,
    ignore: Union[List[str], str] = DatasetArguments().tracing_ignore,
    modality: str = "text",
    trust_remote_code: bool = True,
    skip_weights: bool = True,
    device_map: Union[str, Dict] = "cpu",
) -> Tuple[PreTrainedModel, List[Subgraph], Dict[str, torch.Tensor]]:
    """
    Debug traceability by tracing a pre-trained model into subgraphs

    :param model_id: stub of the model to load
    :param model_class: class constructor of the pre-trained model. Can use either
        HF transformers classes or `Traceable` classes defined by LLM Compressor
    :param sequential_targets: targets for sequential tracing, defaults to automatic
        inference
    :param ignore: patterns to ignore during tracing
    :param modality: data modality for dummy tracing data, defaults to 'text'
    :param trust_remote_code: trust remote model code

    Example usage from CLI
    llmcompressor.trace \
        --model_id Qwen/Qwen2-VL-2B-Instruct \
        --model_class Qwen2VLForConditionalGeneration \
        --sequential_targets Qwen2VLDecoderLayer \
        --ignore "lm_head" "re:visual.*" \
        --modality text
    """
    # Load model
    with skip_weights_download(model_class) if skip_weights else nullcontext():
        model = model_class.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype="auto",
            trust_remote_code=trust_remote_code,
        )
    processor = AutoProcessor.from_pretrained(
        model_id, trust_remote_code=trust_remote_code
    )
    print("Loaded model")

    # Prepare sample data
    dataset_args = DatasetArguments(**get_dataset_kwargs(modality, ignore))
    dataset = TextGenerationDataset.load_from_registry(
        dataset_args.dataset,
        dataset_args=dataset_args,
        split=dataset_args.splits["calibration"],
        processor=processor,
    )(add_labels=False)
    sample = next(iter(dataset))
    sample = collate_sample(sample, device=model.device)
    print("Loaded sample data")

    # infer sequential targets
    if sequential_targets is None:
        sequential_targets = get_no_split_params(model)
    if isinstance(sequential_targets, str):
        sequential_targets = [sequential_targets]

    # Attempt trace
    print(
        "\nAttempting trace\n"
        f"    model_id={model_id}\n"
        f"    model_class={model_class.__name__}\n"
        f"    dataset={dataset_args.dataset}\n"
        f"    split={dataset.split}\n"
        f"    inputs={sample.keys()}\n"
        f"    sequential_targets={sequential_targets}\n"
        f"    ignore={dataset_args.tracing_ignore}\n"
    )
    subgraphs = trace_subgraphs(
        model, sample, sequential_targets, dataset_args.tracing_ignore
    )
    print(f"Successfully traced model into {len(subgraphs)} subgraphs!\n")

    return model, subgraphs, sample


def get_dataset_kwargs(modality: str, ignore: List[str]) -> Dict[str, str]:
    dataset_kwargs = {
        "text": {
            "dataset": "ultrachat-200k",
            "splits": {"calibration": "test_sft[:1]"},
            "max_seq_length": 4096,
        },
        "vision": {
            "dataset": "flickr",
            "splits": {"calibration": "test[:1]"},
            "max_seq_length": 4096,
        },
        "audio": {
            "dataset": "peoples_speech",
            "splits": {"calibration": "test[:1]"},
            "max_seq_length": 4096,
        },
    }
    common_kwargs = {
        "max_seq_length": 4096,
        "tracing_ignore": ignore,
    }

    if modality not in dataset_kwargs:
        raise ValueError(f"Modality must be one of {list(dataset_kwargs.keys())}")

    return dataset_kwargs[modality] | common_kwargs


def collate_sample(sample: Dict[str, Any], device: str) -> Dict[str, torch.Tensor]:
    for name, value in sample.items():
        if name in ("input_ids", "attention_mask") and torch.tensor(value).ndim == 1:
            sample[name] = torch.tensor([value], device=device)

        else:
            sample[name] = torch.tensor(value, device=device)

    return sample


def main():
    args = parse_args()

    if isinstance(args.ignore, str):
        args.ignore = [args.ignore]

    trace(
        model_id=args.model_id,
        model_class=getattr(transformers, args.model_class),
        sequential_targets=args.sequential_targets,
        ignore=args.ignore,
        modality=args.modality,
        trust_remote_code=args.trust_remote_code,
        skip_weights=args.skip_weights,
        device_map=args.device_map,
    )


if __name__ == "__main__":
    main()
