from typing import List, Type, Union, Optional

import torch
from transformers import AutoProcessor

from llmcompressor.pipelines.sequential.helpers import trace_subgraphs
from llmcompressor.transformers import DataTrainingArguments, TextGenerationDataset
from llmcompressor.utils.pytorch.module import get_no_split_params


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


if __name__ == "__main__":
    # TODO: use argparse
    from llmcompressor.transformers.tracing import TraceableMllamaForConditionalGeneration
    TraceableMllamaForConditionalGeneration
    from llmcompressor.transformers.tracing.debug import attempt_trace

    attempt_trace(
        model_id="meta-llama/Llama-3.2-11B-Vision-Instruct",
        model_class=TraceableMllamaForConditionalGeneration,
        multimodal_data=True,
        sequential_targets=None,
        ignore=["re:.*lm_head", "re:multi_modal_projector.*", "re:vision_model.*"],
    )