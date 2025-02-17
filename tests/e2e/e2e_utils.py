import torch
from datasets import load_dataset
from loguru import logger
from transformers import AutoProcessor

from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
from llmcompressor.transformers import oneshot
from tests.test_timer.timer_utils import log_time
from llmcompressor.transformers.tracing import get_model_class
from tests.testing_utils import process_dataset


@log_time
def _load_model_and_processor(
    model: str,
    model_class: str,
    device: str,
):
    pretrained_model_class = get_model_class(model_class)
    loaded_model = pretrained_model_class.from_pretrained(
        model,
        device_map=device,
        torch_dtype="auto",
        trust_remote_code=True,
        _attn_implementation="eager",
    )
    processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)
    return loaded_model, processor


@log_time
def _run_oneshot(device: str, **oneshot_kwargs):
    oneshot(
        **oneshot_kwargs,
        oneshot_device=device,
    )


def run_oneshot_for_e2e_testing(
    model: str,
    model_class: str,
    device: str,
    num_calibration_samples: int,
    max_seq_length: int,
    dataset_id: str,
    recipe: str,
    dataset_split: str,
    dataset_config: str,
    scheme: str,
    quant_type: str,
):
    # Load model.
    oneshot_kwargs = {}

    loaded_model, processor = _load_model_and_processor(model=model, model_class=model_class, device=device)

    if dataset_id:
        ds = load_dataset(dataset_id, name=dataset_config, split=dataset_split)
        ds = ds.shuffle(seed=42).select(range(num_calibration_samples))
        ds = process_dataset(ds, processor, max_seq_length)
        oneshot_kwargs["dataset"] = ds
        oneshot_kwargs["max_seq_length"] = max_seq_length
        oneshot_kwargs["num_calibration_samples"] = num_calibration_samples

        # TODO better conditional on when multimodal data-collator should be added
        if "flickr30k" in dataset_id:
            # Define a oneshot data collator for multimodal inputs.
            def data_collator(batch):
                assert len(batch) == 1
                return {key: torch.tensor(value) for key, value in batch[0].items()}

            oneshot_kwargs["data_collator"] = data_collator

    oneshot_kwargs["model"] = loaded_model
    if recipe:
        oneshot_kwargs["recipe"] = recipe
    else:
        # Test assumes that if a recipe was not provided, using
        # a compatible preset sceme
        if quant_type == "GPTQ":
            oneshot_kwargs["recipe"] = GPTQModifier(
                targets="Linear", scheme=scheme, ignore=["lm_head"]
            )
        else:
            oneshot_kwargs["recipe"] = QuantizationModifier(
                targets="Linear", scheme=scheme, ignore=["lm_head"]
            )

    # Apply quantization.
    logger.info("ONESHOT KWARGS", oneshot_kwargs)
    _run_oneshot(device=device, **oneshot_kwargs)

    return oneshot_kwargs["model"], processor
