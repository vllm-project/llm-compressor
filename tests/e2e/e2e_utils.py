from datasets import load_dataset
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
from llmcompressor.transformers import oneshot
from tests.testing_utils import preprocess_tokenize_dataset


def run_oneshot_for_e2e_testing(
    model: str,
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
    loaded_model = AutoModelForCausalLM.from_pretrained(
        model, device_map=device, torch_dtype="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model)

    if dataset_id:
        ds = load_dataset(dataset_id, name=dataset_config, split=dataset_split)
        ds = ds.shuffle(seed=42).select(range(num_calibration_samples))
        ds = preprocess_tokenize_dataset(ds, tokenizer, max_seq_length)
        oneshot_kwargs["dataset"] = ds
        oneshot_kwargs["max_seq_length"] = max_seq_length
        oneshot_kwargs["num_calibration_samples"] = num_calibration_samples

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
    oneshot(
        **oneshot_kwargs,
        oneshot_device=device,
    )
    return oneshot_kwargs["model"], tokenizer
