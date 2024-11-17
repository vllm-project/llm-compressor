from test.testing_utils import preprocess_tokenize_dataset

from datasets import load_dataset
from transformers import AutoTokenizer

from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
from llmcompressor.transformers import SparseAutoModelForCausalLM, oneshot


def run_oneshot_for_e2e_testing(
    model: str,
    device: str,
    oneshot_kwargs: dict,
    num_calibration_samples: int,
    max_seq_length: int,
    dataset_id: str,
    save_dir: str,
    recipe: str,
    dataset_split: str,
    dataset_config: str,
    scheme: str,
    quant_type: str,
):
    # Load model.
    loaded_model = SparseAutoModelForCausalLM.from_pretrained(
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

    if save_dir is None:
        save_dir = model.split("/")[1] + f"-{scheme}"

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
    print("ONESHOT KWARGS", oneshot_kwargs)
    oneshot(
        **oneshot_kwargs,
        clear_sparse_session=True,
        oneshot_device=device,
    )
    oneshot_kwargs["model"].save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    print("================= UPLOADING TO HUB ======================")
    oneshot_kwargs["model"].push_to_hub(f"nm-testing/{save_dir}-e2e")
    tokenizer.push_to_hub(f"nm-testing/{save_dir}-e2e")

    return save_dir
