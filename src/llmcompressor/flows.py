from typing import Dict, Optional, Union


def quantize_w8a8(
    model_id: str,
    dataset: str,
    output_dir: str,
    splits: Dict[str, str],
    max_seq_length: int,
    pad_to_max_length: bool,
    num_calibration_samples: int,
    smoothquant_kwargs: Optional[Dict[str, Union[int, float, str]]] = None,
    gptq_kwargs: Optional[Dict[str, Union[int, float, str]]] = None,
) -> None:
    """
    Apply W8A8 and SmoothQuant + GPTQ quantization to a model using the oneshot method.

    :param model_id: The HF model id, or local path to the model to be quantized.
    :param dataset: The dataset to be used for calibration.
    :param output_dir: The directory where the quantized model will be saved.
    :param splits: The data splits to be used for calibration
        ex. {"calibration": "train_gen[:5%]"}
    :param max_seq_length: The maximum sequence length for the model.
    :param pad_to_max_length: Whether to pad sequences to the maximum length.
    :param num_calibration_samples: The number of samples to be used for calibration.
    :param smoothquant_kwargs: Optional dictionary of keyword arguments for
        the SmoothQuantModifier.
    :param gptq_kwargs: Optional dictionary of keyword arguments for the GPTQModifier.
    """
    from llmcompressor.modifiers.quantization.gptq import GPTQModifier
    from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
    from llmcompressor.transformers import SparseAutoModelForCausalLM, oneshot

    default_gptq_kwargs = dict(
        targets="Linear", scheme="W8A8", ignore=["lm_head"], sequential_update=False
    )
    gptq_kwargs = {**default_gptq_kwargs, **(gptq_kwargs or {})}
    # Define the quantization recipe with SmoothQuant and GPTQ
    recipe = [
        SmoothQuantModifier(**(smoothquant_kwargs or {})),
        GPTQModifier(**gptq_kwargs),
    ]

    # Load the model
    model = SparseAutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype="auto"
    )

    # Apply the quantization
    oneshot(
        model=model,
        dataset=dataset,
        recipe=recipe,
        output_dir=output_dir,
        splits=splits,
        max_seq_length=max_seq_length,
        pad_to_max_length=pad_to_max_length,
        num_calibration_samples=num_calibration_samples,
        save_compressed=True,
    )
