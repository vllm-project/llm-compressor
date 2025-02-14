from transformers import AutoModelForCausalLM

from llmcompressor import Oneshot
from llmcompressor.entrypoints.oneshot import parse_oneshot_args


def test_oneshot_from_args():
    # Select model and load it.
    stub = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model = AutoModelForCausalLM.from_pretrained(stub)
    dataset = "HuggingFaceH4/ultrachat_200k"

    NUM_CALIBRATION_SAMPLES = 512
    MAX_SEQUENCE_LENGTH = 2048

    recipe = "foo_recipe"

    output_dir = "bar_output_dir"

    model_args, data_args, recipe_args, output_dir = parse_oneshot_args(
        model=model,
        dataset=dataset,
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
        output_dir=output_dir,
    )

    oneshot = Oneshot.from_args(model_args, data_args, recipe_args, output_dir)
    assert oneshot.model == model
    assert oneshot.model_args is model_args
    assert oneshot.data_args is data_args
    assert oneshot.recipe_args is recipe_args
    assert oneshot.model_args is model_args
    assert oneshot.output_dir is output_dir
