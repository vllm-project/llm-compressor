import pytest
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot, train
from tests.testing_utils import parse_params, requires_gpu

CONFIGS_DIRECTORY = "tests/llmcompressor/transformers/finetune/finetune_tokenizer"


@pytest.mark.integration
@requires_gpu
@pytest.mark.parametrize("config", parse_params(CONFIGS_DIRECTORY))
def test_oneshot_and_finetune_with_tokenizer(config, tmp_path):
    model = config["model"]
    dataset = config["dataset"]
    dataset_config_name = config["dataset_config_name"]

    output = tmp_path / "sparsity_finetune_output"
    # finetune workflows in general seem to have trouble with multi-gpus
    # use just one atm

    recipe_str = "tests/llmcompressor/transformers/finetune/test_alternate_recipe.yaml"
    tokenizer = AutoTokenizer.from_pretrained(
        model,
    )
    model_loaded = AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto")

    dataset_loaded = load_dataset(dataset, dataset_config_name, split="train[:50%]")

    concatenate_data = True
    run_stages = True
    max_steps = 50
    splits = {"train": "train[:50%]", "calibration": "train[50%:60%]"}

    model_and_data_kwargs = dict(
        dataset=dataset_loaded,
        dataset_config_name=dataset_config_name,
        recipe=recipe_str,
        concatenate_data=concatenate_data,
        splits=splits,
        tokenizer=tokenizer,
    )

    oneshot_model = oneshot(
        model=model_loaded,
        **model_and_data_kwargs,
        stage="test_oneshot_stage",
    )

    finetune_model = train(
        run_stages=run_stages,
        model=oneshot_model,
        max_steps=max_steps,
        stage="test_train_stage",
        **model_and_data_kwargs,
        output_dir=output,
    )

    input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to("cuda")
    output = finetune_model.generate(input_ids, max_new_tokens=20)
    print(tokenizer.decode(output[0]))
