import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from tests.llmcompressor.transformers.oneshot.dataset_processing import get_data_utils
from tests.testing_utils import parse_params

CONFIGS_DIRECTORY = "tests/llmcompressor/transformers/oneshot/oneshot_configs"

# TODO: Seems better to mark test type (smoke, sanity, regression) as a marker as
# opposed to using a field in the config file?


@pytest.fixture(params=parse_params(CONFIGS_DIRECTORY))
def one_shot_args(request):
    config = request.param
    # config: {model, dataset, recipe, dataset_config_name, tokenize}

    tokenizer = AutoTokenizer.from_pretrained(config["model"])
    model = AutoModelForCausalLM.from_pretrained(config["model"])

    if config["tokenize"]:
        data_utils = get_data_utils(config.get("dataset"))

        def wrapped_preprocess_func(sample):
            preprocess_func = data_utils.get("preprocess")
            return tokenizer(
                preprocess_func(sample), padding=False, max_length=512, truncation=True
            )

        loaded_dataset = data_utils.get("dataload")()
        dataset = loaded_dataset.map(wrapped_preprocess_func)
        tokenizer = None
    else:
        dataset = config["dataset"]

    args = dict(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        recipe=config["recipe"],
        dataset_config_name=config.get("dataset_config_name"),
    )

    return args


@pytest.mark.smoke
@pytest.mark.integration
def test_one_shot_inputs(one_shot_args, tmp_path):
    oneshot(
        **one_shot_args,
        output_dir=tmp_path,
        num_calibration_samples=10,
        pad_to_max_length=False,
    )
