import pytest
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DefaultDataCollator

from llmcompressor import oneshot
from llmcompressor.args import DatasetArguments
from llmcompressor.pytorch.model_load.helpers import get_session_model
from llmcompressor.pytorch.utils import tensors_to_device
from llmcompressor.transformers.compression.compressed_tensors_utils import (
    get_model_compressor,
)
from llmcompressor.transformers.finetune.data import TextGenerationDataset
from tests.testing_utils import parse_params, requires_gpu

CONFIGS_DIRECTORY = (
    "tests/llmcompressor/transformers/sparsegpt/sparsegpt_configs/completion"
)
GPU_CONFIGS_DIRECTORY = (
    "tests/llmcompressor/transformers/sparsegpt/sparsegpt_configs/completion/gpu"
)


def labeled_dataloader(dataset_name, model_name, num_samples):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset_args = DatasetArguments(
        dataset=dataset_name,
        max_seq_length=512,
        pad_to_max_length=False,
    )
    dataset_manager = TextGenerationDataset.load_from_registry(
        dataset_args.dataset,
        dataset_args=dataset_args,
        split=f"train[:{num_samples}]",
        processor=tokenizer,
    )
    calib_dataset = dataset_manager()
    data_loader = DataLoader(
        calib_dataset, batch_size=1, collate_fn=DefaultDataCollator()
    )

    return data_loader


@requires_gpu
@pytest.mark.integration
@pytest.mark.parametrize("config", parse_params(CONFIGS_DIRECTORY))
def test_oneshot_completion(config, tmp_path):
    _test_oneshot_completion(
        config["model"],
        config["dataset"],
        config["recipe"],
        config["num_samples"],
        None,
        config["perplexity"],
        tmp_path,
    )


@requires_gpu
@pytest.mark.integration
@pytest.mark.parametrize("config", parse_params(GPU_CONFIGS_DIRECTORY))
def test_oneshot_completion_gpu(config, tmp_path):
    model = AutoModelForCausalLM.from_pretrained(
        config["model"], device_map=config["device"], torch_dtype=torch.bfloat16
    )
    _test_oneshot_completion(
        model,
        config["dataset"],
        config["recipe"],
        config["num_samples"],
        config["model"],
        config["perplexity"],
        tmp_path,
    )


def _test_oneshot_completion(
    model, dataset, recipe, num_samples, model_name, perplexity, tmp_path
):
    oneshot(
        model=model,
        dataset=dataset,
        splits={"calibration": f"train[:{num_samples}]"},
        recipe=recipe,
        max_seq_length=512,
        num_calibration_samples=num_samples,
        pad_to_max_length=False,
        output_dir=tmp_path,
        precision="bfloat16",
    )

    first_tiny_model = get_session_model()
    compressor = get_model_compressor(
        model=first_tiny_model,
        save_compressed=True,
        skip_sparsity_compression_stats=False,
    )
    if compressor is not None:
        compressor.decompress_model(first_tiny_model)

    dataset = "open_platypus"

    iter = 10
    if model_name:
        dataloader = labeled_dataloader(dataset, model_name, num_samples)
    else:
        dataloader = labeled_dataloader(dataset, model, num_samples)

    total_new_ppl = 0.0
    model_device = next(first_tiny_model.parameters()).device
    for idx, sample in enumerate(dataloader):
        if idx >= iter:
            break

        with torch.no_grad():
            new_output = first_tiny_model(**(tensors_to_device(sample, model_device)))
        new_ppl = torch.exp(new_output.loss)
        total_new_ppl += new_ppl

    avg_new_ppl = total_new_ppl / iter
    assert avg_new_ppl < perplexity
