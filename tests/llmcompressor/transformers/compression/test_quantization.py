import pytest
import torch
from compressed_tensors.quantization.utils import is_module_quantized
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DefaultDataCollator

from llmcompressor import oneshot
from llmcompressor.args import DatasetArguments
from llmcompressor.pytorch.utils import tensors_to_device
from llmcompressor.transformers.finetune.data import TextGenerationDataset
from llmcompressor.utils.dev import dispatch_for_generation
from tests.testing_utils import parse_params, requires_gpu

CONFIGS_DIRECTORY = "tests/llmcompressor/transformers/compression/configs"


def _get_dataloader(dataset_args, tokenizer):
    dataset_manager = TextGenerationDataset.load_from_registry(
        dataset_args.dataset,
        dataset_args=dataset_args,
        split="train_gen[:5%]",
        processor=tokenizer,
    )
    calib_dataset = dataset_manager()
    data_loader = DataLoader(
        calib_dataset,
        batch_size=1,
        collate_fn=DefaultDataCollator(),
        sampler=torch.utils.data.RandomSampler(calib_dataset),
    )

    return data_loader


def _get_quant_info(model):
    quant_info_weights = {}
    quant_info_inputs = {}
    for name, module in model.named_modules():
        if is_module_quantized(module):
            if module.quantization_scheme.weights is not None:
                quant_info_weights[name] = (
                    module.weight_scale,
                    module.weight_zero_point,
                    module.weight,
                )

            if module.quantization_scheme.input_activations is not None:
                is_dynamic = module.quantization_scheme.input_activations.dynamic
                if not is_dynamic:
                    quant_info_inputs[name] = (
                        module.input_scale,
                        module.input_zero_point,
                    )

    return quant_info_weights, quant_info_inputs


@pytest.fixture(params=parse_params(CONFIGS_DIRECTORY), scope="module")
def setup_model_and_config(request, tmpdir_factory):
    base_config = {
        "new_recipe": None,
        "ppl_threshold": None,
        "model_stub": None,
        "dataset": "ultrachat-200k",
        "output": "tiny_llama_out",
        "max_seq_length": 512,
        "weight_dtype": torch.float16,
        "num_eval": 64,
    }
    config = {**base_config, **request.param}

    num_calibration_samples = 64
    max_seq_length = 512
    pad_to_max_length = False

    output_dir = tmpdir_factory.mktemp("setup_model_and_config") / config["output"]
    model = AutoModelForCausalLM.from_pretrained(
        config["model_stub"], torch_dtype=config["weight_dtype"]
    )
    model = oneshot(
        model=model,
        dataset=config["dataset"],
        output_dir=output_dir,
        max_seq_length=max_seq_length,
        num_calibration_samples=num_calibration_samples,
        recipe=config["new_recipe"],
        pad_to_max_length=pad_to_max_length,
        splits={"calibration": "train_gen[:1%]"},
        save_compressed=False,
    )

    yield model, config, output_dir

    del model
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


@requires_gpu
@pytest.mark.integration
def test_quantization_reload(setup_model_and_config):
    model, config, output_dir = setup_model_and_config

    model_reloaded = AutoModelForCausalLM.from_pretrained(
        output_dir, torch_dtype="auto"
    )

    og_weights, og_inputs = _get_quant_info(model)
    reloaded_weights, reloaded_inputs = _get_quant_info(model_reloaded)
    # TODO: can remove `to` calls after
    # https://github.com/neuralmagic/compressed-tensors/pull/427

    for name, (o_scale, o_zp, o_weight) in og_weights.items():
        n_scale, n_zp, n_weight = reloaded_weights[name]
        assert o_scale.dtype == n_scale.dtype == config["weight_dtype"]
        assert torch.equal(o_scale, n_scale.to(o_scale.device))
        assert o_zp.dtype == n_zp.dtype
        assert torch.equal(o_zp, n_zp.to(o_zp.device))

        # we don't expect an exact match here because o_weight still has the
        # original weight and n_weight has been fake_quantized
        assert n_weight.dtype == o_weight.dtype == config["weight_dtype"]

    for name, (o_scale, o_zp) in og_inputs.items():
        n_scale, n_zp = reloaded_inputs[name]
        assert o_scale.dtype == n_scale.dtype == config["weight_dtype"]
        assert torch.equal(o_scale, n_scale.to(o_scale.device))
        assert o_zp.dtype == n_zp.dtype
        assert torch.equal(o_zp, n_zp.to(o_zp.device))


@requires_gpu
@pytest.mark.integration
@torch.no_grad()
def test_perplexity(setup_model_and_config):
    model, config, output_dir = setup_model_and_config
    if config["ppl_threshold"] is None:
        pytest.skip("Skipping perplexity calculation.")
    tokenizer = AutoTokenizer.from_pretrained(config["model_stub"])
    dataset_args = DatasetArguments(
        dataset="ultrachat-200k",
        max_seq_length=config["max_seq_length"],
    )
    dataloader = _get_dataloader(dataset_args, tokenizer)
    dispatch_for_generation(model)

    total_ppl = 0.0
    total_samples = 0
    for sample in dataloader:
        if total_samples >= config["num_eval"]:
            break
        # -100 in labels indicates that the token is not part of the loss calculation
        pct_labels_in_sample = (sample["labels"] != -100).to(torch.float).mean().item()
        if pct_labels_in_sample <= 0.25:
            # At least 25% of the tokens in the sample must be part of loss calculation
            # otherwise the perplexity is too volatile and can skew the results
            continue
        output = model(**tensors_to_device(sample, "cuda:0"))
        total_ppl += torch.exp(output.loss).item()
        total_samples += 1

    avg_ppl = total_ppl / total_samples
    assert avg_ppl <= config["ppl_threshold"]
