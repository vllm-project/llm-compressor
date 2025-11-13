import pytest
import torch
from auto_round.calib_dataset import get_dataset
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.autoround import AutoRoundModifier
from tests.testing_utils import requires_gpu

recipe_str = """
quant_stage:
    quant_modifiers:
        AutoRoundModifier:
            ignore: ["lm_head"]
            iters: 10
            config_groups:
                group_0:
                    targets:
                        - "Linear"
                    input_activations: null
                    output_activations: null
                    weights:
                        num_bits: 4
                        type: "int"
                        symmetric: true
                        strategy: group
                        group_size: 128
"""

recipe_modifier_full = AutoRoundModifier(
    ignore=["lm_head"],
    iters=10,
    config_groups={
        "group_0": QuantizationScheme(
            targets=["Linear"],
            weights=QuantizationArgs(num_bits=4, strategy="group", group_size=128),
        )
    },
)


@requires_gpu(1)
@pytest.mark.parametrize(
    "recipe",
    [
        recipe_str,
        recipe_modifier_full,
    ],
)
def test_oneshot_application(recipe, tmp_path):
    output = tmp_path / "oneshot_output"
    model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model)
    dataset = get_dataset(
        tokenizer=tokenizer,
        seqlen=1024,
        nsamples=32,
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    oneshot(
        model=model,
        dataset=dataset,
        output_dir=output,
        recipe=recipe,
    )
    model_loaded = AutoModelForCausalLM.from_pretrained(output, device_map=device)

    # Check that the model is quantized
    # for compression_config - decompress() will attach a quantization_config
    # to the model as we decompress right away
    # for quantization_config - we have CompressedLinear which will only
    # decompress on the forward pass and does not call decompress(). Results
    # in a slightly different parameter tree to access the quant config
    quantization_config = model_loaded.config.quantization_config.quantization_config
    assert quantization_config is not None

    # check config is set properly
    assert "lm_head" in quantization_config.ignore
    assert len(quantization_config.config_groups) == 1
    quant_scheme = quantization_config.config_groups["group_0"]
    assert isinstance(quant_scheme, QuantizationScheme)

    weight_args = quantization_config.config_groups["group_0"].weights
    assert isinstance(weight_args, QuantizationArgs)
    assert weight_args.num_bits == 4

    # Check a specific layer is quantized
    targetted_linear_layer = model_loaded.model.layers[2].self_attn.q_proj
    assert hasattr(targetted_linear_layer, "quantization_scheme")

    # Check lm-head is not quantized
    not_targetted = model_loaded.lm_head
    assert not hasattr(not_targetted, "quantization_scheme")
