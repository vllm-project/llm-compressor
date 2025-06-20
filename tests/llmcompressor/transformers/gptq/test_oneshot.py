import os
import shutil
import unittest

from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from parameterized import parameterized_class
from transformers import AutoModelForCausalLM

from llmcompressor.modifiers.quantization.gptq import GPTQModifier

recipe_str = """
quant_stage:
    quant_modifiers:
        GPTQModifier:
            ignore: ["lm_head"]
            config_groups:
                group_0:
                    weights:
                        num_bits: 4
                        type: "int"
                        symmetric: true
                        strategy: "channel"
                    targets: ["Linear"]
"""

recipe_modifier_full = GPTQModifier(
    ignore=["lm_head"],
    config_groups={
        "group_0": QuantizationScheme(
            targets=["Linear"], weights=QuantizationArgs(num_bits=4, strategy="channel")
        )
    },
)

recipe_modifier_full_group = GPTQModifier(
    ignore=["lm_head"],
    config_groups={
        "group_0": QuantizationScheme(
            targets=["Linear"],
            weights=QuantizationArgs(num_bits=4, strategy="group", group_size=128),
        )
    },
)

recipe_modifier_shorthand_a = GPTQModifier(
    ignore=["lm_head"], targets="Linear", scheme="W4A16"
)

recipe_modifier_shorthand_b = GPTQModifier(
    ignore=["lm_head"], scheme={"W4A16": ["Linear"]}
)


@parameterized_class(
    [
        {"recipe": recipe_str},
        {"recipe": recipe_modifier_full},
        {"recipe": recipe_modifier_full_group},
        {"recipe": recipe_modifier_shorthand_a},
        {"recipe": recipe_modifier_shorthand_b},
    ]
)
class TestGPTQOneShotWithFullScheme(unittest.TestCase):
    def setUp(self):
        import torch

        self.output = "./oneshot_output"
        self.model = "Xenova/llama2.c-stories110M"
        self.dataset = "open_platypus"
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def test_oneshot_application(self):
        from llmcompressor import oneshot

        oneshot(
            model=self.model,
            dataset=self.dataset,
            output_dir=self.output,
            recipe=self.recipe,
            num_calibration_samples=9,
        )
        model_loaded = AutoModelForCausalLM.from_pretrained(
            self.output, device_map=self.device
        )

        # Check that the model is quantized
        # for compression_config - decompress() will attach a quantization_config
        # to the model as we decompress right away
        # for quantization_config - we have CompressedLinear which will only
        # decompress on the forward pass and does not call decompress(). Results
        # in a slightly different parameter tree to access the quant config
        quantization_config = (
            model_loaded.config.quantization_config.quantization_config
        )
        assert quantization_config is not None

        # check config is set properly
        assert quantization_config.ignore == ["lm_head"]
        assert len(quantization_config.config_groups) == 1
        quant_scheme = quantization_config.config_groups["group_0"]
        assert isinstance(quant_scheme, QuantizationScheme)
        assert quant_scheme.targets == ["Linear"]
        weight_args = quantization_config.config_groups["group_0"].weights
        assert isinstance(weight_args, QuantizationArgs)
        assert weight_args.num_bits == 4

        # Check a specific layer is quantized
        targetted_linear_layer = model_loaded.model.layers[0].self_attn.k_proj
        assert hasattr(targetted_linear_layer, "quantization_scheme")

        # Check lm-head is not quantized
        not_targetted = model_loaded.lm_head
        assert not hasattr(not_targetted, "quantization_scheme")

    def tearDown(self):
        if os.path.isdir(self.output):
            shutil.rmtree(self.output)
