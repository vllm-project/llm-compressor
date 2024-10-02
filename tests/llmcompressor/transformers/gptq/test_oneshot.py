import shutil
import unittest

from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from parameterized import parameterized_class

from llmcompressor.modifiers.quantization.gptq import GPTQModifier
from llmcompressor.transformers.sparsification.sparse_model import (
    SparseAutoModelForCausalLM,
)
from tests.testing_utils import requires_torch

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

recipe_modifier_shorthand_a = GPTQModifier(
    ignore=["lm_head"], targets="Linear", scheme="W4A16"
)

recipe_modifier_shorthand_b = GPTQModifier(
    ignore=["lm_head"], scheme={"W4A16": ["Linear"]}
)


@requires_torch
@parameterized_class(
    [
        {"recipe": recipe_str},
        {"recipe": recipe_modifier_full},
        {"recipe": recipe_modifier_shorthand_a},
        {"recipe": recipe_modifier_shorthand_b},
    ]
)
class TestGPTQOneShotWithFullScheme(unittest.TestCase):
    def setUp(self):
        import torch

        self.output = "./oneshot_output"
        self.model = "roneneldan/TinyStories-1M"
        self.dataset = "open_platypus"
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def test_oneshot_application(self):
        from llmcompressor.transformers import oneshot

        oneshot(
            model=self.model,
            dataset=self.dataset,
            output_dir=self.output,
            overwrite_output_dir=True,
            recipe=self.recipe,
            oneshot_device=self.device,
            num_calibration_samples=9,
        )

        model_loaded = SparseAutoModelForCausalLM.from_pretrained(self.output)

        # Check that the model is quantized
        assert model_loaded.quantization_config is not None

        # check config is set properly
        assert model_loaded.quantization_config.ignore == ["lm_head"]
        assert len(model_loaded.quantization_config.config_groups) == 1
        quant_scheme = model_loaded.quantization_config.config_groups["group_0"]
        assert isinstance(quant_scheme, QuantizationScheme)
        assert quant_scheme.targets == ["Linear"]
        weight_args = model_loaded.quantization_config.config_groups["group_0"].weights
        assert isinstance(weight_args, QuantizationArgs)
        assert weight_args.num_bits == 4

        # Check a specific layer is quantized
        targetted_linear_layer = model_loaded.transformer.h[0].attn.attention.k_proj
        assert hasattr(targetted_linear_layer, "quantization_scheme")

        # Check lm-head is not quantized
        not_targetted = model_loaded.lm_head
        assert not hasattr(not_targetted, "quantization_scheme")

    def tearDown(self):
        shutil.rmtree(self.output)
