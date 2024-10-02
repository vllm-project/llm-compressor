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
            sequential_update: false
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
    sequential_update=False,
    config_groups={
        "group_0": QuantizationScheme(
            targets=["Linear"], weights=QuantizationArgs(num_bits=4, strategy="channel")
        )
    },
)

recipe_modifier_shorthand_a = GPTQModifier(
    ignore=["lm_head"], sequential_update=False, targets="Linear", scheme="W4A16"
)

recipe_modifier_shorthand_b = GPTQModifier(
    ignore=["lm_head"], sequential_update=False, scheme={"W4A16": ["Linear"]}
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
        targetted_linear_layer = model_loaded.transformer.h[0].attn.attention.k_proj
        assert hasattr(targetted_linear_layer, "quantization_scheme")

        # Check lm-head is not quantized
        not_targetted = model_loaded.lm_head
        assert not hasattr(not_targetted, "quantization_scheme")

    def tearDown(self):
        shutil.rmtree(self.output)
