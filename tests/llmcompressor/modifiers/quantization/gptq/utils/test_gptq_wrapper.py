from collections import OrderedDict

import torch
from compressed_tensors.quantization.lifecycle.apply import apply_quantization_config
from compressed_tensors.quantization.quant_config import QuantizationConfig
from compressed_tensors.quantization.quant_scheme import preset_name_to_scheme
from loguru import logger

from llmcompressor.modifiers.quantization.gptq.utils.gptq_wrapper import GPTQWrapper


def test_ignore():
    model = torch.nn.Sequential(
        OrderedDict(
            [
                ("first_layer", torch.nn.Linear(2, 3)),
                ("second_layer", torch.nn.Linear(3, 5)),
            ]
        )
    )

    config = QuantizationConfig(
        config_groups={"group_0": preset_name_to_scheme("W8A8", targets=["Linear"])},
        ignore=["first_layer"],
    )
    apply_quantization_config(model, config)

    messages = []
    logger.add(lambda m: messages.append(m))

    with torch.no_grad():
        first_compressor = GPTQWrapper("first_layer", model.first_layer)
        first_compressor.add_batch(torch.ones(2), None)
        first_compressor.compress()

        first_compressor = GPTQWrapper("second_layer", model.second_layer)
        first_compressor.add_batch(torch.ones(3), None)
        first_compressor.compress()

    assert sum("Skipping unquantized layer first_layer" in m for m in messages) == 1
    assert sum("Skipping unquantized layer second_layer" in m for m in messages) == 0
