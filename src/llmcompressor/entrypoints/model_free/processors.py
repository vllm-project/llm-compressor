from typing import Iterable, Protocol

import torch

from llmcompressor.entrypoints.model_free.helpers import iter_quantizable_tensors


class Processor(Protocol):
    """
    Interface for any other processing that needs to be applied
    during model_free_ptq
    """

    def process(self, tensors: dict[str, torch.Tensor]):
        pass


class ModelOptNvfp4Processor(Processor):
    """
    Convert params from modelopt NVFP4 to CT NVFP4 convention
    nvidia/DeepSeek-R1-NVFP4's nvfp4-quantized layers, found by inspection
    - model.layers.0.mlp.down_proj.weight
    - model.layers.0.mlp.gate_proj.weight
    - model.layers.0.mlp.up_proj.weight
    - model.layers.3.mlp.shared_experts.down_proj.weight
    - model.layers.3.mlp.shared_experts.gate_proj.weight
    - model.layers.3.mlp.shared_experts.up_proj.weight
    - model.layers.3.mlp.experts.0.down_proj.weight
    - model.layers.3.mlp.experts.0.gate_proj.weight
    - model.layers.3.mlp.experts.0.up_proj.weight
    """

    def __init__(
        self,
        ignore: Iterable[str] = tuple(),
        targets: Iterable[str] = tuple("re:.*mlp.*\.(gate|up|down)_proj$"),
    ):
        self.ignore = ignore
        self.targets = targets

    def process(self, tensors: dict[str, torch.Tensor]):
        for module_name, name in iter_quantizable_tensors(
            tensors, self.ignore, self.targets
        ):
            param_name = name.rsplit(".", 1)[-1]

            match param_name:
                # input_scale -> input_global_scale F32
                case "input_scale":
                    # convert modelopt input_scale x -> 1/x
                    # https://github.com/vllm-project/vllm/blob/v0.13.0/vllm/model_executor/layers/quantization/modelopt.py#L1070-L1073
                    # https://github.com/vllm-project/vllm/blob/v0.13.0/vllm/model_executor/layers/quantization/modelopt.py#L1134
                    # https://github.com/vllm-project/vllm/blob/v0.13.0/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py#L190
                    tensors[f"{module_name}.input_global_scale"] = 1 / tensors[name]
                    del tensors[name]
                # weight -> weight_packed U8
                case "weight":
                    tensors[f"{module_name}.weight_packed"] = tensors[name]
                    del tensors[name]
                # weight_scale -> weight_scale F8_E4M3
                case "weight_scale":
                    pass
                # weight_scale_2 -> weight_global_scale F32
                case "weight_scale_2":
                    # convert modelopt weight_scale_2 x -> 1/x
                    # https://github.com/vllm-project/vllm/blob/v0.13.0/vllm/model_executor/layers/quantization/modelopt.py#L1066-L1068
                    # https://github.com/vllm-project/vllm/blob/v0.13.0/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py#L163-L166
                    tensors[f"{module_name}.weight_global_scale"] = 1 / tensors[name]
                    del tensors[name]
                case _:
                    raise RuntimeError(f"Hit unexpected tensor {name}")
