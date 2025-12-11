import glob
import os
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal, cast

import torch
import transformers
from auto_round.export.export_to_awq.utils import (
    reverse_awq_order,
    unpack_awq,
)
from compressed_tensors import ModelCompressor
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
    QuantizationStrategy,
    QuantizationType,
)
from huggingface_hub import load_state_dict_from_file, snapshot_download


def is_autoawq_model(model_path: Path) -> bool:
    config = transformers.AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if not hasattr(config, "quantization_config"):
        return False

    quantization_config = cast(dict[str, Any], config.quantization_config)
    return quantization_config.get("quant_method") == "awq"


def resolve_model_path(model_name_or_path: str) -> Path:
    """Locate the model path.

    If the input is a repository ID, download the model from the Hugging Face Hub and
    return the path to the local directory.
    """
    if os.path.isdir(model_name_or_path):
        return Path(model_name_or_path)

    return Path(snapshot_download(model_name_or_path))


def load_state_dict_from_model_dir(model_path: Path) -> dict[str, torch.Tensor]:
    weight_files = glob.glob(str(model_path / "*.safetensors"))
    if not weight_files:
        weight_files = glob.glob(str(model_path / "*.bin"))

    state_dict = {}
    for weight_file in weight_files:
        state_dict.update(
            load_state_dict_from_file(
                weight_file, map_location="cpu", weights_only=True
            )
        )
    return state_dict


def dequantize_gemm(
    state_dict: dict[str, torch.Tensor], prefix: str, autoawq_config: dict[str, Any]
) -> None:
    num_bits = cast(int, autoawq_config.get("bits"))
    group_size = cast(int, autoawq_config.get("group_size"))

    qweight = state_dict.pop(f"{prefix}.qweight")
    scales = state_dict.pop(f"{prefix}.scales")
    qzeros = state_dict.pop(f"{prefix}.qzeros")

    def dequantize_gemm_original(
        qweight: torch.Tensor,
        qzeros: torch.Tensor,
        scales: torch.Tensor,
        bits: int,
        group_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Modified from auto_round.export.export_to_awq.utils.dequantize_gemm."""
        # Unpack the qweight and qzeros tensors
        iweight, izeros = unpack_awq(qweight, qzeros, bits)
        # Reverse the order of the iweight and izeros tensors
        iweight, izeros = reverse_awq_order(iweight, izeros, bits)

        # overflow checks
        iweight = torch.bitwise_and(iweight, (2**bits) - 1)
        izeros = torch.bitwise_and(izeros, (2**bits) - 1)

        # fp16 weights
        scales_interleaved = scales.repeat_interleave(group_size, dim=0)
        izeros_interleaved = izeros.repeat_interleave(group_size, dim=0)
        fweight = (iweight - izeros_interleaved) * scales_interleaved

        return fweight, izeros

    weight, zero_point = dequantize_gemm_original(
        qweight, qzeros, scales, num_bits, group_size
    )

    # AutoAWQ uses [0, 2^bits - 1], e.g., [0, 15], for quantized weights, but
    # compressed-tensors uses [-2^(bits - 1), 2^(bits - 1) - 1], e.g., [-8, 7].
    # Therefore, we need to shift the zero point by 2^(bits - 1) to match the range
    # of compressed-tensors and to allow correct quant/dequantization.
    shifted_zero_point = zero_point - 2 ** (num_bits - 1)

    state_dict.update(
        {
            f"{prefix}.weight": weight.T,
            f"{prefix}.weight_scale": scales.T,
            f"{prefix}.weight_zero_point": shifted_zero_point.T,
        }
    )


def dequantize_autoawq_state_dict(
    state_dict: dict[str, torch.Tensor], autoawq_config: dict[str, Any]
) -> dict[str, torch.Tensor]:
    version = cast(str, autoawq_config.get("version"))

    # TODO: maybe add support for other versions?
    match version:
        case "gemm":
            dequantize_fn = dequantize_gemm
        case _:
            raise ValueError(f"Unsupported version: {version}")

    keys = list(state_dict.keys())
    for key in filter(lambda k: k.endswith("qweight"), keys):
        prefix = key.removesuffix(".qweight")
        dequantize_fn(state_dict, prefix, autoawq_config)

    return state_dict


def convert_and_save(
    model_name_or_path: str,
    output_dir: str,
    quantization_format: str,
    overwrite: bool = False,
    trust_remote_code: bool = False,
) -> None:
    """Convert an AutoAWQ model to a compressed model and save it.

    Steps:

    1. Load the model weights directly.
    2. Dequantize the weights accordingly.
    3. Load the model with the dequantized weights.
    4. Add the quantization parameters to the model.
    5. Re-pack the weights using `ModelCompressor` with the correct configuration.
    6. Save the model to the output directory.

    :param model_name_or_path: Model ID on huggingface hub or path to local model.
    :param output_dir: Path to save the converted model.
    :param quantization_format: Compression format to be saved.
    :param overwrite: Overwrite the existing output directory if it exists.
    :param trust_remote_code: Whether to trust remote code.
    """
    if os.path.exists(output_dir) and not overwrite:
        raise FileExistsError(
            f"Output directory {output_dir} already exists. Set `overwrite=True` to"
            " overwrite the existing directory."
        )

    model_path = resolve_model_path(model_name_or_path)
    if not is_autoawq_model(model_path):
        raise ValueError("Model is not an AutoAWQ model")

    config = transformers.AutoConfig.from_pretrained(
        model_path, trust_remote_code=trust_remote_code
    )
    autoawq_config = cast(dict[str, Any], config.quantization_config)
    num_bits = cast(int, autoawq_config.get("bits"))
    is_symmetric = not autoawq_config.get("zero_point")
    group_size = cast(int, autoawq_config.get("group_size"))

    # TODO: check syntax of modules_to_not_convert
    ignore = autoawq_config.get("modules_to_not_convert")
    if ignore is None:
        ignore = ["lm_head"]

    # 1. Load the model weights directly.
    state_dict = load_state_dict_from_model_dir(model_path)

    # 2. Dequantize the weights accordingly.
    state_dict = dequantize_autoawq_state_dict(state_dict, autoawq_config)

    # 3. Load the model with the dequantized weights.
    del config.quantization_config  # remove to avoid loading with AutoAWQ.
    with transformers.modeling_utils.no_init_weights():
        model = transformers.AutoModelForCausalLM.from_config(
            config, torch_dtype=torch.float16, trust_remote_code=trust_remote_code
        )

    model.load_state_dict(state_dict, strict=False)

    # 4. Add the quantization parameters to the model.
    quantization_scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=num_bits,
            type=QuantizationType.INT,
            symmetric=is_symmetric,
            group_size=group_size,
            strategy=QuantizationStrategy.GROUP,
        ),
    )

    for key in filter(lambda k: k.endswith("weight_zero_point"), state_dict.keys()):
        module_name = key.removesuffix(".weight_zero_point")
        setattr(
            model.get_submodule(module_name), "quantization_scheme", quantization_scheme
        )

    quant_config = QuantizationConfig(
        config_groups={"group_0": quantization_scheme},
        quant_method="compressed-tensors",
        quantization_status=QuantizationStatus.COMPRESSED,
        format=quantization_format,
        ignore=ignore,
    )

    # 5. Re-pack the weights using `ModelCompressor`.
    compressor = ModelCompressor(quantization_config=quant_config)
    compressed_state_dict = compressor.compress(model, state_dict, show_progress=True)

    # 6. Save the model.
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
    model.save_pretrained(output_dir, state_dict=compressed_state_dict)
    tokenizer.save_pretrained(output_dir)
    compressor.update_config(output_dir)


def load_and_convert_from_autoawq(
    model_name_or_path: str,
    quantization_format: str = "pack-quantized",
    trust_remote_code: bool = False,
) -> transformers.modeling_utils.PreTrainedModel:
    """
    Load an AutoAWQ checkpoint and convert it to a compressed model.

    :param model_name_or_path: Model ID on huggingface hub or path to local model.
    :param quantization_format: Compression format to be saved.
    :param trust_remote_code: Whether to trust remote code.
    :return: A compressed model.
    """
    with TemporaryDirectory() as temp_dir:
        convert_and_save(model_name_or_path, temp_dir, quantization_format)
        return transformers.AutoModelForCausalLM.from_pretrained(
            temp_dir, torch_dtype=torch.float16, trust_remote_code=trust_remote_code
        )


@dataclass
class ConversionArgs:
    model_name_or_path: str = field(
        metadata={"help": "Model ID on huggingface hub or path to local model."},
    )
    output_dir: str = field(
        metadata={"help": "Path to save the converted model."},
    )
    quantization_format: Literal["naive-quantized", "packed-quantized"] = field(
        default="naive-quantized",
        metadata={"help": "Compression format to be saved."},
    )  # TODO: switch default to packed-quantized once supported by llm-compressor.
    overwrite: bool = field(
        default=False,
        metadata={"help": "Overwrite the existing output directory if it exists."},
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Whether to trust remote code."},
    )


__all__ = ["convert_and_save", "load_and_convert_from_autoawq"]


if __name__ == "__main__":
    parser = transformers.HfArgumentParser(ConversionArgs)
    args = parser.parse_args_into_dataclasses()[0]
    convert_and_save(
        args.model_name_or_path,
        args.output_dir,
        args.quantization_format,
        args.overwrite,
        args.trust_remote_code,
    )
