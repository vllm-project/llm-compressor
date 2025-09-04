import json
from pathlib import Path
from typing import List, Union

import pytest
from transformers import AutoConfig


def requires_gpu_count(num_required_gpus: int) -> pytest.MarkDecorator:
    """
    Pytest decorator to skip based on number of available GPUs. This plays nicely with
    the CUDA_VISIBLE_DEVICES environment variable.
    """
    import torch

    num_gpus = torch.cuda.device_count()
    reason = f"{num_required_gpus} GPUs required, {num_gpus} GPUs detected"
    return pytest.mark.skipif(num_required_gpus > num_gpus, reason=reason)


def requires_gpu_mem(required_amount: Union[int, float]) -> pytest.MarkDecorator:
    """
    Pytest decorator to skip based on total available GPU memory (across all GPUs). This
    plays nicely with the CUDA_VISIBLE_DEVICES environment variable.

    Note: make sure to account for measured memory vs. simple specs. For example, H100
    has '80 GiB' VRAM, however, the actual number, at least per PyTorch, is ~79.2 GiB.

    :param amount: amount of required GPU memory in GiB
    """
    import torch

    vram_bytes = sum(
        torch.cuda.mem_get_info(device_id)[1]
        for device_id in range(torch.cuda.device_count())
    )
    actual_vram = vram_bytes / 1024**3
    reason = (
        f"{required_amount} GiB GPU memory required, "
        f"{actual_vram:.1f} GiB GPU memory found"
    )
    return pytest.mark.skipif(required_amount > actual_vram, reason=reason)


def replace_2of4_w4a16_recipe(content: str) -> str:
    return content.replace("2of4_w4a16_recipe.yaml", "2of4_w4a16_group-128_recipe.yaml")


def verify_2of4_w4a16_output(tmp_path: Path, example_dir: str):
    output_dir = Path("output_llama7b_2of4_w4a16_channel")

    stages = {
        "quantization": {
            "path": Path("quantization_stage"),
            "format": "marlin-24",
        },
        "sparsity": {
            "path": Path("sparsity_stage"),
            "format": "sparse-24-bitmask",
        },
        "finetuning": {
            "path": Path("finetuning_stage"),
            "format": "sparse-24-bitmask",
        },
    }

    for stage, stage_info in stages.items():
        stage_path = tmp_path / example_dir / output_dir / stage_info["path"]
        recipe_path = stage_path / "recipe.yaml"
        config_path = stage_path / "config.json"

        assert recipe_path.exists(), f"Missing recipe file in {stage}: {recipe_path}"
        assert config_path.exists(), f"Missing config file in {stage}: {config_path}"

        config = AutoConfig.from_pretrained(stage_path)
        assert config is not None, f"Failed to load config in {stage}"

        quant_config = getattr(config, "quantization_config", {})
        if stage == "quantization":
            actual_format = quant_config.get("format")
        else:
            actual_format = quant_config.get("sparsity_config", {}).get("format")

        assert actual_format, f"Missing expected format field in {stage} config"
        assert actual_format == stage_info["format"], (
            f"Unexpected format in {stage}: got '{actual_format}', "
            f"expected '{stage_info['format']}'"
        )


def verify_w4a4_fp4_output(tmp_path: Path, example_dir: str):
    # verify the expected directory was generated
    nvfp4_dirs: List[Path] = [p for p in tmp_path.rglob("*-NVFP4") if p.is_dir()]
    assert (
        len(nvfp4_dirs)
    ) == 1, f"did not find exactly one generated folder: {nvfp4_dirs}"

    # verify the format in the generated config
    config_json = json.loads((nvfp4_dirs[0] / "config.json").read_text())
    config_format = config_json["quantization_config"]["format"]
    assert config_format == "nvfp4-pack-quantized"
