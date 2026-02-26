from __future__ import annotations

import json
import shlex
import shutil
import sys
from pathlib import Path
from typing import Callable, List, NamedTuple

import pytest
from compressed_tensors.config import CompressionFormat

from tests.testing_utils import requires_gpu, run_cli_command


def verify_quantization_config(
    tmp_path: Path, prefix: str, compressed_format: CompressionFormat
):
    # verify the expected directory was generated
    dirs: List[Path] = [p for p in tmp_path.rglob(f"*-{prefix}") if p.is_dir()]
    assert (len(dirs)) == 1, f"did not find exactly one generated folder: {dirs}"

    # verify the format in the generated config
    config_json = json.loads((dirs[0] / "config.json").read_text())
    config_format = config_json["quantization_config"]["format"]
    assert config_format == compressed_format.value


class TestCase(NamedTuple):
    path: str
    flags: tuple[str] = ()
    preprocess_fn: None | Callable[[str], str] = None
    # verify_fn(tmp_path, prefix, compressed_format)
    verify_fn: Callable[[Path, str, CompressionFormat], None] | None = (
        verify_quantization_config
    )
    compressed_format: CompressionFormat | None = None
    prefix: str | None = None

    def __repr__(self):
        values = [f"'{self.path}'"]
        for attr_name in [
            "flags",
            "preprocess_fn",
            "verify_fn",
            "prefix",
            "compressed_format",
        ]:
            attr = getattr(self, attr_name)
            if attr:
                if callable(attr):
                    attr_repr = attr.__name__
                else:
                    attr_repr = repr(attr)

                values.append(f"{attr_name}={attr_repr}")

        return f"{self.__class__.__name__}({', '.join(values)})"


@pytest.mark.example
@requires_gpu(1)
@pytest.mark.parametrize(
    "test_case",
    [
        pytest.param(
            TestCase(
                "quantization_w4a16/llama3_ddp_example.py",
                compressed_format=CompressionFormat.pack_quantized,
                prefix="W4A16-G128-DDP2",
            ),
            marks=(requires_gpu(2), pytest.mark.multi_gpu),
        ),
        TestCase(
            "awq/llama_example.py",
            compressed_format=CompressionFormat.pack_quantized,
            prefix="awq-asym",
        ),
        TestCase(
            "awq/qwen3_moe_example.py",
            compressed_format=CompressionFormat.pack_quantized,
            prefix="awq-sym",
        ),
        TestCase(
            "big_models_with_sequential_onloading/llama3.3_70b.py",
            compressed_format=CompressionFormat.int_quantized,
            prefix="W8A8",
        ),
        TestCase(
            "quantization_kv_cache/llama3_fp8_kv_example.py",
            compressed_format=CompressionFormat.float_quantized,
            prefix="FP8-KV",
        ),
        TestCase(
            "quantization_w4a16/llama3_example.py",
            compressed_format=CompressionFormat.pack_quantized,
            prefix="W4A16-G128",
        ),
        TestCase(
            "quantization_w8a8_fp8/fp8_block_example.py",
            compressed_format=CompressionFormat.float_quantized,
            prefix="30B-A3B-FP8-BLOCK",
        ),
        TestCase(
            "quantization_w8a8_fp8/llama3_example.py",
            compressed_format=CompressionFormat.float_quantized,
            prefix="FP8-Dynamic",
        ),
        TestCase(
            "quantization_w8a8_fp8/qwen3_vl_moe_fp8_example.py",
            compressed_format=CompressionFormat.float_quantized,
            prefix="235B-A22B-Instruct-FP8-DYNAMIC",
        ),
        TestCase(
            "disk_offloading/qwen3_example.py",
            compressed_format=CompressionFormat.nvfp4_pack_quantized,
            prefix="NVFP4-Disk-Offload",
        ),
        TestCase(
            "model_free_ptq/qwen3_fp8_block.py",
            compressed_format=CompressionFormat.float_quantized,
            prefix="FP8-BLOCK",
        ),
        TestCase(
            "multimodal_vision/qwen3_vl_example.py",
            compressed_format=CompressionFormat.pack_quantized,
            prefix="W4A16",
        ),
        TestCase(
            "quantization_w4a16_fp4/mxfp4/qwen3_example.py",
            compressed_format=CompressionFormat.mxfp4_pack_quantized,
            prefix="MXFP4A16",
        ),
        TestCase(
            "sparse_2of4_quantization_fp8/llama3_8b_2of4.py",
            flags=["--fp8"],
            compressed_format=CompressionFormat.float_quantized,
            prefix="W8A8-FP8-Dynamic-Per-Token",
        ),
        TestCase(
            "quantization_w4a4_fp4/llama4_example.py",
            compressed_format=CompressionFormat.nvfp4_pack_quantized,
            prefix="NVFP4",
        ),
        TestCase(
            "quantization_w4a4_fp4/qwen_30b_a3b.py",
            compressed_format=CompressionFormat.nvfp4_pack_quantized,
            prefix="30B-A3B-NVFP4",
        ),
        TestCase(
            "quantization_w4a4_fp4/llama3_gptq_example.py",
            compressed_format=CompressionFormat.nvfp4_pack_quantized,
            prefix="NVFP4-GPTQ",
        ),
    ],
    ids=repr,
)
def test_example_scripts(
    test_case: str | TestCase, tmp_path: Path, request: pytest.FixtureRequest
):
    if isinstance(test_case, str):
        test_case = TestCase(test_case)

    example_subdir, filename = test_case.path.rsplit("/", 1)
    example_dir = f"examples/{example_subdir}"

    # Check if this is a multi-GPU test
    is_multi_gpu = "multi_gpu" in [mark.name for mark in request.node.iter_markers()]

    if is_multi_gpu:
        command = ["torchrun", "--standalone", "--nproc_per_node=2", filename]
    else:
        command = [sys.executable, filename]

    if test_case.flags:
        command.extend(test_case.flags)

    script_working_dir = tmp_path / example_dir
    shutil.copytree(Path.cwd() / example_dir, script_working_dir)

    if test_case.preprocess_fn:
        path = script_working_dir / filename
        content = path.read_text(encoding="utf-8")
        content = test_case.preprocess_fn(content)
        path.write_text(content, encoding="utf-8")

    result = run_cli_command(command, cwd=script_working_dir)

    assert result.returncode == 0, (
        f"command failed with exit code {result.returncode}:\n"
        f"Command:\n{shlex.join(command)}\nOutput:\n{result.stdout}"
    )

    if test_case.verify_fn:
        test_case.verify_fn(tmp_path, test_case.prefix, test_case.compressed_format)
