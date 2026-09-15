import json
from pathlib import Path

import pytest
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoTokenizer, Glm5NextForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context

MODEL_ID = "inference-optimization/GLM-5.3-Flash-0.1B-A0.1B-MTP"
MODEL_REVISION = "443ac6c54ba0d65ad8a7c701af4fd22a960c9e9c"
MTP_PREFIX = "model.language_model.layers.5"


@pytest.fixture(scope="module")
def mtp_checkpoint():
    # About 171 MB of weights, including a 2.6 MB MTP shard. The unsuffixed
    # model does not contain MTP weights and cannot exercise this save path.
    return Path(
        snapshot_download(
            MODEL_ID,
            revision=MODEL_REVISION,
            allow_patterns=["*.json", "*.safetensors", "*.jinja"],
        )
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    "deferred_save", [False, True], ids=["output-dir", "save-pretrained"]
)
@pytest.mark.parametrize(
    "mtp_scheme", [None, "BF16", "NVFP4A16"], ids=["preserve", "bf16", "nvfp4a16"]
)
def test_oneshot_mtp_checkpoint(mtp_checkpoint, tmp_path, mtp_scheme, deferred_save):
    """Exercise real loading, backbone compression, and unloaded MTP saving."""
    source = load_file(mtp_checkpoint / "model_mtp.safetensors")
    assert source

    with load_context(Glm5NextForConditionalGeneration):
        model = Glm5NextForConditionalGeneration.from_pretrained(
            mtp_checkpoint, dtype=torch.bfloat16, device_map="cpu"
        )
    assert not any(
        name.startswith(f"{MTP_PREFIX}.") for name, _ in model.named_parameters()
    )

    # Explicitly supply the text tokenizer: this test needs neither vision
    # processor dependencies nor calibration data.
    oneshot(
        model=model,
        processor=AutoTokenizer.from_pretrained(mtp_checkpoint),
        recipe=QuantizationModifier(
            targets=["re:.*mlp.shared_experts.*_proj$"], scheme="FP8_DYNAMIC"
        ),
        mtp_scheme=mtp_scheme,
        output_dir=None if deferred_save else str(tmp_path),
    )
    if deferred_save:
        model.save_pretrained(tmp_path)

    saved = load_file(tmp_path / "model_mtp.safetensors")
    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    actual_weight_map = {}
    for shard in tmp_path.glob("*.safetensors"):
        with safe_open(shard, framework="pt", device="cpu") as reader:
            for name in reader.keys():
                assert name not in actual_weight_map, f"Duplicate tensor: {name}"
                actual_weight_map[name] = shard.name
    assert weight_map == actual_weight_map

    # Ensure the test actually compressed the backbone before saving MTP.
    backbone_weight = (
        "model.language_model.layers.3.mlp.shared_experts.gate_proj.weight"
    )
    with safe_open(tmp_path / weight_map[backbone_weight], framework="pt") as reader:
        assert reader.get_tensor(backbone_weight).dtype == torch.float8_e4m3fn

    config = json.loads((tmp_path / "config.json").read_text())
    assert config["architectures"] == ["Glm5NextForConditionalGeneration"]
    quantization = config["quantization_config"]
    groups = quantization["config_groups"]
    assert quantization["quantization_status"] == "compressed"

    # These expectations describe the pinned fixture independently of the
    # production layout helpers, so a policy regression cannot update both.
    discarded = {
        f"{MTP_PREFIX}.hc_{block}_{suffix}"
        for block in ("attn", "ffn")
        for suffix in ("base", "fn", "scale")
    }
    expert_prefixes = [
        *(f"{MTP_PREFIX}.mlp.experts.{index}" for index in range(8)),
        f"{MTP_PREFIX}.mlp.shared_experts",
    ]
    projections = {
        f"{expert}.{projection}"
        for expert in expert_prefixes
        for projection in ("gate_proj", "up_proj", "down_proj")
    }
    quantized_weights = (
        {f"{projection}.weight" for projection in projections}
        if mtp_scheme == "NVFP4A16"
        else set()
    )
    dense_weights = set(source) - discarded - quantized_weights
    expected_keys = dense_weights | {
        f"{name.removesuffix('.weight')}.{suffix}"
        for name in quantized_weights
        for suffix in ("weight_packed", "weight_scale", "weight_global_scale")
    }
    assert set(saved) == expected_keys
    for name in dense_weights:
        expected = source[name]
        if mtp_scheme == "BF16" and expected.is_floating_point():
            expected = expected.bfloat16()
        assert saved[name].dtype == expected.dtype
        assert torch.equal(saved[name], expected), name

    if mtp_scheme in (None, "BF16"):
        assert "mtp_group" not in groups
        assert quantization["format"] == "float-quantized"
        return

    # A silent dense fallback must fail this test, even if oneshot returns normally.
    assert list(groups)[0] == "mtp_group"
    mtp_group = groups["mtp_group"]
    assert quantization["format"] == "mixed-precision"
    assert mtp_group["format"] == "nvfp4-pack-quantized"
    assert mtp_group.get("input_activations") is None
    assert mtp_group.get("output_activations") is None
    for projection in projections:
        rows, columns = source[f"{projection}.weight"].shape
        packed = saved[f"{projection}.weight_packed"]
        assert packed.dtype == torch.uint8
        assert packed.shape == (rows, columns // 2)
