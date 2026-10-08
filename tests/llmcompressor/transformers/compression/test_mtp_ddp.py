"""Exercise MTP loading, quantization and ordinary sharded saving on two ranks."""

import gc
import shutil
import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from compressed_tensors.offload import OffloadCache, init_dist, set_onload_device
from compressed_tensors.offload.utils import as_single_threaded
from compressed_tensors.utils.safetensors_load import get_weight_mappings
from safetensors.torch import load_file
from transformers import (
    DeepseekV3ForCausalLM,
    Glm4MoeForCausalLM,
    Glm4MoeLiteForCausalLM,
    GlmMoeDsaForCausalLM,
    InklingForCausalLM,
)

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context
from tests.llmcompressor.transformers.compression.test_mtp_upstream import (
    _source_glm_model,
    _source_model,
)
from tests.testing_utils import requires_gpu, torchrun


@pytest.mark.multi_gpu
@requires_gpu(2)
@torchrun(world_size=2)
def test_mtp_two_rank_quantization():
    init_dist()
    models = (
        InklingForCausalLM,
        Glm4MoeForCausalLM,
        Glm4MoeLiteForCausalLM,
        GlmMoeDsaForCausalLM,
        DeepseekV3ForCausalLM,
    )
    for model_cls in models:
        print(f"Checking {model_cls.__name__} on rank {dist.get_rank()}", flush=True)
        root = [tempfile.mkdtemp(prefix="mtp-ddp-") if dist.get_rank() == 0 else None]
        dist.broadcast_object_list(root, src=0)
        root = Path(root[0])
        if dist.get_rank() == 0:
            with as_single_threaded():
                if model_cls is InklingForCausalLM:
                    source = _source_model(root)
                else:
                    source = _source_glm_model(root, model_cls)
            del source
        dist.barrier()

        with load_context(model_cls, load_mtp=True):
            model = model_cls.from_pretrained(
                root / "source", device_map="cpu", local_files_only=True
            )
        assert isinstance(model.mtp.layers[0].eh_proj._parameters, OffloadCache)
        for name, param in model.mtp.named_parameters():
            assert torch.isfinite(param).all(), (model_cls.__name__, name, "loaded")
        set_onload_device(model, torch.device("cuda", dist.get_rank()))
        recipe = QuantizationModifier(
            scheme="FP8_DYNAMIC", ignore=["lm_head", r"re:.*\.eh_proj$"]
        )
        oneshot(model=model, recipe=recipe)
        reference = {
            name: module.weight.detach().cpu().clone()
            for name, module in model.mtp.named_modules()
            if hasattr(module, "quantization_scheme")
        }
        for name, weight in reference.items():
            assert torch.isfinite(weight).all(), (model_cls.__name__, name, "quantized")
        model.save_pretrained(root / "output", max_shard_size="100KB")
        dist.barrier()

        mapping = get_weight_mappings(root / "output")
        prefix = (
            "model.mtp.layers.0."
            if model_cls is InklingForCausalLM
            else f"model.layers.{model.config.num_hidden_layers}."
        )
        mtp_scales = [
            name
            for name in mapping
            if name.startswith(prefix) and name.endswith(".weight_scale")
        ]
        assert mtp_scales, model_cls.__name__
        assert not (root / "output" / "model_mtp.safetensors").exists()
        assert model.mtp.shared_head is model.lm_head
        for name, weight in reference.items():
            gathered = [torch.empty_like(weight, device="cuda") for _ in range(2)]
            dist.all_gather(gathered, weight.to("cuda"))
            torch.testing.assert_close(
                gathered[0],
                gathered[1],
                rtol=0,
                atol=0,
                msg=f"{model_cls.__name__}: {name}",
            )
        for name in mtp_scales:
            scale = load_file(mapping[name])[name]
            assert torch.isfinite(scale).all() and (scale > 0).all()

        del model, reference
        gc.collect()
        torch.accelerator.empty_cache()
        dist.barrier()
        if dist.get_rank() == 0:
            shutil.rmtree(root)
        dist.barrier()
    dist.destroy_process_group()
