import contextlib
import shutil
import tempfile

import pytest
import torch
import torch.distributed as dist
from compressed_tensors.offload import init_dist, load_offloaded_model
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from tests.testing_utils import requires_gpu, torchrun

MODEL_ID = "nm-testing/tinysmokellama-3.2"
NUM_CALIBRATION_SAMPLES = 16
MAX_SEQ_LENGTH = 128
MAX_CPU_MEMORY = 6e5  # 600KB forces ~half the modules to disk on tinysmokellama-3.2
PERPLEXITY_THRESHOLD = 3000
EVAL_TEXT = "Paris is the capital of France"


@contextlib.contextmanager
def _disk_offloaded_model():
    """Load a disk-offloaded model using a temporary offload folder."""
    offload_dir = tempfile.mkdtemp(prefix="disk_offload_test_")
    try:
        with load_offloaded_model():
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float32,
                device_map="auto_offload",
                max_memory={"cpu": MAX_CPU_MEMORY},
                offload_folder=offload_dir,
            )
        yield model
    finally:
        shutil.rmtree(offload_dir, ignore_errors=True)


def _dataset():
    return Dataset.from_dict(
        {"text": ["Paris is the capital of France. " * 16] * NUM_CALIBRATION_SAMPLES}
    )


def _recipe():
    return GPTQModifier(
        ignore=["lm_head"],
        config_groups={
            "group_0": QuantizationScheme(
                targets=["Linear"],
                weights=QuantizationArgs(
                    num_bits=4, strategy="channel", symmetric=True
                ),
            ),
        },
    )


@torch.no_grad()
def _compute_perplexity(model, text):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    inputs["labels"] = inputs["input_ids"]
    output = model(**inputs)
    return torch.exp(output.loss).item()


@pytest.mark.integration
@pytest.mark.multi_gpu
@requires_gpu(2)
@torchrun(world_size=2)
def test_gptq_distributed_disk_offload():
    """
    Verify distributed GPTQ with disk offloading produces a model with
    reasonable perplexity on a simple sentence.
    """

    init_dist()
    with _disk_offloaded_model() as model:
        oneshot(
            model=model,
            dataset=_dataset(),
            recipe=_recipe(),
            num_calibration_samples=NUM_CALIBRATION_SAMPLES,
            max_seq_length=MAX_SEQ_LENGTH,
        )

        torch.distributed.barrier()

        if dist.get_rank() == 0:
            ppl = _compute_perplexity(model, EVAL_TEXT)
            assert (
                ppl < PERPLEXITY_THRESHOLD
            ), f"Perplexity {ppl:.1f} exceeds threshold {PERPLEXITY_THRESHOLD}"
