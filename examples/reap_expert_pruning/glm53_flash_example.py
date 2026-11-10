import torch
from compressed_tensors.offload import init_dist
from transformers import (
    AutoTokenizer,
    CompressedTensorsConfig,
    Glm5NextForConditionalGeneration,
)

from llmcompressor import oneshot
from llmcompressor.datasets.utils import get_rank_partition
from llmcompressor.modifiers.pruning import RazorPruningModifier
from llmcompressor.utils import load_context

# Select model and load it.
MODEL_ID = "RedHatAI/GLM-5.3-Flash-NVFP4"
REPORT_PATH = "glm53_flash_razorcal_report.json"
NUM_CALIBRATION_SAMPLES = 2048

init_dist()
with load_context(Glm5NextForConditionalGeneration, load_mtp=False):
    model = Glm5NextForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="disk",
        offload_folder="offload_folder",
        quantization_config=CompressedTensorsConfig(decompress=False),
    )
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


# skip mtp layer for now
model.config.text_config.num_nextn_predict_layers = 0
recipe = RazorPruningModifier(sparsity=0.0, report_path=REPORT_PATH)

oneshot(
    model=model,
    processor=tokenizer,
    dataset="razorcal",
    splits=get_rank_partition("train", NUM_CALIBRATION_SAMPLES),
    recipe=recipe,
    max_seq_length=2048,
    propagate_error=False,
    batch_size=32,
)

torch.distributed.destroy_process_group()
