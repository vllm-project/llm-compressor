#!/usr/bin/env python3

import os
import time
from pathlib import Path

import modal

_resolved = Path(__file__).resolve()
REPO_ROOT = _resolved.parents[2] if len(_resolved.parents) >= 3 else Path("/repo")

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 2048
RECIPE_PATH = "tests/e2e/vLLM/recipes/WNA16/recipe_w4a16_awq_sym.yaml"
SAVE_DIR = "llama3-8b-w4a16-awq-baseline"

LMEVAL_TASK = "gsm8k"
LMEVAL_LIMIT = 100
LMEVAL_NUM_FEWSHOT = 5
LMEVAL_BATCH_SIZE = 4


def _make_image() -> modal.Image:
    return (
        modal.Image.debian_slim(python_version="3.11")
        .add_local_dir(
            REPO_ROOT,
            remote_path="/repo",
            copy=True,
            ignore=[".git", ".venv", "*.pyc", "__pycache__", "*.egg-info"],
        )
        .env({"SETUPTOOLS_SCM_PRETEND_VERSION": "0.1.0"})
        .run_commands(
            "cd /repo && pip install -e .",
            "pip install lm_eval==0.4.9.2",
            "pip install pytest",
        )
    )


image = _make_image()
app = modal.App("awq-baseline-standalone")


@app.function(
    image=image,
    gpu="H100",
    timeout=14400,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    env={
        "PYTHONPATH": "/repo/src:/repo",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    },
)
def run_baseline(skip_lm_eval: bool = False) -> dict:
    """Run AWQ baseline (no smooth) + optional lm_eval."""
    os.chdir("/repo")
    from tests.e2e.e2e_utils import run_oneshot_for_e2e_testing

    recipe_abs = f"/repo/{RECIPE_PATH}"
    if not os.path.isfile(recipe_abs):
        return {"error": f"Recipe not found: {recipe_abs}", "awq_time_s": None, "lm_eval": None}

    t0 = time.perf_counter()
    model, processor = run_oneshot_for_e2e_testing(
        model=MODEL_ID,
        model_class="AutoModelForCausalLM",
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        scheme="W4A16_awq_sym",
        dataset_id=DATASET_ID,
        dataset_config=None,
        dataset_split=DATASET_SPLIT,
        recipe=recipe_abs,
        quant_type=None,
    )
    os.makedirs(SAVE_DIR, exist_ok=True)
    model.save_pretrained(SAVE_DIR)
    processor.save_pretrained(SAVE_DIR)
    from llmcompressor.core import active_session
    active_session().reset()
    awq_time_s = time.perf_counter() - t0

    # Compression-only: return AWQ runtime; evaluation is handled by a separate script.
    return {"awq_time_s": awq_time_s}


@app.local_entrypoint()
def main(skip_lm_eval: bool = False):
    """Entrypoint: run baseline and print result."""
    result = run_baseline.remote(skip_lm_eval=skip_lm_eval)
    print("Baseline result:", result)
