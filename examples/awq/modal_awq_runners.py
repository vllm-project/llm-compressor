#!/usr/bin/env python3

import os
import sys
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
RECIPE_WITHOUT_SMOOTH = "tests/e2e/vLLM/recipes/WNA16/recipe_w4a16_awq_sym.yaml"
RECIPE_WITH_SMOOTH = (
    "tests/e2e/vLLM/recipes/WNA16/recipe_w4a16_awq_sym_with_smooth.yaml"
)

LMEVAL_TASK = "gsm8k"
LMEVAL_LIMIT = 100
LMEVAL_NUM_FEWSHOT = 5
LMEVAL_BATCH_SIZE = 4


def _make_image() -> modal.Image:
    """Build image with repo and deps (used at build time)."""
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
app = modal.App("awq-smooth-layer-runners")


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
    """Run AWQ compression (baseline, no smooth). Returns {'awq_time_s': ...}."""
    return _run_one(
        recipe_path=RECIPE_WITHOUT_SMOOTH,
        save_dir="llama3-8b-w4a16-awq-baseline",
        skip_lm_eval=skip_lm_eval,
    )


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
def run_with_smooth(skip_lm_eval: bool = False) -> dict:
    """Run AWQ compression (with smooth_layer_quantization). Returns {'awq_time_s': ...}."""
    return _run_one(
        recipe_path=RECIPE_WITH_SMOOTH,
        save_dir="llama3-8b-w4a16-awq-with-smooth",
        skip_lm_eval=skip_lm_eval,
    )


def _run_one(recipe_path: str, save_dir: str, skip_lm_eval: bool) -> dict:
    """Run compression; optionally run lm_eval in same container (so save_dir is on disk)."""
    os.chdir("/repo")
    from tests.e2e.e2e_utils import run_oneshot_for_e2e_testing

    recipe_abs = f"/repo/{recipe_path}"
    if not os.path.isfile(recipe_abs):
        return {"error": f"Recipe not found: {recipe_abs}", "awq_time_s": None, "lm_eval": None}

    # Compression
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
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    from llmcompressor.core import active_session
    active_session().reset()
    awq_time_s = time.perf_counter() - t0

    out = {"awq_time_s": awq_time_s, "lm_eval": None}
    if skip_lm_eval:
        return out

    # lm_eval in same container so save_dir exists on disk
    try:
        import lm_eval
        import lm_eval.api.registry
        import lm_eval.models  # noqa: F401
    except ImportError:
        return out

    from transformers import AutoTokenizer
    from tests.e2e.e2e_utils import load_model

    lm_eval_cls = lm_eval.api.registry.get_model("hf")
    model = load_model(save_dir, "AutoModelForCausalLM", device_map="cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(save_dir, fix_mistral_regex=True)

    try:
        results = lm_eval.simple_evaluate(
            model=lm_eval_cls(
                pretrained=model,
                tokenizer=tokenizer,
                batch_size=LMEVAL_BATCH_SIZE,
                add_bos_token=True,
                dtype="bfloat16",
            ),
            tasks=[LMEVAL_TASK],
            num_fewshot=LMEVAL_NUM_FEWSHOT,
            limit=LMEVAL_LIMIT,
            apply_chat_template=False,
            batch_size=LMEVAL_BATCH_SIZE,
            log_samples=True,
        )
    except TypeError:
        results = lm_eval.simple_evaluate(
            model=lm_eval_cls(
                pretrained=model,
                tokenizer=tokenizer,
                batch_size=LMEVAL_BATCH_SIZE,
                add_bos_token=True,
                dtype="bfloat16",
            ),
            tasks=[LMEVAL_TASK],
            num_fewshot=LMEVAL_NUM_FEWSHOT,
            limit=LMEVAL_LIMIT,
            apply_chat_template=False,
            batch_size=LMEVAL_BATCH_SIZE,
        )

    out["lm_eval"] = results.get("results", {}).get(LMEVAL_TASK, {})
    return out


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
def run_lm_eval(model_dir: str) -> dict:
    """Run lm_eval for a saved model directory; returns task metrics dict."""
    os.chdir("/repo")
    try:
        import lm_eval
        import lm_eval.api.registry
        import lm_eval.models  # noqa: F401 - populate registry so get_model("hf") works
    except ImportError:
        return {}

    from transformers import AutoTokenizer
    from tests.e2e.e2e_utils import load_model

    lm_eval_cls = lm_eval.api.registry.get_model("hf")
    model = load_model(model_dir, "AutoModelForCausalLM", device_map="cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, fix_mistral_regex=True)

    try:
        results = lm_eval.simple_evaluate(
            model=lm_eval_cls(
                pretrained=model,
                tokenizer=tokenizer,
                batch_size=LMEVAL_BATCH_SIZE,
                add_bos_token=True,
                dtype="bfloat16",
            ),
            tasks=[LMEVAL_TASK],
            num_fewshot=LMEVAL_NUM_FEWSHOT,
            limit=LMEVAL_LIMIT,
            apply_chat_template=False,
            batch_size=LMEVAL_BATCH_SIZE,
            log_samples=True,
        )
    except TypeError:
        results = lm_eval.simple_evaluate(
            model=lm_eval_cls(
                pretrained=model,
                tokenizer=tokenizer,
                batch_size=LMEVAL_BATCH_SIZE,
                add_bos_token=True,
                dtype="bfloat16",
            ),
            tasks=[LMEVAL_TASK],
            num_fewshot=LMEVAL_NUM_FEWSHOT,
            limit=LMEVAL_LIMIT,
            apply_chat_template=False,
            batch_size=LMEVAL_BATCH_SIZE,
        )

    return results.get("results", {}).get(LMEVAL_TASK, {})


@app.local_entrypoint()
def main_baseline(skip_lm_eval: bool = False):
    """Entrypoint: run baseline compression only."""
    result = run_baseline.remote(skip_lm_eval=skip_lm_eval)
    print("Baseline result:", result)


@app.local_entrypoint()
def main_with_smooth(skip_lm_eval: bool = False):
    """Entrypoint: run with-smooth compression only."""
    result = run_with_smooth.remote(skip_lm_eval=skip_lm_eval)
    print("With-smooth result:", result)


@app.local_entrypoint()
def main_both(skip_lm_eval: bool = False):
    """Run baseline and with-smooth AWQ; each run does eval in same container if not skipped."""
    print("Running baseline (no smooth_layer_quantization) on Modal...")
    baseline = run_baseline.remote(skip_lm_eval=skip_lm_eval)
    print("Baseline result:", baseline)

    print("Running with smooth_layer_quantization on Modal...")
    with_smooth = run_with_smooth.remote(skip_lm_eval=skip_lm_eval)
    print("With-smooth result:", with_smooth)
