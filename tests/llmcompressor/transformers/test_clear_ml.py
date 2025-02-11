from pathlib import Path

import pytest
import torch

try:
    from clearml import Task

    is_clearml = True
except Exception:
    is_clearml = False

from llmcompressor.transformers import train


@pytest.mark.skipif(not is_clearml, reason="clearML not installed")
def test_finetune_wout_recipe(tmp_path: Path):
    recipe_str = None
    model = "Xenova/llama2.c-stories15M"
    device = "cuda:0"
    if not torch.cuda.is_available():
        device = "cpu"
    dataset = "open_platypus"
    concatenate_data = False
    output_dir = tmp_path
    max_steps = 50
    splits = "train"

    Task.init(project_name="test", task_name="test_oneshot_and_finetune")

    train(
        model=model,
        dataset=dataset,
        output_dir=output_dir,
        recipe=recipe_str,
        max_steps=max_steps,
        concatenate_data=concatenate_data,
        splits=splits,
        post_train_device=device,
    )
