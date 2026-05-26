import os
import shutil
from functools import wraps
from typing import Callable

import torch
import torch.distributed as dist
import transformers
from compressed_tensors.offload import load_offloaded_model
from datasets import load_dataset
from loguru import logger
from transformers import AutoProcessor, DefaultDataCollator

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from tests.test_timer.timer_utils import log_time
from tests.testing_utils import process_dataset

OFFLOAD_DIR = "./offload_folder"


def cleanup_offload_dir(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        finally:
            if os.path.exists(OFFLOAD_DIR):
                shutil.rmtree(OFFLOAD_DIR)

    return wrapper


def load_model(model: str, model_class: str, max_memory: dict[int | str, int] | None):
    pretrained_model_class = getattr(transformers, model_class)
    device_map = None
    if max_memory is not None or dist.is_initialized():
        device_map = "auto_offload"

    with load_offloaded_model(pretrained_model_class):
        loaded_model = pretrained_model_class.from_pretrained(
            model,
            device_map=device_map,
            max_memory=max_memory,
            offload_folder=OFFLOAD_DIR,
            dtype="auto",
        )
    return loaded_model


@log_time
def _run_oneshot(**oneshot_kwargs):
    oneshot(**oneshot_kwargs)


@cleanup_offload_dir
def run_oneshot_for_e2e_testing(
    model: str,
    model_class: str,
    max_memory: dict[int | str, int] | None,
    num_calibration_samples: int,
    max_seq_length: int,
    dataset_id: str,
    recipe: str,
    dataset_split: str,
    dataset_config: str,
    scheme: str,
    quant_type: str,
    save_dir: str,
    num_gpus: int = 1,
    save_compressed: bool = False,
    shuffle_calibration_samples: bool = True,
    data_collator: str | Callable = DefaultDataCollator(),
):
    if num_gpus == 1:
        run_oneshot_single(
            model=model,
            model_class=model_class,
            max_memory=max_memory,
            num_calibration_samples=num_calibration_samples,
            max_seq_length=max_seq_length,
            dataset_id=dataset_id,
            recipe=recipe,
            dataset_split=dataset_split,
            dataset_config=dataset_config,
            scheme=scheme,
            quant_type=quant_type,
            shuffle_calibration_samples=shuffle_calibration_samples,
            data_collator=data_collator,
            save_dir=save_dir,
            save_compressed=save_compressed,
        )
    else:
        # this just calls launch_ddp -> subprocess -> run_oneshot_ddp
        # the goal is to call run_oneshot_ddp with the correct torchrun invocation
        from tests.e2e.run_oneshot_ddp import launch_ddp

        config = {
            "model": model,
            "model_class": model_class,
            "max_memory": max_memory,
            "num_calibration_samples": num_calibration_samples,
            "max_seq_length": max_seq_length,
            "dataset_id": dataset_id,
            "dataset_split": dataset_split,
            "dataset_config": dataset_config,
            "scheme": scheme,
            "quant_type": quant_type,
            "recipe": recipe,
            "save_dir": save_dir,
            "save_compressed": save_compressed,
            "shuffle_calibration_samples": shuffle_calibration_samples,
        }
        logger.info(f"========== RUNNING DDP oneshot ({num_gpus} GPUs) ==========")
        launch_ddp(num_gpus, config)


def run_oneshot_single(
    model: str,
    model_class: str,
    max_memory: dict[int | str, int] | None,
    num_calibration_samples: int,
    max_seq_length: int,
    dataset_id: str,
    recipe: str,
    dataset_split: str,
    dataset_config: str,
    scheme: str,
    quant_type: str,
    save_dir: str,
    save_compressed: bool = False,
    shuffle_calibration_samples: bool = True,
    data_collator: str | Callable = DefaultDataCollator(),
):
    oneshot_kwargs, loaded_model, processor = prepare_oneshot_kwargs(
        model_id=model,
        model_class=model_class,
        max_memory=max_memory,
        dataset_id=dataset_id,
        dataset_config=dataset_config,
        dataset_split=dataset_split,
        max_seq_length=max_seq_length,
        num_calibration_samples=num_calibration_samples,
        recipe=recipe,
        quant_type=quant_type,
        scheme=scheme,
        shuffle_calibration_samples=shuffle_calibration_samples,
    )
    oneshot_kwargs.setdefault("data_collator", data_collator)

    logger.info(f"ONESHOT KWARGS: {oneshot_kwargs}")
    _run_oneshot(**oneshot_kwargs)

    logger.info("================= SAVING TO DISK ======================")
    save_output(loaded_model, processor, save_dir, save_compressed, reset_session=True)


def run_oneshot_ddp(config: dict, save_compressed: bool = False):
    """
    DDP variant of run_oneshot_for_e2e_testing.

    Assumes init_dist() has already been called. Loads the model with
    device_map="auto_offload", partitions calibration data across ranks, runs
    oneshot, and saves the result from rank 0.

    :param config: dict matching BaseTestConfig / TestConfig fields
    :param save_compressed: if True, saves with save_compressed=True and writes
        recipe.yaml from the active session (needed for e2e vLLM tests)
    """
    rank = dist.get_rank()

    oneshot_kwargs, loaded_model, processor = prepare_oneshot_kwargs(
        model_id=config["model"],
        model_class=config.get("model_class", "AutoModelForCausalLM"),
        max_memory=config.get("max_memory"),
        dataset_id=config.get("dataset_id"),
        dataset_config=config.get("dataset_config"),
        dataset_split=config.get("dataset_split"),
        max_seq_length=config.get("max_seq_length", 2048),
        num_calibration_samples=config.get("num_calibration_samples", 512),
        recipe=config.get("recipe"),
        quant_type=config.get("quant_type"),
        scheme=config.get("scheme"),
        shuffle_calibration_samples=config.get("shuffle_calibration_samples", True),
    )
    _run_oneshot(**oneshot_kwargs)

    if rank == 0:
        logger.info("================= SAVING TO DISK ======================")
        save_output(loaded_model, processor, config["save_dir"], save_compressed)

    dist.barrier()


def prepare_oneshot_kwargs(
    model_id: str,
    model_class: str,
    max_memory: dict[int | str, int] | None,
    dataset_id: str | None,
    dataset_config: str | None,
    dataset_split: str | None,
    max_seq_length: int,
    num_calibration_samples: int,
    recipe,
    quant_type: str | None,
    scheme: str | None,
    shuffle_calibration_samples: bool = True,
) -> tuple:
    from llmcompressor.datasets.utils import get_rank_partition

    loaded_model = load_model(model_id, model_class, max_memory)
    processor = AutoProcessor.from_pretrained(model_id)

    kwargs = {"model": loaded_model}

    if dataset_id:
        split = dataset_split
        if dist.is_initialized():
            split = get_rank_partition(dataset_split, num_calibration_samples)

        ds = load_dataset(dataset_id, name=dataset_config, split=split)
        ds = ds.shuffle(seed=42)

        if not dist.is_initialized():
            ds = ds.select(range(num_calibration_samples))

        ds = process_dataset(ds, processor, max_seq_length)
        kwargs["dataset"] = ds
        kwargs["max_seq_length"] = max_seq_length
        kwargs["num_calibration_samples"] = num_calibration_samples

        if "flickr30k" in dataset_id:

            def data_collator(batch):
                assert len(batch) == 1
                return {key: torch.tensor(value) for key, value in batch[0].items()}

            kwargs["data_collator"] = data_collator
        elif "calibration" in dataset_id:

            def data_collator(batch):
                assert len(batch) == 1
                return {
                    key: (
                        torch.tensor(value)
                        if key != "pixel_values"
                        else torch.tensor(value, dtype=torch.bfloat16).squeeze(0)
                    )
                    for key, value in batch[0].items()
                }

            kwargs["data_collator"] = data_collator

    kwargs["recipe"] = build_recipe(recipe, quant_type, scheme)
    kwargs["shuffle_calibration_samples"] = shuffle_calibration_samples

    return kwargs, loaded_model, processor


def build_recipe(recipe, quant_type, scheme):
    if recipe:
        return recipe
    if quant_type == "GPTQ":
        return GPTQModifier(
            targets="Linear",
            scheme=scheme,
            actorder=None,
            ignore=["lm_head", "re:.*mlp.gate[.].*"],
        )
    return QuantizationModifier(
        targets="Linear",
        scheme=scheme,
        ignore=["lm_head", "re:.*mlp.gate[.].*"],
    )


def save_output(model, processor, save_dir, save_compressed, reset_session=False):
    model.save_pretrained(save_dir, save_compressed=save_compressed)
    processor.save_pretrained(save_dir)
    if save_compressed:
        from llmcompressor.core import active_session

        session = active_session()
        recipe_yaml_str = session.get_serialized_recipe()
        assert recipe_yaml_str is not None, "Session contains no recipe after oneshot"
        with open(os.path.join(save_dir, "recipe.yaml"), "w") as f:
            f.write(recipe_yaml_str)
        if reset_session:
            session.reset()
