import contextlib
import logging
import os
import tempfile
from functools import wraps
from typing import Type

import torch
from compressed_tensors.offload import dispatch_model, load_offloaded_model
from compressed_tensors.utils import deprecated, patch_attr
from huggingface_hub import snapshot_download
from loguru import logger
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.conversion_mapping import extract_weight_conversions_for_model

try:
    # Transformers < v5 support
    from transformers.modeling_utils import TORCH_INIT_FUNCTIONS
except ImportError:
    # Transformers v5 support
    from transformers.initialization import TORCH_INIT_FUNCTIONS
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, WEIGHTS_INDEX_NAME

__all__ = [
    "skip_weights_download",
    "patch_transformers_logger_level",
    "get_main_device",
    "get_high_precision",
    "dispatch_for_generation",
    "load_context",
]


# redundant conversion artifacts which transformers never loads, but which repos
# commonly ship and which are expensive to fetch (ex. `original/consolidated.00.pth`
# in Meta's Llama repos). Kept deliberately narrow: anything broader risks skipping
# a file some checkpoint or custom modeling code genuinely needs
_UNUSED_CHECKPOINT_PATTERNS = ["original/**"]


@contextlib.contextmanager
def download_checkpoint_first(model_cls: Type[PreTrainedModel] = AutoModelForCausalLM):
    """
    Context manager which downloads a checkpoint on local rank zero before any rank
    attempts to load it.

    Ranks which populate and resolve the same hub cache concurrently can raise
    `OSError: ... does not appear to have files named (...)` for shards which are
    present on disk, because transformers resolves the cache through
    `try_to_load_from_cache` while another rank is writing it.
    See https://github.com/vllm-project/llm-compressor/issues/2984

    Prefetching removes that overlap for the ranks of this job: none of them resolve
    the cache until every downloader has finished. One downloader per node is
    required for node-local caches, which would otherwise stay cold on every node
    but one. Note that this does not protect against unrelated processes writing the
    same cache concurrently, and that a cache shared across nodes still has one
    downloader per node.

    Assumes every rank receives the same arguments and sees the same filesystem, as
    is the case for a torchrun job running one script. Ranks which disagree on
    whether the checkpoint is a local directory would not reach the same collectives.

    :param model_cls: The model class to patch, defaults to AutoModelForCausalLM
    """
    if (
        not torch.distributed.is_initialized()
        or torch.distributed.get_world_size() <= 1
    ):
        yield
        return

    original_from_pretrained = model_cls.from_pretrained

    @classmethod
    @wraps(original_from_pretrained)
    def patched(cls, *args, **kwargs):
        stub = args[0] if args else kwargs.get("pretrained_model_name_or_path")

        # local checkpoints and offline loads never populate the hub cache. A
        # missing stub is left for transformers to report
        if stub is None or os.path.isdir(stub) or kwargs.get("local_files_only"):
            return original_from_pretrained(*args, **kwargs)

        # raises if the launcher did not set `LOCAL_RANK`, which is safer than
        # assuming zero and starting a downloader on every rank
        local_rank = torch.distributed.get_node_local_rank()

        error = None
        try:
            if local_rank == 0:
                snapshot_download(
                    stub,
                    revision=kwargs.get("revision"),
                    cache_dir=kwargs.get("cache_dir"),
                    token=kwargs.get("token"),
                    force_download=kwargs.get("force_download", False),
                    ignore_patterns=_UNUSED_CHECKPOINT_PATTERNS,
                )
        except Exception as exception:
            error = exception

        # nccl must reduce on the accelerator, gloo reduces on cpu
        if torch.distributed.get_backend() == "nccl":
            accelerator = torch.accelerator.current_accelerator(check_available=True)
            if accelerator is None:
                raise RuntimeError("NCCL backend requires an available accelerator")
            device = torch.device(
                accelerator.type, torch.accelerator.current_device_index()
            )
        else:
            device = torch.device("cpu")

        # synchronizes ranks and ensures that none of them load against a cache
        # which a downloader failed to populate
        succeeded = torch.tensor(error is None, dtype=torch.uint8, device=device)
        torch.distributed.all_reduce(succeeded, op=torch.distributed.ReduceOp.MIN)
        if not succeeded.item():
            if error is not None:
                raise error
            raise RuntimeError(
                "Checkpoint prefetch failed on another rank. See that rank's "
                "traceback for the underlying error"
            )

        # the checkpoint is cached now, and re-downloading it on every rank would
        # restore exactly the concurrency this context exists to remove
        kwargs["force_download"] = False

        return original_from_pretrained(*args, **kwargs)

    with patch_attr(model_cls, "from_pretrained", patched):
        yield


@contextlib.contextmanager
def load_context(model_cls: Type[PreTrainedModel] = AutoModelForCausalLM):
    """
    Context manager for loading HuggingFace models with both offloading and
    MoE linearization support.

    This context manager combines `download_checkpoint_first`, `load_offloaded_model`
    and `load_quantizable_moe` contexts to provide a unified interface for loading
    models that may require any of those capabilities.

    :param model_cls: The model class to patch, defaults to AutoModelForCausalLM
    """
    from llmcompressor.modeling.moe.linearize import load_quantizable_moe

    with contextlib.ExitStack() as stack:
        stack.enter_context(load_offloaded_model(model_cls))
        stack.enter_context(load_quantizable_moe(model_cls))
        # entered last so that the checkpoint is fetched before any other patch runs
        stack.enter_context(download_checkpoint_first(model_cls))
        yield


@contextlib.contextmanager
def skip_weights_download(model_class: Type[PreTrainedModel] = AutoModelForCausalLM):
    """
    Context manager under which models are initialized without having to download
    the model weight files. This differs from `init_empty_weights` in that weights are
    allocated on to assigned devices with random values, as opposed to being on the meta
    device

    :param model_class: class to patch, defaults to `AutoModelForCausalLM`
    """
    original_fn = model_class.from_pretrained
    weights_files = [
        "*.bin",
        "*.safetensors",
        "*.pth",
        SAFE_WEIGHTS_INDEX_NAME,
        WEIGHTS_INDEX_NAME,
        "*.msgpack",
        "*.pt",
    ]

    @classmethod
    def patched(cls, *args, **kwargs):
        nonlocal tmp_dir

        # intercept model stub
        model_stub = args[0] if args else kwargs.pop("pretrained_model_name_or_path")

        # download files into tmp dir
        os.makedirs(tmp_dir, exist_ok=True)
        snapshot_download(
            repo_id=model_stub, local_dir=tmp_dir, ignore_patterns=weights_files
        )

        # make an empty weights file to avoid errors
        weights_file_path = os.path.join(tmp_dir, "model.safetensors")
        save_file({}, weights_file_path, metadata={"format": "pt"})

        # load from tmp dir
        model = original_fn(tmp_dir, **kwargs)

        # replace model_path
        model.name_or_path = model_stub
        model.config._name_or_path = model_stub

        # normally transformers populates `_weight_conversions` based on which were used
        # but none were used (since no weights were loaded), so populate directly
        # so that saving in original checkpoint format still works
        model._weight_conversions = extract_weight_conversions_for_model(model)

        return model

    with (
        tempfile.TemporaryDirectory() as tmp_dir,
        patch_attr(model_class, "from_pretrained", patched),
        skip_weights_initialize(),
        patch_transformers_logger_level(),
    ):
        yield


@contextlib.contextmanager
def skip_weights_initialize(use_zeros: bool = False):
    """
    Very similar to `transformers.model_utils.no_init_weights`, except that torch.Tensor
    initialization functions are also patched to account for tensors which are
    initialized not on the meta device
    """

    def skip(tensor: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if use_zeros:
            return tensor.fill_(0)
        return tensor

    with contextlib.ExitStack() as stack:
        for name in TORCH_INIT_FUNCTIONS.keys():
            stack.enter_context(patch_attr(torch.nn.init, name, skip))
            stack.enter_context(patch_attr(torch.Tensor, name, skip))
        yield


@contextlib.contextmanager
def patch_transformers_logger_level(level: int = logging.ERROR):
    """
    Context under which the transformers logger's level is modified

    This can be used with `skip_weights_download` to squelch warnings related to
    missing parameters in the checkpoint

    :param level: new logging level for transformers logger. Logs whose level is below
        this level will not be logged
    """
    transformers_logger = logging.getLogger("transformers.modeling_utils")
    restore_log_level = transformers_logger.getEffectiveLevel()

    transformers_logger.setLevel(level=level)
    try:
        yield
    finally:
        transformers_logger.setLevel(level=restore_log_level)


def get_main_device() -> torch.device:
    is_distributed_enable = torch.distributed.is_initialized()
    rank = 0 if not is_distributed_enable else torch.distributed.get_rank()

    # Check for unsupported MPS + distributed combination
    if hasattr(torch, "mps") and torch.mps.is_available():
        if is_distributed_enable:
            raise RuntimeError("Parallelism has not been supported for MPS")
        return torch.device("mps")

    elif torch.accelerator.is_available():
        accel_type = torch.accelerator.current_accelerator().type
        return torch.device(accel_type, rank)
    else:
        logger.warning("No accelerator available! Compressing model on CPU instead")
        return torch.device("cpu")


def get_high_precision() -> torch.dtype:
    main_device = get_main_device()

    if main_device.type == "mps":  # MPS does not support float64
        return torch.float32

    return torch.float64


@deprecated("compressed_tensors.offload::dispatch_model")
@wraps(dispatch_model)
def dispatch_for_generation(*args, **kwargs) -> PreTrainedModel:
    """
    Dispatch a model autoregressive generation. This means that modules are dispatched
    evenly across avaiable devices and kept onloaded if possible.

    :param model: model to dispatch
    :param hint_batch_size: reserve memory for batch size of inputs
    :param hint_batch_seq_len: reserve memory for sequence of length of inputs
    :param hint_model_dtype: reserve memory for model's dtype.
        Will be inferred from model if none is provided
    :param hint_extra_memory: extra memory reserved for model serving
    :param no_split_modules: names of module classes which should not be split
        across multiple devices
    :return: dispatched model
    """
    return dispatch_model(*args, **kwargs)
