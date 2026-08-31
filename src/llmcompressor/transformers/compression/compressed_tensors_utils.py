import datetime
import os
import weakref
from contextlib import contextmanager
from functools import wraps

import torch
import torch.distributed as dist
from compressed_tensors import ModelCompressor, SparsityCompressionConfig
from compressed_tensors.config import CompressionFormat
from compressed_tensors.distributed import is_source_process
from compressed_tensors.offload import OffloadCache, from_accelerate, to_accelerate
from compressed_tensors.utils import deprecated
from loguru import logger
from transformers import PreTrainedModel

from llmcompressor.core import active_session
from llmcompressor.pytorch.model_load.helpers import copy_python_files_from_model_cache
from llmcompressor.transformers.utils import RECIPE_FILE_NAME
from llmcompressor.transformers.utils.helpers import infer_recipe_from_model_path
from llmcompressor.utils.transformers import get_embeddings

__all__ = ["modify_save_pretrained"]


def _named_tensors(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return a module's own (non-None) parameters and buffers keyed by name.

    ``None`` placeholders are skipped so e.g. a biasless ``lm_head`` (which still
    registers ``bias=None``) compares equal to an ``Embedding`` that has none.
    """
    names = list(module._parameters.keys()) + list(module._buffers.keys())
    tensors = {name: getattr(module, name) for name in names}
    return {name: tensor for name, tensor in tensors.items() if tensor is not None}


def _retie_embeddings(model: PreTrainedModel):
    """Re-tie input and output embeddings before saving so one shared table is
    written instead of a duplicate.

    Embeddings can end up as value-identical but separate tensors that defeat
    transformers' save-time de-duplication: offloading splits a tied weight into
    per-module parameters, and embeddings targeted for quantization are untied
    during calibration and compressed independently. Whenever the two embeddings
    hold exactly the same tensors they represent a single table, so point the
    output's at the input's and restore ``tie_word_embeddings``. Embeddings that
    differ -- quantized differently, or an untied model whose head has diverged
    -- are left untouched. Comparing the tensors themselves keeps this agnostic
    to the compression format (packed int, fp8, or dense).

    Args:
        model: The model about to be saved.
    """
    input_embed, output_embed = get_embeddings(model)
    if input_embed is None or output_embed is None or input_embed is output_embed:
        return

    input_tensors = _named_tensors(input_embed)
    output_tensors = _named_tensors(output_embed)
    if input_tensors.keys() != output_tensors.keys() or not all(
        torch.equal(input_tensors[name], output_tensors[name]) for name in input_tensors
    ):
        return

    # Share the input's storage so transformers' save-time de-duplication keeps a
    # single copy. ``disable_onloading`` makes the read and assignment go by
    # reference (a plain assignment to an offloaded module copies); it can be
    # dropped once compressed-tensors makes ``__setitem__`` a non-copying
    # replacement (vllm-project/compressed-tensors#709).
    with OffloadCache.disable_onloading():
        for name in input_tensors:
            setattr(output_embed, name, getattr(input_embed, name))

    config = getattr(model, "config", None)
    get_text_config = getattr(config, "get_text_config", None)
    text_config = get_text_config(decoder=True) if callable(get_text_config) else config
    if text_config is not None:
        text_config.tie_word_embeddings = True
    logger.info("Re-tied input/output embeddings; saving a single shared table.")


def _extract_mtp_scheme(quant_config_dict: dict):
    """
    Derive a QuantizationScheme suitable for MTP layers from the saved
    quantization_config. Returns None if the config cannot be parsed.

    Microscale formats (NVFP4/MXFP4) are not yet fully supported for MTP
    because they require special fused-set handling. Those cases fall back to
    FP8-block quantization.
    # TODO: add full microscale support for MTP layer quantization
    """
    from compressed_tensors.quantization import (
        QuantizationArgs,
        QuantizationConfig,
        QuantizationScheme,
        QuantizationStrategy,
        QuantizationType,
    )

    if not quant_config_dict:
        return None
    try:
        qconfig = QuantizationConfig.model_validate(quant_config_dict)
    except Exception:
        return None

    if not qconfig.config_groups:
        return None

    primary_scheme = next(iter(qconfig.config_groups.values()))
    weights_args = primary_scheme.weights
    if weights_args is None:
        return None

    # microscale schemes (NVFP4/MXFP4) require fused-set handling not yet
    # supported for MTP layers — fall back to FP8-block
    is_microscale = weights_args.num_bits == 4 and weights_args.type in (
        QuantizationType.FLOAT,
        "float",
    )
    if is_microscale:
        logger.warning(
            "Main model uses a microscale scheme (NVFP4/MXFP4). MTP layer "
            "quantization falls back to FP8-block. "
            "TODO: add full microscale support for MTP layer quantization."
        )
        weights_args = QuantizationArgs(
            num_bits=8,
            type=QuantizationType.FLOAT,
            strategy=QuantizationStrategy.BLOCK,
            block_structure=[128, 128],
        )

    return QuantizationScheme(targets=["re:.*\\.weight"], weights=weights_args)


def _quantize_and_save_mtp_tensors(
    source_model: str,
    dest_dir: str,
    mtp_prefix: str,
    shard_name: str = "model_mtp.safetensors",
):
    """
    Load MTP tensors from source_model, quantize them with the scheme from
    dest_dir's quantization_config, and save the result as a new shard.
    Falls back to saving unquantized tensors and marking them ignored when
    no quantization config is found.
    """
    import json

    # Load MTP tensors from the original (unquantized) source checkpoint.
    # AutoModel never downloads MTP shards (those keys are in
    # _keys_to_ignore_on_load_unexpected), so we read the index directly and
    # trigger a targeted download of only the shard(s) that contain MTP keys.
    import json as _json

    from compressed_tensors.base import QUANTIZATION_CONFIG_NAME
    from compressed_tensors.compressors import compress_module
    from compressed_tensors.utils.safetensors_load import (
        find_config_path,
        get_safetensors_folder,
        get_weight_mappings,
        update_safetensors_index,
    )
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from safetensors.torch import save_file

    from llmcompressor.entrypoints.model_free.lifecycle import (
        calibrate_weight,
        initialize_quantized_linear,
    )

    source_dir = get_safetensors_folder(source_model)
    index_path = os.path.join(source_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            raw_map = _json.load(f)["weight_map"]
        # find which shard file(s) hold the MTP keys and download if missing
        mtp_shards = {v for k, v in raw_map.items() if k.startswith(mtp_prefix)}
        for shard_file in mtp_shards:
            local_shard = os.path.join(source_dir, shard_file)
            if not os.path.exists(local_shard):
                try:
                    hf_hub_download(source_model, shard_file)
                except Exception:
                    pass  # local-only model, nothing to download
        full_weight_map = {k: os.path.join(source_dir, v) for k, v in raw_map.items()}
    else:
        full_weight_map = get_weight_mappings(source_dir)

    mtp_tensors = {}
    for key, filepath in full_weight_map.items():
        if key.startswith(mtp_prefix):
            with safe_open(filepath, framework="pt", device="cpu") as f:
                mtp_tensors[key] = f.get_tensor(key)

    if not mtp_tensors:
        raise ValueError(
            f"No tensors with prefix '{mtp_prefix}' found in {source_model}"
        )

    # read quantization scheme from the already-saved dest config
    config_path = find_config_path(dest_dir)
    scheme = None
    config = None
    if config_path is not None:
        with open(config_path) as f:
            config = json.load(f)
        quant_config_dict = config.get(QUANTIZATION_CONFIG_NAME)
        if quant_config_dict:
            scheme = _extract_mtp_scheme(quant_config_dict)

    if scheme is not None:
        output_tensors = {}
        for key, tensor in mtp_tensors.items():
            if not key.endswith(".weight"):
                output_tensors[key] = tensor
                continue
            module_name = key[: -len(".weight")]
            try:
                module = initialize_quantized_linear(tensor, scheme, "cpu")
                calibrate_weight(module)
                compress_module(module)
                for k, v in module.state_dict(prefix=module_name + ".").items():
                    output_tensors[k] = v.cpu()
            except Exception as exc:
                logger.warning(
                    f"Could not quantize MTP tensor {key}: {exc}. "
                    "Keeping at full precision."
                )
                output_tensors[key] = tensor
    else:
        # no quantization config found — save unquantized and add to ignore
        output_tensors = mtp_tensors

    # save the shard
    save_file(output_tensors, os.path.join(dest_dir, shard_name))

    # update the safetensors index to include the new shard
    dest_weight_map = {
        k: os.path.basename(v) for k, v in get_weight_mappings(dest_dir).items()
    }
    dest_weight_map.update({key: shard_name for key in output_tensors})
    total_size = sum(
        os.path.getsize(os.path.join(dest_dir, s))
        for s in set(dest_weight_map.values())
    )
    update_safetensors_index(dest_dir, total_size, dest_weight_map)

    if config is not None and config_path is not None:
        quant_config = config.get(QUANTIZATION_CONFIG_NAME)
        if quant_config is not None:
            if scheme is not None:
                # add an explicit config group so inference engines know the
                # quantization scheme applied to MTP layers
                groups = quant_config.get("config_groups") or {}
                groups["mtp_group"] = {
                    "targets": [f"re:^{mtp_prefix}\\..*\\.weight"],
                    "weights": scheme.weights.model_dump(),
                }
                quant_config["config_groups"] = groups
            else:
                # unquantized fallback — mark MTP as ignored
                ignore_list = quant_config.get("ignore") or []
                mtp_ignore_pattern = f"re:^{mtp_prefix}.*"
                if mtp_ignore_pattern not in ignore_list:
                    ignore_list.append(mtp_ignore_pattern)
                    quant_config["ignore"] = ignore_list
            config[QUANTIZATION_CONFIG_NAME] = quant_config
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)


def _get_mtp_prefix(source_model: str, text_config) -> str:
    """
    Detect the tensor key prefix used for MTP layers in a checkpoint.
    Qwen/Nemotron/DeepSeek use a top-level "mtp" prefix; GLM-style models
    store MTP as the last N layers of model.layers (or model.language_model.layers
    for VLMs).

    Reads from the safetensors index rather than the downloaded files so that
    the MTP shard does not need to be present locally (AutoModel never downloads
    it because the keys are in _keys_to_ignore_on_load_unexpected).
    """
    import json

    from compressed_tensors.utils.safetensors_load import get_safetensors_folder
    from huggingface_hub import hf_hub_download

    source_dir = get_safetensors_folder(source_model)
    index_path = os.path.join(source_dir, "model.safetensors.index.json")

    # fall back to downloading the index from the hub if not on disk
    if not os.path.exists(index_path):
        try:
            index_path = hf_hub_download(source_model, "model.safetensors.index.json")
        except Exception:
            index_path = None

    if index_path and os.path.exists(index_path):
        with open(index_path) as f:
            all_keys = list(json.load(f).get("weight_map", {}).keys())
    else:
        # single-shard model — read keys directly from the safetensors metadata
        from safetensors import safe_open

        shard = os.path.join(source_dir, "model.safetensors")
        with safe_open(shard, framework="pt", device="cpu") as f:
            all_keys = list(f.keys())

    if any(k.startswith("mtp") for k in all_keys):
        return "mtp"

    # GLM-style: MTP is stored as the last N layers of model.layers
    num_hidden = getattr(text_config, "num_hidden_layers", None)
    if num_hidden is not None:
        for prefix in (
            f"model.language_model.layers.{num_hidden}",  # VLM (e.g. GLM5)
            f"model.layers.{num_hidden}",  # text-only
        ):
            if any(k.startswith(prefix) for k in all_keys):
                return prefix

    raise ValueError(
        f"Could not detect MTP tensor prefix in {source_model}. "
        "Check the checkpoint structure or set mtp_prefix manually."
    )


def modify_save_pretrained(model: PreTrainedModel):
    """
    Overrides a PreTrainedModel's save_pretrained() method with a wrapped version that
    supports compression. The new save_pretrained function performs the following saving
    operations:

    1. Saves the model state, potentially in a compressed format
    2. Saves the recipe, appending any current recipes to existing recipe files
    3. Copies any necessary python files from the model cache
    """

    def save_pretrained_compressed(save_pretrained_method):
        if getattr(save_pretrained_method, "_overridden", False):
            # `model.save_pretrained` has already been replaced, return.
            return save_pretrained_method

        # Keep a weak reference to the model class and unbound save_pretrained
        # method so we can call the original
        model_ref = weakref.ref(save_pretrained_method.__self__)
        original_save_fn = save_pretrained_method.__func__
        model_class = model_ref().__class__
        del save_pretrained_method

        @wraps(original_save_fn)
        def save_pretrained_wrapper(
            save_directory: str,
            quantization_format: str | None = None,
            save_compressed: bool = True,
            **kwargs,
        ):
            """
            Wrapper around PreTrainedModel.save_pretrained(), adds functionality for
            saving models in a compressed format on disk. The compression format is
            saved to the model's config file

            :param save_directory: output directory to save model to
            :param quantization_format: optional compression format override. If none
                is provided, the compression format will be inferred from the model
            :param save_compressed: whether or not to compress the model. If true,
                weights will be compressed. Otherwise, weights will remain in full
                precision in the "FROZEN" state.
            :param kwargs: additional kwargs to pass on to model.save_pretrained
            """

            kwargs.setdefault("max_shard_size", "20GB")

            # compress model using compressor
            compressor = ModelCompressor.from_pretrained_model(
                model, quantization_format=quantization_format
            )
            if save_compressed:
                compressor.compress_model(model, skip_compressed=True)

            # Re-tie input and output embeddings before offload conversion so a
            # shared table is written once. Offloading splits a tied weight into
            # separate params, and quantized embeddings are untied during
            # calibration; either way identical tensors would otherwise be saved
            # twice. Doing this before `to_accelerate` keeps accelerate's
            # tied-parameter bookkeeping consistent.
            _retie_embeddings(model)

            # convert to accelerate offloaded for optimal saving with transformers
            to_accelerate(model)

            with suspend_distributed_timeout():
                if is_source_process():
                    # save model structure
                    original_save_fn.__get__(model, model_class)(
                        save_directory, **kwargs
                    )

                    # update config to reflect quantization
                    compressor.update_config(save_directory)

                    # update existing recipe
                    update_and_save_recipe(model.name_or_path, save_directory)

                    # copy python files from cache dir to save_path if any
                    copy_python_files_from_model_cache(model, save_directory)

                    # copy mtp tensors (not loaded by transformers) and update config
                    text_config = model.config.get_text_config()
                    has_mtp = (
                        getattr(text_config, "num_mtp_layers", 0)
                        or getattr(text_config, "mtp_num_hidden_layers", 0)
                        or getattr(text_config, "num_nextn_predict_layers", 0)
                    )
                    if has_mtp:
                        mtp_prefix = _get_mtp_prefix(model.name_or_path, text_config)
                        _quantize_and_save_mtp_tensors(
                            model.name_or_path, save_directory, mtp_prefix=mtp_prefix
                        )

            # convert back from accelerate to restore model to original form
            from_accelerate(model)

        save_pretrained_wrapper._overridden = True
        return save_pretrained_wrapper

    # wrap save_pretrained if not already
    if not getattr(model.save_pretrained, "_overridden", False):
        model.save_pretrained = save_pretrained_compressed(model.save_pretrained)


@deprecated("ModelCompressor.from_pretrained_model")
def get_model_compressor(
    model: torch.nn.Module,
    sparsity_config: SparsityCompressionConfig | None = None,
    quantization_format: str | None = None,
    save_compressed: bool = True,
    skip_sparsity_compression_stats: bool = True,
    disable_sparse_compression: bool = False,
):
    """
    Obtain the compressor based on the config and the quantization_format

    :param model: torch model
    :param sparsify_config: Sparsity Compression config
    :param quantization_format: Format that the model was quantized to.
        if not provivided, will be extrapolated from `infer_quantization_format`
    :param save_compressed: boolean representing to save in a compressed
        format
    :param skip_sparsity_compression_stats: bool allowing compression stats on std out
    :param disable_sparse_compression: bool to skip sparse compression
    """

    if (
        sparsity_config is not None
        or not skip_sparsity_compression_stats
        or disable_sparse_compression
    ):
        logger.warning(
            "Sparse compression is no longer supported by compressed-tensors"
        )

    if not save_compressed:
        if quantization_format not in (None, CompressionFormat.dense.value):
            raise ValueError(
                "A quantizatiom format was provided but "
                "save_compressed is set to False. "
                "A compression format can only be applied when "
                "saving the model compressed"
            )
        quantization_format = CompressionFormat.dense.value

    return ModelCompressor.from_pretrained_model(
        model,
        quantization_format=quantization_format,
    )


def update_and_save_recipe(model_stub: str, save_directory: str):
    """
    Save a recipe ontop of any existing recipe files located at model_stub

    :param model_stub: path to existing model or model stub which may contain an
        existing recipe
    :param save_directory: path to save combined existing recipe and current recipe
    """

    existing_recipe = infer_recipe_from_model_path(model_stub)

    recipe = active_session().lifecycle.recipe

    recipe_path = os.path.join(save_directory, RECIPE_FILE_NAME)
    recipe.yaml(file_path=recipe_path, existing_recipe_path=existing_recipe)


@contextmanager
def suspend_distributed_timeout(
    timeout: datetime.timedelta = datetime.timedelta(hours=3),
    current_group: dist.ProcessGroup | None = None,
):
    """
    Context manager that extends the timeout for distributed operations.

    Creates a temporary process group with an extended timeout to prevent
    timeout errors during long-running operations (e.g., model saving) in
    distributed training environments. The context manager synchronizes all
    processes before and after the operation using barriers.

    :param timeout: The extended timeout for the temporary process group.
        Defaults to 3 hours
    :param current_group: The current process group to synchronize. If None,
        defaults to dist.group.WORLD
    """
    if not dist.is_initialized():
        yield
        return

    if current_group is None:
        current_group = dist.group.WORLD
    suspend_group = dist.new_group(backend="gloo", timeout=timeout)

    try:
        dist.barrier(group=current_group)
        yield
    finally:
        dist.barrier(group=suspend_group)
        dist.barrier(group=current_group)
        dist.destroy_process_group(group=suspend_group)
