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


# MTP scheme values that mean "leave MTP layers at full precision".
_MTP_UNQUANTIZED_ALIASES = frozenset(
    {"bf16", "bfloat16", "none", "dense", "unquantized"}
)

# MTP fusion-projection names that must stay full precision: these are plain
# nn.Linear layers with no associated scale parameter in vLLM MTP loaders
# (verified against vLLM 0.26.0 Qwen3_5MTP, DeepseekV3MTP, GLM4MTP).
_FUSION_PROJ_NAMES = frozenset({"eh_proj", "fc"})


def _drop_uncalibratable_input_activations(scheme):
    """
    Return ``scheme`` with input-activation quantization that requires
    calibration removed, keeping the weights quantized.

    MTP layers are never run during calibration, so any input-activation scale
    that depends on observed activations cannot be produced. This covers two
    cases:

    - ``dynamic=False`` (fully static): a per-tensor scale calibrated from
      activations.
    - ``dynamic="local"`` (e.g. NVFP4 ``tensor_group``): per-group micro-scales
      are computed at runtime, but a per-tensor ``input_global_scale`` is still
      static and must be calibrated.

    Both would save a checkpoint whose ``input_global_scale`` is meaningless
    (never observed), producing ``1/scale -> inf`` at load time; drop them to
    weight-only. Only fully dynamic activation quant (``dynamic is True``, e.g.
    FP8_DYNAMIC), whose scales are computed entirely at runtime, needs no
    calibration and is kept.
    """
    input_acts = scheme.input_activations
    # Keep only fully-dynamic activations (dynamic is True). Static (False) and
    # local (DynamicType.LOCAL) both carry a calibration-dependent scale.
    if input_acts is not None and getattr(input_acts, "dynamic", False) is not True:
        logger.warning(
            "mtp_scheme uses input-activation quantization with a "
            "calibration-dependent scale, which cannot be calibrated for MTP "
            "layers; they will be weight-only quantized."
        )
        return scheme.model_copy(update={"input_activations": None})
    return scheme


def _resolve_mtp_scheme(mtp_scheme):
    """
    Resolve the caller-supplied ``mtp_scheme`` into a QuantizationScheme, or
    None to keep the MTP layers full precision (bf16).

    MTP quantization is opt-in and explicit: ``save_pretrained`` leaves MTP
    layers unquantized unless the caller passes ``mtp_scheme``. Accepted values:

    - ``None`` (default): keep MTP full precision (bf16).
    - a ``QuantizationScheme``: quantize the MTP linears with it directly.
    - a preset name string (e.g. ``"NVFP4"``, ``"FP8_DYNAMIC"``): resolve it
      with compressed-tensors' ``preset_name_to_scheme``.
    - ``"bf16"`` / ``"none"`` / ``"dense"`` / ``"unquantized"``: same as None.

    Input-activation quantization whose scale must be calibrated (static, or
    NVFP4-style ``dynamic="local"`` with a static ``input_global_scale``) is
    always dropped to weight-only (see
    ``_drop_uncalibratable_input_activations``), since MTP layers are never
    observed. Only fully dynamic activation quant is kept.
    """
    from compressed_tensors.quantization import (
        QuantizationScheme,
        preset_name_to_scheme,
    )

    if mtp_scheme is None:
        return None
    if isinstance(mtp_scheme, QuantizationScheme):
        scheme = mtp_scheme
    elif isinstance(mtp_scheme, str):
        normalized = mtp_scheme.strip()
        if normalized.lower() in _MTP_UNQUANTIZED_ALIASES:
            return None
        # Otherwise treat it as a preset scheme name (e.g. "NVFP4").
        scheme = preset_name_to_scheme(normalized.upper(), targets=["re:.*\\.weight"])
    else:
        raise TypeError(
            "mtp_scheme must be None, a QuantizationScheme, or a preset name "
            f"str, got {type(mtp_scheme).__name__}"
        )

    return _drop_uncalibratable_input_activations(scheme)


def _compress_mtp_linears(
    to_quantize: dict,
    scheme,
    output_tensors: dict,
) -> list:
    """
    Quantize and compress a set of 2D MTP linear weights using
    ``ModelFreePtqConverter``, which handles both per-tensor and microscale
    (NVFP4/MXFP4) schemes including fused-set global-scale coordination.
    Compressed tensors are written into ``output_tensors``; the list of
    quantized module names is returned.
    """
    from compressed_tensors.quantization import QuantizationConfig, QuantizationScheme

    from llmcompressor.entrypoints.model_free.converter import ModelFreePtqConverter

    # Build a minimal QuantizationConfig targeting every key in to_quantize.
    # The converter matches on the module name (key without ".weight"), so
    # "re:.*" covers all of them without needing exact names up front.
    mtp_scheme = QuantizationScheme(
        targets=["re:.*"],
        weights=scheme.weights,
        input_activations=scheme.input_activations,
    )
    config = QuantizationConfig(config_groups={"mtp": mtp_scheme})
    converter = ModelFreePtqConverter(config)

    compressed = converter.process(dict(to_quantize))
    output_tensors.update(compressed)

    # Module names = keys that lost the ".weight" suffix (i.e. the inputs that
    # now have derived keys like ".weight_packed", ".weight_scale", etc.)
    input_weight_keys = set(to_quantize.keys())
    quantized_modules = list(
        {
            k.rsplit(".", 1)[0]
            for k in compressed
            if k not in input_weight_keys
        }
    )
    return quantized_modules


def _quantize_and_save_mtp_tensors(
    source_model: str,
    dest_dir: str,
    mtp_prefix: str,
    vocab_size: int | None = None,
    shard_name: str = "model_mtp.safetensors",
    mtp_scheme=None,
):
    """
    Load MTP tensors from source_model and save them as a new shard.

    MTP quantization is opt-in via ``mtp_scheme`` (see ``_resolve_mtp_scheme``).
    When it resolves to a scheme, the MTP linears are quantized with it;
    otherwise (the default) the tensors are saved full precision (bf16) and the
    MTP prefix is added to the quantization_config's ``ignore`` list so
    inference engines skip it.
    """
    import json
    import re

    from compressed_tensors.base import QUANTIZATION_CONFIG_NAME
    from compressed_tensors.utils.safetensors_load import (
        find_config_path,
        get_safetensors_folder,
        get_weight_mappings,
        update_safetensors_index,
    )
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from safetensors.torch import save_file

    # Load MTP tensors from the original (unquantized) source checkpoint.
    # AutoModel never downloads MTP shards (those keys are in
    # _keys_to_ignore_on_load_unexpected), so we read the index directly and
    # trigger a targeted download of only the shard(s) that contain MTP keys.
    source_dir = get_safetensors_folder(source_model)
    index_path = os.path.join(source_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            raw_map = json.load(f)["weight_map"]
        # find which shard file(s) hold the MTP keys and download if missing;
        # build a per-shard resolved path so source_dir is never mutated (which
        # would break paths for shards that were already present locally)
        mtp_shards = {v for k, v in raw_map.items() if k.startswith(mtp_prefix + ".")}
        resolved_shard = {}  # shard_filename -> absolute local path
        for shard_file in mtp_shards:
            local_shard = os.path.join(source_dir, shard_file)
            if not os.path.exists(local_shard):
                try:
                    local_shard = hf_hub_download(source_model, shard_file)
                except Exception as exc:
                    logger.warning(
                        f"Could not download MTP shard {shard_file}: {exc}. "
                        "Proceeding; tensors will be skipped if absent locally."
                    )
            resolved_shard[shard_file] = local_shard
        full_weight_map = {
            k: resolved_shard.get(v, os.path.join(source_dir, v))
            for k, v in raw_map.items()
        }
    else:
        full_weight_map = get_weight_mappings(source_dir)

    # Group MTP keys by their shard file and read each file once.
    from collections import defaultdict

    mtp_keys_by_file = defaultdict(list)
    for key, filepath in full_weight_map.items():
        if key.startswith(mtp_prefix + "."):
            mtp_keys_by_file[filepath].append(key)

    mtp_tensors = {}
    for filepath, keys in mtp_keys_by_file.items():
        if not os.path.exists(filepath):
            logger.warning(
                f"MTP shard {filepath} not found (local-only model or "
                "failed download); skipping tensors."
            )
            continue
        with safe_open(filepath, framework="pt", device="cpu") as f:
            for key in keys:
                mtp_tensors[key] = f.get_tensor(key)

    if not mtp_tensors:
        raise ValueError(
            f"No tensors with prefix '{mtp_prefix}' found in {source_model}"
        )

    # Load the already-saved dest config; it is needed to emit the MTP config
    # group (or, when nothing is quantized, the ignore entry).
    config_path = find_config_path(dest_dir)
    config = None
    if config_path is not None:
        with open(config_path) as f:
            config = json.load(f)

    # MTP quantization is opt-in: without an explicit mtp_scheme the layers stay
    # full precision (bf16) and are added to the ignore list below.
    scheme = _resolve_mtp_scheme(mtp_scheme)

    # module names of the MTP layers we actually quantized; used to emit config
    # group targets that match modules exactly (targets match module names, not
    # parameter names, so a "...\\.weight" pattern would match nothing)
    quantized_modules = []
    if scheme is not None:
        output_tensors = {}
        # First pass: separate the 2D linear weights we will quantize from
        # everything that must stay full precision.
        to_quantize = {}
        for key, tensor in mtp_tensors.items():
            # only 2D linear weights are quantized; norms/biases/1D tensors
            # (e.g. mtp.norm.weight) stay full precision
            if not key.endswith(".weight") or tensor.ndim != 2:
                output_tensors[key] = tensor
                continue
            # Embedding and output-head weights have the vocab dimension as
            # their first axis; keep them full precision. vLLM only supports
            # packed-INT (WNA16) quantization for these, never FP8/NVFP4, and
            # MTP embeddings/heads are built unquantized there, so quantizing
            # them yields a checkpoint vLLM cannot load (this matches how the
            # main model leaves its own embeddings/lm_head unquantized). We
            # detect these by checking shape[0] == vocab_size; false positives
            # are unlikely given the typical size difference.
            if vocab_size is not None and tensor.shape[0] == vocab_size:
                output_tensors[key] = tensor
                continue
            # The MTP fusion projection (eh_proj / fc, which combines the
            # previous hidden state with the next-token embedding) is a plain
            # nn.Linear with no scale param in some engines (e.g. deepseek/glm
            # MTP in vLLM), so a quantized weight would load without its scale
            # and silently corrupt outputs. Keep it full precision everywhere.
            # NOTE: This is a hardcoded list verified against vLLM 0.26.0 MTP
            # implementations. New MTP architectures with different fusion names
            # would bypass this check and potentially create broken checkpoints.
            if key[: -len(".weight")].rsplit(".", 1)[-1] in _FUSION_PROJ_NAMES:
                output_tensors[key] = tensor
                continue
            to_quantize[key] = tensor

        quantized_modules = _compress_mtp_linears(to_quantize, scheme, output_tensors)
    else:
        # no quantization config found; save unquantized and add to ignore
        output_tensors = mtp_tensors

    # save the shard
    save_file(output_tensors, os.path.join(dest_dir, shard_name))

    # update the safetensors index to include the new shard
    # Read the dest index directly when it exists: get_weight_mappings prefers a
    # bare model.safetensors over the index file, so it would try to parse any
    # placeholder shard header (and fail). Reading the index directly is also
    # more reliable than parsing shard headers.
    _dest_index = os.path.join(dest_dir, "model.safetensors.index.json")
    if os.path.exists(_dest_index):
        with open(_dest_index) as _f:
            dest_weight_map = dict(json.load(_f)["weight_map"])
    else:
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
            if quantized_modules:
                # Config-group targets match *module* names, not parameter
                # names. vLLM fuses sibling projections before matching
                # (q/k/v -> qkv_proj, gate/up -> gate_up_proj), so the exact
                # per-component names we quantized (e.g. mtp...gate_proj) never
                # match the fused runtime module (mtp...gate_up_proj). Worse,
                # vLLM resolves the *first* matching target across all groups in
                # order, before its fused-layer fallback runs -- so the main
                # model's broad mlp/attn regexes (which DO match the fused name)
                # would capture the MTP modules unless the MTP group is both
                # listed first and carries a target matching the fused names. We
                # therefore emit an mtp-anchored regex covering the fused and
                # unfused projection names and prepend the group. The exact
                # names are kept as belt-and-suspenders for unfused projections.
                #
                # Each config group must carry its own compression `format`
                # (the main model's groups do, and it is required when the
                # top-level format is "mixed-precision"). Infer it from the
                # scheme exactly as the main model does, so vLLM loads the MTP
                # group with the correct compressor (e.g. nvfp4-pack-quantized).
                from compressed_tensors.compressors.format import (
                    infer_module_format,
                )

                mtp_proj_regex = (
                    rf"re:^{re.escape(mtp_prefix)}\..*\."
                    r"(q_proj|k_proj|v_proj|o_proj|qkv_proj|"
                    r"gate_proj|up_proj|down_proj|gate_up_proj)$"
                )
                group = {
                    "format": infer_module_format(torch.nn.Linear, scheme).value,
                    "targets": [mtp_proj_regex, *sorted(quantized_modules)],
                    "weights": scheme.weights.model_dump(),
                }
                if scheme.input_activations is not None:
                    group["input_activations"] = scheme.input_activations.model_dump()
                # Prepend (dropping any stale mtp_group from a prior save) so
                # vLLM's first-match wins for MTP modules over the main groups.
                groups = {
                    k: v
                    for k, v in (quant_config.get("config_groups") or {}).items()
                    if k != "mtp_group"
                }
                quant_config["config_groups"] = {"mtp_group": group, **groups}
            else:
                # nothing was quantized; mark MTP as ignored so inference
                # engines skip it
                ignore_list = quant_config.get("ignore") or []
                mtp_ignore_pattern = f"re:^{re.escape(mtp_prefix)}\\."
                if mtp_ignore_pattern not in ignore_list:
                    ignore_list.append(mtp_ignore_pattern)
                    quant_config["ignore"] = ignore_list
            config[QUANTIZATION_CONFIG_NAME] = quant_config
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)


def _get_mtp_prefix(source_model: str, text_config) -> str:
    """
    Detect the tensor key prefix used for MTP layers in a checkpoint.

    Reads from the safetensors index rather than the downloaded files so that
    the MTP shard does not need to be present locally (AutoModel never downloads
    it because the keys are in _keys_to_ignore_on_load_unexpected).
    """
    import json

    from compressed_tensors.utils.safetensors_load import get_safetensors_folder
    from huggingface_hub import hf_hub_download

    source_dir = get_safetensors_folder(source_model)
    index_path = os.path.join(source_dir, "model.safetensors.index.json")

    if not os.path.exists(index_path):
        try:
            index_path = hf_hub_download(source_model, "model.safetensors.index.json")
        except Exception:
            index_path = None

    if index_path and os.path.exists(index_path):
        with open(index_path) as f:
            all_keys = list(json.load(f).get("weight_map", {}).keys())
    else:
        # single-shard model; read keys directly from the safetensors metadata
        from safetensors import safe_open

        shard = os.path.join(source_dir, "model.safetensors")
        if not os.path.exists(shard):
            raise FileNotFoundError(
                "Could not find model.safetensors or "
                f"model.safetensors.index.json in {source_dir}"
            )
        with safe_open(shard, framework="pt", device="cpu") as f:
            all_keys = list(f.keys())

    if any(k.startswith("mtp.") for k in all_keys):
        return "mtp"

    # GLM-style: MTP is the layer at index num_hidden_layers, stored either
    # under model.language_model.layers (VLM, e.g. GLM-5.3-Flash) or
    # model.layers (text-only)
    num_hidden = getattr(text_config, "num_hidden_layers", None)
    if num_hidden is not None:
        for prefix in (
            f"model.language_model.layers.{num_hidden}",
            f"model.layers.{num_hidden}",
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
            mtp_scheme=None,
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
            :param mtp_scheme: how to quantize Multi-Token Prediction (MTP) layers,
                which transformers does not load or compress. MTP quantization is
                opt-in: by default (None) MTP layers are saved full precision (bf16)
                and marked ignored. Pass a ``QuantizationScheme`` or a preset name
                (e.g. "FP8_DYNAMIC", "NVFP4") to quantize them. "bf16"/"none"/
                "dense"/"unquantized" are treated the same as None. Input-
                activation quant whose scale must be calibrated (static, or
                NVFP4-style dynamic="local") is dropped to weight-only since MTP
                layers cannot be calibrated; only fully dynamic activation quant
                (e.g. FP8_DYNAMIC) is kept. Ignored for models without MTP layers.
            :param kwargs: additional kwargs to pass on to model.save_pretrained
            """

            save_dir = save_directory
            kwargs.setdefault("max_shard_size", "20GB")

            # without this, quantization format will be inferred from the model
            if not save_compressed and quantization_format is None:
                quantization_format = CompressionFormat.dense.value

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
                    original_save_fn.__get__(model, model_class)(save_dir, **kwargs)

                    # update config to reflect quantization
                    compressor.update_config(save_dir)

                    # update existing recipe
                    update_and_save_recipe(model.name_or_path, save_dir)

                    # copy python files from cache dir to save_path if any
                    copy_python_files_from_model_cache(model, save_dir)

                    # quantize mtp tensors (not loaded by transformers) and
                    # update config
                    text_config = model.config.get_text_config()
                    has_mtp = (
                        getattr(text_config, "num_mtp_layers", 0)
                        or getattr(text_config, "mtp_num_hidden_layers", 0)
                        or getattr(text_config, "num_nextn_predict_layers", 0)
                    )
                    if has_mtp:
                        mtp_prefix = _get_mtp_prefix(model.name_or_path, text_config)
                        _quantize_and_save_mtp_tensors(
                            model.name_or_path,
                            save_dir,
                            mtp_prefix=mtp_prefix,
                            vocab_size=getattr(text_config, "vocab_size", None),
                            mtp_scheme=mtp_scheme,
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
