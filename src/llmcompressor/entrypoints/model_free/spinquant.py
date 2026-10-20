import os
import re
from collections import defaultdict
from typing import Literal

import torch
from compressed_tensors.entrypoints.convert import Converter
from compressed_tensors.quantization import QuantizationConfig
from compressed_tensors.transform.utils.hadamard import (
    deterministic_hadamard_matrix,
    random_hadamard_matrix,
)
from compressed_tensors.utils.moe import load_model_config
from compressed_tensors.utils.safetensors_load import (
    get_checkpoint_files,
    get_weight_map,
)
from loguru import logger
from pydantic import BaseModel, Field
from safetensors import safe_open

__all__ = ["SpinQuantConverter", "SpinQuantConverterMapping"]


class SpinQuantConverterMapping(BaseModel):
    """
    Name-based description of where SpinQuant rotations are fused in a checkpoint.
    Patterns are regexes searched against module names (tensor names without the
    trailing `.weight` / `.bias`). Defaults target pre-norm decoders with
    Llama-style names, including MoE routers and experts.

    :param embedding: embedding whose rows are rotated by R1
    :param attn_in: attention layers reading the residual stream
    :param attn_out: attention layers writing the residual stream
    :param attn_v: layer whose output heads are rotated by R2
    :param attn_o: layer whose input heads are rotated by R2
    :param mlp_in: mlp layers reading the residual stream
    :param mlp_out: mlp layers writing the residual stream
    :param lm_head: language model head
    :param residual: bare parameters, without a `.weight` suffix, whose last
        dimension is one or more residual streams. Each stream is rotated by R1,
        i.e. a last dimension of `k * hidden_size` is rotated by a block-diagonal
        R1. Defaults include DeepSeek-V4 mHC mixers (`hc_attn_fn`, `hc_ffn_fn`,
        `hc_head_fn`)
    :param attn_norm: norm fused into `attn_in`, relative to the decoder layer
    :param mlp_norm: norm fused into `mlp_in`, relative to the decoder layer
    :param final_norm: norm fused into `lm_head`, relative to the decoder prefix
    """

    embedding: str = r"(^|\.)embed_tokens$"
    attn_in: list[str] = Field(default_factory=lambda: [r"\.self_attn\.(q|k|v)_proj$"])
    attn_out: list[str] = Field(default_factory=lambda: [r"\.self_attn\.o_proj$"])
    attn_v: str = r"\.self_attn\.v_proj$"
    attn_o: str = r"\.self_attn\.o_proj$"
    mlp_in: list[str] = Field(
        default_factory=lambda: [
            r"\.mlp\.(.+\.)?(gate|up|gate_up)_proj$",
            r"\.mlp\.gate$",
            r"\.mlp\.shared_expert_gate$",
        ]
    )
    mlp_out: list[str] = Field(default_factory=lambda: [r"\.mlp\.(.+\.)?down_proj$"])
    lm_head: str = r"(^|\.)lm_head$"
    residual: list[str] = Field(
        default_factory=lambda: [r"(^|\.)hc_(attn|ffn|head)_fn$"]
    )
    attn_norm: str = "input_layernorm"
    mlp_norm: str = "post_attention_layernorm"
    final_norm: str = "norm"


_LAYER_RE = re.compile(r"^(?P<layer>(.*\.)?layers\.\d+)\.")
# Parameter names produced by compressed-tensors and by GPTQ/AWQ checkpoints.
# Matched exactly, so a tensor whose name merely ends in one of these is left alone.
_QPARAM_NAMES = frozenset(
    {
        "weight_scale",
        "weight_scale_inv",
        "input_scale",
        "weight_zero_point",
        "weight_global_scale",
        "input_global_scale",
        "weight_packed",
        "weight_shape",
        "qweight",
        "qzeros",
        "scales",
    }
)
_QUANTIZED_DTYPES = {
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.uint8,
    torch.int8,
    torch.int32,
}


class SpinQuantConverter(Converter):
    """
    Model-free implementation of the offline SpinQuant rotations
    (https://arxiv.org/abs/2405.16406). Norm scales are fused into the linears
    which follow them, then R1 is fused into every layer which reads or writes the
    residual stream and R2 is fused into each attention head's value/output
    projections. The resulting checkpoint is numerically equivalent to the source
    model, but has weights which are easier to quantize. No online transforms are
    added, so the checkpoint loads like any dense (or subsequently quantized) model.

    Construct with :meth:`from_pretrained` and chain before quantization, e.g.

    ```python
    model_free_ptq(
        MODEL_ID,
        SAVE_DIR,
        scheme="MXFP4",
        converter=[
            FP8BlockDequantizer(),
            SpinQuantConverter.from_pretrained(MODEL_ID),
        ],
    )
    ```

    :param hidden_size: size of the residual stream
    :param head_dim: size of each attention head
    :param norms: module name -> norm weight, for every norm being fused
    :param tie_word_embeddings: whether the source model ties `lm_head` to the
        embedding. If so, the saved config is untied, since the rotated `lm_head`
        and embedding differ
    :param emit_lm_head: whether to create `lm_head.weight` from the embedding,
        for tied checkpoints which do not save `lm_head.weight`
    :param rotations: offline rotations to apply. `from_pretrained` defaults an
        MLA model (one with `kv_lora_rank`) to R1 only: R2 needs an uncompressed
        `num_heads * head_dim` path between `v_proj` and `o_proj`, which MLA
        replaces with a compressed latent
    :param transform_type: `"hadamard"` supports power-of-two sizes,
        `"random-hadamard"` also supports sizes with a known hadamard divisor
    :param transform_block_size: R1 block size. Defaults to `hidden_size`. The
        rotation is block-diagonal when smaller than `hidden_size`
    :param seed: seed for `"random-hadamard"`
    :param precision: dtype used to fuse rotations into weights
    :param mapping: names of the layers to rotate
    :param ignore: regexes of tensor names which are left untouched, for modules
        outside of the rotated residual stream such as a vision tower
    """

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        norms: dict[str, torch.Tensor],
        tie_word_embeddings: bool = False,
        emit_lm_head: bool = False,
        rotations: tuple[Literal["R1", "R2"], ...] = ("R1", "R2"),
        transform_type: Literal["hadamard", "random-hadamard"] = "hadamard",
        transform_block_size: int | None = None,
        seed: int = 0,
        precision: torch.dtype = torch.float64,
        mapping: SpinQuantConverterMapping | None = None,
        ignore: list[str] = (),
        num_attention_heads: int | None = None,
        num_key_value_heads: int | None = None,
    ):
        rotations = tuple(r.upper() for r in rotations)
        if not rotations or not set(rotations) <= {"R1", "R2"}:
            raise ValueError(
                f"SpinQuantConverter supports R1 and R2 only, got {rotations}"
            )

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.norms = norms
        self.tie_word_embeddings = tie_word_embeddings
        self.emit_lm_head = emit_lm_head
        self.rotations = rotations
        self.precision = precision
        self.mapping = mapping or SpinQuantConverterMapping()

        block_size = transform_block_size or hidden_size
        if hidden_size % block_size != 0:
            raise ValueError(
                f"transform_block_size {block_size} must divide "
                f"hidden_size {hidden_size}"
            )
        generator = torch.Generator().manual_seed(seed)
        self.r1 = _hadamard(block_size, transform_type, precision, generator)
        # R2's head-sized matrix is only needed, and only constructible for every
        # head_dim, when R2 is actually applied.
        self.r2 = (
            _hadamard(head_dim, transform_type, precision, generator)
            if "R2" in rotations
            else None
        )
        self.r2_v_dim = (
            None if num_key_value_heads is None else num_key_value_heads * head_dim
        )
        self.r2_o_dim = (
            None if num_attention_heads is None else num_attention_heads * head_dim
        )

        m = self.mapping
        self._embedding = re.compile(m.embedding)
        self._lm_head = re.compile(m.lm_head)
        self._attn_in = [re.compile(p) for p in m.attn_in]
        self._attn_out = [re.compile(p) for p in m.attn_out]
        self._mlp_in = [re.compile(p) for p in m.mlp_in]
        self._mlp_out = [re.compile(p) for p in m.mlp_out]
        self._attn_v = re.compile(m.attn_v)
        self._attn_o = re.compile(m.attn_o)
        self._residual = [re.compile(p) for p in m.residual]
        self._ignore = [re.compile(p) for p in ignore]

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: str | os.PathLike, **kwargs
    ) -> "SpinQuantConverter":
        """
        Build the converter from a checkpoint's config.json and norm weights. Norm
        weights are small, so they are loaded up front rather than declared as
        dependencies of every linear which reads them.

        :param model_name_or_path: HuggingFace stub or local checkpoint path
        :param kwargs: additional arguments passed to the constructor
        """
        model_files = get_checkpoint_files(model_name_or_path)
        weight_map = get_weight_map(model_files)
        config = load_model_config(model_files)
        text_config = config.get("text_config", config)

        hidden_size = text_config["hidden_size"]
        head_dim = text_config.get("head_dim") or (
            hidden_size // text_config["num_attention_heads"]
        )
        tie_word_embeddings = config.get(
            "tie_word_embeddings", text_config.get("tie_word_embeddings", False)
        )

        mapping = kwargs.get("mapping") or SpinQuantConverterMapping()
        ignore = [re.compile(p) for p in kwargs.get("ignore", ())]
        names = [name for name in weight_map if not _search_any(ignore, name)]
        lm_head = re.compile(mapping.lm_head)
        has_lm_head = any(
            lm_head.search(name.removesuffix(".weight"))
            for name in names
            if name.endswith(".weight")
        )

        norm_names = _find_norm_names(names, mapping)
        if not norm_names:
            raise ValueError("No norm weights found to fuse into linear layers")

        names_by_file: dict[str, list[str]] = defaultdict(list)
        for name in norm_names:
            names_by_file[model_files[weight_map[name]]].append(name)

        norms = {}
        for path, names in names_by_file.items():
            with safe_open(path, framework="pt") as file:
                for name in names:
                    norms[name.removesuffix(".weight")] = file.get_tensor(name)

        if "rotations" not in kwargs and _is_mla(text_config):
            kwargs["rotations"] = ("R1",)
            logger.warning(
                "MLA model detected (kv_lora_rank is set). SpinQuant R2 needs an "
                "uncompressed head dimension between v_proj and o_proj, so "
                "rotations default to R1 only. Pass rotations explicitly to override"
            )
        kwargs.setdefault("num_attention_heads", text_config.get("num_attention_heads"))
        kwargs.setdefault("num_key_value_heads", text_config.get("num_key_value_heads"))

        return cls(
            hidden_size=hidden_size,
            head_dim=head_dim,
            norms=norms,
            tie_word_embeddings=tie_word_embeddings,
            emit_lm_head=tie_word_embeddings and not has_lm_head,
            **kwargs,
        )

    def process(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        result = {}
        for name, tensor in tensors.items():
            if _search_any(self._ignore, name):
                result[name] = tensor
                continue

            module_name, param = _split_param(name, tensor)
            if param in _QPARAM_NAMES or (
                param == "weight" and tensor.dtype in _QUANTIZED_DTYPES
            ):
                raise ValueError(
                    f"{name} is quantized. Please dequantize the checkpoint before "
                    "SpinQuantConverter, e.g. by chaining a dequantizer before it"
                )

            if param == "weight" and self._embedding.search(module_name):
                result[name] = self._rotate_embedding(tensor)
                if self.emit_lm_head:
                    result["lm_head.weight"] = self._rotate_lm_head("lm_head", tensor)
            elif param == "weight" and self._lm_head.search(module_name):
                result[name] = self._rotate_lm_head(module_name, tensor)
            elif param in ("weight", "bias"):
                result[name] = self._rotate_layer(name, module_name, param, tensor)
            elif self._is_residual(name, module_name):
                result[name] = self._rotate_residual_streams(name, tensor)
            elif self._touches_residual(tensor):
                raise ValueError(
                    _unmapped_residual_message(name, tensor, self.hidden_size)
                )
            else:
                result[name] = tensor
        return result

    def validate(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.process(tensors)

    def update_config(
        self, config: QuantizationConfig | None
    ) -> QuantizationConfig | None:
        return config

    def update_model_config(self, model_config: dict) -> dict:
        if self.tie_word_embeddings:
            model_config["tie_word_embeddings"] = False
            if isinstance(model_config.get("text_config"), dict):
                model_config["text_config"]["tie_word_embeddings"] = False
            logger.info("Untied word embeddings to fuse SpinQuant rotations")
        return model_config

    def get_dependencies(self, weight_name: str) -> set[str]:
        return set()

    def _rotate_layer(
        self, name: str, module_name: str, param: str, tensor: torch.Tensor
    ) -> torch.Tensor:
        if module_name in self.norms:
            if param == "bias":
                raise ValueError(
                    f"{module_name} has a bias, but SpinQuantConverter only supports "
                    "norms without bias, such as RMSNorm"
                )
            return torch.ones_like(tensor)

        is_attn_in = _search_any(self._attn_in, module_name)
        is_mlp_in = _search_any(self._mlp_in, module_name)
        is_out = _search_any(self._attn_out + self._mlp_out, module_name)
        is_v = "R2" in self.rotations and self._attn_v.search(module_name)
        is_o = "R2" in self.rotations and self._attn_o.search(module_name)

        if param == "bias":
            if is_out and "R1" in self.rotations:
                return self._apply(_rotate_bias, tensor, self.r1)
            if is_v:
                return self._apply(_rotate_bias, tensor, self.r2)
            if not (is_attn_in or is_mlp_in) and self._touches_residual(tensor):
                raise ValueError(
                    _unmapped_residual_message(name, tensor, self.hidden_size)
                )
            return tensor

        if is_attn_in or is_mlp_in:
            self._check_dim(module_name, tensor, -1)
            norm_name = self.mapping.attn_norm if is_attn_in else self.mapping.mlp_norm
            tensor = self._fuse_norm(tensor, self._norm_for(module_name, norm_name))
            if "R1" in self.rotations:
                tensor = self._apply(_rotate_cols, tensor, self.r1)
            if is_v:
                self._check_r2_dim(name, tensor, -2, self.r2_v_dim)
                tensor = self._apply(_rotate_rows, tensor, self.r2)
            return tensor

        if is_out:
            if is_o:
                self._check_r2_dim(name, tensor, -1, self.r2_o_dim)
                tensor = self._apply(_rotate_cols, tensor, self.r2)
            if "R1" in self.rotations:
                self._check_dim(module_name, tensor, -2)
                tensor = self._apply(_rotate_rows, tensor, self.r1)
            return tensor

        if self._is_residual(name, module_name):
            return self._rotate_residual_streams(name, tensor)
        if self._touches_residual(tensor):
            raise ValueError(_unmapped_residual_message(name, tensor, self.hidden_size))
        return tensor

    def _rotate_embedding(self, tensor: torch.Tensor) -> torch.Tensor:
        if "R1" not in self.rotations:
            return tensor
        return self._apply(_rotate_cols, tensor, self.r1)

    def _rotate_lm_head(self, module_name: str, tensor: torch.Tensor) -> torch.Tensor:
        self._check_dim(module_name, tensor, -1)
        final_norms = [
            norm
            for name, norm in self.norms.items()
            if not _LAYER_RE.match(name)
            and name.rpartition(".")[2] == self.mapping.final_norm
        ]
        if len(final_norms) != 1:
            raise ValueError(
                f"Expected one final norm for {module_name}, found {len(final_norms)}"
            )
        tensor = self._fuse_norm(tensor, final_norms[0])
        if "R1" in self.rotations:
            tensor = self._apply(_rotate_cols, tensor, self.r1)
        return tensor

    def _norm_for(self, module_name: str, norm: str) -> torch.Tensor:
        match = _LAYER_RE.match(module_name)
        norm_name = f"{match.group('layer')}.{norm}" if match else None
        if norm_name not in self.norms:
            raise ValueError(f"Could not find norm {norm_name} for {module_name}")
        return self.norms[norm_name]

    def _fuse_norm(self, tensor: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        norm = norm.to(device=tensor.device, dtype=self.precision)
        return (tensor.to(self.precision) * norm).to(tensor.dtype)

    def _apply(self, fn, tensor: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
        rotation = rotation.to(device=tensor.device)
        return fn(tensor.to(self.precision), rotation).to(tensor.dtype)

    def _is_residual(self, name: str, module_name: str) -> bool:
        return _search_any(self._residual, name) or _search_any(
            self._residual, module_name
        )

    def _touches_residual(self, tensor: torch.Tensor) -> bool:
        """A dimension is one or more residual streams (`k * hidden_size`)."""
        return any(size > 0 and size % self.hidden_size == 0 for size in tensor.shape)

    def _rotate_residual_streams(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        if "R1" not in self.rotations:
            return tensor
        last = tensor.shape[-1] if tensor.ndim >= 1 else 0
        if last == 0 or last % self.hidden_size != 0:
            raise ValueError(
                f"{name} is marked as a residual tensor but its last dimension "
                f"{last} is not a multiple of hidden_size={self.hidden_size}"
            )
        return self._apply(_rotate_cols, tensor, self.r1)

    def _check_r2_dim(
        self, name: str, tensor: torch.Tensor, dim: int, expected: int | None
    ):
        size = tensor.shape[dim]
        if expected is not None and size != expected:
            raise ValueError(
                f"R2 cannot rotate {name}: dimension {dim} has size {size}, "
                f"expected {expected} (number of heads times head_dim="
                f"{self.head_dim}). Compressed attention paths such as MLA do not "
                "have that shape; use rotations=('R1',)"
            )
        if size % self.head_dim != 0:
            raise ValueError(
                f"R2 cannot rotate {name}: dimension {dim} has size {size}, "
                f"which is not divisible by head_dim={self.head_dim}"
            )

    def _check_dim(self, module_name: str, tensor: torch.Tensor, dim: int):
        if tensor.ndim < 2 or tensor.shape[dim] != self.hidden_size:
            raise ValueError(
                f"Expected dim {dim} of {module_name} with shape "
                f"{tuple(tensor.shape)} to equal hidden_size={self.hidden_size}"
            )


def _is_mla(text_config: dict) -> bool:
    return "kv_lora_rank" in text_config


def _unmapped_residual_message(
    name: str, tensor: torch.Tensor, hidden_size: int
) -> str:
    return (
        f"{name} with shape {tuple(tensor.shape)} touches the residual stream "
        f"(a dimension is a multiple of hidden_size={hidden_size}) but is not "
        "covered by the SpinQuant mapping. Rotating the stream without rotating "
        "this tensor would change the model's outputs. Bare parameters such as "
        "mHC mixers belong in mapping.residual, which rotates the last dimension "
        "by a block-diagonal R1, one block per stream"
    )


def _find_norm_names(names, mapping: SpinQuantConverterMapping) -> list[str]:
    layer_norms = re.compile(
        rf"(^|\.)layers\.\d+\.({mapping.attn_norm}|{mapping.mlp_norm})\.weight$"
    )
    embeddings = re.compile(mapping.embedding)
    module_names = [n.removesuffix(".weight") for n in names if n.endswith(".weight")]
    final_norms = {
        ".".join(filter(None, (name.rpartition(".")[0], mapping.final_norm, "weight")))
        for name in module_names
        if embeddings.search(name)
    }
    return [name for name in names if layer_norms.search(name) or name in final_norms]


def _split_param(name: str, tensor: torch.Tensor) -> tuple[str, str]:
    """
    Split a tensor name into its module name and parameter type. Fused 3D experts
    such as `mlp.experts.gate_up_proj` have no `.weight` suffix, so they are
    treated as weights of a module of the same name
    """
    module_name, _, param = name.rpartition(".")
    if param in ("weight", "bias"):
        return module_name, param
    if tensor.ndim == 3:
        return name, "weight"
    return name, param


def _search_any(patterns: list[re.Pattern], name: str) -> bool:
    return any(pattern.search(name) for pattern in patterns)


def _hadamard(
    size: int,
    transform_type: str,
    precision: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    if transform_type == "hadamard":
        matrix = deterministic_hadamard_matrix(size, precision)
    elif transform_type == "random-hadamard":
        matrix = random_hadamard_matrix(size, precision, gen=generator)
    else:
        raise ValueError(f"Unsupported transform_type {transform_type}")
    # The constructors return a ±1 Hadamard, whose square is `size * I`. Dividing
    # by sqrt(size) once, as HadamardTransform.forward does, makes the rotation
    # orthogonal: R @ R.T = I. Both R1 and R2 need this so an input rotation and
    # the matching output rotation cancel.
    return matrix / torch.tensor(size, dtype=precision).sqrt()


def _rotate_cols(tensor: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """`tensor @ blockdiag(rotation)` over the last dim"""
    block = rotation.shape[0]
    blocks = tensor.unflatten(-1, (-1, block))
    return (blocks @ rotation).flatten(-2)


def _rotate_rows(tensor: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """`blockdiag(rotation).T @ tensor` over the second-to-last dim"""
    block = rotation.shape[0]
    blocks = tensor.unflatten(-2, (-1, block))
    return (rotation.T @ blocks).flatten(-3, -2)


def _rotate_bias(tensor: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """`bias @ blockdiag(rotation)`, matching an output rotation of the weight"""
    return _rotate_cols(tensor, rotation)
