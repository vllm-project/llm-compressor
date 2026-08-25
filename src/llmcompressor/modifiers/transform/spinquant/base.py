from collections.abc import Iterable
from enum import Enum
from typing import Literal

import torch
import torch.nn.utils.parametrize as P
from compressed_tensors import (
    TRANSFORM_CONFIG_NAME,
    match_modules_set,
    match_named_modules,
)
from compressed_tensors.offload import OffloadCache, disable_offloading
from compressed_tensors.offload.module import remove_module_offload
from compressed_tensors.transform import (
    TransformArgs,
    TransformConfig,
    TransformLocation,
    TransformScheme,
    apply_transform_config,
)
from compressed_tensors.transform.factory.base import TransformBase
from compressed_tensors.utils import TorchDtype, get_head_dim, update_offload_parameter
from loguru import logger
from pydantic import Field, ValidationInfo, field_validator
from torch.utils._pytree import tree_leaves
from transformers import PreTrainedModel

from llmcompressor.core import Event, State
from llmcompressor.modeling import center_embeddings, fuse_norm_linears
from llmcompressor.modifiers import Modifier
from llmcompressor.typing import NamedModules
from llmcompressor.utils import (
    get_high_precision,
    get_main_device,
    untie_word_embeddings,
)

from .mappings import SpinQuantMapping, infer_mapping_from_model
from .norm_mappings import NormMapping, infer_norm_mapping_from_model


class SpinquantRotation(str, Enum):
    R1 = "R1"
    R2 = "R2"
    R3 = "R3"
    R4 = "R4"


class SpinQuantModifier(Modifier, use_enum_values=True):
    """
    Implements the transforms according to "SpinQuant: LLM quantization
    with learned rotations" (https://arxiv.org/abs/2405.16406)

    Transforms (rotations) are extra layers added to a model which reduce the accuracy
    loss induced by quantization. This is achieved through "rotating" weights and
    activations into a space with a smaller dynamic range of values, thus decreasing
    the range of scales required for quantization.

    The SpinQuant authors describe four different rotations which can be applied to a
    model. R1 and R2 are "offline" rotations, meaning that they can be fused into
    existing weights and therefore do not induce runtime cost. R3 and R4 are "online"
    rotations, meaning that they require additional computation at runtime.

    **Learned rotations** (``learnable=True``): the key contribution of the SpinQuant
    paper is that the rotation matrices are *learned* via SGD on a small calibration
    set rather than left as fixed Hadamard matrices (as in QuaRot). When
    ``learnable=True``, the rotation matrices are optimized to minimize the language
    modeling loss over the calibration data (see ``_learn_rotations``) before the
    quantization algorithm runs. Only the rotation parameters are trained; the rest of
    the model is frozen. This requires a calibration ``dataset`` to be passed to
    ``oneshot``. After learning, the offline rotations (R1/R2) are fused back into the
    weights so the serialized checkpoint is identical in structure to the
    non-learnable path.

    Lifecycle:

    - on_initialize
        - infer SpinQuantMappings & NormMappings
        - as needed, create transform schemes for R1, R2, R3, & R4
    - on_calibration_start
        - normalize embeddings
        - fuse norm layers into subsequent Linear layers
        - apply TransformConfig
            - fuse transforms into weights for mergeable transforms
            - add hooks for online transforms
        - if learnable: learn the rotation matrices over the calibration set, then
            fuse the offline rotations into weights and clear ``requires_grad`` so
            the saved checkpoint matches the non-learnable path

    :param rotations: A list containing the names of rotations to apply to the model.
        Possible rotations include R1, R2, R3, and R4
    :param transform_type: The type of transform to apply to the model.
        `"hadamard"` has the least performance cost but only supports sizes which are
        powers of power of two.
        `"random-matrix"` has more performance cost, but supports a much larger set of
            sizes.
        `"random-matrix"` has the greatest performance cost, but supports any size
    :param randomize: if True, create distinct transforms for each application.
        Not currently supported and raises if set.
    :param learnable: if True, learn the rotation matrices via SGD/AdamW over the
        calibration set (SpinQuant's contribution) instead of using fixed transforms.
        Requires a calibration `dataset` to be passed to `oneshot`. Default False
        (QuaRot-style fixed rotations).
    :param learn_lr: learning rate for rotation learning (default 1e-3)
    :param learn_steps: number of gradient steps for rotation learning; each step
        consumes one batch from the calibration set, which is iterated cyclically
        (default 100)
    :param learn_optimizer: optimizer used for rotation learning, "adamw" or "sgd"
        (SGD uses momentum 0.9) (default "adamw")
    :param learn_grad_clip: max norm for gradient clipping during rotation learning;
        None disables clipping (default 1.0)
    :param precision: Precision at which all transforms should be applied. This applies
        to both weight fusing and online rotations
    :param transform_block_size: Block size to use for rotation matrices. The model's
        hidden_size and head_dim must be evenly divisible by transform_block_size.
        Layers will be transformed by a block-diagonal matrix where each block is a
        matrix of this size.
        If None is provided, model's hidden_size will be used for R1, R3, and R4
        and model's head_dim will be used for R2
    :param mappings: Specifies layers within a model to target for transforms.
        A mapping will be inferred if None is provided
    :param norm_mappings: Specifies layers within a model to target for norm fusing.
        A mapping will be inferred if None is provided
    :param transform_config: Optional transform config for overriding provided arguments
    """

    rotations: list[SpinquantRotation] = Field(default_factory=lambda: ["R1", "R2"])
    transform_type: Literal["hadamard", "random-hadamard", "random-matrix"] = Field(
        default="hadamard"
    )
    randomize: bool = Field(default=False)
    learnable: bool = Field(default=False)
    learn_lr: float = Field(default=1e-3)
    learn_steps: int = Field(default=100)
    learn_optimizer: Literal["adamw", "sgd"] = Field(default="adamw")
    learn_grad_clip: float | None = Field(default=1.0)
    precision: TorchDtype = Field(default=get_high_precision())
    transform_block_size: int | None = Field(default=None)

    # norm mappings separate from spinquant mappings to allow users to
    # override spinquant mappings with transform_config without overriding norms
    mappings: SpinQuantMapping | None = Field(
        default=None,
        repr=False,
        exclude=True,
    )
    norm_mappings: list[NormMapping] | None = Field(
        default=None,
        repr=False,
        exclude=True,
    )

    # optional override for more fine-grained control
    # also included in recipe serialization
    transform_config: TransformConfig | None = Field(default=None, repr=False)

    @field_validator("randomize", mode="before")
    def validate_randomize_not_implemented(cls, value, info: ValidationInfo):
        if value:
            raise NotImplementedError(f"{info.field_name} is not supported as of now")
        return value

    @field_validator("rotations", mode="before")
    def validate_rotations(cls, value):
        if isinstance(value, Iterable):
            return tuple(v.upper() for v in value)
        return value

    def on_initialize(self, state: State, **kwargs) -> bool:
        if self.transform_config is not None:
            return True

        self.mappings = infer_mapping_from_model(state.model)
        self.norm_mappings = infer_norm_mapping_from_model(state.model)
        head_dim = get_head_dim(state.model.config)

        config_groups = {}
        if SpinquantRotation.R1 in self.rotations:
            config_groups["R1"] = self._create_r1_scheme()

        if SpinquantRotation.R2 in self.rotations:
            config_groups["R2"] = self._create_r2_scheme(head_dim)

        if SpinquantRotation.R3 in self.rotations:
            config_groups["R3"] = self._create_r3_scheme(head_dim)

        if SpinquantRotation.R4 in self.rotations:
            config_groups["R4"] = self._create_r4_scheme()

        self.transform_config = TransformConfig(config_groups=config_groups)

        return True

    def on_calibration_start(self, state: State, event: Event, **kwargs):
        model = state.model

        with torch.no_grad():
            # untie embeddings to avoid unintended effects of `_center_embeddings`
            untie_word_embeddings(model)

            # needs to happen after the model has been hooked to execute on the GPU
            # otherwise we're applying weight transforms on CPU
            self._center_embeddings(model)
            self._fuse_norms(model)

        if self.learnable:
            # Learn the rotation matrices over the calibration set (the "learned
            # rotations" contribution of SpinQuant). Rotations are applied, learned,
            # and fused ONE AT A TIME because the compressed-tensors factories cannot
            # compose multiple parametrized (requires_grad) transforms on the same
            # Linear (e.g. R1 and R2 both target attn_v/attn_o). Fusing between
            # rotations keeps every Linear a plain ``nn.Linear`` for the next apply.
            self._apply_and_learn_rotations(state, model)
        else:
            with torch.no_grad():
                apply_transform_config(model, self.transform_config)

    def _get_targets(self, model: torch.nn.Module) -> NamedModules:
        return [
            (name, module)
            for scheme in self.transform_config.config_groups.values()
            for arg in scheme.apply
            for name, module in match_named_modules(model, arg.targets, arg.ignore)
        ]

    def _center_embeddings(self, model: PreTrainedModel):
        for _, embedding in match_named_modules(
            model, [self.mappings.embedding], warn_on_fail=True
        ):
            center_embeddings(embedding)

    def _fuse_norms(self, model: PreTrainedModel):
        for mapping in self.norm_mappings:
            for norm, *linears in match_modules_set(
                model, (mapping.norm, *mapping.linears)
            ):
                # match_modules_set returns a list of lists
                assert len(norm) == 1
                fuse_norm_linears(norm[0], tree_leaves(linears))

    def _create_r1_scheme(self) -> TransformScheme:
        return TransformScheme(
            type=self.transform_type,
            randomize=self.randomize,
            requires_grad=self.learnable,
            precision=self.precision,
            head_dim=self.transform_block_size,
            apply=[
                TransformArgs(
                    targets=[
                        self.mappings.embedding,
                        self.mappings.attn_o,
                        *self.mappings.mlp_out,
                    ],
                    location="weight_output",
                ),
                TransformArgs(
                    targets=[
                        self.mappings.attn_q,
                        self.mappings.attn_k,
                        self.mappings.attn_v,
                        *self.mappings.mlp_in,
                        self.mappings.lm_head,
                    ],
                    location="weight_input",
                    inverse=True,
                ),
            ],
        )

    def _create_r2_scheme(self, head_dim: int) -> TransformScheme:
        return TransformScheme(
            type=self.transform_type,
            randomize=self.randomize,
            requires_grad=self.learnable,
            precision=self.precision,
            head_dim=head_dim,
            apply=[
                TransformArgs(targets=[self.mappings.attn_v], location="weight_output"),
                TransformArgs(
                    targets=[self.mappings.attn_o],
                    location="weight_input",
                    inverse=True,
                ),
            ],
        )

    def _create_r3_scheme(self, head_dim: int) -> TransformScheme:
        return TransformScheme(
            type=self.transform_type,
            randomize=self.randomize,
            requires_grad=self.learnable,
            precision=self.precision,
            head_dim=head_dim,
            apply=[
                TransformArgs(
                    targets=[self.mappings.attn],
                    location="q_attn",
                ),
                TransformArgs(
                    targets=[self.mappings.attn],
                    location="k_cache",
                ),
            ],
        )

    def _create_r4_scheme(self) -> TransformScheme:
        return TransformScheme(
            type=self.transform_type,
            randomize=self.randomize,
            requires_grad=self.learnable,
            precision=self.precision,
            head_dim=self.transform_block_size,
            apply=[
                TransformArgs(
                    targets=[*self.mappings.mlp_out],
                    location="input",
                ),
                TransformArgs(
                    targets=[*self.mappings.mlp_out],
                    location="weight_input",
                    inverse=True,
                ),
            ],
        )

    def _wrap_bias_for_training(self, model: torch.nn.Module):
        """
        For models with biased projections (e.g. Qwen), rotating an output weight
        without also rotating the bias breaks the math: ``y = R W x + b`` instead of
        ``R (W x + b) = R W x + R b``. The factories only parametrize the weight, so
        wrap the bias of any parametrized ``WEIGHT_OUTPUT`` linear with an adapter
        that applies the same rotation, keeping training forward passes consistent.
        The adapter is removed in :meth:`_finalize_offline_transforms`.
        """
        for _, module in model.named_modules():
            parametrizations = getattr(module, "parametrizations", None)
            if parametrizations is None or "weight" not in parametrizations:
                continue
            transform = parametrizations.weight[0]
            if not isinstance(transform, TransformBase):
                continue
            if (
                transform.args.location == TransformLocation.WEIGHT_OUTPUT
                and "bias" not in parametrizations
                and getattr(module, "bias", None) is not None
            ):
                P.register_parametrization(module, "bias", _BiasTransform(transform))

    def _remove_offload(self, model: torch.nn.Module):
        """
        Bring the whole model back onto the accelerator as plain parameters.

        The calibration pipelines call ``set_onload_device`` before firing
        ``calibration_start``, which offloads every module (wraps ``_parameters`` in
        an ``OffloadCache``). The compressed-tensors factories refuse to create
        ``requires_grad`` parametrizations on offloaded modules ("Offloaded training
        is not supported"), so un-offload the model before applying + learning the
        rotations. The next sub-pipeline re-offloads as needed.
        """
        for module in model.modules():
            if isinstance(module._parameters, OffloadCache):
                remove_module_offload(module, onload_tensors=True)

    def _apply_and_learn_rotations(self, state: State, model: torch.nn.Module):
        """
        Apply each requested rotation, learn its matrices over the calibration set,
        and fuse it back into the weights before moving to the next rotation.

        The compressed-tensors factories create parametrizations for
        ``requires_grad`` transforms, and a parametrized Linear becomes a
        ``ParametrizedLinear`` which the factories cannot transform again. Several
        SpinQuant rotations target the same Linear (R1 and R2 both hit attn_v/attn_o,
        R1 and R4 both hit mlp_out), so applying them all at once is impossible under
        ``learnable=True``. Applying, learning, and fusing one rotation at a time
        avoids that while still learning every rotation.

        ``learn_steps`` is split evenly across the requested rotations so the total
        training budget is independent of how many rotations are used.
        """
        learnable = [
            rotation
            for rotation in self.rotations
            if rotation in self.transform_config.config_groups
        ]
        if not learnable:
            logger.warning(
                "SpinQuantModifier(learnable=True) requested rotations {} but none "
                "were built",
                self.rotations,
            )
            return

        self._remove_offload(model)

        steps_per_rotation = max(1, self.learn_steps // len(learnable))
        for rotation in learnable:
            logger.info(
                "SpinQuant: applying + learning {} ({} steps)",
                rotation,
                steps_per_rotation,
            )
            sub_config = TransformConfig(
                config_groups={
                    rotation: self.transform_config.config_groups[rotation]
                }
            )
            with torch.no_grad():
                apply_transform_config(model, sub_config)
            self._wrap_bias_for_training(model)
            self._learn_rotations(state, model, steps=steps_per_rotation)
            self._finalize_offline_transforms(model)

        # rotations are fused into the weights; the runtime (vLLM) must never try to
        # re-learn them, and the serialized config should match the non-learnable path
        for scheme in self.transform_config.config_groups.values():
            scheme.requires_grad = False
        setattr(model, TRANSFORM_CONFIG_NAME, self.transform_config)

    def _learn_rotations(
        self, state: State, model: torch.nn.Module, steps: int | None = None
    ):
        """
        Learn the currently-applied rotation matrices by optimizing the language
        modeling loss over the calibration set, per SpinQuant (arXiv:2405.16406,
        Sec. 3.2).

        Only the rotation parameters are trained; the rest of the model is frozen.
        The rotations applied with ``requires_grad`` (i.e. the rotation currently
        being processed in :meth:`_apply_and_learn_rotations`) are optimized jointly:
        each step is one forward/backward over a calibration batch with gradients
        flowing to every rotation simultaneously. Because the factories share a single
        rotation Parameter across all applications (as in QuaRot/SpinQuant), a shared
        rotation matrix is learned per R-group.

        :param state: session state; ``state.data.calib`` must hold a dataloader
        :param model: model with the current rotation's transforms already applied
        :param steps: number of gradient steps for this rotation; defaults to
            ``self.learn_steps``
        """
        steps = self.learn_steps if steps is None else steps

        dataloader = state.data.calib
        if dataloader is None:
            raise ValueError(
                "SpinQuantModifier(learnable=True) requires calibration data to learn "
                "rotations, but none was provided. Pass a `dataset` to oneshot()."
            )

        # collect trainable transform parameters (dedupe: transforms are shared)
        transform_params: list[torch.nn.Parameter] = []
        seen_ids: set[int] = set()
        for _, module in model.named_modules():
            if not isinstance(module, TransformBase):
                continue
            for param in module.parameters(recurse=False):
                if param.requires_grad and id(param) not in seen_ids:
                    seen_ids.add(id(param))
                    transform_params.append(param)

        if not transform_params:
            raise ValueError(
                "SpinQuantModifier(learnable=True) found no trainable transform "
                "parameters. Request at least one rotation (e.g. rotations=['R1', "
                "'R2']) on a supported architecture."
            )

        # freeze the model, keep only the rotations trainable
        for param in model.parameters():
            param.requires_grad_(False)
        for param in transform_params:
            param.requires_grad_(True)

        if self.learn_optimizer == "adamw":
            optimizer = torch.optim.AdamW(transform_params, lr=self.learn_lr)
        else:
            optimizer = torch.optim.SGD(
                transform_params, lr=self.learn_lr, momentum=0.9
            )

        device = get_main_device()
        if device.type == "cuda":
            model.to(device)
        elif device.type != "cpu":
            # MPS/other backends do not support the float64 transforms used by
            # SpinQuant; fall back to CPU rather than crashing mid-training
            logger.warning(
                "SpinQuant learned rotations are only supported on CUDA/CPU; got "
                "{}, falling back to CPU",
                device,
            )
            device = torch.device("cpu")

        num_batches = len(dataloader)
        restore_training = model.training
        model.train()

        logger.info(
            "SpinQuant: learning {} rotation matrices over {} steps "
            "({} batches/set, lr={}, optimizer={})",
            len(transform_params),
            steps,
            num_batches,
            self.learn_lr,
            self.learn_optimizer,
        )

        # keep the model on the accelerator for the duration of training
        with disable_offloading():
            data_iter = iter(dataloader)
            for step in range(1, steps + 1):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    batch = next(data_iter)

                batch = {
                    key: value.to(device)
                    if isinstance(value, torch.Tensor)
                    else value
                    for key, value in batch.items()
                }

                optimizer.zero_grad(set_to_none=True)
                outputs = model(**batch)
                loss = outputs.loss
                if loss is None:
                    # fallback for models/dataloaders that do not wire labels
                    logits = outputs.logits
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = batch["labels"][..., 1:].contiguous()
                    loss = torch.nn.functional.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100,
                    )

                loss.backward()
                if self.learn_grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        transform_params, self.learn_grad_clip
                    )
                optimizer.step()

                if step == 1 or step % 10 == 0 or step == steps:
                    logger.info(
                        "SpinQuant: rotation step {}/{} loss={:.4f}",
                        step,
                        steps,
                        loss.item(),
                    )

        model.train(restore_training)

    @torch.no_grad()
    def _finalize_offline_transforms(self, model: torch.nn.Module):
        """
        Fuse the currently-applied offline rotations into their target weights (and
        biases) and remove the parametrization machinery added by the factories for
        ``requires_grad``. This returns every affected Linear to a plain
        ``nn.Linear`` (so the next rotation can be applied on top) and makes the
        serialized checkpoint structurally identical to the non-learnable path:
        rotated weights on disk, no transform submodules for offline rotations.
        Online (R3/R4) transforms are left as runtime hooks and their trained weights
        are saved as model parameters.
        """
        for _, module in list(model.named_modules()):
            parametrizations = getattr(module, "parametrizations", None)
            if parametrizations is None or "weight" not in parametrizations:
                continue

            transform = parametrizations.weight[0]
            if not isinstance(transform, TransformBase):
                continue

            raw_weight = parametrizations.weight.original.detach()
            fused_weight = transform(raw_weight)
            P.remove_parametrizations(module, "weight", leave_parametrized=True)
            update_offload_parameter(module, "weight", fused_weight)

            if "bias" in parametrizations and getattr(module, "bias", None) is not None:
                raw_bias = parametrizations.bias.original.detach()
                fused_bias = transform(raw_bias.unsqueeze(-1)).squeeze(-1)
                P.remove_parametrizations(module, "bias", leave_parametrized=True)
                update_offload_parameter(module, "bias", fused_bias)

            # drop the transform submodule, mirroring the non-learnable path
            for sub_name, sub_module in module.named_modules():
                if sub_module is transform:
                    delattr(module, sub_name.split(".")[0])
                    break


class _BiasTransform(torch.nn.Module):
    """
    Adapter that applies a 2D rotation transform to a 1D bias vector, keeping biased
    linears mathematically consistent while rotations are learned. Mirrors the
    unsqueeze/squeeze used by the non-learnable path when fusing bias into weights.
    """

    def __init__(self, transform: TransformBase):
        super().__init__()
        self.transform = transform

    def forward(self, bias: torch.Tensor) -> torch.Tensor:
        return self.transform(bias.unsqueeze(-1)).squeeze(-1)

    def right_inverse(self, value: torch.Tensor) -> torch.Tensor:
        return value
