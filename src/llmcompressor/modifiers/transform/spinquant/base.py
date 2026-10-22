from collections.abc import Iterable
from enum import Enum
from typing import Literal

import torch
from compressed_tensors import match_modules_set, match_named_modules
from compressed_tensors.transform import (
    TransformArgs,
    TransformConfig,
    TransformFactory,
    TransformScheme,
    apply_transform_config,
)
from compressed_tensors.utils import TorchDtype, get_head_dim
from pydantic import Field, PrivateAttr, ValidationInfo, field_validator
from torch.utils._pytree import tree_leaves
from transformers import PreTrainedModel

from llmcompressor.core import Event, State
from llmcompressor.modeling import center_embeddings, fuse_norm_linears
from llmcompressor.modifiers import Modifier
from llmcompressor.typing import NamedModules
from llmcompressor.utils import (
    get_high_precision,
    targets_embeddings,
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

    Lifecycle:

    - on_initialize
        - infer SpinQuantMappings & NormMappings
        - as needed, create transform schemes for R1, R2, R3, & R4
    - on_sequential_epoch_initialize
        - normalize embeddings
        - fuse norm layers into subsequent Linear layers
        - apply TransformConfig
            - fuse transforms into weights for mergeable transforms
            - add hooks for online transforms

    :param rotations: A list containing the names of rotations to apply to the model.
        Possible rotations include R1, R2, R3, and R4
    :param transform_type: The type of transform to apply to the model.
        `"hadamard"` has the least performance cost but only supports sizes which are
        powers of power of two.
        `"random-matrix"` has more performance cost, but supports a much larger set of
            sizes.
        `"random-matrix"` has the greatest performance cost, but supports any size
    :param randomize: if True, create distinct transforms for each application
    :param learnable: if True, attach gradients to transform weights for training
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
    _transform_factories: dict[str, TransformFactory] = PrivateAttr(
        default_factory=dict
    )

    @field_validator("randomize", "learnable", mode="before")
    def validate_not_implemented(cls, value, info: ValidationInfo):
        if value:
            raise NotImplementedError(f"{info.field_name} is not supported as of now")
        return value

    @field_validator("rotations", mode="before")
    def validate_rotations(cls, value):
        if isinstance(value, Iterable):
            return tuple(v.upper() for v in value)
        return value

    def on_initialize(self, state: State, **kwargs) -> bool:
        if self.mappings is None:
            self.mappings = infer_mapping_from_model(state.model)
        if self.norm_mappings is None:
            self.norm_mappings = infer_norm_mapping_from_model(state.model)
        if self.transform_config is None:
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

        self._transform_factories = {
            name: TransformFactory.from_scheme(scheme, name=name)
            for name, scheme in self.transform_config.config_groups.items()
        }

        return True

    @torch.no_grad()
    def on_sequential_epoch_initialize(
        self,
        state: State,
        event: Event,
        modules: list[torch.nn.Module] | None = None,
        **kwargs,
    ):
        self._apply_transform_scope(state.model, modules=modules, **kwargs)

    def _apply_transform_scope(
        self,
        model: PreTrainedModel,
        modules: list[torch.nn.Module] | None = None,
        onloaded_modules: dict[str, torch.nn.Module] | None = None,
        offload_kwargs: dict[str, dict] | None = None,
        **kwargs,
    ):
        targets = self._get_targets(model, modules)
        if targets_embeddings(model, targets):
            # Untying may update a tied output head outside this scope, so do it before
            # applying any scoped transforms.
            untie_word_embeddings(model)

        self._center_embeddings(model, modules)
        self._fuse_norms(model, modules)
        if onloaded_modules is not None:
            for factory in self._transform_factories.values():
                factory.cache_device = torch.device("cpu")
        apply_transform_config(
            model,
            self.transform_config,
            modules=modules,
            factories=self._transform_factories,
        )
        if onloaded_modules is not None and offload_kwargs is not None:
            for name, module in dict(targets).items():
                for child_name, child in module.named_children():
                    child_name = f"{name}.{child_name}" if name else child_name
                    onloaded_modules[child_name] = child
                    if name in offload_kwargs:
                        offload_kwargs.setdefault(
                            child_name, offload_kwargs[name].copy()
                        )

    def on_finalize(self, state: State, **kwargs) -> bool:
        self._transform_factories.clear()
        return True

    def _get_targets(
        self,
        model: torch.nn.Module,
        modules: list[torch.nn.Module] | None = None,
    ) -> NamedModules:
        return [
            (name, module)
            for scheme in self.transform_config.config_groups.values()
            for arg in scheme.apply
            for name, module in match_named_modules(model, arg.targets, arg.ignore)
            if modules is None or module in modules
        ]

    def _center_embeddings(
        self, model: PreTrainedModel, modules: list[torch.nn.Module] | None = None
    ):
        for _, embedding in match_named_modules(
            model,
            [self.mappings.embedding],
            warn_on_fail=modules is None,
        ):
            if modules is not None and embedding not in modules:
                continue
            center_embeddings(embedding)

    def _fuse_norms(
        self, model: PreTrainedModel, modules: list[torch.nn.Module] | None = None
    ):
        module_scope = None if modules is None else set(modules)
        for mapping in self.norm_mappings:
            for norm, *linears in match_modules_set(
                model, (mapping.norm, *mapping.linears)
            ):
                # match_modules_set returns a list of lists
                assert len(norm) == 1
                norm_layer = norm[0]
                linear_layers = tree_leaves(linears)
                if module_scope is not None:
                    mapping_modules = [norm_layer, *linear_layers]
                    in_scope = [module in module_scope for module in mapping_modules]
                    if any(in_scope) and not all(in_scope):
                        raise ValueError(
                            "SpinQuant norm mapping spans sequential subgraphs. "
                            "Choose sequential targets that keep each norm and its "
                            "linears in the same subgraph."
                        )
                    if not all(in_scope):
                        continue

                fuse_norm_linears(norm_layer, linear_layers)

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
