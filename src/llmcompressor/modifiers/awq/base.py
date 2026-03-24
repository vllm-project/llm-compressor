import inspect
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Dict, Iterator, List, Literal, Optional, Set, Union

import torch
from compressed_tensors.offload.dist_utils import as_broadcastable, is_distributed
from compressed_tensors.quantization import (
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
    QuantizationStrategy,
    apply_quantization_config,
    disable_quantization,
    forward_quantize,
    is_preset_scheme,
    preset_name_to_scheme,
)
from compressed_tensors.utils import (
    align_modules,
    get_execution_device,
    get_lowest_common_ancestor_name,
    getattr_chain,
    match_modules_set,
    match_named_modules,
    patch_attrs,
    update_offload_parameter,
)
from loguru import logger
from pydantic import ConfigDict, Field, PrivateAttr, field_validator
from torch import distributed as dist
from torch.nn import Module
from torch.utils._pytree import tree_leaves
from tqdm import tqdm

from llmcompressor.core import Event, EventType, State, active_session
from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.awq.mappings import (
    AWQMapping,
    ResolvedMapping,
    get_layer_mappings_from_architecture,
)
from llmcompressor.modifiers.quantization.calibration import (
    call_observer,
    initialize_observer,
    reset_quantization_status,
)
from llmcompressor.modifiers.utils import update_fused_layer_weight_global_scales
from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.modifiers.utils.pytorch_helpers import is_moe_model
from llmcompressor.observers.base import Observer
from llmcompressor.pipelines.cache import IntermediatesCache
from llmcompressor.sentinel import Sentinel
from llmcompressor.utils import wait_for_comms
from llmcompressor.utils.helpers import calibration_forward_context
from llmcompressor.utils.pytorch.module import (
    get_module_to_name_dict,
)

__all__ = ["AWQModifier"]

_MISSING = object()  # sentinel for snapshot/restore of temp quant schemes


@dataclass
class AWQSearchResult:
    """Result of a grid search for the best smoothing scales for a single mapping."""

    best_scales: torch.Tensor
    best_error: float
    initial_error: float
    best_ratio: float
    history: list = field(default_factory=list)


class AWQModifier(Modifier):
    """
    Implements the AWQ (Activation-Weighted Quantization) smoothing algorithm,
    as described in https://arxiv.org/pdf/2306.00978.

    AWQ is a **pre-quantization transform**: it computes per-channel smoothing
    scales that reduce quantization error, then applies those scales to model
    weights in-place. It does **not** produce final quantized weights, scales,
    or zero-points. A downstream quantizer (e.g. QuantizationModifier or
    GPTQModifier) must follow AWQ in the recipe to finalize quantization.

    example recipe (stacked with a downstream quantizer):
    ```yaml
    AWQModifier:
      mappings:
        - smooth_layer: "re:.*self_attn_layer_norm"
          balance_layers: ["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"]
        - smooth_layer: "re:.*final_layer_norm"
          balance_layers: ["re:.*fc1"]
        # activation_hook_target specifies which submodule of the parent to hook
        # for activation caching.
        # This change is only useful for MoE models with parallel transformer blocks,
        # and one should use the default value (None) in most cases.
      ignore: ["lm_head"]
      scheme: "W4A16"
    QuantizationModifier:
      targets: ["Linear"]
      scheme: "W4A16"
      ignore: ["lm_head"]
    ```

    Lifecycle:

    - on_initialize
        - validate recipe (downstream quantizer exists, config compatible)
        - resolve mappings
    - on_start
        - temporarily apply quant schemes for grid search
        - set up activation cache hooks
    - on sequential/calibration epoch end
        - apply smoothing (grid search + scale application)
    - on_end
        - strip temporary quant schemes (restore prior module state)
        - remove hooks
    - on_finalize
        - clear caches and mappings

    :param sequential_targets: list of module names to compress in
        the same calibration pass
    :param mappings: list of activation layers to smooth and which layers to
        scale the output such that activations are smoothed.
        If regex is used, it matches layers with the largest overlap in module name.
        Each mapping may also include an ``activation_hook_target``: a dotted
        attribute path relative to the parent module (lowest common ancestor)
        specifying which submodule to hook for activation caching. This is useful
        for parallel transformer blocks where the default (hooking
        ``balance_layers[0]``) would capture the wrong activations.
    :param ignore: list of layers to ignore (not smoothed).
    :param offload_device: offload cached args to this device to reduce memory.
        Defaults to None (no offloading). Consider torch.device("cpu") for OOM.
    :param duo_scaling: whether to use both activations and weights for scaling.
        True: both used. False: activations only. "both": half grid each way.
    :param n_grid: number of grid points for the smoothing scale search. Default 20.
    :param scheme: quantization scheme used for pseudo-quantization during the
        AWQ grid search. This does NOT apply final quantization. It tells AWQ
        how the downstream quantizer will quantize, so AWQ can optimize for it.
    :param config_groups: alternative to scheme for specifying quantization config.
    :param targets: layer types targeted for quantization (for grid search only).
    """

    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)

    # AWQ smoothing parameters
    sequential_targets: str | list[str] | None = None
    mappings: list[AWQMapping] | None = None
    offload_device: torch.device | None | Sentinel = Sentinel("not_provided")
    duo_scaling: bool | Literal["both"] = True
    n_grid: int = 20

    # Quantization config used ONLY for pseudo-quantization during grid search.
    # AWQ does not own final quantization — a downstream quantizer does.
    config_groups: Optional[Dict[str, QuantizationScheme]] = None
    targets: Union[str, List[str]] = Field(default_factory=lambda: ["Linear"])
    ignore: List[str] = Field(default_factory=list)
    scheme: Optional[Union[str, Dict[str, Any]]] = None

    # Private state
    _resolved_mappings: list[ResolvedMapping] = PrivateAttr(default_factory=list)
    _parent_args_cache: dict[Module, IntermediatesCache] = PrivateAttr(
        default_factory=dict
    )
    # Dict[smooth layer name, [activation sums, activation counts]]
    _smooth_activation_stats: dict[str, list[torch.Tensor]] = PrivateAttr(
        default_factory=dict
    )
    _error_metrics: list[dict] = PrivateAttr(default_factory=list)
    _fp16_baseline_cache: dict[Module, IntermediatesCache] = PrivateAttr(
        default_factory=dict
    )
    # Snapshot of module state before temp quant schemes were applied
    _temp_quant_snapshot: dict[str, dict[str, Any]] = PrivateAttr(default_factory=dict)
    _resolved_config: Optional[QuantizationConfig] = PrivateAttr(None)

    # ------------------------------------------------------------------ #
    #  Quantization config resolution (local, NOT from QuantizationMixin) #
    # ------------------------------------------------------------------ #

    @field_validator("targets", mode="before")
    @classmethod
    def validate_targets(cls, value: Union[str, List[str]]) -> List[str]:
        if isinstance(value, str):
            return [value]
        return value

    @field_validator("scheme", mode="before")
    @classmethod
    def validate_scheme(
        cls, value: Optional[Union[str, Dict[str, Any]]]
    ) -> Optional[Union[str, Dict[str, Any]]]:
        if isinstance(value, str) and not is_preset_scheme(value):
            raise ValueError(
                "`scheme` must either be a preset scheme name or a dictionary "
                "of preset scheme names"
            )
        if isinstance(value, dict):
            for scheme_name in value.keys():
                cls.validate_scheme(scheme_name)
            for key, target in value.items():
                value[key] = cls.validate_targets(target)
        return value

    @field_validator("duo_scaling")
    @classmethod
    def validate_duo_scaling(cls, v):
        """Validate that duo_scaling is either True, False, or 'both' (lowercase)"""
        if v not in (True, False, "both"):
            raise ValueError(f"duo_scaling must be True, False, or 'both', got {v!r}")
        return v

    @property
    def resolved_config(self) -> QuantizationConfig:
        if self._resolved_config is None:
            self._resolved_config = self._resolve_quantization_config()
        return self._resolved_config

    @property
    def resolved_targets(self) -> Set[str]:
        targets = set()
        for config_group in self.resolved_config.config_groups.values():
            for target in config_group.targets:
                targets.add(target)
        return targets

    def _has_quant_config(self) -> bool:
        return not (
            self.config_groups is None
            and self.targets == ["Linear"]
            and self.ignore == []
            and self.scheme is None
        )

    def _resolve_quantization_config(self) -> QuantizationConfig:
        scheme = self.scheme
        targets = self.targets
        config_groups = self.config_groups
        ignore = self.ignore

        if scheme is not None and config_groups is not None:
            raise ValueError("Please specify either `scheme` or `config_groups`")

        if scheme is not None:
            if isinstance(scheme, str) and is_preset_scheme(scheme):
                scheme = {scheme: targets}
            config_groups = {}
            for idx, key in enumerate(scheme.keys()):
                if is_preset_scheme(key):
                    scheme_obj = preset_name_to_scheme(key, scheme[key])
                else:
                    scheme_obj = QuantizationScheme.model_validate(
                        {"targets": scheme[key], **scheme}
                    )
                group_name = f"group_{idx}"
                config_groups[group_name] = scheme_obj

        if config_groups is None or len(config_groups) == 0:
            default_quant_scheme = QuantizationScheme(targets=targets)
            config_groups = {"group_0": default_quant_scheme}

        return QuantizationConfig(
            config_groups=config_groups,
            quantization_status=QuantizationStatus.INITIALIZED,
            ignore=ignore,
        )

    # ------------------------------------------------------------------ #
    #  Temporary quant scheme management (context-managed)               #
    # ------------------------------------------------------------------ #

    @contextmanager
    def _temporary_quant_schemes(self, model: Module, with_observers: bool = False):
        """
        Apply quantization schemes to modules temporarily, restoring prior
        state on exit regardless of success or failure.

        This is the single code path for all temporary quant scheme usage in
        AWQModifier. Both ``on_initialize`` (for mapping resolution and
        duo_scaling validation) and ``_apply_smoothing`` (for the grid search)
        go through this context manager so that snapshot/restore logic is never
        duplicated.

        :param model: the model whose modules will be temporarily modified
        :param with_observers: if True, also initialize weight observers on
            each targeted module after applying schemes. Required for the grid
            search in ``_apply_smoothing``; not needed during ``on_initialize``
            where we only need the scheme metadata on modules.

        On entry:
            1. Snapshots every quant-related attribute on every targeted module.
            2. Resets any prior quantization status.
            3. Applies AWQ's resolved quantization config.
            4. Disables quantization (schemes present but forward pass unchanged).
            5. Optionally initializes weight observers.

        On exit:
            For each targeted module, restores every snapshotted attribute:
            - If the attribute did not exist before entry, it is deleted.
            - If the attribute existed before entry, it is set back to the
              prior value (even if AWQ overwrote it).
            This guarantees that downstream modifiers' state is never lost.
        """
        snapshot: dict[str, dict[str, Any]] = {}
        # All attributes that apply_quantization_config and initialize_observer
        # may add to modules. We must snapshot and restore every one of them.
        # apply_quantization_config also wraps the module's forward method
        # by adding it to the instance __dict__, so we must track that too.
        quant_attrs = (
            "quantization_scheme",
            "quantization_status",
            "quantization_enabled",
            "weight_scale",
            "weight_zero_point",
            "weight_observer",
        )

        # 1. Snapshot prior state for all modules we will touch
        for name, module in match_named_modules(
            model, self.resolved_targets, self.ignore
        ):
            prior = {}
            for attr in quant_attrs:
                prior[attr] = getattr(module, attr, _MISSING)
            # forward may be overridden on the instance by apply_quantization_config
            prior["forward"] = module.__dict__.get("forward", _MISSING)
            snapshot[name] = prior

        # 2-4. Apply temp quant config with quantization disabled
        for _, module in match_named_modules(model, self.resolved_targets, self.ignore):
            reset_quantization_status(module)

        apply_quantization_config(model, self.resolved_config)
        model.apply(disable_quantization)

        # 5. Optionally initialize weight observers
        if with_observers:
            for _, module in match_named_modules(
                model, self.resolved_targets, self.ignore
            ):
                if hasattr(module, "quantization_scheme"):
                    scheme = module.quantization_scheme
                    if scheme.weights is not None:
                        initialize_observer(module, base_name="weight")

        self._temp_quant_snapshot = snapshot
        try:
            yield
        finally:
            self._strip_temp_quant_schemes(model)

    def _strip_temp_quant_schemes(self, model: Module):
        """Restore modules to their state before temp quant schemes were applied."""
        quant_attrs = (
            "quantization_scheme",
            "quantization_status",
            "quantization_enabled",
            "weight_scale",
            "weight_zero_point",
            "weight_observer",
        )

        for name, module in match_named_modules(
            model, self.resolved_targets, self.ignore
        ):
            prior = self._temp_quant_snapshot.get(name, {})
            for attr in quant_attrs:
                prev_val = prior.get(attr, _MISSING)
                if prev_val is _MISSING:
                    # AWQ added this — remove it
                    if hasattr(module, attr):
                        delattr(module, attr)
                else:
                    # Restore prior value
                    setattr(module, attr, prev_val)

            # Restore or remove the instance-level forward override
            prev_forward = prior.get("forward", _MISSING)
            if prev_forward is _MISSING:
                # forward was not overridden before — remove the override
                module.__dict__.pop("forward", None)
            else:
                # Restore prior forward override
                module.__dict__["forward"] = prev_forward

        self._temp_quant_snapshot.clear()

    # ------------------------------------------------------------------ #
    #  Recipe validation                                                  #
    # ------------------------------------------------------------------ #

    def _validate_recipe(self, state: State):
        """
        Validate that the recipe contains a compatible downstream quantizer
        after this AWQModifier, with matching targets/ignore/scheme.
        """
        from llmcompressor.core import active_session
        from llmcompressor.modifiers.quantization.quantization import (
            QuantizationMixin,
        )

        try:
            modifiers = active_session().lifecycle.recipe.modifiers
        except Exception:
            modifiers = []
        if not modifiers:
            return

        # Find our position in the modifier list
        my_idx = None
        for i, m in enumerate(modifiers):
            if m is self:
                my_idx = i
                break

        if my_idx is None:
            return

        # Find downstream quantizer (must come after AWQ)
        downstream_quantizer = None
        for m in modifiers[my_idx + 1 :]:
            if isinstance(m, QuantizationMixin):
                downstream_quantizer = m
                break

        # Check for reversed ordering (quantizer before AWQ)
        for m in modifiers[:my_idx]:
            if isinstance(m, QuantizationMixin):
                logger.warning(
                    "A quantizer modifier appears before AWQModifier in the recipe. "
                    "AWQ smoothing should run before the quantizer for best results. "
                    "Consider reordering your recipe."
                )

        if downstream_quantizer is None:
            logger.warning(
                "AWQModifier is used without a downstream quantizer. "
                "AWQ only applies smoothing — add QuantizationModifier or "
                "GPTQModifier after AWQ to produce a quantized model."
            )
            return

        # Validate target/ignore compatibility if AWQ has quant config
        if self._has_quant_config():
            awq_targets = self.resolved_targets
            ds_targets = downstream_quantizer.resolved_targets
            awq_ignore = set(self.ignore) if self.ignore else set()
            ds_ignore = (
                set(downstream_quantizer.ignore)
                if downstream_quantizer.ignore
                else set()
            )

            # Check that AWQ targets are a subset of downstream targets
            if awq_targets and ds_targets and not awq_targets.issubset(ds_targets):
                diff = awq_targets - ds_targets
                logger.warning(
                    f"AWQModifier targets {diff} are not targeted by the "
                    "downstream quantizer. These modules will be smoothed but "
                    "not quantized."
                )

            # Check ignore alignment
            if awq_ignore != ds_ignore:
                logger.warning(
                    f"AWQModifier ignore list {awq_ignore} differs from "
                    f"downstream quantizer ignore list {ds_ignore}. "
                    "Mismatched ignore lists may cause unexpected behavior."
                )

            # Validate weight quantization scheme compatibility
            self._validate_scheme_compatibility(downstream_quantizer)

    def _validate_scheme_compatibility(self, downstream_quantizer):
        """
        Validate that AWQ's quant config (used for grid search) matches the
        downstream quantizer's config field-by-field.
        """
        awq_groups = self.resolved_config.config_groups
        ds_groups = downstream_quantizer.resolved_config.config_groups

        for group_name, awq_scheme in awq_groups.items():
            if awq_scheme.weights is None:
                continue

            # Find a matching downstream scheme by target overlap
            matched_ds_scheme = None
            for ds_scheme in ds_groups.values():
                if set(awq_scheme.targets) & set(ds_scheme.targets):
                    matched_ds_scheme = ds_scheme
                    break

            if matched_ds_scheme is None or matched_ds_scheme.weights is None:
                raise ValueError(
                    f"AWQModifier config group '{group_name}' targets "
                    f"{awq_scheme.targets} but no downstream quantizer scheme "
                    "targets the same modules with weight quantization. "
                    "Ensure the downstream quantizer covers the same targets."
                )

            awq_w = awq_scheme.weights
            ds_w = matched_ds_scheme.weights

            mismatches = []
            if awq_w.num_bits != ds_w.num_bits:
                mismatches.append(
                    f"num_bits: AWQ={awq_w.num_bits}, downstream={ds_w.num_bits}"
                )
            if awq_w.symmetric != ds_w.symmetric:
                mismatches.append(
                    f"symmetric: AWQ={awq_w.symmetric}, downstream={ds_w.symmetric}"
                )
            if awq_w.strategy != ds_w.strategy:
                mismatches.append(
                    f"strategy: AWQ={awq_w.strategy}, downstream={ds_w.strategy}"
                )
            if (
                awq_w.group_size is not None
                and ds_w.group_size is not None
                and awq_w.group_size != ds_w.group_size
            ):
                mismatches.append(
                    f"group_size: AWQ={awq_w.group_size}, downstream={ds_w.group_size}"
                )

            if mismatches:
                raise ValueError(
                    f"AWQModifier quantization config does not match downstream "
                    f"quantizer for targets {awq_scheme.targets}. Mismatches: "
                    + ", ".join(mismatches)
                    + ". The AWQ scheme must match the downstream quantizer "
                    "so the grid search optimizes for the correct quantization."
                )

    # ------------------------------------------------------------------ #
    #  Modifier lifecycle                                                 #
    # ------------------------------------------------------------------ #

    def on_initialize(self, state: State, **kwargs) -> bool:
        """
        Initialize AWQ: validate recipe, resolve mappings, validate config.

        Quant schemes are applied temporarily (via ``_temporary_quant_schemes``)
        so that ``_set_resolved_mappings`` can identify which modules are
        targeted for quantization and so that duo_scaling validation can inspect
        the effective quantization strategy. The schemes are stripped on exit
        from the context manager — they are re-applied later in
        ``_apply_smoothing`` with observers for the actual grid search.
        """
        self._validate_recipe(state)

        if self.mappings is None:
            logger.info("No AWQModifier.mappings provided, inferring from model...")
            self.mappings = get_layer_mappings_from_architecture(
                architecture=state.model.__class__.__name__
            )

        # Set default offload_device
        if self.offload_device == Sentinel("not_provided"):
            if is_moe_model(state.model):
                self.offload_device = torch.device("cpu")
                logger.info(
                    "MoE model detected: setting offload_device to 'cpu' by default "
                    "to reduce memory usage. You can override this by explicitly "
                    "setting offload_device in your recipe."
                )
            else:
                self.offload_device = None

        if self._has_quant_config():
            # Temporarily apply quant schemes so that:
            #   1. _set_resolved_mappings can resolve targeted_names
            #   2. duo_scaling validation can inspect quantization strategies
            # No observers needed — this is metadata-only.
            with self._temporary_quant_schemes(state.model, with_observers=False):
                self._set_resolved_mappings(state.model)
                self._validate_duo_scaling(state.model)
        else:
            self._set_resolved_mappings(state.model)

        return True

    def _validate_duo_scaling(self, model: Module):
        """
        Validate that duo_scaling is not used with TENSOR quantization strategy.

        Must be called while temporary quant schemes are applied to the model
        (i.e. inside the ``_temporary_quant_schemes`` context) so that
        ``module.quantization_scheme`` is available for inspection.
        """
        if self.duo_scaling is False:
            return

        for _, module in match_named_modules(model, self.resolved_targets, self.ignore):
            if (
                hasattr(module, "quantization_scheme")
                and hasattr(module.quantization_scheme, "weights")
                and module.quantization_scheme.weights.strategy
                == QuantizationStrategy.TENSOR
            ):
                raise ValueError(
                    "duo_scaling is only supported with per-channel quantization "
                    "strategies (group or channel), but found TENSOR strategy. "
                    "Please set duo_scaling=False or use a per-channel "
                    "quantization strategy."
                )

    def on_start(self, state: State, event: Event, **kwargs):
        self.started_ = True

        # Check for unsupported token masking with MoE up_proj -> down_proj mappings
        if state.loss_masks is not None and self._has_moe_up_down_proj_mapping():
            raise ValueError(
                "Token masking (use_loss_mask=True) is not supported with "
                "up_proj -> down_proj mappings in MoE models. The MoE routing "
                "mechanism dispatches tokens to different experts, and the loss mask "
                "cannot be properly aligned with this dispatch. Please either "
                "disable token masking or exclude the up_proj -> down_proj mapping "
                "for MoE layers from the AWQ configuration."
            )

        self._setup_activation_cache_hooks()

    def on_event(self, state: State, event: Event, **kwargs):
        if event.type_ == EventType.CALIBRATION_EPOCH_START:
            if not self.started_:
                self.on_start(state, None)

        elif event.type_ == EventType.SEQUENTIAL_EPOCH_END:
            self._apply_smoothing(state.model)

        elif event.type_ == EventType.CALIBRATION_EPOCH_END:
            self._apply_smoothing(state.model)

            if not self.ended_:
                self.on_end(state, None)

    def on_end(self, state: State, event: Event, **kwargs):
        """
        Finish smoothing. AWQ does NOT set scales/zero-points or finalize
        quantization — that is the downstream quantizer's responsibility.
        """
        self._assert_all_activations_consumed()
        self.ended_ = True
        self.remove_hooks()

    def on_finalize(self, state: State, **kwargs) -> bool:
        if not self.ended_:
            self.on_end(state, None)

        self._log_error_metrics()

        self._parent_args_cache.clear()
        self._smooth_activation_stats.clear()
        self._resolved_mappings.clear()
        self._error_metrics.clear()

        return True

    # ------------------------------------------------------------------ #
    #  Mapping resolution                                                 #
    # ------------------------------------------------------------------ #

    def _set_resolved_mappings(self, model: Module) -> None:
        """
        Transforms the list of activations to smooth and their corresponding weights
        into ResolvedMapping objects, resolving regular expressions.
        """
        resolved_mappings: list[ResolvedMapping] = []
        module_to_name = get_module_to_name_dict(model)

        # Get names of modules targeted for quantization (excludes ignored)
        targeted_names = set(
            name
            for name, _ in match_named_modules(
                model, self.resolved_targets, self.ignore
            )
        )

        for mapping in self.mappings:
            for smooth_layers, *nested_balance_layers in match_modules_set(
                model, (mapping.smooth_layer, *mapping.balance_layers)
            ):
                if len(smooth_layers) > 1:
                    raise ValueError(
                        "AWQ needs to match a single smoothlayer for each mapping but "
                        f"got {[module_to_name.get(s) for s in smooth_layers]}"
                        f" for mapping: {mapping}"
                    )
                smooth_layer = smooth_layers[0]
                smooth_name = module_to_name.get(smooth_layer)

                balance_layers = tree_leaves(nested_balance_layers)
                balance_names = [
                    module_to_name.get(balance_layer)
                    for balance_layer in balance_layers
                ]

                any_targeted = smooth_name in targeted_names or any(
                    bn in targeted_names for bn in balance_names
                )

                all_compatible = _check_layers_are_compatible(
                    smooth_layer, smooth_name, balance_layers, balance_names
                )

                skip_message: str | None = None
                if not all_compatible:
                    skip_message = " because found incompatible balance layers"
                elif not any_targeted:
                    skip_message = " because no layers are targeted for quantization"
                elif len(balance_layers) == 0:
                    skip_message = " because no balance layers were found"

                if skip_message:
                    logger.warning(
                        f"skipping AWQ for {smooth_name} for mapping {mapping}"
                        + skip_message
                    )
                    continue

                ancestor_name, ancestor = get_lowest_common_ancestor_with_avoid(
                    balance_names, model, torch.nn.ModuleList
                )

                activation_hook_target = None
                if mapping.activation_hook_target:
                    activation_hook_target = getattr_chain(
                        ancestor, mapping.activation_hook_target
                    )
                    if activation_hook_target is None:
                        raise ValueError(
                            f"activation_hook_target '{mapping.activation_hook_target}'"
                            f" not found on parent module '{ancestor_name}'"
                        )

                resolved_mappings.append(
                    ResolvedMapping(
                        smooth_name,
                        smooth_layer,
                        balance_layers,
                        balance_names=balance_names,
                        parent=ancestor,
                        parent_name=ancestor_name,
                        activation_hook_target=activation_hook_target,
                    )
                )
        self._resolved_mappings = resolved_mappings

    # ------------------------------------------------------------------ #
    #  Activation caching hooks                                           #
    # ------------------------------------------------------------------ #

    def _setup_activation_cache_hooks(self) -> None:
        """
        Attach forward hooks to cache parent kwargs and smooth activation means.
        """

        def cache_parent_kwargs_hook(
            module: Module,
            args: tuple[torch.Tensor, ...],
            kwargs,
        ):
            values = inspect.signature(module.forward).bind(*args, **kwargs)
            self._parent_args_cache[module].append(values.arguments)

        def create_cache_smooth_activations_hook_fn(smooth_name):
            def cache_smooth_activations_hook(
                _module: Module,
                args: tuple[torch.Tensor, ...],
                _output: torch.Tensor,
            ):
                activations = args[0].abs().detach()

                # Get loss mask for current batch from state
                session = active_session()
                state = session.state
                loss_masks = state.loss_masks if state else None
                batch_idx = state.current_batch_idx if state else -1
                loss_mask = (
                    loss_masks[batch_idx] if loss_masks and batch_idx >= 0 else None
                )

                if loss_mask is not None:
                    # Mask: [batch, seq] -> [batch, seq, 1]
                    mask = loss_mask.to(activations.device).unsqueeze(-1)
                    flat_activations = activations.flatten(0, -2)  # [batch*seq, hidden]
                    flat_mask = mask.flatten(0, -2).squeeze(-1)
                    masked_activations = flat_activations[flat_mask.bool()]
                else:
                    masked_activations = activations.flatten(0, -2)

                # accumulate activation sum&count
                new_sum = masked_activations.float().sum(dim=0).cpu()
                new_count = torch.tensor(masked_activations.size(0)).cpu()
                if smooth_name not in self._smooth_activation_stats:
                    self._smooth_activation_stats[smooth_name] = [
                        torch.zeros_like(new_sum),
                        torch.zeros_like(new_count),
                    ]
                self._smooth_activation_stats[smooth_name][0] += new_sum
                self._smooth_activation_stats[smooth_name][1] += new_count

            return cache_smooth_activations_hook

        for mapping in self._resolved_mappings:
            if mapping.parent not in self._parent_args_cache:
                self._parent_args_cache[mapping.parent] = IntermediatesCache(
                    None,
                    self.offload_device,
                )
                self.register_hook(
                    mapping.parent,
                    cache_parent_kwargs_hook,
                    "forward_pre",
                    with_kwargs=True,
                )

            # input activations to balance layers needed for loss function
            # storing inputs to first balance layer is sufficient
            # other balance layers get the same input
            #
            # For parallel transformer blocks (e.g. Command A, Gemma 3) the first
            # balance layer may not receive the right activations.  When
            # activation_hook_target is set on the mapping, hook that module
            # instead of balance_layers[0].
            layer_to_hook = mapping.activation_hook_target or mapping.balance_layers[0]
            self.register_hook(
                layer_to_hook,
                create_cache_smooth_activations_hook_fn(mapping.smooth_name),
                "forward",
            )

    # ------------------------------------------------------------------ #
    #  Smoothing (grid search + scale application)                        #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _apply_smoothing(self, model: Module) -> None:
        """
        For each mapping: run grid search to find best smoothing scales,
        then apply those scales to model weights in-place.

        Quant schemes are applied temporarily via context manager for the
        duration of the grid search, then stripped.
        """
        mappings_to_smooth = [
            mapping
            for mapping in self._resolved_mappings
            if mapping.smooth_name in self._smooth_activation_stats
        ]

        if not mappings_to_smooth:
            return

        # Apply temp quant schemes with observers for pseudo-quantization
        # during the grid search. Observers are needed so that
        # _evaluate_candidate can call call_observer + forward_quantize.
        with self._temporary_quant_schemes(model, with_observers=True):
            for mapping in tqdm(mappings_to_smooth, desc="Smoothing"):
                smooth_layer = mapping.smooth_layer
                balance_layers = mapping.balance_layers
                parent_module = mapping.parent

                with (
                    align_modules([parent_module, smooth_layer, *balance_layers]),
                    calibration_forward_context(model),
                    HooksMixin.disable_hooks(),
                ):
                    # Compute output of unquantized module
                    fp16_outputs = self._run_samples(parent_module)
                    if len(fp16_outputs) == 0 or all(
                        f.numel() == 0 for f in fp16_outputs
                    ):
                        logger.info(
                            f"Skipping smooth_layer {mapping.smooth_name}, no "
                            "activations found to scale. This can occasionally "
                            "occur in MoE models when certain experts are not "
                            "activated by calibration samples."
                        )
                        del self._smooth_activation_stats[mapping.smooth_name]
                        continue
                    if not all(
                        fp16_output.isfinite().all() for fp16_output in fp16_outputs
                    ):
                        logger.warning(
                            f"Skipping smooth_layer {mapping.smooth_name}, NaN or "
                            "inf outputs found during forward pass of the parent "
                            f"module {mapping.parent_name}. The model is either "
                            "generating NaN output with provided calibration data "
                            "set, or the mappings are incorrectly set and modifying "
                            "the model in undesired ways. If you encounter this "
                            "consistently, raise an issue at "
                            "https://github.com/vllm-project/llm-compressor/issues"
                        )
                        del self._smooth_activation_stats[mapping.smooth_name]
                        continue

                    orig_layer_weights = {
                        balance_layer: balance_layer.weight.clone()
                        for balance_layer in mapping.balance_layers
                        if hasattr(balance_layer, "quantization_scheme")
                        and hasattr(balance_layer.quantization_scheme, "weights")
                    }

                    search_result = self._select_best_scale(
                        mapping, fp16_outputs, orig_layer_weights
                    )

                    self._apply_best_scales(
                        mapping, search_result.best_scales, orig_layer_weights
                    )

                    del self._smooth_activation_stats[mapping.smooth_name]
                    del orig_layer_weights

        for v in self._parent_args_cache.values():
            v.batch_intermediates.clear()
        self._assert_all_activations_consumed()

    @torch.no_grad()
    def _run_samples(self, module: Module) -> list[torch.Tensor]:
        cache = self._parent_args_cache[module]
        use_prefetch = active_session().state.sequential_prefetch
        batch_iter = cache.iter_prefetch() if use_prefetch else cache
        outputs = [module(**batch_kwargs) for batch_kwargs in batch_iter]
        return [
            output[0] if isinstance(output, tuple) else output for output in outputs
        ]

    # ------------------------------------------------------------------ #
    #  Grid search — decomposed into pure helpers                         #
    # ------------------------------------------------------------------ #

    def _collect_activation_stats(
        self, mapping: ResolvedMapping, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Collect activation and weight statistics needed for scale generation.

        :returns: (x_mean, w_mean) where w_mean is None if duo_scaling is False
        """
        x_sum, count = self._smooth_activation_stats[mapping.smooth_name]
        if is_distributed():
            x_sum, count = _allreduce_data_sum([x_sum, count])
        x_mean = x_sum.to(device) / count.to(device)
        w_mean = None
        if self.duo_scaling:
            w_mean = self._compute_layer_means(mapping.balance_layers).to(device)
        return x_mean, w_mean

    @staticmethod
    def _generate_scale_candidates(
        x_mean: torch.Tensor,
        w_mean: torch.Tensor | None,
        n_grid: int,
        duo_scalings: list[bool],
    ) -> Iterator[tuple[float, bool, torch.Tensor]]:
        """
        Yield (ratio, use_duo_scaling, scales) for each grid point.
        """
        for grid_idx, use_duo_scaling in product(range(n_grid), duo_scalings):
            ratio = grid_idx / n_grid

            if use_duo_scaling and w_mean is not None:
                scales = (x_mean.pow(ratio) / (w_mean.pow(1 - ratio) + 1e-4)).clamp(
                    min=1e-4
                )
            else:
                scales = x_mean.pow(ratio).clamp(min=1e-4).view(-1)

            scales = scales / (scales.max() * scales.min()).sqrt()

            # avoid scaling values that overflow
            scales[torch.isinf(scales)] = 1
            scales[torch.isnan(scales)] = 1

            yield ratio, use_duo_scaling, scales

    def _evaluate_candidate(
        self,
        mapping: ResolvedMapping,
        scales: torch.Tensor,
        fp16_outputs: list[torch.Tensor],
        orig_layer_weights: dict[Module, torch.Tensor],
        balance_layers_to_patch: list[Module],
        device: torch.device,
    ) -> float:
        """
        Pseudo-quantize weights with candidate scales, run forward, return MSE loss.
        """
        _scalesview = scales.view(1, -1).to(device)

        # Q(W * s)
        for balance_layer in balance_layers_to_patch:
            if not hasattr(balance_layer, "quantization_scheme") or not hasattr(
                balance_layer.quantization_scheme, "weights"
            ):
                continue

            w_qscheme = balance_layer.quantization_scheme.weights
            balance_layer.weight.data.copy_(
                orig_layer_weights[balance_layer].to(_scalesview.device) * _scalesview
            )

            should_calculate_gparam = (
                w_qscheme.strategy == QuantizationStrategy.TENSOR_GROUP
            )
            call_observer(
                balance_layer,
                "weight",
                balance_layer.weight,
                should_calculate_gparam=should_calculate_gparam,
            )
            balance_layer.weight.data = (
                forward_quantize(
                    balance_layer,
                    balance_layer.weight,
                    "weight",
                    w_qscheme,
                )
                / _scalesview
            ).to(balance_layer.weight.dtype)

        # Apply fused global scales for TENSOR_GROUP during grid search
        if balance_layers_to_patch and all(
            getattr(layer.quantization_scheme.weights, "strategy", None)
            == QuantizationStrategy.TENSOR_GROUP
            for layer in balance_layers_to_patch
        ):
            update_fused_layer_weight_global_scales(mapping.parent)

        # W * X
        int_w_outputs = self._run_samples(mapping.parent)

        # compute mean squared error
        loss = self._compute_loss(fp16_outputs, int_w_outputs)
        del int_w_outputs

        return loss

    def _select_best_scale(
        self,
        mapping: ResolvedMapping,
        fp16_outputs: list[torch.Tensor],
        orig_layer_weights: dict[Module, torch.Tensor],
    ) -> AWQSearchResult:
        """
        Orchestrate grid search: generate candidates, evaluate each, return best.
        """
        device = get_execution_device(mapping.parent)
        x_mean, w_mean = self._collect_activation_stats(mapping, device)

        match self.duo_scaling:
            case "both":
                n_grid = int(self.n_grid / 2)
                duo_scalings = [False, True]
            case _:
                n_grid = self.n_grid
                duo_scalings = [self.duo_scaling]

        balance_layers_to_patch = [
            bl
            for bl in mapping.balance_layers
            if hasattr(bl, "quantization_scheme")
            and hasattr(bl.quantization_scheme, "weights")
        ]

        history = []
        best_ratio = -1.0
        best_scales = None
        best_error = float("inf")
        initial_error = None

        # Replace observers with memoryless_minmax for duration of grid search
        with patch_attrs(
            balance_layers_to_patch,
            "weight_observer",
            [
                Observer.load_from_registry(
                    "memoryless_minmax",
                    base_name="weight",
                    args=bl.quantization_scheme.weights,
                    module=bl,
                )
                for bl in balance_layers_to_patch
            ],
        ):
            total_iterations = n_grid * len(duo_scalings)
            candidates = self._generate_scale_candidates(
                x_mean, w_mean, n_grid, duo_scalings
            )
            pbar = tqdm(
                candidates,
                total=total_iterations,
                desc=f"Grid search for {mapping.smooth_name}",
                leave=False,
            )
            for ratio, use_duo_scaling, scales in pbar:
                loss = self._evaluate_candidate(
                    mapping,
                    scales,
                    fp16_outputs,
                    orig_layer_weights,
                    balance_layers_to_patch,
                    device,
                )

                if initial_error is None:
                    initial_error = loss

                history.append(
                    {"ratio": ratio, "duo_scaling": use_duo_scaling, "error": loss}
                )
                if loss < best_error:
                    best_error = loss
                    best_ratio = ratio
                    best_scales = scales.clone()
                pbar.set_postfix({"best_error": f"{best_error:.3e}"})

        if best_ratio == -1:
            logger.debug(history)
            raise Exception(
                "No finite loss was found in best scales grid search. This typically "
                "means NaN values are appearing in the forward pass of the parent "
                "module. If you encounter this error, raise an issue at "
                "https://github.com/vllm-project/llm-compressor/issues"
            )

        err_reduction = best_error / initial_error if initial_error > 0 else 1.0
        logger.debug(
            f"AWQ grid search for {mapping.smooth_name}: "
            f"initial error = {initial_error:.3e}, "
            f"best error = {best_error:.3e}, "
            f"error reduction rate (best/initial) = {err_reduction * 100:.3f}%"
        )

        self._error_metrics.append(
            {
                "layer_name": mapping.smooth_name,
                "parent_name": mapping.parent_name,
                "initial_error": initial_error,
                "best_error": best_error,
                "reduction": err_reduction,
            }
        )

        assert torch.isnan(best_scales).sum() == 0, (
            f"Nan found in scales: {best_scales}"
        )

        return AWQSearchResult(
            best_scales=best_scales.detach().cpu(),
            best_error=best_error,
            initial_error=initial_error,
            best_ratio=best_ratio,
            history=history,
        )

    # ------------------------------------------------------------------ #
    #  Scale application (smoothing)                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    @torch.no_grad()
    def _apply_best_scales(
        mapping: ResolvedMapping,
        best_scales: torch.Tensor,
        orig_layer_weights: dict[Module, torch.Tensor],
    ):
        """Apply the best smoothing scales to smooth_layer and balance_layers."""
        smooth_layer = mapping.smooth_layer
        balance_layers = mapping.balance_layers

        for layer in balance_layers:
            if layer in orig_layer_weights:
                scales = best_scales.to(layer.weight.device)
                update_offload_parameter(
                    layer,
                    "weight",
                    orig_layer_weights[layer].to(layer.weight.device)
                    * scales.view(1, -1),
                )

        # Apply inverse to smooth layer
        scales = best_scales.to(smooth_layer.weight.device)
        if smooth_layer.weight.ndim == 1:
            update_offload_parameter(
                smooth_layer,
                "weight",
                smooth_layer.weight.div_(scales),
            )
        else:
            # Edge case: smooth layer out_features != balance layer in_features
            # e.g. fused qkv_proj smoothing o_proj — scale last output features
            # https://github.com/casper-hansen/AutoAWQ/blob/main/awq/quantize/scale.py#L123
            weight = smooth_layer.weight
            weight[-scales.size(0) :].div_(scales.view(-1, 1))
            update_offload_parameter(smooth_layer, "weight", weight)

        if hasattr(smooth_layer, "bias") and smooth_layer.bias is not None:
            update_offload_parameter(
                smooth_layer,
                "bias",
                smooth_layer.bias.div_(scales),
            )

    # ------------------------------------------------------------------ #
    #  Utility methods                                                    #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _compute_loss(
        self,
        fp16_outputs: list[torch.Tensor],
        int_w_outputs: list[torch.Tensor],
    ) -> float:
        session = active_session()
        loss_masks = session.state.loss_masks if session.state else None

        device = fp16_outputs[0].device
        loss = torch.tensor(0.0, device=device)
        num_elements = torch.tensor(0, device=device)

        # Compute the MSE loss for each batch
        for batch_idx, (fp16_batch, int_w_batch) in enumerate(
            zip(fp16_outputs, int_w_outputs)
        ):
            loss_mask = loss_masks[batch_idx] if loss_masks else None

            if loss_mask is not None:
                token_mask = loss_mask.to(fp16_batch.device) == 1  # (batch, seq)
                fp16_masked = fp16_batch[token_mask]  # (num_masked_tokens, hidden)
                int_w_masked = int_w_batch.to(fp16_batch.device)[token_mask]
                loss += torch.nn.functional.mse_loss(
                    fp16_masked, int_w_masked, reduction="sum"
                )
                num_elements += fp16_masked.numel()
            else:
                loss += torch.nn.functional.mse_loss(
                    fp16_batch, int_w_batch.to(fp16_batch.device), reduction="sum"
                )
                num_elements += fp16_batch.numel()

        if is_distributed():
            loss, num_elements = _allreduce_data_sum([loss, num_elements])
        # Normalize the loss by the total number of elements
        return (loss / num_elements).item()

    def _log_error_metrics(self):
        if not self._error_metrics:
            return

        metrics_data = {
            "quantization_config": {
                "duo_scaling": self.duo_scaling,
                "n_grid": self.n_grid,
            },
            "total_layers": len(self._error_metrics),
            "metrics": self._error_metrics,
        }

        logger.debug(f"AWQ per-mapping error metrics: {metrics_data}")

        reductions = [m["reduction"] for m in self._error_metrics]
        avg_reduction = sum(reductions) / len(reductions)
        min_reduction = min(reductions)
        max_reduction = max(reductions)
        sorted_reductions = sorted(reductions)
        median_reduction = sorted_reductions[len(sorted_reductions) // 2]
        logger.debug(
            f"Error reduction statistics: "
            f"avg={avg_reduction:.4f}, median={median_reduction:.4f}, "
            f"min={min_reduction:.4f}, max={max_reduction:.4f}"
        )

    def _assert_all_activations_consumed(self):
        """
        Confirm all activations have been consumed
        If not, something has gone wrong
        """
        if len(self._smooth_activation_stats) != 0:
            raise RuntimeError("Some cached activations were not used")

    def _has_moe_up_down_proj_mapping(self) -> bool:
        """
        Check if any resolved mapping is an up_proj -> down_proj mapping
        where the balance layers are MoE experts (indicated by '.experts.'
        in the name).

        Token masking is not supported for such mappings because the MoE
        routing mechanism dispatches tokens to different experts, and the
        loss mask cannot be properly aligned with this dispatch.
        """
        for mapping in self._resolved_mappings:
            # Check if this is an up_proj -> down_proj mapping
            if mapping.smooth_name.endswith("up_proj"):
                for balance_name in mapping.balance_names:
                    if (
                        balance_name.endswith("down_proj")
                        and ".experts." in balance_name
                    ):
                        return True
        return False

    @staticmethod
    def _compute_layer_means(layers: list[Module]) -> torch.Tensor:
        """
        Compute per-channel/group/block/tensor mean of normalised weights
        for all passed in layers taking into account the quantization_scheme.
        """
        weight_total_count = 0
        weight_total_sum = 0

        for layer in layers:
            if not hasattr(layer, "weight"):
                logger.warning(
                    "Unable to find weight param for targeted"
                    f" layer {type(layer)}, skipping"
                )
                continue
            weight = layer.weight.clone()
            orig_shape = weight.shape

            q_args = getattr_chain(layer, "quantization_scheme.weights", None)
            if not q_args:
                logger.warning(
                    "Unable to find quantization scheme for "
                    f"targeted layer {type(layer)}, skipping"
                )
                continue

            match q_args.strategy:
                case QuantizationStrategy.TENSOR:
                    chunk_size = weight.numel()
                case QuantizationStrategy.CHANNEL:
                    chunk_size = weight.size(1)
                case QuantizationStrategy.GROUP | QuantizationStrategy.TENSOR_GROUP:
                    chunk_size = q_args.group_size
                case QuantizationStrategy.BLOCK:
                    block_height, block_width = q_args.block_structure
                    weight = (
                        weight.unflatten(0, (-1, block_height))
                        .unflatten(-1, (-1, block_width))
                        .transpose(1, 2)
                    )
                    intermediate_shape = weight.shape
                    chunk_size = block_height * block_width

            weight = weight.reshape(-1, chunk_size)
            weight.abs_()
            weight.div_(weight.amax(dim=1, keepdim=True) + 1e-6)
            if q_args.strategy == QuantizationStrategy.BLOCK:
                weight = weight.view(intermediate_shape).transpose(1, 2)

            weight = weight.reshape(orig_shape)
            weight_total_count += weight.size(0)
            weight_sum = weight.sum(0, dtype=torch.float64)
            weight_total_sum += weight_sum

        return weight_total_sum / weight_total_count


# ------------------------------------------------------------------ #
#  Module-level helpers (unchanged from original)                      #
# ------------------------------------------------------------------ #


def _check_layers_are_compatible(
    smooth_layer, smooth_name, balance_layers, balance_names
):
    """
    returns True if they are all compatible
    returns False if any smooth & balance layers are incompatible
    """
    for balance_layer, balance_name in zip(balance_layers, balance_names):
        # exclude v_proj->o_proj mappings whose shapes are incompatible
        # https://github.com/mit-han-lab/llm-awq/pull/67#issuecomment-1681632777
        if (
            isinstance(smooth_layer, torch.nn.Linear)
            and isinstance(balance_layer, torch.nn.Linear)
            and balance_name.endswith(".o_proj")
            and (
                (
                    smooth_name.endswith(".v_proj")
                    and smooth_layer.out_features != balance_layer.in_features
                )
                or (
                    smooth_name.endswith(".qkv_proj")
                    and smooth_layer.out_features != 3 * balance_layer.in_features
                )
            )
        ):
            return False
    return True


def get_lowest_common_ancestor_with_avoid(
    balance_names: Iterator[str], model: Module, avoid=torch.nn.ModuleList
):
    """
    Get the lowest ancestor that is not the avoided class/type.
    see compressed_tensors.utils.get_lowest_common_ancestor_name
    for detail on case handling.

    NOTE: primarily used to exclude parents of type ModuleList, which don't play
    nicely with hooks because their forward method is never directly
    called for MoE models. See Qwen3MoeSparseMoeBlock for example, experts
    are selected based on router output and their forward method is called.
    https://github.com/huggingface/transformers/blob/v4.52.4/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py#L233
    """
    ancestor_name = get_lowest_common_ancestor_name(balance_names)

    while True:
        if ancestor_name == "":
            return "", model
        ancestor = model.get_submodule(ancestor_name)
        if not isinstance(ancestor, avoid):
            return ancestor_name, ancestor
        ancestor_name = ".".join(ancestor_name.split(".")[:-1])


def _allreduce_data_sum(data: list[torch.Tensor]) -> list[torch.Tensor]:
    # needs to be on device to broadcast
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    data = [datum.to(device) for datum in data]

    pending_comms = []
    for datum in data:
        pending_comms.append(
            dist.all_reduce(
                as_broadcastable(datum), op=dist.ReduceOp.SUM, async_op=True
            )
        )
    wait_for_comms(pending_comms)
    return data
