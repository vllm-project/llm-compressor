import math
from functools import partial

import torch
from compressed_tensors.distributed import is_distributed
from compressed_tensors.offload import get_execution_device
from compressed_tensors.quantization import enable_quantization
from compressed_tensors.quantization.lifecycle.forward import forward_quantize
from compressed_tensors.registry import standardize_lookup_name
from compressed_tensors.utils import (
    getattr_chain,
    match_named_modules,
    update_offload_parameter,
)
from loguru import logger
from pydantic import Field, PrivateAttr
from torch.utils._pytree import tree_map_only

from llmcompressor.core import Event, State
from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.quantization.calibration import observe, update_qparams
from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.pipelines.cache import IntermediatesCache, IntermediateValue
from llmcompressor.utils.pytorch import infer_sequential_targets

__all__ = ["QADModifier"]

# Each FP16 overflow halves GradScaler's scale; 32 attempts take its default
# 2**16 scale down to 2**-16
_LOSS_SCALE_ATTEMPTS = 32


class QADModifier(Modifier):
    """
    Quantization-aware distillation (QAD) trains the weights of each sequential
    subgraph's targets, such as decoder layers, so that their fake-quantized
    outputs match their unquantized outputs. A subgraph's targets train jointly as
    a chain: each target takes the previous target's output, and the loss compares
    the last target's outputs. ``sequential_targets_per_subgraph`` sets how many
    targets train together; training memory grows with it. For each subgraph,
    this modifier:

    1. Caches the targets' inputs and the last target's unquantized outputs for
       every calibration batch. The unquantized outputs are the teacher; no
       teacher model is loaded
    2. Splits the cached batches into training and validation batches
    3. After the preceding modifier calibrates the targets' weight qparams, trains
       their weights through compressed-tensors fake quantization, re-observing
       weight qparams before training and after every epoch
    4. Restores the weights with the lowest validation loss, which may be the
       untrained weights
    5. Re-observes weight qparams and writes the fake-quantized weights back

    With ``propagate_error=True`` (default, recommended), each subgraph trains on
    the preceding subgraphs' quantized outputs, matching its inputs at inference.
    With ``propagate_error=False``, it trains on the original model's activations.
    Validation batches are withheld from QAD updates but still calibrate the
    quantizer.

    Place QADModifier after a weight quantization modifier in the same recipe and
    use the sequential pipeline. RTN and GPTQ are tested. Other preceding methods
    must keep weights unquantized during calibration, initialize qparams before
    QAD's ``sequential_epoch_end``, and use floating weights compatible with
    compressed-tensors fake quantization.

    Sample usage::

        from llmcompressor import oneshot
        from llmcompressor.modifiers.qad import QADModifier
        from llmcompressor.modifiers.quantization import QuantizationModifier

        recipe = [
            QuantizationModifier(
                targets="Linear", scheme="NVFP4A16", ignore=["lm_head"]
            ),
            QADModifier(num_epochs=1, lr=2e-6),
        ]
        oneshot(model=model, dataset=ds, recipe=recipe, pipeline="sequential")

    :param reobserve_weights: re-observe weight qparams before training, after
        every epoch, and before writing the final weights. Requires the preceding
        modifier to keep live weight observers. Set to False to keep its qparams
        fixed. Defaults to True
    :param num_epochs: training epochs per subgraph. Defaults to 1
    :param lr: AdamW learning rate. Defaults to 2e-6
    :param weight_decay: AdamW weight decay. Defaults to 0
    :param gradient_accumulation_steps: calibration batches per optimizer update.
        Defaults to 1
    :param max_grad_norm: gradient norm clipping threshold, or None to disable
        clipping. Defaults to 1.0
    :param seed: seed for the validation split and per-epoch shuffling.
        Defaults to 42
    :param offload_device: device that holds cached batches and best-weight
        snapshots between uses. Defaults to "cpu"
    :param validation_fraction: fraction of calibration batches held out from
        training to select the best epoch; at least one batch is held out.
        Defaults to 0.1
    """

    requires_calibration_data: bool = True
    reobserve_weights: bool = True
    num_epochs: int = Field(default=1, ge=1)
    lr: float = Field(default=2.0e-6, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    max_grad_norm: float | None = Field(default=1.0, gt=0)
    seed: int = 42
    offload_device: str = "cpu"
    validation_fraction: float = Field(default=0.1, gt=0, lt=1)

    # sequential targets containing quantized weights, in model execution order
    _seq_target_names: dict[torch.nn.Module, str] = PrivateAttr(default_factory=dict)
    # sequential target -> cache of captured call arguments, one entry per batch
    _input_caches: dict[torch.nn.Module, IntermediatesCache] = PrivateAttr(
        default_factory=dict
    )
    # sequential target -> cache of unquantized outputs; intermediate outputs deleted
    _output_caches: dict[torch.nn.Module, IntermediatesCache] = PrivateAttr(
        default_factory=dict
    )
    # sequential targets whose distillation has finished
    _distilled: set[torch.nn.Module] = PrivateAttr(default_factory=set)
    # sequential target -> the target whose output is its first input, or None if
    # it starts its subgraph's chain
    _predecessors: dict[torch.nn.Module, torch.nn.Module | None] = PrivateAttr(
        default_factory=dict
    )
    # most recently run sequential target and its hidden states
    _cur_seq_target: torch.nn.Module | None = PrivateAttr(default=None)
    _cur_seq_target_output: torch.Tensor | None = PrivateAttr(default=None)

    def on_initialize(self, state: State, **kwargs) -> bool:
        pipeline = kwargs.get("pipeline")
        if pipeline and standardize_lookup_name(pipeline) != "sequential":
            raise ValueError(
                'QADModifier requires pipeline="sequential" so the preceding '
                "weight quantization modifier and QAD share each subgraph's "
                f"calibration stage; got pipeline={pipeline!r}"
            )
        modules = _modules_to_quantize(state.model.modules())
        if not modules:
            raise ValueError(
                "Place QADModifier after a weight quantization modifier, such as "
                "QuantizationModifier (RTN) or GPTQModifier, in the same recipe"
            )
        # TODO add distributed support for QAD calibration
        if is_distributed() and torch.distributed.get_world_size() > 1:
            raise ValueError("QAD currently supports single-process calibration")
        seq_target_patterns = infer_sequential_targets(
            state.model, kwargs.get("sequential_targets")
        )
        self._seq_target_names = {
            module: name or type(module).__name__
            for name, module in match_named_modules(state.model, seq_target_patterns)
            if _modules_to_quantize(module.modules())
        }
        return True

    def on_calibration_start(self, state: State, event: Event, **kwargs):
        for seq_target in self._seq_target_names:
            self.register_hook(
                seq_target,
                partial(self._input_capture_hook, state=state),
                "forward_pre",
                with_kwargs=True,
            )
            self.register_hook(
                seq_target, self._output_capture_hook, "forward", with_kwargs=True
            )

    def _input_capture_hook(self, module, args, kwargs, *, state):
        if module not in self._input_caches:
            self._input_caches[module] = IntermediatesCache(
                offload_device=self.offload_device
            )
        batches = self._input_caches[module]
        # One entry per batch, so the entry count must equal the batch index
        if module in self._distilled or state.current_batch_idx != len(batches):
            raise ValueError(
                "QAD requires each target to run once per calibration batch "
                "in one sequential stage"
            )
        # A target whose first input is the previous target's output extends that
        # target's chain. Training recomputes this input from the chain, and only
        # the chain's last output is a teacher.
        predecessor = (
            self._cur_seq_target
            if args and args[0] is self._cur_seq_target_output
            else None
        )
        self._predecessors[module] = predecessor
        if predecessor is not None:
            args = (None, *args[1:])
            self._output_caches[predecessor].delete(
                state.current_batch_idx, ["teacher_output"]
            )
        batches.append(_copy_tensors({"args": args, "kwargs": kwargs}))

    def _output_capture_hook(self, module, args, kwargs, output):
        if module not in self._output_caches:
            self._output_caches[module] = IntermediatesCache(
                offload_device=self.offload_device
            )
        self._output_caches[module].append(_copy_tensors({"teacher_output": output}))
        self._cur_seq_target = module
        self._cur_seq_target_output = _get_first(output)

    def on_sequential_epoch_end(
        self, state: State, event: Event, modules: list[torch.nn.Module], **kwargs
    ):
        self._cur_seq_target = None
        self._cur_seq_target_output = None
        # The cache holds this subgraph's targets in execution order
        subgraph = set(modules)
        seq_targets = [
            seq_target for seq_target in self._input_caches if seq_target in subgraph
        ]
        if not seq_targets:
            return
        if any(
            self._predecessors[seq_target] is not previous
            for previous, seq_target in zip([None, *seq_targets], seq_targets)
        ):
            raise ValueError(
                "QAD trains a subgraph's targets jointly, so each target after the "
                "first must take the previous target's output as its first input"
            )
        # Ordered caches for this subgraph, removed from the capture maps once consumed.
        subgraph_inputs = [
            self._input_caches.pop(seq_target) for seq_target in seq_targets
        ]
        subgraph_outputs = [
            self._output_caches.pop(seq_target) for seq_target in seq_targets
        ]
        batch_counts = {len(cache) for cache in subgraph_inputs + subgraph_outputs}
        if len(batch_counts) != 1:
            raise ValueError(
                "QAD requires each sequential target to capture one input and output "
                "for every calibration batch"
            )
        batch_count = len(subgraph_inputs[0])
        train_indices, validation_indices = self._split_batch_indices(batch_count)
        if torch.accelerator.is_available():
            for cache in subgraph_inputs + subgraph_outputs:
                for index in range(batch_count):
                    cache.pin_memory(index)
        logger.info(
            "QAD cached {} local teacher batches for {}",
            batch_count,
            self._first_last_name(seq_targets),
        )
        first_input_cache = subgraph_inputs[0]
        last_output_cache = subgraph_outputs[-1]
        chain_batches = []
        for batch_index in range(batch_count):
            first_inputs = first_input_cache.batch_intermediates[batch_index]
            links = [
                IntermediateValue(
                    {
                        "args": cache.batch_intermediates[batch_index]["args"],
                        "kwargs": cache.batch_intermediates[batch_index]["kwargs"],
                    },
                    None,
                )
                for cache in subgraph_inputs[1:]
            ]
            chain_batches.append(
                {
                    "args": first_inputs["args"],
                    "kwargs": first_inputs["kwargs"],
                    "links": IntermediateValue(links, None),
                    "teacher_output": last_output_cache.batch_intermediates[
                        batch_index
                    ]["teacher_output"],
                }
            )
        with HooksMixin.disable_hooks():
            self._apply_distillation(
                seq_targets,
                [chain_batches[i] for i in train_indices],
                [chain_batches[i] for i in validation_indices],
            )

    def _split_batch_indices(self, batch_count):
        if batch_count < 2:
            raise ValueError("QAD validation requires at least two calibration batches")
        validation_count = min(
            batch_count - 1, max(1, math.ceil(batch_count * self.validation_fraction))
        )
        order = torch.randperm(
            batch_count, generator=torch.Generator().manual_seed(self.seed)
        ).tolist()
        return order[validation_count:], order[:validation_count]

    def _first_last_name(self, seq_targets):
        # A chain's targets run consecutively, so its ends identify it
        first, last = (
            self._seq_target_names[seq_targets[0]],
            self._seq_target_names[seq_targets[-1]],
        )
        return first if len(seq_targets) == 1 else f"{first}..{last}"

    def _apply_distillation(self, seq_targets, train, validation):
        name = self._first_last_name(seq_targets)
        modules = _modules_to_quantize(
            module for seq_target in seq_targets for module in seq_target.modules()
        )
        self._check_weight_scales(modules)
        # A weight shared by several modules is optimized once
        trainable = list(dict.fromkeys(module.weight for module in modules))
        # FP32 optimizer state/master weights avoid Adam's epsilon underflow and
        # sub-ULP updates when the model's execution weights are FP16/BF16.
        masters = [
            torch.nn.Parameter(p.detach().float().clone())
            if p.dtype in (torch.float16, torch.bfloat16)
            else p
            for p in trainable
        ]
        optimizer = torch.optim.AdamW(
            masters, lr=self.lr, weight_decay=self.weight_decay
        )
        original_requires_grad = {
            p: p.requires_grad
            for seq_target in seq_targets
            for p in seq_target.parameters()
        }
        for seq_target in seq_targets:
            seq_target.requires_grad_(False)
        for parameter in trainable:
            parameter.requires_grad_(True)
        for module in modules:
            enable_quantization(module)
        try:
            self._reobserve_weights(seq_targets, modules, "before_training")
            initial_train = self._evaluate(seq_targets, train)
            steps = self._train_with_validation(
                seq_targets, modules, optimizer, trainable, masters, train, validation
            )
            self._reobserve_weights(seq_targets, modules, "before_materialization")
            self._materialize_quantized_weights(modules)
            final_train = self._evaluate(seq_targets, train)
            final_validation = self._evaluate(seq_targets, validation)
            self._distilled.update(seq_targets)
            logger.info(
                "QAD {}: {} updates over {} epochs; train MSE {:.6e} -> {:.6e}, "
                "final validation MSE {:.6e}",
                name,
                steps,
                self.num_epochs,
                initial_train,
                final_train,
                final_validation,
            )
        finally:
            for parameter, requires_grad in original_requires_grad.items():
                parameter.grad = None
                parameter.requires_grad_(requires_grad)

    @staticmethod
    def _check_weight_scales(modules):
        for module in modules:
            keys = ["weight_scale"]
            # Only some schemes, such as NVFP4, have a global scale
            if getattr(module, "weight_global_scale", None) is not None:
                keys.append("weight_global_scale")
            for key in keys:
                scale = getattr(module, key, None)
                if (
                    scale is None
                    or scale.is_meta
                    or not torch.isfinite(scale.float()).all()
                    or not (scale.float() > 0).all()
                ):
                    raise ValueError(
                        f"QAD requires initialized {key}; run the preceding "
                        "quantization modifier before QAD at the subgraph boundary"
                    )

    @torch.no_grad()
    def _reobserve_weights(self, seq_targets, modules, stage):
        if not self.reobserve_weights:
            return
        if any(getattr(module, "weight_observer", None) is None for module in modules):
            raise ValueError("QAD weight re-observation requires live weight observers")
        # Observe the entire group before updating scales: fused Q/K/V and
        # gate/up projections can share a global-scale observer.
        observe(modules, "weight")
        update_qparams(modules, "weight")
        logger.info(
            "QAD {} re-observed weights: {}",
            self._first_last_name(seq_targets),
            stage,
        )

    def _train_with_validation(
        self, seq_targets, modules, optimizer, trainable, masters, train, validation
    ):
        name = self._first_last_name(seq_targets)
        generator = torch.Generator().manual_seed(self.seed)
        scaler = torch.amp.GradScaler(
            get_execution_device(modules[0]).type,
            enabled=any(p.dtype == torch.float16 for p in trainable),
        )
        best_loss = self._evaluate(seq_targets, validation)
        best_weights = self._snapshot_weights(trainable)
        logger.info("QAD {} initial validation MSE {:.6e}", name, best_loss)
        steps = 0
        for epoch in range(1, self.num_epochs + 1):
            order = torch.randperm(len(train), generator=generator).tolist()
            steps += self._train_epoch(
                seq_targets,
                optimizer,
                trainable,
                masters,
                [train[i] for i in order],
                scaler,
            )
            self._reobserve_weights(seq_targets, modules, f"epoch_{epoch}")
            loss = self._evaluate(seq_targets, validation)
            if loss < best_loss:
                best_loss = loss
                best_weights = self._snapshot_weights(trainable)
            logger.info(
                "QAD {} epoch {} validation MSE {:.6e}, best {:.6e}",
                name,
                epoch,
                loss,
                best_loss,
            )
        self._copy_weights(trainable, best_weights)
        return steps

    def _train_epoch(self, seq_targets, optimizer, trainable, masters, batches, scaler):
        size = self.gradient_accumulation_steps
        groups = [batches[i : i + size] for i in range(0, len(batches), size)]
        for group in groups:
            self._accumulate_gradients(
                seq_targets, optimizer, trainable, masters, group, scaler
            )
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(masters, self.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            # Write the updated master weights back to the model
            self._copy_weights(trainable, masters)
        return len(groups)

    @torch.enable_grad()
    def _accumulate_gradients(
        self, seq_targets, optimizer, trainable, masters, group, scaler
    ):
        """
        Accumulate the group's mean loss gradients into the master weights, unscaled
        and ready for clipping.

        Small reconstruction losses can underflow during FP16 backward, so the loss
        is scaled by GradScaler, whose scale stays fixed during accumulation. If the
        scaled gradients overflow, GradScaler lowers its scale and the same group is
        retried, so every calibration batch contributes to an update.
        """
        for _ in range(_LOSS_SCALE_ATTEMPTS):
            for parameter, master in zip(trainable, masters):
                parameter.grad = master.grad = None
            for batch in IntermediatesCache(group).iter_prefetch():
                loss = self._batch_loss(seq_targets, batch) / len(group)
                scaler.scale(loss).backward()
            for parameter, master in zip(trainable, masters):
                if master is not parameter and parameter.grad is not None:
                    master.grad = parameter.grad.float()
            scaler.unscale_(optimizer)
            if all(p.grad is None or torch.isfinite(p.grad).all() for p in masters):
                return
            if not scaler.is_enabled():
                break
            # GradScaler skips this step because of the overflow, then lowers its
            # scale for the retry
            scaler.step(optimizer)
            scaler.update()
        raise ValueError(
            f"Nonfinite QAD gradient in {self._first_last_name(seq_targets)}"
        )

    def _snapshot_weights(self, parameters):
        return [p.detach().to(self.offload_device, copy=True) for p in parameters]

    @staticmethod
    @torch.no_grad()
    def _copy_weights(parameters, weights):
        for parameter, weight in zip(parameters, weights):
            parameter.copy_(weight)

    @torch.no_grad()
    def _evaluate(self, seq_targets, batches):
        return sum(
            self._batch_loss(seq_targets, batch).item()
            for batch in IntermediatesCache(batches).iter_prefetch()
        ) / len(batches)

    def _batch_loss(self, seq_targets, batch):
        # Run the chain, feeding each target the previous target's hidden states
        output = seq_targets[0](*batch["args"], **batch["kwargs"])
        for seq_target, inputs in zip(seq_targets[1:], batch["links"]):
            args = (_get_first(output), *inputs["args"][1:])
            output = seq_target(*args, **inputs["kwargs"])
        prediction = _get_first(output)
        teacher_output = _get_first(batch["teacher_output"])
        # FP32 keeps the reduction accurate for FP16/BF16 outputs
        loss = torch.nn.functional.mse_loss(prediction.float(), teacher_output.float())
        if not torch.isfinite(loss):
            raise ValueError(
                f"Nonfinite QAD loss in {self._first_last_name(seq_targets)}"
            )
        return loss

    @staticmethod
    @torch.no_grad()
    def _materialize_quantized_weights(modules):
        for module in modules:
            weight = forward_quantize(
                module, module.weight, "weight", module.quantization_scheme.weights
            )
            update_offload_parameter(module, "weight", weight)

    def on_calibration_end(self, state: State, event: Event, **kwargs):
        try:
            if len(self._distilled) != len(self._seq_target_names):
                raise ValueError(
                    "QAD did not optimize every target; use the sequential pipeline "
                    "with each target inside a single subgraph"
                )
        finally:
            self._clear_cache()

    def on_finalize(self, state: State, **kwargs) -> bool:
        self._clear_cache()
        return True

    def _clear_cache(self):
        self.remove_hooks()
        self._input_caches.clear()
        self._output_caches.clear()
        self._predecessors.clear()
        self._cur_seq_target = None
        self._cur_seq_target_output = None


def _modules_to_quantize(modules):
    # Unlike compressed-tensors' is_module_quantized, skip activation-only schemes,
    # which leave no quantized weights for QAD to train
    return [
        module
        for module in modules
        if getattr_chain(module, "quantization_scheme.weights", None) is not None
    ]


def _get_first(output):
    # Decoder layers return hidden states, or a tuple starting with them
    return output[0] if isinstance(output, tuple) else output


def _copy_tensors(values):
    # QAD reuses cached inputs and teacher outputs across epochs, so they must not
    # alias live tensors. IntermediatesCache stores tensors that are already on the
    # offload device by reference.
    return tree_map_only(torch.Tensor, lambda t: t.detach().clone(), values)
