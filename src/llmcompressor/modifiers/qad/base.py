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
from llmcompressor.pipelines.cache import IntermediatesCache
from llmcompressor.utils.pytorch import infer_sequential_targets

__all__ = ["QADModifier"]

# Each FP16 overflow halves GradScaler's scale; 32 attempts take its default
# 2**16 scale down to 2**-16
_LOSS_SCALE_ATTEMPTS = 32


def _quantized_modules(modules):
    # Unlike compressed-tensors' is_module_quantized, skip activation-only schemes,
    # which leave no quantized weights for QAD to train
    return [
        module
        for module in modules
        if getattr_chain(module, "quantization_scheme.weights", None) is not None
    ]


def _snapshot(values):
    # QAD reuses cached inputs and teacher outputs across epochs, so they must not
    # alias live tensors. IntermediatesCache stores tensors that are already on the
    # offload device by reference.
    return tree_map_only(torch.Tensor, lambda t: t.detach().clone(), values)


class QADModifier(Modifier):
    """
    Quantization-aware distillation (QAD) trains each sequential target's weights
    so that its fake-quantized outputs match its unquantized outputs. For each
    target, such as a decoder layer, this modifier:

    1. Caches the target's inputs and unquantized outputs for every calibration
       batch. The unquantized outputs are the teacher; no teacher model is loaded
    2. Splits the cached batches into training and validation batches
    3. After the preceding modifier calibrates the target's weight qparams, trains
       its weights through compressed-tensors fake quantization, re-observing
       weight qparams before training and after every epoch
    4. Restores the weights with the lowest validation loss, which may be the
       untrained weights
    5. Re-observes weight qparams and writes the fake-quantized weights back

    With ``propagate_error=True`` (default, recommended), each target trains on the
    preceding targets' quantized outputs, matching its inputs at inference. With
    ``propagate_error=False``, it trains on the original model's activations.
    Validation batches are withheld from QAD updates but still calibrate the
    quantizer.

    Place QADModifier after a weight quantization modifier in the same recipe and
    use the sequential pipeline with ``sequential_targets_per_subgraph=1``. RTN and
    GPTQ are tested. Other preceding methods must keep weights unquantized during
    calibration, initialize qparams before QAD's ``sequential_epoch_end``, and use
    floating weights compatible with compressed-tensors fake quantization.

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
    :param num_epochs: training epochs per target. Defaults to 1
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

    # sequential target -> name, for targets containing quantized weights
    _module_names: dict[torch.nn.Module, str] = PrivateAttr(default_factory=dict)
    # sequential target -> captured inputs and teacher outputs, one entry per batch
    _block_cache: dict[torch.nn.Module, IntermediatesCache] = PrivateAttr(
        default_factory=dict
    )
    # sequential targets whose distillation has finished
    _distilled: set[torch.nn.Module] = PrivateAttr(default_factory=set)
    # whether to prefetch cached batches, taken from the pipeline state
    _sequential_prefetch: bool = PrivateAttr(default=False)

    def on_initialize(self, state: State, **kwargs) -> bool:
        pipeline = kwargs.get("pipeline")
        if pipeline and standardize_lookup_name(pipeline) != "sequential":
            raise ValueError(
                'QADModifier requires pipeline="sequential" so the preceding '
                "weight quantization modifier and QAD share each block's "
                f"calibration stage; got pipeline={pipeline!r}"
            )
        modules = _quantized_modules(state.model.modules())
        if not modules:
            raise ValueError(
                "Place QADModifier after a weight quantization modifier, such as "
                "QuantizationModifier (RTN) or GPTQModifier, in the same recipe"
            )
        # TODO add distributed support for QAD calibration
        if is_distributed() and torch.distributed.get_world_size() > 1:
            raise ValueError("QAD currently supports single-process calibration")
        targets = infer_sequential_targets(
            state.model, kwargs.get("sequential_targets")
        )
        self._module_names = {
            module: name or type(module).__name__
            for name, module in match_named_modules(state.model, targets)
            if _quantized_modules(module.modules())
        }
        return True

    def on_calibration_start(self, state: State, event: Event, **kwargs):
        for block in self._module_names:
            self.register_hook(
                block,
                partial(self._input_capture_hook, state=state),
                "forward_pre",
                with_kwargs=True,
            )
            self.register_hook(
                block, self._output_capture_hook, "forward", with_kwargs=True
            )

    def _input_capture_hook(self, module, args, kwargs, *, state):
        if module not in self._block_cache:
            self._block_cache[module] = IntermediatesCache(
                offload_device=self.offload_device
            )
        batches = self._block_cache[module]
        # One entry per batch, so the entry count must equal the batch index
        if module in self._distilled or state.current_batch_idx != len(batches):
            raise ValueError(
                "QAD requires each target to run once per calibration batch "
                "in one sequential stage"
            )
        batches.append(_snapshot({"args": args, "kwargs": kwargs}))

    def _output_capture_hook(self, module, args, kwargs, output):
        # Complete the entry the input hook just appended
        batches = self._block_cache[module]
        batches.update(len(batches) - 1, _snapshot({"target": output}))

    def on_sequential_epoch_end(
        self, state: State, event: Event, modules: list[torch.nn.Module], **kwargs
    ):
        blocks = [module for module in modules if module in self._module_names]
        if not blocks:
            return
        if len(blocks) != 1:
            raise ValueError(
                "QAD requires one target module per subgraph; set "
                "sequential_targets_per_subgraph=1"
            )
        block = blocks[0]
        batches = self._block_cache.pop(block, IntermediatesCache())
        train_indices, validation_indices = self._split_batch_indices(len(batches))
        self._sequential_prefetch = state.sequential_prefetch
        if self._sequential_prefetch and torch.accelerator.is_available():
            for index in range(len(batches)):
                batches.pin_memory(index)
        logger.info(
            "QAD cached {} local teacher batches for {}",
            len(batches),
            self._module_names[block],
        )
        entries = batches.batch_intermediates
        with HooksMixin.disable_hooks():
            self._apply_distillation(
                block,
                [entries[i] for i in train_indices],
                [entries[i] for i in validation_indices],
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

    def _apply_distillation(self, block, train, validation):
        name = self._module_names[block]
        modules = _quantized_modules(block.modules())
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
        original_requires_grad = {p: p.requires_grad for p in block.parameters()}
        block.requires_grad_(False)
        for parameter in trainable:
            parameter.requires_grad_(True)
        for module in modules:
            enable_quantization(module)
        try:
            self._reobserve_weights(block, modules, "before_training")
            initial_train = self._evaluate(block, train)
            steps = self._train_with_validation(
                block, modules, optimizer, trainable, masters, train, validation
            )
            self._reobserve_weights(block, modules, "before_materialization")
            self._materialize_quantized_weights(modules)
            final_train = self._evaluate(block, train)
            final_validation = self._evaluate(block, validation)
            self._distilled.add(block)
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
                        "quantization modifier before QAD at the block boundary"
                    )

    @torch.no_grad()
    def _reobserve_weights(self, block, modules, stage):
        if not self.reobserve_weights:
            return
        if any(getattr(module, "weight_observer", None) is None for module in modules):
            raise ValueError("QAD weight re-observation requires live weight observers")
        # Observe the entire group before updating scales: fused Q/K/V and
        # gate/up projections can share a global-scale observer.
        observe(modules, "weight")
        update_qparams(modules, "weight")
        logger.info("QAD {} re-observed weights: {}", self._module_names[block], stage)

    def _train_with_validation(
        self, block, modules, optimizer, trainable, masters, train, validation
    ):
        name = self._module_names[block]
        generator = torch.Generator().manual_seed(self.seed)
        scaler = torch.amp.GradScaler(
            get_execution_device(modules[0]).type,
            enabled=any(p.dtype == torch.float16 for p in trainable),
        )
        best_loss = self._evaluate(block, validation)
        best_weights = self._snapshot_weights(trainable)
        logger.info("QAD {} initial validation MSE {:.6e}", name, best_loss)
        steps = 0
        for epoch in range(1, self.num_epochs + 1):
            order = torch.randperm(len(train), generator=generator).tolist()
            steps += self._train_epoch(
                block, optimizer, trainable, masters, [train[i] for i in order], scaler
            )
            self._reobserve_weights(block, modules, f"epoch_{epoch}")
            loss = self._evaluate(block, validation)
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

    def _train_epoch(self, block, optimizer, trainable, masters, batches, scaler):
        size = self.gradient_accumulation_steps
        groups = [batches[i : i + size] for i in range(0, len(batches), size)]
        for group in groups:
            self._accumulate_gradients(
                block, optimizer, trainable, masters, group, scaler
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
        self, block, optimizer, trainable, masters, group, scaler
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
            for batch in self._iter_batches(group):
                loss = self._batch_loss(block, batch) / len(group)
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
        raise ValueError(f"Nonfinite QAD gradient in {self._module_names[block]}")

    def _snapshot_weights(self, parameters):
        return [p.detach().to(self.offload_device, copy=True) for p in parameters]

    @staticmethod
    @torch.no_grad()
    def _copy_weights(parameters, weights):
        for parameter, weight in zip(parameters, weights):
            parameter.copy_(weight)

    @torch.no_grad()
    def _evaluate(self, block, batches):
        return sum(
            self._batch_loss(block, batch).item()
            for batch in self._iter_batches(batches)
        ) / len(batches)

    def _batch_loss(self, block, batch):
        prediction, target = block(*batch["args"], **batch["kwargs"]), batch["target"]
        # Decoder layers return hidden states, or a tuple starting with them
        if isinstance(prediction, tuple):
            prediction, target = prediction[0], target[0]
        # FP32 keeps the reduction accurate for FP16/BF16 outputs
        loss = torch.nn.functional.mse_loss(prediction.float(), target.float())
        if not torch.isfinite(loss):
            raise ValueError(f"Nonfinite QAD loss in {self._module_names[block]}")
        return loss

    def _iter_batches(self, batches):
        # Wrap the given entries, in the given order, to reuse the cache's
        # onloading and prefetching; the entries are shared, not copied
        cache = IntermediatesCache(batches, self.offload_device)
        return cache.iter_prefetch() if self._sequential_prefetch else cache.iter()

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
            if len(self._distilled) != len(self._module_names):
                raise ValueError(
                    "QAD did not optimize every target; use the sequential pipeline "
                    "with one complete target module per subgraph"
                )
        finally:
            self._clear_cache()

    def on_finalize(self, state: State, **kwargs) -> bool:
        self._clear_cache()
        return True

    def _clear_cache(self):
        self.remove_hooks()
        self._block_cache.clear()
