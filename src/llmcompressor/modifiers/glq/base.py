import contextlib
from typing import Dict, List, Tuple, Union

import torch
from compressed_tensors.utils import (
    align_module_device,
    get_execution_device,
    match_named_modules,
    update_offload_parameter,
)
from loguru import logger
from pydantic import PrivateAttr

from llmcompressor.core import Event, EventType, State
from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.glq.glq_quantize import glq_quantize_weight
from llmcompressor.modifiers.gptq.gptq_quantize import (
    accumulate_hessian,
    make_empty_hessian,
)
from llmcompressor.utils.metric_logging import CompressionLogger

__all__ = ["GLQModifier"]


class GLQModifier(Modifier):
    """
    GLQ: E8 lattice codebook + Randomized Hadamard Transform + LDLQ quantization.

    Implements Hessian-aware vector quantization using the E8 lattice shell codebook
    (65536 entries, 8 dimensions) with RHT incoherence processing and block-LDL
    sequential error propagation.

    Lifecycle (mirrors GPTQModifier):

    - on_initialize: find target modules
    - on_start: register Hessian accumulation hooks
    - calibrate_module: accumulate H += x.T @ x per forward pass
    - on_event(SEQUENTIAL_EPOCH_END): compress all calibrated modules
    - on_end: remove hooks
    - on_finalize: validate all modules compressed

    Sample usage:

    ```python
    from llmcompressor import oneshot
    from llmcompressor.modifiers.glq import GLQModifier

    recipe = [GLQModifier(bits=2, ignore=["lm_head"])]
    oneshot(model=model, dataset=ds, recipe=recipe, ...)
    ```

    :param bits: bits per weight (2, 3, or 4)
    :param sequential_targets: layer names for sequential compression
    :param targets: module class names to quantize (default: Linear)
    :param ignore: module names or class names to skip
    :param dampening_frac: Hessian damping as fraction of diagonal mean
    :param tune_iters: LDLQ refinement iterations (0 = no refinement)
    :param offload_hessians: offload Hessians to CPU to save GPU memory
    """

    bits: int = 2
    sequential_targets: Union[str, List[str], None] = None
    targets: Union[str, List[str]] = "Linear"
    ignore: Union[str, List[str]] = "lm_head"
    dampening_frac: float = 0.01
    tune_iters: int = 0
    offload_hessians: bool = False

    _module_names: Dict[torch.nn.Module, str] = PrivateAttr(default_factory=dict)
    _hessians: Dict[torch.nn.Module, torch.Tensor] = PrivateAttr(default_factory=dict)
    _num_samples: Dict[torch.nn.Module, torch.Tensor] = PrivateAttr(
        default_factory=dict
    )
    _codebook: object = PrivateAttr(default=None)
    _codebook_small: object = PrivateAttr(default=None)

    def _resolve_targets(self) -> list:
        if isinstance(self.targets, str):
            return [self.targets]
        return list(self.targets)

    def _resolve_ignore(self) -> list:
        if isinstance(self.ignore, str):
            return [self.ignore]
        if self.ignore is None:
            return []
        return list(self.ignore)

    def _ensure_codebook(self, device):
        """Create shared codebook(s) on first use."""
        if self._codebook is not None:
            if self._codebook.device != device:
                self._codebook = self._codebook.to(device)
                if self._codebook_small is not None:
                    self._codebook_small = self._codebook_small.to(device)
            return

        from glq.codebook import E8ShellCodebook

        logger.info(f"GLQ: creating E8ShellCodebook on {device}")
        self._codebook = E8ShellCodebook.build(device=device, verbose=True)

        if self.bits == 3:
            self._codebook_small = self._codebook.make_small(256)
            logger.info("GLQ: created small codebook (256 entries) for 3bpw")

    def on_initialize(self, state: State, **kwargs) -> bool:
        targets = self._resolve_targets()
        ignore = self._resolve_ignore()

        self._module_names = {
            m: name for name, m in match_named_modules(state.model, targets, ignore)
        }
        return True

    def on_start(self, state: State, event: Event, **kwargs):
        self.started_ = True

        targets = self._resolve_targets()
        ignore = self._resolve_ignore()
        added_hook = False

        for _, module in match_named_modules(state.model, targets, ignore):
            if isinstance(module, torch.nn.Linear):
                self.register_hook(module, self.calibrate_module, "forward")
                added_hook = True

        if not added_hook:
            raise ValueError(
                "GLQModifier found no Linear modules matching targets. "
                "Check your targets/ignore settings."
            )

    def on_event(self, state: State, event: Event, **kwargs):
        if event.type_ == EventType.CALIBRATION_EPOCH_START:
            if not self.started_:
                self.on_start(state, None)

        if event.type_ == EventType.SEQUENTIAL_EPOCH_END:
            self.compress_modules()

        if event.type_ == EventType.CALIBRATION_EPOCH_END:
            self.compress_modules()
            if not self.ended_:
                self.on_end(state, None)

    def calibrate_module(
        self,
        module: torch.nn.Module,
        args: Tuple[torch.Tensor, ...],
        _output: torch.Tensor,
    ):
        """Accumulate Hessian from forward pass input."""
        inp = args[0]

        if module not in self._num_samples:
            init_device = (
                "cpu" if self.offload_hessians else get_execution_device(module)
            )
            self._hessians[module] = make_empty_hessian(module, device=init_device)
            self._num_samples[module] = torch.zeros(
                tuple(), device=get_execution_device(module)
            )

        with self._maybe_onload_hessian(module):
            self._hessians[module], self._num_samples[module] = accumulate_hessian(
                inp,
                module,
                self._hessians[module],
                self._num_samples[module],
            )

    def compress_modules(self):
        """Quantize all calibrated modules."""
        for module in list(self._num_samples.keys()):
            name = self._module_names.get(module, "<unknown>")
            num_samples = self._num_samples[module]

            logger.info(
                f"GLQ: quantizing {name} ({self.bits}bpw) using {num_samples} samples"
            )

            with (
                torch.no_grad(),
                align_module_device(module),
                self._maybe_onload_hessian(module),
                CompressionLogger(module) as comp_logger,
            ):
                H = self._hessians.pop(module) / self._num_samples.pop(module)
                device = get_execution_device(module)
                self._ensure_codebook(device)

                W = module.weight.data.clone()
                W_hat, proxy_loss = glq_quantize_weight(
                    W=W,
                    H=H,
                    bits=self.bits,
                    codebook=self._codebook,
                    codebook_small=self._codebook_small,
                    dampening_frac=self.dampening_frac,
                    tune_iters=self.tune_iters,
                )
                comp_logger.set_results(loss=proxy_loss)

            update_offload_parameter(module, "weight", W_hat)

    def on_end(self, state: State, event: Event, **kwargs):
        self.ended_ = True
        self.remove_hooks()

    def on_finalize(self, state: State, **kwargs) -> bool:
        if not self.ended_:
            self.on_end(state, None)

        if len(self._num_samples) > 0:
            raise ValueError(
                f"GLQ: failed to compress {len(self._num_samples)} modules"
            )

        self._hessians = dict()
        self._num_samples = dict()
        return True

    @contextlib.contextmanager
    def _maybe_onload_hessian(self, module: torch.nn.Module):
        if self.offload_hessians:
            device = get_execution_device(module)
            self._hessians[module] = self._hessians[module].to(device=device)

        yield

        if self.offload_hessians:
            if module in self._hessians:
                self._hessians[module] = self._hessians[module].to(device="cpu")
