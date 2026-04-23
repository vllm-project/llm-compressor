from typing import Dict, List, Union

from compressed_tensors.quantization import QuantizationArgs
from compressed_tensors.utils import match_named_modules
from pydantic import Field

from llmcompressor.core import Event, EventType, State
from llmcompressor.modifiers import Modifier
from llmcompressor.observers.base import Observer

__all__ = ["IMatrixGatherer"]


class IMatrixGatherer(Modifier):
    """
    Lifecycle trigger for iMatrix importance collection.

    Triggers a calibration pass so that ``IMatrixMSEObserver`` can collect
    E[x²] via its ``attach()`` hook.  Does **not** quantize weights — the
    actual quantization is done by the subsequent
    ``QuantizationModifier`` / ``GPTQModifier``.

    The observer's ``detach()`` method computes ``_imatrix_importance``
    from the accumulated statistics and leaves it on the module for the
    next quantization pass to consume.

    Example recipe::

        recipe:
          - IMatrixGatherer:
              ignore: ["lm_head"]
          - QuantizationModifier:
              config_groups:
                group_0:
                  targets: ["Linear"]
                  weights:
                    observer: imatrix_mse

    Or composed with GPTQ::

        recipe:
          - IMatrixGatherer:
              ignore: ["lm_head"]
          - GPTQModifier:
              config_groups:
                group_0:
                  targets: ["Linear"]
                  weights:
                    observer: imatrix_mse

    .. note::
        Auto-prepend (inserting the gatherer automatically when
        ``imatrix_mse`` is detected in a recipe) is planned for a
        follow-up PR.

    :param targets: module types to instrument (default: ``["Linear"]``)
    :param ignore: layer name patterns to skip (default: ``["lm_head"]``)
    :param weight_observer: observer to attach during calibration.
        Must be ``"imatrix_mse"`` (default).
    """

    targets: Union[str, List[str]] = Field(default_factory=lambda: ["Linear"])
    ignore: List[str] = Field(default_factory=lambda: ["lm_head"])
    weight_observer: str = "imatrix_mse"

    # ------------------------------------------------------------------ #
    #  Lifecycle
    # ------------------------------------------------------------------ #

    def on_initialize(self, state: State, **kwargs) -> bool:
        """
        Attach iMatrix observers to target modules for E[x²] collection
        """
        self._resolved_targets = (
            self.targets if isinstance(self.targets, list) else [self.targets]
        )
        self._observers: Dict[str, Observer] = {}

        # Minimal QuantizationArgs — only used to instantiate the observer,
        # no quantization config is applied to the model.
        observer_args = QuantizationArgs(observer=self.weight_observer)

        for name, module in match_named_modules(
            state.model, self._resolved_targets, self.ignore
        ):
            observer = Observer.load_from_registry(
                self.weight_observer,
                base_name="weight",
                args=observer_args,
                module=module,
            )
            module.register_module("weight_observer", observer)
            observer.attach(module)
            self._observers[name] = observer

        return True

    def on_start(self, state: State, event: Event, **kwargs):
        self.started_ = True

    def on_event(self, state: State, event: Event, **kwargs):
        if event.type_ == EventType.CALIBRATION_EPOCH_START:
            if not self.started_:
                self.on_start(state, None)

        if event.type_ == EventType.CALIBRATION_EPOCH_END:
            if not self.ended_:
                self.on_end(state, None)

    def on_end(self, state: State, event: Event, **kwargs):
        self.ended_ = True
        for name, observer in self._observers.items():
            module = observer.module() if observer.module is not None else None
            if module is not None:
                observer.detach(module)
                if hasattr(module, "weight_observer"):
                    delattr(module, "weight_observer")
        self._observers.clear()

    def on_finalize(self, state: State, **kwargs) -> bool:
        """
        Clean up importance tensors so they don't end up in the checkpoint
        """
        if not self.ended_:
            self.on_end(state, None)

        # Clean up importance tensors so they don't end up in checkpoint
        for _, module in match_named_modules(
            state.model, self._resolved_targets, self.ignore
        ):
            if hasattr(module, "_imatrix_importance"):
                del module._imatrix_importance

        return True
