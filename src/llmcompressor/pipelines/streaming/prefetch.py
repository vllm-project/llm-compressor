"""
Background prefetching of subgraph weights for the streaming pipeline.

While the current subgraph is being calibrated, the next subgraph's weights
are read from the original checkpoint shards on a background thread (see
`stage_modules`), overlapping disk I/O with calibration compute.
"""

import threading

import torch

from llmcompressor.pipelines.streaming.checkpoint import CheckpointMap, stage_modules

__all__ = ["SubgraphPrefetcher"]


class SubgraphPrefetcher:
    """
    Prefetches the next subgraph's weights from the original checkpoint into
    CPU memory in a background thread, overlapping disk reads with the
    current subgraph's calibration.
    """

    def __init__(self, model: torch.nn.Module, ckpt_map: CheckpointMap):
        self.model = model
        self.ckpt_map = ckpt_map
        self._thread: threading.Thread | None = None
        self._staged = None
        self._error: BaseException | None = None

    def start(self, modules: list[torch.nn.Module]):
        def _run():
            try:
                self._staged = stage_modules(self.model, modules, self.ckpt_map)
            except BaseException as e:  # surfaced on join
                self._error = e

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def join(self) -> dict:
        self._thread.join()
        self._thread = None
        if self._error is not None:
            raise self._error
        staged, self._staged = self._staged, None
        return staged
