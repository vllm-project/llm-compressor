"""Write sequentially compressed model tensors directly to checkpoint shards."""

from __future__ import annotations

import json
import os
import queue
import re
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import torch
from compressed_tensors.distributed import is_source_process
from loguru import logger
from safetensors.torch import load_file, save_file


def _size_in_bytes(size: int | str) -> int:
    if isinstance(size, int):
        return size
    match = re.fullmatch(r"\s*(\d+)\s*(B|KB|MB|GB|TB)?\s*", size, re.IGNORECASE)
    if match is None:
        raise ValueError(f"Invalid max shard size: {size!r}")
    amount = int(match.group(1))
    unit = (match.group(2) or "B").upper()
    return amount * (1024 ** ("B", "KB", "MB", "GB", "TB").index(unit))


class _WriteJob:
    def __init__(self, path: Path, tensors: dict[str, torch.Tensor]):
        self.path = path
        self.tensors = tensors
        self.done = threading.Event()
        self.error: BaseException | None = None


class StreamingCheckpointWriter:
    """Bounded asynchronous safetensors writer used as a disk offload backing store.

    Tensors are copied to CPU when submitted, then written in bounded-size shards by
    one worker thread. The shard paths and keys are returned immediately so the disk
    offload cache can read the same files once a write completes.
    """

    INDEX_NAME = "model.safetensors.index.json"
    SHARD_PREFIX = "model-streaming-"

    def __init__(
        self,
        output_dir: str | os.PathLike,
        max_shard_size: int | str = "20GB",
    ):
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.max_shard_size = _size_in_bytes(max_shard_size)
        if self.max_shard_size <= 0:
            raise ValueError("max_shard_size must be positive")

        self.is_source = is_source_process()
        self.run_id = uuid.uuid4().hex[:12]
        self._lock = threading.RLock()
        self._jobs: dict[str, _WriteJob] = {}
        self._weight_map: dict[str, str] = {}
        self._tensor_sizes: dict[str, tuple[int, int]] = {}
        self._current_path: Path | None = None
        self._current_tensors: dict[str, torch.Tensor] = {}
        self._current_size = 0
        self._shard_number = 0
        self._in_batch = False
        self._closed = False
        self._published = False
        self._queue: queue.Queue[_WriteJob | None] | None = None
        self._worker: threading.Thread | None = None

        if self.is_source:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._queue = queue.Queue(maxsize=1)
            self._worker = threading.Thread(
                target=self._write_worker,
                name="llmcompressor-checkpoint-writer",
                daemon=True,
            )
            self._worker.start()

    @contextmanager
    def batch(self) -> Iterator[None]:
        """Group cache submissions from one subgraph under the writer."""
        if not self.is_source:
            yield
            return
        with self._lock:
            if self._closed:
                raise RuntimeError("Checkpoint writer is already closed")
            if self._in_batch:
                raise RuntimeError("Checkpoint writer batches cannot be nested")
            self._in_batch = True
        try:
            yield
        finally:
            with self._lock:
                self._in_batch = False

    def add_tensor(self, name: str, tensor: torch.Tensor) -> tuple[str, str]:
        """Snapshot a tensor and return its assigned shard path and safetensors key."""
        if not self.is_source:
            raise RuntimeError("Only the source rank can submit checkpoint tensors")
        if tensor.device.type == "meta":
            raise ValueError(f"Cannot stream meta tensor {name!r}")

        with self._lock:
            if self._closed:
                raise RuntimeError("Checkpoint writer is already closed")
            if not self._in_batch:
                raise RuntimeError("Checkpoint tensors must be submitted in a batch")
            if name in self._weight_map:
                raise ValueError(f"Duplicate streamed model tensor: {name}")

            tensor_bytes = tensor.numel() * tensor.element_size()
            if (
                self._current_tensors
                and self._current_size + tensor_bytes > self.max_shard_size
            ):
                self._flush_locked()
            if self._current_path is None:
                self._shard_number += 1
                self._current_path = self.output_dir / (
                    f"{self.SHARD_PREFIX}{self.run_id}-"
                    f"{self._shard_number:05d}.safetensors"
                )

            # This copy releases the accelerator tensor once CT replaces the module
            # parameter with its meta offload handle. The worker owns only CPU copies.
            snapshot = tensor.detach().contiguous().to(device="cpu", copy=True)
            self._current_tensors[name] = snapshot
            self._current_size += tensor_bytes
            self._weight_map[name] = self._current_path.name
            self._tensor_sizes[name] = (tensor.numel(), tensor_bytes)
            assigned_path = self._current_path

            if self._current_size >= self.max_shard_size:
                self._flush_locked()

            return str(assigned_path), name

    def _flush_locked(self) -> None:
        if not self._current_tensors:
            return
        assert self._queue is not None and self._current_path is not None
        job = _WriteJob(self._current_path, self._current_tensors)
        self._jobs[str(job.path)] = job
        self._queue.put(job)
        self._current_path = None
        self._current_tensors = {}
        self._current_size = 0

    def _write_worker(self) -> None:
        assert self._queue is not None
        while True:
            job = self._queue.get()
            try:
                if job is None:
                    return
                temp_path = job.path.with_name(
                    f".{job.path.name}.{uuid.uuid4().hex}.tmp"
                )
                try:
                    save_file(job.tensors, str(temp_path), metadata={"format": "pt"})
                    os.replace(temp_path, job.path)
                finally:
                    if temp_path.exists():
                        temp_path.unlink()
            except BaseException as error:
                if job is not None:
                    job.error = error
                    marker = job.path.with_suffix(job.path.suffix + ".error")
                    try:
                        marker.write_text(repr(error), encoding="utf-8")
                    except OSError:
                        pass
            finally:
                if job is not None:
                    job.tensors.clear()
                    job.done.set()
                self._queue.task_done()

    def wait(self, path: str | os.PathLike) -> None:
        """Wait until a shard is visible, propagating worker errors."""
        shard_path = Path(path)
        if not self.is_source:
            self._wait_for_shared_file(shard_path)
            return

        with self._lock:
            if self._current_path == shard_path:
                self._flush_locked()
            job = self._jobs.get(str(shard_path))
        if job is not None:
            job.done.wait()
            if job.error is not None:
                raise RuntimeError(
                    f"Failed to write checkpoint shard {shard_path}"
                ) from job.error
        elif not shard_path.exists():
            raise FileNotFoundError(f"No checkpoint shard was assigned at {shard_path}")

    @staticmethod
    def _wait_for_shared_file(path: Path, timeout: float = 600.0) -> None:
        deadline = time.monotonic() + timeout
        error_path = path.with_suffix(path.suffix + ".error")
        while not path.exists():
            if error_path.exists():
                raise RuntimeError(
                    f"Source rank failed to write checkpoint shard {path}: "
                    f"{error_path.read_text(encoding='utf-8')}"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for checkpoint shard {path}")
            time.sleep(0.05)

    def update_tensor(self, path: str | os.PathLike, name: str, tensor: torch.Tensor):
        """Replace a tensor in an already-written shard after lifecycle finalization."""
        if not self.is_source:
            return
        shard_path = Path(path)
        self.wait(shard_path)
        with self._lock:
            tensors = load_file(str(shard_path), device="cpu")
            if name not in tensors:
                raise KeyError(f"Tensor {name!r} is missing from {shard_path}")
            updated = tensor.detach().contiguous().to(device="cpu", copy=True)
            if tensors[name].shape != updated.shape:
                raise ValueError(
                    f"Streaming checkpoint update changed the shape of {name}: "
                    f"{tuple(tensors[name].shape)} to {tuple(updated.shape)}"
                )
            old = tensors[name]
            tensors[name] = updated
            temp_path = shard_path.with_name(
                f".{shard_path.name}.{uuid.uuid4().hex}.tmp"
            )
            try:
                save_file(tensors, str(temp_path), metadata={"format": "pt"})
                os.replace(temp_path, shard_path)
            finally:
                if temp_path.exists():
                    temp_path.unlink()
            self._tensor_sizes[name] = (
                updated.numel(),
                updated.numel() * updated.element_size(),
            )
            del old, tensors

    def wait_all(self) -> None:
        """Flush pending tensors and propagate all shard-write failures."""
        if not self.is_source:
            return
        with self._lock:
            self._flush_locked()
            jobs = list(self._jobs.values())
        errors = []
        for job in jobs:
            job.done.wait()
            if job.error is not None:
                errors.append((job.path, job.error))
        if errors:
            path, error = errors[0]
            raise RuntimeError(f"Failed to write checkpoint shard {path}") from error

    def validate_coverage(self, expected_tensor_names: set[str]) -> None:
        if not self.is_source:
            return
        actual = set(self._weight_map)
        missing = expected_tensor_names - actual
        unexpected = actual - expected_tensor_names
        if missing or unexpected:
            details = []
            if missing:
                details.append(
                    f"missing {len(missing)} tensors, e.g. {sorted(missing)[:5]}"
                )
            if unexpected:
                details.append(
                    f"found {len(unexpected)} unexpected tensors, e.g. "
                    f"{sorted(unexpected)[:5]}"
                )
            raise ValueError(
                "Streaming checkpoint tensors do not cover the model state: "
                + "; ".join(details)
                + ". Ensure every saved tensor belongs to a traced subgraph."
            )

    def publish(self) -> None:
        """Write the safetensors index last, then remove shards from older runs."""
        if not self.is_source:
            return
        self.wait_all()
        if not self._weight_map:
            raise ValueError("Streaming checkpoint contains no tensors")

        total_size = sum(size for _, size in self._tensor_sizes.values())
        total_parameters = sum(count for count, _ in self._tensor_sizes.values())
        index = {
            "metadata": {
                "total_parameters": total_parameters,
                "total_size": total_size,
            },
            "weight_map": self._weight_map,
        }
        index_path = self.output_dir / self.INDEX_NAME
        temp_path = index_path.with_name(f".{index_path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temp_path.write_text(
                json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            os.replace(temp_path, index_path)
            self._published = True
        finally:
            if temp_path.exists():
                temp_path.unlink()

        self._close_worker()
        keep = set(self._weight_map.values())
        for old_shard in self.output_dir.glob(f"{self.SHARD_PREFIX}*.safetensors"):
            if old_shard.name not in keep:
                old_shard.unlink(missing_ok=True)
        for old_shard in self.output_dir.glob("model-?????-of-?????.safetensors"):
            old_shard.unlink(missing_ok=True)
        (self.output_dir / "model_mtp.safetensors").unlink(missing_ok=True)
        (self.output_dir / "model_backbone.safetensors").unlink(missing_ok=True)
        # A single-file checkpoint takes precedence over an index in Transformers.
        (self.output_dir / "model.safetensors").unlink(missing_ok=True)
        (self.output_dir / "model.safetensors.index.json.error").unlink(missing_ok=True)
        logger.info(
            "Streamed {} tensors into {} checkpoint shards",
            len(self._weight_map),
            len(set(self._weight_map.values())),
        )

    def abort(self) -> None:
        """Stop the worker and remove shards produced by this incomplete run."""
        if not self.is_source or self._published:
            return
        try:
            self.wait_all()
        except Exception:
            pass
        self._close_worker()
        pattern = f"{self.SHARD_PREFIX}{self.run_id}-*.safetensors*"
        for path in self.output_dir.glob(pattern):
            path.unlink(missing_ok=True)

    def _close_worker(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._queue is not None and self._worker is not None:
            self._queue.put(None)
            self._worker.join()
