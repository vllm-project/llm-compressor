"""Persistent Python process for staging disk tensors through shared memory."""

from __future__ import annotations

import mmap
import os
import socket
import subprocess
import sys
from multiprocessing.connection import Connection
from multiprocessing.reduction import recv_handle
from pathlib import Path

import torch

from llmcompressor.pipelines.sequential import _staging_io


class ProcessReader:
    """Persistent CPU-only worker with batched metadata and shared-memory IPC.

    A standalone interpreter avoids re-executing an unguarded user script. The
    child imports only the standard library, receives paths and keys, and returns
    one fresh shared file containing the batch. The parent maps the same pages and
    handles PyTorch wrapping, dtype conversion and optional pinning. Mappings
    remain alive through torch.frombuffer references for pageable results.
    """

    def __init__(self):
        if not sys.platform.startswith("linux"):
            raise RuntimeError("Process staging currently requires Linux")
        parent, child = socket.socketpair()
        try:
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    str(Path(_staging_io.__file__)),
                    str(child.fileno()),
                ],
                pass_fds=(child.fileno(),),
                env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            )
        except BaseException:
            parent.close()
            raise
        finally:
            child.close()
        self.connection = Connection(parent.detach())
        self.closed = False
        try:
            if not self.connection.poll(30):
                raise RuntimeError("Staging worker startup timed out")
            self.ready = self.connection.recv()
            if self.ready[0] != "ready" or self.ready[2]:
                raise RuntimeError(f"Invalid staging worker startup: {self.ready}")
        except BaseException:
            self.close()
            raise

    def read_batch(
        self, requests: list[tuple[str, str, torch.dtype, bool]]
    ) -> list[torch.Tensor | None]:
        """Read a batch in the child and wrap/pin its shared storage in the parent."""
        if self.closed:
            raise RuntimeError("Staging worker is closed")
        try:
            self.connection.send([(r[0], r[1]) for r in requests])
            response = self.connection.recv()
            if response[0] == "error":
                raise RuntimeError(f"Staging worker failed:\n{response[1]}")
            _, infos, capacity = response
            fd = recv_handle(self.connection)
        except (EOFError, OSError) as exc:
            raise RuntimeError("Staging worker exited or lost its connection") from exc
        try:
            mapping = mmap.mmap(fd, capacity)
        finally:
            os.close(fd)
        base = torch.frombuffer(mapping, dtype=torch.uint8)
        results = []
        for request, info in zip(requests, infos):
            if info is None or not hasattr(torch, info["dtype"]):
                results.append(None)
                continue
            _, _, dtype, pinned = request
            source = base.narrow(0, info["shared_offset"], info["size"])
            source = source.view(getattr(torch, info["dtype"])).view(info["shape"])
            if pinned or source.dtype != dtype:
                output = torch.empty(
                    info["shape"], dtype=dtype, device="cpu", pin_memory=pinned
                )
                output.copy_(source)
                source = output
            results.append(source)
        return results

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            if self.process.poll() is None:
                try:
                    self.connection.send(None)
                except (BrokenPipeError, OSError):
                    pass
        finally:
            self.connection.close()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
