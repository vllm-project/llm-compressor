"""Standard-library disk reader and standalone staging process entry point.

Keep this module independent of torch and llmcompressor imports. The process
backend executes this file directly, without importing the user's main script.
"""

from __future__ import annotations

import json
import math
import mmap
import os
import stat
import sys
import tempfile
import traceback
from collections import defaultdict
from contextlib import closing
from multiprocessing.connection import Connection
from multiprocessing.reduction import send_handle
from typing import BinaryIO, TypedDict


class _TensorInfo(TypedDict):
    shape: list[int]
    dtype: str
    offset: int
    size: int
    shared_offset: int


_FileSignature = tuple[int, int, int, int]
_FileReads = tuple[str, _FileSignature, list[int]]

# safetensors dtype -> (torch dtype attribute, bits per element)
DTYPES = {
    "BOOL": ("bool", 8),
    "U8": ("uint8", 8),
    "I8": ("int8", 8),
    "I16": ("int16", 16),
    "I32": ("int32", 32),
    "I64": ("int64", 64),
    "U16": ("uint16", 16),
    "U32": ("uint32", 32),
    "U64": ("uint64", 64),
    "F16": ("float16", 16),
    "BF16": ("bfloat16", 16),
    "F32": ("float32", 32),
    "F64": ("float64", 64),
    "C64": ("complex64", 64),
    "F8_E4M3": ("float8_e4m3fn", 8),
    "F8_E5M2": ("float8_e5m2", 8),
    "F8_E4M3FNUZ": ("float8_e4m3fnuz", 8),
    "F8_E5M2FNUZ": ("float8_e5m2fnuz", 8),
}


def read_into(fd: int, buffer, offset: int) -> None:
    """Read exactly into a contiguous byte buffer, including short reads."""
    with memoryview(buffer).cast("B") as view:
        done = 0
        while done < len(view):
            try:
                count = os.preadv(fd, [view[done : done + (1 << 30)]], offset + done)
            except InterruptedError:
                continue
            if not count:
                raise ValueError("Truncated safetensors payload")
            done += count


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate safetensors header key: {key}")
        result[key] = value
    return result


def _check_nesting(header: bytes | bytearray) -> None:
    depth, quoted, escaped = 0, False, False
    for char in header:
        if quoted:
            if escaped:
                escaped = False
            elif char == 92:
                escaped = True
            elif char == 34:
                quoted = False
        elif char == 34:
            quoted = True
        elif char in (91, 123):
            depth += 1
            if depth > 16:
                raise ValueError("Safetensors header nesting too deep")
        elif char in (93, 125):
            depth -= 1


def _integer(value: object) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= sys.maxsize
    )


def file_signature(fd: int) -> _FileSignature:
    st = os.fstat(fd)
    if not stat.S_ISREG(st.st_mode):
        raise ValueError("Expected a regular safetensors file")
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns


def read_header(fd: int) -> tuple[dict, int, _FileSignature]:
    signature = file_signature(fd)
    prefix = bytearray(8)
    read_into(fd, prefix, 0)
    size = int.from_bytes(prefix, "little")
    if not 0 < size <= min(100_000_000, signature[2] - 8):
        raise ValueError("Invalid safetensors header length")
    encoded = bytearray(size)
    read_into(fd, encoded, 8)
    if encoded[0] != 123:
        raise ValueError("Invalid safetensors header")
    _check_nesting(encoded)
    header = json.loads(encoded, object_pairs_hook=_unique_object)
    if not isinstance(header, dict):
        raise ValueError("Invalid safetensors JSON header")
    data_start = size + 8
    ranges = []
    for name, info in header.items():
        if name == "__metadata__":
            if not isinstance(info, dict) or not all(
                isinstance(v, str) for v in info.values()
            ):
                raise ValueError("Invalid safetensors metadata")
            continue
        if not isinstance(info, dict):
            raise ValueError("Invalid tensor metadata")
        shape, offsets, dtype = (
            info.get("shape"),
            info.get("data_offsets"),
            info.get("dtype"),
        )
        if (
            not isinstance(shape, list)
            or not all(_integer(dim) for dim in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(_integer(offset) for offset in offsets)
            or not isinstance(dtype, str)
        ):
            raise ValueError("Invalid tensor shape, dtype or offsets")
        begin, end = offsets
        numel = math.prod(shape)
        if numel > sys.maxsize or not 0 <= begin <= end <= signature[2] - data_start:
            raise ValueError("Invalid tensor shape or payload bounds")
        if dtype in DTYPES and numel * DTYPES[dtype][1] != (end - begin) * 8:
            raise ValueError("Tensor shape/dtype does not match payload size")
        ranges.append((begin, end))
    cursor = 0
    for begin, end in sorted(ranges):
        if begin != cursor:
            raise ValueError("Overlapping or incomplete tensor data")
        cursor = end
    if cursor != signature[2] - data_start:
        raise ValueError("Safetensors payload has trailing data")
    return header, data_start, signature


def plan_reads(
    requests: list[tuple[str, str]],
) -> tuple[list[_TensorInfo | None], list[_FileReads], int]:
    """Resolve shared shards once and describe source and shared-memory offsets."""
    grouped = defaultdict(list)
    resolved = {}
    for index, (path, key) in enumerate(requests):
        if path not in resolved:
            resolved[path] = os.path.realpath(path)
        grouped[resolved[path]].append((index, key))
    infos: list[_TensorInfo | None] = [None] * len(requests)
    files: list[_FileReads] = []
    capacity = 0
    for path, entries in grouped.items():
        with open(path, "rb", buffering=0) as file:
            header, data_start, signature = read_header(file.fileno())
        indices = []
        for index, key in entries:
            if key == "__metadata__" or key not in header:
                raise ValueError(f"Tensor key not found: {key} in {path}")
            info = header[key]
            if info["dtype"] not in DTYPES:
                continue  # Upstream reader handles future dtypes.
            begin, end = info["data_offsets"]
            size = end - begin
            infos[index] = {
                "shape": info["shape"],
                "dtype": DTYPES[info["dtype"]][0],
                "offset": data_start + begin,
                "size": size,
                "shared_offset": capacity,
            }
            capacity += (size + 63) // 64 * 64
            indices.append(index)
        files.append((path, signature, indices))
    return infos, files, capacity


def open_source(path: str, signature: _FileSignature) -> BinaryIO:
    file = open(path, "rb", buffering=0)
    try:
        if file_signature(file.fileno()) != signature:
            raise ValueError(f"Safetensors file changed during staging: {path}")
    except BaseException:
        file.close()
        raise
    return file


def serve(fd: int) -> None:
    """Serve batches using a fresh shared file per batch and one descriptor transfer."""
    with Connection(fd) as connection:
        connection.send(("ready", os.getpid(), "torch" in sys.modules))
        while True:
            try:
                requests = connection.recv()
            except EOFError:
                return
            if requests is None:
                return
            try:
                infos, files, capacity = plan_reads(requests)
                # Empty batches still need a valid mapping for descriptor passing.
                capacity = max(1, capacity)
                directory = os.environ.get("LLMCOMPRESSOR_STAGING_SHM_DIR", "/dev/shm")
                with tempfile.TemporaryFile(dir=directory) as shared_file:
                    shared_fd = shared_file.fileno()
                    # Reserve capacity up front so a full tmpfs raises ENOSPC
                    # instead of SIGBUS while writing the mapping.
                    os.posix_fallocate(shared_fd, 0, capacity)
                    with closing(mmap.mmap(shared_fd, capacity)) as mapping:
                        for path, signature, indices in files:
                            with open_source(path, signature) as file:
                                for index in indices:
                                    info = infos[index]
                                    assert info is not None
                                    start = info["shared_offset"]
                                    with memoryview(mapping)[
                                        start : start + info["size"]
                                    ] as view:
                                        read_into(file.fileno(), view, info["offset"])
                        connection.send(("ok", infos, capacity))
                        send_handle(connection, shared_fd, os.getppid())
            except Exception:
                connection.send(("error", traceback.format_exc()))


if __name__ == "__main__":
    serve(int(sys.argv[1]))
