"""Read safetensors payloads directly into PyTorch CPU allocations."""

from __future__ import annotations

import json
import math
import os
import stat
import sys
from collections import defaultdict
from typing import BinaryIO, TypedDict

import torch


class _TensorInfo(TypedDict):
    shape: list[int]
    dtype: str
    offset: int


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
    """Validate tensor ranges before any payload is copied into tensor storage."""
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
) -> tuple[list[_TensorInfo | None], list[_FileReads]]:
    """Resolve shared shards once and describe their tensor payloads."""
    grouped = defaultdict(list)
    resolved = {}
    for index, (path, key) in enumerate(requests):
        if path not in resolved:
            resolved[path] = os.path.realpath(path)
        grouped[resolved[path]].append((index, key))
    infos: list[_TensorInfo | None] = [None] * len(requests)
    files: list[_FileReads] = []
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
            begin, _ = info["data_offsets"]
            infos[index] = {
                "shape": info["shape"],
                "dtype": DTYPES[info["dtype"]][0],
                "offset": data_start + begin,
            }
            indices.append(index)
        files.append((path, signature, indices))
    return infos, files


def open_source(path: str, signature: _FileSignature) -> BinaryIO:
    file = open(path, "rb", buffering=0)
    try:
        if file_signature(file.fileno()) != signature:
            raise ValueError(f"Safetensors file changed during staging: {path}")
    except BaseException:
        file.close()
        raise
    return file


def read_tensors(
    requests: list[tuple[str, str, torch.dtype, bool]],
) -> list[torch.Tensor | None]:
    """Read (filename, key, dtype, pinned) requests into fresh CPU tensors.

    A NumPy byte view exposes each tensor's storage without copying its payload.
    Return None for unsupported source dtypes so the caller can use its upstream
    reader. Malformed files and I/O failures propagate without partial results.
    """
    infos, files = plan_reads([(r[0], r[1]) for r in requests])
    results: list[torch.Tensor | None] = [None] * len(requests)
    for path, signature, indices in files:
        with open_source(path, signature) as file:
            for index in indices:
                info = infos[index]
                assert info is not None  # Only supported header entries are scheduled.
                source_dtype = getattr(torch, info["dtype"], None)
                if source_dtype is None:
                    continue
                _, _, dtype, pinned = requests[index]
                source = torch.empty(
                    info["shape"],
                    dtype=source_dtype,
                    device="cpu",
                    pin_memory=pinned and dtype == source_dtype,
                )
                # Viewing as uint8 also supports BF16 and float8, which cannot
                # be represented directly by NumPy. NumPy shares this storage.
                raw = source.reshape(-1).view(torch.uint8).numpy()
                read_into(file.fileno(), raw, info["offset"])
                if dtype != source_dtype:
                    output = torch.empty(
                        info["shape"], dtype=dtype, device="cpu", pin_memory=pinned
                    )
                    output.copy_(source)
                    source = output
                results[index] = source
    return results
