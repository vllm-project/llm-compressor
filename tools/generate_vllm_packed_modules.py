#!/usr/bin/env python3
"""
Collect the linear-layer groups that vLLM packs into one weight, for the
FUSED_MODULE_MAPPINGS contract test.

Reads the model definitions of a vLLM git checkout at one commit and writes
every group declared by a literal `packed_modules_mapping` dict (also one
merged into another mapping with `|`), a literal
`stacked_params_mapping` style list of (packed, sub, shard) tuples, or a
literal `orig_to_new_stacked` dict (both of kind "stacked_params_mapping").
Stacked tuples are merged per function or class body, so tuples appended under
a condition join the group they extend. Groups are keyed by packed name and set
of sub names, relative to their common parent, and each one lists the files
that declare it.

It also writes the linear layers that each class of those files assigns to
`self`, e.g. `self.kv_proj = MergedColumnParallelLinear(...)`, for the classes
that construct one of their file's packed layers. They tell which layers sit
next to each other in a module, which the packed groups alone do not. Limits:
layers that a class builds under mutually exclusive conditions are merged
into one set (e.g. Bailing MoE V3 qkv_proj and b_proj, or qkvb_proj), layers
assigned through a tuple, walrus or setattr() target, built by a helper call
or inherited from a base class are not seen, and only classes of the file that
declares the packed groups are scanned.

Usage:
    python tools/generate_vllm_packed_modules.py /path/to/vllm --rev <commit>
"""

import argparse
import ast
import io
import json
import subprocess
import tarfile
from pathlib import Path

MODEL_DIRS = ("vllm/model_executor/models", "vllm/models")
DEFAULT_OUTPUT = (
    Path(__file__).parent.parent
    / "tests/llmcompressor/observers/vllm_packed_modules.json"
)


def _name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _is_packed_name(name: str) -> bool:
    # packed_modules_mapping, STEP5_PACKED_MODULES_MAPPING
    return name.lower().endswith("packed_modules_mapping")


def _is_stacked_name(name: str) -> bool:
    # stacked_params_mapping, mla_params_mapping, _STEP5_WEIGHT_SHARDS, ...
    name = name.lower()
    return "stacked" in name or name.endswith(("mapping", "mappings", "shards"))


def _str(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value.strip(".")
    return None


def _packed_groups(node: ast.AST) -> list[tuple[str, list[str]]]:
    """Groups of a `packed_modules_mapping = {packed: [subs]}` literal, also
    when merged into another mapping with `|`"""
    groups = []
    if isinstance(node, ast.BinOp):
        return _packed_groups(node.left) + _packed_groups(node.right)
    if isinstance(node, ast.Dict):
        for key, value in zip(node.keys, node.values):
            packed = _str(key)
            if packed and isinstance(value, (ast.List, ast.Tuple)):
                subs = [_str(elt) for elt in value.elts]
                if subs and all(subs):
                    groups.append((packed, subs))
    return groups


def _pair(packed: ast.AST, sub: ast.AST) -> list[tuple[str, str]]:
    packed, sub = _str(packed), _str(sub)
    return [(packed, sub)] if packed and sub else []


def _shard_pair(node: ast.AST) -> list[tuple[str, str]]:
    """(packed, sub) of a `(packed, sub, shard)` tuple"""
    if isinstance(node, ast.Tuple) and len(node.elts) == 3:
        return _pair(node.elts[0], node.elts[1])
    return []


def _dict_pair(sub: ast.AST, value: ast.AST) -> list[tuple[str, str]]:
    """(packed, sub) of a `{sub: (packed, shard)}` dict entry"""
    if isinstance(value, ast.Tuple) and len(value.elts) == 2:
        return _pair(value.elts[0], sub)
    return []


def _stacked_pairs(node: ast.AST) -> list[tuple[str, str]]:
    """(packed, sub) pairs of stacked tuples or `{sub: (packed, shard)}` dicts"""
    if pair := _shard_pair(node):
        return pair
    if isinstance(node, (ast.List, ast.Tuple)):
        return [pair for elt in node.elts for pair in _stacked_pairs(elt)]
    if isinstance(node, ast.BinOp):
        return _stacked_pairs(node.left) + _stacked_pairs(node.right)
    if isinstance(node, ast.Dict):
        return [
            pair for entry in zip(node.keys, node.values) for pair in _dict_pair(*entry)
        ]
    return []


def _scopes(node: ast.AST):
    """Yield the nodes of each scope (module, class or function body), in
    source order"""
    owned, nested = [], []
    todo = list(ast.iter_child_nodes(node))[::-1]
    while todo:
        child = todo.pop()
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            nested.append(child)
        else:
            owned.append(child)
            todo.extend(list(ast.iter_child_nodes(child))[::-1])
    yield owned
    for child in nested:
        yield from _scopes(child)


def _targets(node: ast.AST) -> list[ast.AST]:
    if isinstance(node, ast.Assign):
        return node.targets
    if isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        return [node.target]
    return []


def _method_args(node: ast.AST, methods: tuple[str, ...], is_name) -> list[ast.AST]:
    """Arguments of a `name.method(...)` call whose name passes `is_name`"""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in methods
        and is_name(_name(node.func.value))
    ):
        return node.args
    return []


def _packed_values(node: ast.AST) -> list[ast.AST]:
    """The literals one node declares packed_modules_mapping groups with"""
    targets = _targets(node)
    # self.packed_modules_mapping["qkv_proj"] = ["q_proj", ...]
    values = [
        ast.Dict(keys=[target.slice], values=[node.value])
        for target in targets
        if isinstance(target, ast.Subscript) and _is_packed_name(_name(target.value))
    ]
    if any(_is_packed_name(_name(target)) for target in targets):
        values.append(node.value)
    return values + _method_args(node, ("update",), _is_packed_name)


def _stacked_values(node: ast.AST) -> list[ast.AST]:
    """The literals one node declares stacked params pairs with"""
    names = [_name(target) for target in _targets(node)]
    values = []
    if not any(map(_is_packed_name, names)) and any(map(_is_stacked_name, names)):
        values.append(node.value)
    if isinstance(node, ast.keyword) and node.arg == "orig_to_new_stacked":
        values.append(node.value)
    return values + _method_args(node, ("append", "extend", "insert"), _is_stacked_name)


def _merge_pairs(pairs: list[tuple[str, str]]) -> list[tuple[str, list[str]]]:
    """One group per packed name, with its sub names in first-seen order"""
    groups: dict[str, list[str]] = {}
    for packed, sub in pairs:
        subs = groups.setdefault(packed, [])
        if sub not in subs:
            subs.append(sub)
    return list(groups.items())


def collect_file(source: str) -> dict[str, list[tuple[str, list[str]]]]:
    """Packed groups of one model file, by kind of declaration"""
    found = {"packed_modules_mapping": [], "stacked_params_mapping": []}
    for nodes in _scopes(ast.parse(source)):
        values = [value for node in nodes for value in _packed_values(node)]
        found["packed_modules_mapping"] += [
            group for value in values for group in _packed_groups(value)
        ]
        values = [value for node in nodes for value in _stacked_values(node)]
        pairs = [pair for value in values for pair in _stacked_pairs(value)]
        found["stacked_params_mapping"] += _merge_pairs(pairs)
    return found


def _is_linear(node: ast.AST) -> bool:
    """A `...Linear(...)` call, or a conditional expression with one"""
    if isinstance(node, ast.IfExp):
        return _is_linear(node.body) or _is_linear(node.orelse)
    return isinstance(node, ast.Call) and _name(node.func).endswith("Linear")


def _own_nodes(node: ast.AST):
    """Nodes under `node`, without those of nested classes"""
    for child in ast.iter_child_nodes(node):
        if not isinstance(child, ast.ClassDef):
            yield child
            yield from _own_nodes(child)


def collect_modules(source: str) -> list[tuple[str, list[str]]]:
    """Names of the linear layers each class assigns to `self`"""
    modules = []
    for cls in ast.walk(ast.parse(source)):
        if not isinstance(cls, ast.ClassDef):
            continue
        layers = {
            target.attr
            for node in _own_nodes(cls)
            if getattr(node, "value", None) is not None and _is_linear(node.value)
            for target in _targets(node)
            if isinstance(target, ast.Attribute) and _name(target.value) == "self"
        }
        if layers:
            modules.append((cls.name, sorted(layers)))
    return modules


def read_model_files(vllm_path: Path, rev: str) -> tuple[str, dict[str, str]]:
    """Commit SHA and model sources of `rev`, read from git, not the worktree"""
    git = ["git", "-C", str(vllm_path)]
    commit = subprocess.check_output([*git, "rev-parse", f"{rev}^{{commit}}"])
    commit = commit.decode().strip()
    # git archive fails on a path that does not exist at that commit
    dirs = [
        path
        for path in MODEL_DIRS
        if subprocess.run(
            [*git, "cat-file", "-e", f"{commit}:{path}"], stderr=subprocess.DEVNULL
        ).returncode
        == 0
    ]
    if not dirs:
        # with no paths, git archive would archive the whole tree
        raise SystemExit(f"none of {', '.join(MODEL_DIRS)} exists at {commit}")
    archive = subprocess.check_output([*git, "archive", commit, "--", *dirs])
    sources = {}
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name.endswith(".py"):
                sources[member.name] = tar.extractfile(member).read().decode()
    return commit, sources


def _relative(packed: str, names: list[str]) -> tuple[str, list[str]]:
    """Strip the path components that the packed and sub names share, e.g.
    `self_attn.qkv_proj` from `self_attn.q_proj`, or a `.weight` suffix"""
    parts = [name.split(".") for name in (packed, *names)]
    shortest = min(map(len, parts)) - 1
    start = 0
    while start < shortest and len({p[start] for p in parts}) == 1:
        start += 1
    end = 0
    while start + end < shortest and len({p[-1 - end] for p in parts}) == 1:
        end += 1
    packed, *names = (".".join(p[start : len(p) - end]) for p in parts)
    return packed, names


def build_snapshot(commit: str, sources: dict[str, str]) -> dict:
    groups: dict[tuple[str, frozenset[str]], dict] = {}
    modules = []
    for path in sorted(sources):
        packed_names = set()
        for kind, file_groups in collect_file(sources[path]).items():
            for group in file_groups:
                packed, names = _relative(*group)
                packed_names.add(packed)
                group = groups.setdefault(
                    (packed, frozenset(names)),
                    {"packed": packed, "names": names, "kinds": [], "files": []},
                )
                if kind not in group["kinds"]:
                    group["kinds"] = sorted([*group["kinds"], kind])
                if path not in group["files"]:
                    group["files"].append(path)
        modules += [
            {"file": path, "class": name, "layers": layers}
            for name, layers in collect_modules(sources[path])
            if packed_names.intersection(layers)
        ]

    return {
        "vllm_commit": commit,
        "groups": sorted(
            groups.values(), key=lambda group: (group["packed"], group["names"])
        ),
        "modules": modules,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("vllm_path", type=Path, help="vLLM git checkout")
    parser.add_argument("--rev", default="HEAD", help="vLLM commit to read")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    snapshot = build_snapshot(*read_model_files(args.vllm_path, args.rev))
    args.output.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {len(snapshot['groups'])} groups and "
        f"{len(snapshot['modules'])} modules to {args.output}"
    )


if __name__ == "__main__":
    main()
