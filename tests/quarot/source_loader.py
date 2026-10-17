# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Execute selected original source definitions without package import side effects.

Definitions are compiled unchanged, retaining their source filename/line numbers.
This is a limited CPU oracle adapter, not a ModelSlim processor emulation. No
algorithm code is copied, rewritten, or mocked. Changed source is hashed in reports.
"""

import ast
import hashlib
import sys
import types
from pathlib import Path


def source_definitions(path, names, namespace, *, class_name=None):
    path = Path(path)
    contents = path.read_bytes()
    tree = ast.parse(contents, filename=str(path))
    nodes = tree.body
    if class_name is not None:
        nodes = next(
            node
            for node in nodes
            if isinstance(node, ast.ClassDef) and node.name == class_name
        ).body
    selected, found = [], set()
    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            keys = {node.name}
        elif isinstance(node, ast.Assign):
            keys = {
                target.id for target in node.targets if isinstance(target, ast.Name)
            }
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            keys = {node.target.id}
        else:
            continue
        if keys & set(names):
            selected.append(node)
            found.update(keys & set(names))
    if found != set(names):
        raise RuntimeError(
            f"Oracle source changed: missing {set(names) - found} in {path}"
        )
    module_name = (
        "_quarot_source_"
        + hashlib.sha256((str(path) + str(class_name)).encode()).hexdigest()[:16]
    )
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    module.__dict__.update(namespace)
    sys.modules[module_name] = module
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"),
        module.__dict__,
    )
    return module, hashlib.sha256(contents).hexdigest()
