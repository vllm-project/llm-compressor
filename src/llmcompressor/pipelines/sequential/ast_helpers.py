import ast
import contextlib
import inspect
import linecache
import sys
import textwrap
from typing import List

import torch

from llmcompressor.pipelines.sequential.ast_utils.AutoWrapper import AutoWrapper
from llmcompressor.utils import patch_attr


@contextlib.contextmanager
def autowrap_forwards(modules: List[torch.nn.Module], ignore: List[str]):
    with contextlib.ExitStack() as stack:
        for module in modules:
            stack.enter_context(autowrap_forward(module, ignore))
        yield


@contextlib.contextmanager
def autowrap_forward(module: torch.nn.Module, ignore: List[str]):
    # get source code of module forward
    source = inspect.getsource(module.forward)
    source = textwrap.dedent(source)
    tree = ast.parse(source)

    # construct namespace
    defining_module = sys.modules[module.__class__.__module__]
    namespace = defining_module.__dict__  # original module context
    namespace.update({"self": module})
    namespace.update({"torch.fx.wrap": torch.fx.wrap})

    # autowrap untraceable code
    auto_wrapper = AutoWrapper(namespace, ignore)
    tree = auto_wrapper.auto_wrap(tree)

    # compile new forward function from autowrapped code
    filename = f"{module.__class__.__name__}_{hash(module)}_autowrapped"
    code = compile(tree, filename=filename, mode="exec")
    exec(code, namespace)

    # enable better tracebacks if autowrapped code fails
    source_str = ast.unparse(tree)
    linecache.cache[filename] = (
        len(source_str),
        None,
        [line + "\n" for line in source_str.splitlines()],
        filename,
    )

    # some modules (such as ModuleList) do not implement a forward function,
    # so fall back to existing forward definition
    new_forward = namespace.get("forward", module.forward.__func__)
    new_forward = new_forward.__get__(module)  # curry self
    with patch_attr(module, "forward", new_forward):
        yield
