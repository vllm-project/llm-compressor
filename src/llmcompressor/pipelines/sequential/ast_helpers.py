import ast
import contextlib
import inspect
import linecache
import sys
import textwrap
from typing import List

import torch

from llmcompressor.pipelines.sequential.ast_utils.auto_wrapper import AutoWrapper
from llmcompressor.utils import patch_attr

__all__ = ["autowrap_forwards"]


@contextlib.contextmanager
def autowrap_forwards(modules: List[torch.nn.Module], ignore: List[str]):
    """
    Replace the `forward` method of the given modules with a recompiled version where
    all untraceble code patterns are removed and replaced with torch.fx function
    wrappers

    :param modules: list of modules whose forward methods should be replaced
    :param ignore: explicit list of function names to wrap
    """
    with contextlib.ExitStack() as stack:
        for module in modules:
            if not isinstance(module, (torch.nn.ModuleList, torch.nn.ModuleDict)):
                stack.enter_context(autowrap_forward(module, ignore))
        yield


@contextlib.contextmanager
def autowrap_forward(module: torch.nn.Module, ignore: List[str]):
    """
    Replace the `forward` method of the given module with a recompiled version where
    all untraceble code patterns are removed and replaced with torch.fx function
    wrappers.

    For a list of untraceable code patterns and their explainations, see
    https://github.com/vllm-project/llm-compressor/pull/1411

    :param module: module whose forward method should be replaced
    :param ignore: explicit list of function names to wrap
    """
    # get source code of module forward
    source = inspect.getsource(module.forward)
    source = textwrap.dedent(source)
    tree = ast.parse(source)

    # construct namespace for our new code
    defining_module = sys.modules[module.__class__.__module__]
    namespace = defining_module.__dict__.copy()
    namespace.update({"torch.fx.wrap": torch.fx.wrap})
    namespace.update({"self": module})

    # autowrap untraceable code
    auto_wrapper = AutoWrapper(namespace, ignore)
    tree = auto_wrapper.auto_wrap(tree)

    # compile new forward function from autowrapped code
    filename = f"{module.__class__.__name__}_{hash(module)}_autowrapped"
    code = compile(tree, filename=filename, mode="exec")
    exec(code, namespace)  # ensure ns of functions is the same ns as torch.fx.wrap

    # enable better tracebacks if autowrapped code fails
    source_str = ast.unparse(tree)
    linecache.cache[filename] = (
        len(source_str),
        None,
        [line + "\n" for line in source_str.splitlines()],
        filename,
    )

    # patch forward with autowrapped forward
    new_forward = namespace["forward"].__get__(module)
    with patch_attr(module, "forward", new_forward):
        yield
