import ast
import contextlib
import inspect
import linecache
import sys
import textwrap
import traceback

import torch
from compressed_tensors.utils import patch_attr

from llmcompressor.pipelines.sequential.ast_utils import AutoWrapper

__all__ = ["autowrap_forwards", "append_autowrap_source_on_fail"]


@contextlib.contextmanager
def autowrap_forwards(modules: list[torch.nn.Module], ignore: list[str]):
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
def autowrap_forward(module: torch.nn.Module, ignore: list[str]):
    """
    Replace the `forward` method of the given module with a recompiled version where
    all untraceble code patterns are removed and replaced with torch.fx function
    wrappers.

    For a list of untraceable code patterns and their explainations, see
    https://github.com/vllm-project/llm-compressor/pull/1411

    :param module: module whose forward method should be replaced
    :param ignore: explicit list of function names to wrap
    """
    # check forward method is implemented
    if module.forward.__name__ == "_forward_unimplemented":
        raise ValueError(
            "Cannot calibrate model which does not implement `forward` method. Please "
            "either implement a forward method on the model, or pass a submodule to "
            "`oneshot`. For example, `oneshot(model.thinker, ...)`"
        )

    # get source code of module forward
    target = inspect.unwrap(module.forward)
    # Some decorators (e.g. transformers' `force_accelerate_hooks`, see
    # transformers/integrations/accelerate.py) wrap `forward` with an inner
    # function but do not apply `functools.wraps`. inspect.unwrap therefore
    # cannot recover the original (there is no `__wrapped__` attribute) and
    # inspect.getsource returns the *wrapper's* source, so the re-exec'd code
    # defines a function named `wrapped` rather than `forward` and the
    # `namespace["forward"]` lookup below raises KeyError. Recover the original
    # `forward` by searching the wrapper's closure cells. inspect.getsource on
    # the original includes its decorator line, so exec re-applies the
    # decorator and preserves its behaviour (e.g. accelerate hook setup).
    if getattr(target, "__name__", "") != "forward" and getattr(
        target, "__closure__", None
    ):
        from types import FunctionType

        for cell in target.__closure__:
            try:
                contents = cell.cell_contents
            except ValueError:
                continue
            if isinstance(contents, FunctionType) and contents.__name__ == "forward":
                target = contents
                break
    source = inspect.getsource(target)
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
    source = ast.unparse(tree)

    # compile new forward function from autowrapped code
    filename = f"<Autowrapped {module.__class__.__name__} {id(module)}>"
    code = compile(source, filename=filename, mode="exec")
    with append_autowrap_source_on_fail():
        exec(code, namespace)  # ensure ns of functions is the same ns as torch.fx.wrap

    # enable better tracebacks if autowrapped code fails
    linecache.cache[filename] = (
        len(source),
        None,
        [line + "\n" for line in source.splitlines()],
        filename,
    )

    # patch forward with autowrapped forward
    new_forward = namespace["forward"].__get__(module)
    with patch_attr(module, "forward", new_forward):
        yield


@contextlib.contextmanager
def append_autowrap_source_on_fail():
    try:
        yield
    except Exception as exception:
        _exc_type, _exc_value, exc_tb = sys.exc_info()
        tb_list = traceback.extract_tb(exc_tb)

        for frame in reversed(tb_list):
            if "Autowrapped" in frame.filename:
                source_lines = linecache.getlines(frame.filename)
                lineno = frame.lineno

                # annotate failing line
                source_lines = [
                    ("> " if i + 1 == lineno else "  ") + line
                    for i, line in enumerate(source_lines)
                ]

                message = f"--- {frame.filename}:{lineno} ---\n"
                message += "".join(source_lines)
                message += f"\n\n{exception}"
                raise RuntimeError(message) from exception

        raise exception
