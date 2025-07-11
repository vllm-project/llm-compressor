import ast
import textwrap
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from llmcompressor.pipelines.sequential.ast_utils.auto_wrapper import AutoWrapper


def check_wrapping(
    source: str,
    output: Optional[str] = None,
    num_wrapped: int = 0,
    namespace: Optional[Dict[str, Any]] = None,
    ignore: Optional[List[str]] = None,
):
    namespace = namespace or {}
    ignore = ignore or []

    tree = ast.parse(textwrap.dedent(source))
    wrapper = AutoWrapper(namespace, ignore)
    wrapped = wrapper.auto_wrap(tree)

    if output is not None:
        wrapped_lines = ast.unparse(wrapped).splitlines()
        output_lines = textwrap.dedent(output).splitlines()[1:]
        assert wrapped_lines == output_lines

    assert len(wrapper._wrapper_fn_defs) == num_wrapped


def test_static_if():
    source = """
    def forward():
        if 1 + 1 == 2:
            pass
    """
    output = """
    def forward():
        if True:
            pass
    """
    check_wrapping(source, output, 0)


def test_static_if_global_vars():
    source = """
    def forward():
        if config.is_false:
            pass
    """
    output = """
    def forward():
        if False:
            pass
    """
    config = SimpleNamespace(is_false=False)
    check_wrapping(source, output, 0, namespace={"config": config})


def test_dynamic_if():
    source = """
    def forward():
        test = ...
        if test:
            pass
    """
    check_wrapping(source, None, 1)


def test_ignore_functions():
    def func_one():
        pass

    def func_two():
        pass

    source = """
    def forward():
        func_one()
        func_two()
    """
    namespace = {"func_one": func_one, "func_two": func_two}
    check_wrapping(source, None, 1, namespace=namespace, ignore=["func_one"])


def test_ignore_methods():
    class Model:
        def meth_one(self):
            pass

        def meth_two(self):
            pass

    source = """
    def forward(self):
        self.meth_one()
        self.meth_two()
    """
    namespace = {"self": Model()}
    check_wrapping(source, None, 1, namespace=namespace, ignore=["meth_one"])


def test_branch_with_self_assignment():
    source = """
    def forward(x, y):
        if y > 0:
            x = x + 1
        else:
            x = x - 1
        return x
    """

    tree = ast.parse(textwrap.dedent(source))
    wrapper = AutoWrapper(namespace={}, ignore=[])
    wrapper.auto_wrap(tree)

    assert len(wrapper._wrapper_fn_defs) == 1

    # Check if both x, y are included in args
    wrapped_fn = wrapper._wrapper_fn_defs[0]
    arg_names = {arg.arg for arg in wrapped_fn.args.args}

    assert arg_names == {
        "x",
        "y",
    }, f"Expected arguments {{'x', 'y'}}, but got {arg_names}"
