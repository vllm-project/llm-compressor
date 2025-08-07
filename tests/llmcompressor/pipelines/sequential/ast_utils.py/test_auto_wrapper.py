import ast
import textwrap
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from llmcompressor.pipelines.sequential.ast_utils.auto_wrapper import AutoWrapper


def check_wrapping(
    source: str,
    output: str,
    namespace: Optional[Dict[str, Any]] = None,
    ignore: Optional[List[str]] = None,
):
    namespace = namespace or {}
    ignore = ignore or []

    tree = ast.parse(textwrap.dedent(source))
    wrapper = AutoWrapper(namespace, ignore)
    wrapped = wrapper.auto_wrap(tree)

    wrapped_lines = ast.unparse(wrapped).splitlines()
    output_lines = textwrap.dedent(output).splitlines()[1:]

    assert len(wrapped_lines) == len(output_lines)
    for wrapped_line, output_line in zip(wrapped_lines, output_lines):
        if "# skip" in output:
            continue

        assert wrapped_line == output_line


def test_static_if():
    """Checks that resolvable if statements are replaced"""

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
    check_wrapping(source, output)


def test_static_if_global_vars():
    """Checks that resolvable if statements are replaced"""

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
    check_wrapping(source, output, namespace={"config": config})


def test_dynamic_if():
    """Checks that non-resolvable if statements are ignored"""

    source = """
    def forward():
        test = ...
        if test:
            pass
    """
    output = """
    @torch.fx.wrap
    def wrapped_0(test):
        if test:
            pass
        return ()

    def forward():
        test = ...
        () = wrapped_0(test)
    """
    check_wrapping(source, output)


def test_ignore_functions():
    """Checks that ignored functions are wrapped"""

    def func_one():
        pass

    def func_two():
        pass

    source = """
    def forward():
        func_one()
        func_two()
    """
    output = """
    @torch.fx.wrap
    def wrapped_0():
        return func_one()
        return ()

    def forward():
        wrapped_0()
        func_two()
    """
    namespace = {"func_one": func_one, "func_two": func_two}
    check_wrapping(source, output, namespace=namespace, ignore=["func_one"])


def test_ignore_methods():
    """Checks that ignored class methods are wrapped"""

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
    output = """
    @torch.fx.wrap
    def wrapped_0():
        return self.meth_one()
        return ()

    def forward(self):
        wrapped_0()
        self.meth_two()
    """
    namespace = {"self": Model()}
    check_wrapping(source, output, namespace=namespace, ignore=["meth_one"])


def test_branch_with_self_assignment():
    """Checks that names referenced in self assignment are included in fn args"""

    source = """
    def forward(x, y):
        if y > 0:
            x = x + 1
        else:
            x = x - 1
        return x
    """
    output = """
    @torch.fx.wrap
    def wrapped_0(x, y):
        if y > 0:
            x = x + 1
        else:
            x = x - 1
        return (x,)

    def forward(x, y):
        (x,) = wrapped_0(x, y)  # skip: some envs use "(x,)" -> "x,"
        return x
    """
    check_wrapping(source, output)


def test_function_variadic():
    """Checks for handling variadic names created via function def"""

    source = """
    def forward(a, *b, c=5, **d):
        if a == b and c == d:
            pass
    """
    output = """
    @torch.fx.wrap
    def wrapped_0(a, b, c, d):
        if a == b and c == d:
            pass
        return ()

    def forward(a, *b, c=5, **d):
        () = wrapped_0(a, b, c, d)
    """
    check_wrapping(source, output)
