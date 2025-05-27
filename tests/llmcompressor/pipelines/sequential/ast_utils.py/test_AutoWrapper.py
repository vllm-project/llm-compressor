import ast
import textwrap
from types import SimpleNamespace

from llmcompressor.pipelines.sequential.ast_utils.AutoWrapper import AutoWrapper


def test_static_if():
    source = textwrap.dedent("""
    def forward():
        if 1 + 1 == 2:
            return True
        else:
            return False
    """)
    tree = ast.parse(source)

    wrapper = AutoWrapper({}, [])
    new_tree = wrapper.auto_wrap(tree)

    assert len(wrapper._wrapper_fn_defs) == 0
    output = textwrap.dedent("""
    def forward():
        if True:
            return True
        else:
            return False
    """)
    assert ast.unparse(new_tree).splitlines() == output.splitlines()[1:]


def test_static_if_global_vars():
    source = textwrap.dedent("""
    def forward():
        if config.is_false:
            return True
        else:
            return False
    """)
    tree = ast.parse(source)

    config = SimpleNamespace(is_false=False)
    wrapper = AutoWrapper({"config": config}, [])
    new_tree = wrapper.auto_wrap(tree)

    assert len(wrapper._wrapper_fn_defs) == 0
    output = textwrap.dedent("""
    def forward():
        if False:
            return True
        else:
            return False
    """)
    assert ast.unparse(new_tree).splitlines() == output.splitlines()[1:]
