from typing import List, Union

import ast
import sys
import torch
import inspect
import textwrap
import contextlib

from llmcompressor.utils import patch_attr


@contextlib.contextmanager
def autowrap_forward(module: torch.nn.Module):
    source = inspect.getsource(module.forward)
    source = textwrap.dedent(source)
    tree = ast.parse(source)
    breakpoint()

    auto_wrapper = AutoWrapper(tree)
    tree = ast.fix_missing_locations(auto_wrapper.visit(tree))
    print(ast.unparse(tree))
    breakpoint()

    code = compile(tree, filename="<autowrapped>", mode="exec")
    defining_module = sys.modules[module.__class__.__module__].__dict__
    namespace = defining_module.__dict__  # original module context
    exec(code, namespace)
    new_forward = namespace["forward"]
    breakpoint()

    with patch_attr(module, "forward", new_forward):
        yield


class AutoWrapper(ast.NodeTransformer):
    def __init__(self, tree: ast.Module):
        self.tree = tree
        
    def visit_If(self, node):
        return self._wrap(node)
        #return super().visit_If(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """
        Remove decorators which prevent forward function recompilation
        """
        if "wrapped" in node.name:
            return node
            
        node.decorator_list = []
        return super().generic_visit(node)
    
    def _wrap(self, nodes: Union[List[ast.AST], ast.AST]) -> ast.Call:
        nodes = nodes if isinstance(nodes, List) else [nodes]

        analyzer = VarAnalyzer()
        for node in nodes:
            analyzer.visit(node)
        
        args = analyzer.unbound_names
        returns = analyzer.all_names  # - {"self"}

        # Create function arguments
        args_ast = [ast.arg(arg=name) for name in args]
        args_obj = ast.arguments(
            args=args_ast,
            posonlyargs=[], kwonlyargs=[], kw_defaults=[],
            defaults=[], vararg=None, kwarg=None
        )

        # Build return statement
        return_tuple = ast.Tuple(
            elts=[ast.Name(id=name, ctx=ast.Load()) for name in sorted(returns)],
            ctx=ast.Load()
        )
        return_stmt = ast.Return(value=return_tuple)

        # Build body with return at the end
        body = [ast.Expr(node) for node in nodes] + [return_stmt]

        fn_name = f"wrapped_{hash(node)}"
        fn_def = ast.FunctionDef(
            name=fn_name,
            args=args_obj,
            body=body,
            #decorator_list=[],
            decorator_list=[ast.Name(id="torch.fx.wrap", ctx=ast.Load())],
        )
        fn_call = ast.Call(
            func=ast.Name(id=fn_name, ctx=ast.Load()),
            args=args_ast,
            keywords=list()
        )
        fn_call_expr = ast.Assign(
            targets=[return_tuple],
            value=fn_call
        )

        assert isinstance(self.tree, ast.Module)
        self.tree.body.insert(0, fn_def)

        return fn_call_expr


class VarAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.unbound_names = set()
        self.all_names = set()

    def generic_visit(self, node):
        """Explicitly define to guarantee DFS"""
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        self.visit(item)
            elif isinstance(value, ast.AST):
                self.visit(value)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and node.id not in self.all_names:
            self.unbound_names.add(node.id)

        self.all_names.add(node.id)
        self.generic_visit(node)