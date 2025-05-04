import ast
import builtins
import contextlib
import inspect
import sys
import textwrap
from types import FunctionType, MethodType
from typing import Any, Dict, List, Union

import torch

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
    tree = ast.fix_missing_locations(auto_wrapper.autowrap(tree))
    print(type(module))
    print(ast.unparse(ast.fix_missing_locations(tree)))

    # compile new forward function from autowrapped code
    filename = f"{module.__class__.__name__}_autowrapped"
    code = compile(tree, filename=filename, mode="exec")
    exec(code, namespace)
    import linecache

    source_str = ast.unparse(tree)
    linecache.cache[filename] = (
        len(source_str),
        None,
        [line + "\n" for line in source_str.splitlines()],
        filename,
    )

    # some modules (such as ModuleList) do not implement a forward function
    new_forward = namespace.get("forward", module.forward.__func__)
    new_forward = new_forward.__get__(module)  # curry self
    with patch_attr(module, "forward", new_forward):
        yield


class AutoWrapper(ast.NodeTransformer):
    def __init__(self, eval_context: Dict[str, Any], ignore: List[str]):
        self.eval_context = eval_context
        self.ignore = ignore
        self._wrapped_fn_defs = list()

    def autowrap(self, tree: ast.Module) -> ast.Module:
        tree = self.visit(tree)

        for fn_def in self._wrapped_fn_defs:
            tree.body.insert(0, fn_def)

        return tree

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """
        Remove decorators which prevent forward function recompilation
        For example, add_start_docstrings_to_model_forward
        """
        node.decorator_list = []
        return super().generic_visit(node)

    def visit_If(self, node: ast.If) -> Union[ast.If, ast.Assign]:
        """
        Attempt static eval, else wrap
        """
        try:
            value = bool(self._eval_expr(node.test))

        except:
            return self._wrap(node)

        else:
            print(
                f"successfully evaled {ast.unparse(ast.fix_missing_locations(node.test))} into {value}"
            )
            node.test = ast.Constant(value=value)
            return super().generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> ast.Assign:
        """
        Check for assignment from ignored functions
        """
        if isinstance(node.value, ast.Call):
            func = node.value.func
            try:
                caller = self._eval_expr(func)

            except:
                pass

            else:
                if (
                    isinstance(caller, (FunctionType, MethodType))
                    and caller.__name__ in self.ignore
                ):
                    return self._wrap(node)

        return super().generic_visit(node)

    def visit_Starred(self, node: ast.Starred):
        """
        Note that args is represented as ast.arugments(..., vararg=ast.arg(arg="args")),
        so starred only represents iterations
        """
        wrap_assign = self._wrap(
            ast.Assign(
                targets=[ast.Name(id="unpacked", ctx=ast.Store())],
                value=ast.Tuple(elts=[node], ctx=ast.Load()),
            )
        )
        return wrap_assign.value
        return super().visit_Starred(node)

    def _eval_expr(self, node: ast.expr) -> Any:
        if not isinstance(node, ast.expr):
            raise TypeError("Expected an ast.expr node")

        module = ast.Expression(body=node)  # wrap in expression in order to compile
        module = ast.fix_missing_locations(module)
        compiled = compile(module, filename="<ast>", mode="eval")
        return eval(compiled, {}, self.eval_context)

    def _wrap(self, nodes: Union[List[ast.AST], ast.AST]) -> ast.Assign:
        nodes = nodes if isinstance(nodes, List) else [nodes]

        analyzer = NameAnalyzer(self.eval_context.keys())
        for node in nodes:
            analyzer.visit(node)

        args = analyzer.unbound_names - analyzer.conditionally_assigned_names
        kwargs = analyzer.conditionally_assigned_names
        returns = analyzer.assigned_names

        # Create function arguments
        args_ast = [ast.arg(arg=name) for name in args] + [ast.arg(arg=name) for name in kwargs]  # ensure defaults last
        args_obj = ast.arguments(
            args=args_ast,
            posonlyargs=[],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[ast.Constant(value=None) for _ in range(len(kwargs))],
            vararg=None,
            kwarg=None,
        )

        # Build return statement
        return_stmt = ast.Return(
            value=ast.Tuple(
                elts=[ast.Name(id=name, ctx=ast.Load()) for name in sorted(returns)],
                ctx=ast.Load(),
            )
        )

        # Build body with return at the end
        body = [node for node in nodes] + [return_stmt]

        fn_name = f"wrapped_{hash(node)}"
        fn_def = ast.FunctionDef(
            name=fn_name,
            args=args_obj,
            body=body,
            # decorator_list=[],
            decorator_list=[ast.Name(id="torch.fx.wrap", ctx=ast.Load())],
        )
        fn_call = ast.Call(
            func=ast.Name(id=fn_name, ctx=ast.Load()),
            args=[ast.Name(id=name, ctx=ast.Load()) for name in args],
            keywords=list(),
        )

        return_tuple = ast.Tuple(
            elts=[ast.Name(id=name, ctx=ast.Store()) for name in sorted(returns)],
            ctx=ast.Store(),
        )
        fn_call_expr = ast.Assign(targets=[return_tuple], value=fn_call)

        self._wrapped_fn_defs.append(fn_def)

        return fn_call_expr


class NameAnalyzer(ast.NodeVisitor):
    def __init__(self, globals):
        self.unbound_names = set()
        self.assigned_names = set()
        self.conditionally_assigned_names = set()
        self._omit_names = builtins.__dict__.keys() | globals

    def generic_visit(self, node):
        """Explicitly define to guarantee DFS"""
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        self.visit(item)
            elif isinstance(value, ast.AST):
                self.visit(value)

    def visit_If(self, node: ast.If):
        self.visit(node.test)
        pre_assigned_names = self.assigned_names.copy()

        for statement in node.body:
            self.visit(statement)

        if_true_assigned_names = self.assigned_names - pre_assigned_names
        self.assigned_names = pre_assigned_names.copy()

        for statement in node.orelse:
            self.visit(statement)

        if_false_assigned_names = self.assigned_names - pre_assigned_names

        self.conditionally_assigned_names |= if_true_assigned_names ^ if_false_assigned_names
        self.assigned_names |= if_true_assigned_names | if_false_assigned_names

    def visit_Name(self, node: ast.Name) -> ast.Name:
        name = node.id

        if isinstance(node.ctx, ast.Load):
            # reading name that has not been written yet
            if name not in self.assigned_names and name not in self._omit_names:
                self.unbound_names.add(node.id)
        else:
            # writing any name that's not omitted
            if name not in self._omit_names:
                self.assigned_names.add(name)

        self.generic_visit(node)
