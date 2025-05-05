import ast
import builtins
import contextlib
import inspect
import sys
import textwrap
from types import FunctionType, MethodType
from typing import Any, Dict, List, Set, Tuple, Type, Union

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
    tree = auto_wrapper.auto_wrap(tree)

    # compile new forward function from autowrapped code
    filename = f"{module.__class__.__name__}_{hash(module)}_autowrapped"
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
        self._wrapper_fn_defs = list()
        self._local_names = set()

    def auto_wrap(self, tree: ast.Module) -> ast.Module:
        tree = self.visit(tree)
        for fn_def in self._wrapper_fn_defs:
            tree.body.insert(0, fn_def)

        return ast.fix_missing_locations(tree)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """
        Remove decorators which prevent forward function recompilation
        For example, add_start_docstrings_to_model_forward

        Because `_wrapper_fn_defs` are appended after `visit` finishes, this function
        will not affect wrapper functions

        :param node: function definition whose decorators will be stripped
        :return: function definition without decorators
        """
        node.decorator_list = []
        if node.name == "forward":
            for arg in node.args.args:
                self._local_names.add(arg.arg)
        return super().generic_visit(node)

    def visit_If(self, node: ast.If) -> Union[ast.If, ast.Assign]:
        """
        Attempt to statically evaluate the condition of the `if` statement. If the
        condition can not be statically evaluated, then wrap the `if` statement if
        possible

        TODO: sometimes module calls happen in if statements.
        Most commonly, this can happen for code like this:
        ```
        if image_embeds is None:
            image_embeds = self.visual(pixel_values)
        ```

        There may be some ways of mitigating this, but there are likely no perfect
        solutions that cover all cases without requiring some user intervention
        1. Add model inputs such as `image_embeds` to the eval context, allowing
            these names to be evaluated (although any intermediate ops will not be
            reflected)
        2. Attempt to infer if a node calls a module from static code analysis alone
            a. We can eval the caller of any Calls (in the self module context) and
                check if the caller is one of our targets (or an ancestor of targets)
            b. We can use type inference libraries like `jedi` to infer the type of
                any callers, which may be more robust but more complicated

        :param node: `if` statement which may be wrapped
        :return: if the `if` statement cannot be statically evaluated, return the
            `if` statement with the condition replaced by `True` or `False`.
            Otherwise, return a wrapper function call
        """
        try:
            value = bool(self._eval_expr(node.test))

        except Exception:
            return self._wrap_if_possible(node)

        else:
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

            except Exception:
                pass

            else:
                if (
                    isinstance(caller, (FunctionType, MethodType))
                    and caller.__name__ in self.ignore
                ):
                    return self._wrap_if_possible(node)

        return super().generic_visit(node)

    def visit_Starred(self, node: ast.Starred) -> Union[ast.Starred, ast.Assign]:
        """
        Note that args is represented as ast.arugments(..., vararg=ast.arg(arg="args")),
        so starred only represents iterations
        """
        assign = ast.Assign(
            targets=[ast.Name(id="unpacked", ctx=ast.Store())],
            value=ast.Tuple(elts=[node], ctx=ast.Load()),
        )
        wrapped_assign = self._wrap_if_possible(assign)
        if wrapped_assign is assign:
            # wrapping failed (although it really shouldn't)
            return node

        return wrapped_assign.value

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Store):
            self._local_names.add(node.id)

        return super().generic_visit(node)

    def visit_Delete(self, node: ast.Delete):
        ret = super().visit_Delete(node)

        for target in node.targets:
            if isinstance(target, ast.Name):
                self._local_names.remove(target.id)

        return ret

    def _eval_expr(self, node: ast.expr) -> Any:
        if not isinstance(node, ast.expr):
            raise TypeError("Expected an ast.expr node")

        module = ast.Expression(body=node)  # wrap in expression in order to compile
        module = ast.fix_missing_locations(module)
        compiled = compile(module, filename="<ast>", mode="eval")
        return eval(compiled, {}, self.eval_context)

    def _can_wrap(self, node: ast.AST) -> bool:
        """
        Some nodes cannot be wrapped because they contain control flow which is invalid
        without its original context
        """
        analyzer = ControlFlowAnalyzer()
        return analyzer.is_valid(node)

    def _wrap_if_possible(self, node: ast.AST) -> ast.Assign:
        """
        Defines a wrapper function containing the wrapped node. Returns a statement
        which calls the newly defined wrapper function with required inputs and outputs

        The new wrapper function definition is stored in `_wrapper_fn_defs` and is later
        appended to the module ast by `AutoWrapper.auto_wrap`

        :param node: node to be wrapped
        :return: an Assign statement which calls the wrapper function
        """
        if not self._can_wrap(node):
            return node

        # unbound := names which are read by node before being assigned
        # assigned := names which are assigned by operations in node
        # cond_assigned := names which may be assigned depending on execution
        analyzer = NameAnalyzer(omit=self.eval_context.keys())
        unbound, assigned, conditionally_assigned = analyzer.analyze(node)

        # args := names which already existed and are needed for ops or wrapped return
        # kwargs := names which are needed for return but did not already exist
        # returns := names which are assigned or could be assigned
        args = (unbound | conditionally_assigned) & self._local_names
        kwargs = conditionally_assigned - self._local_names
        returns = assigned | conditionally_assigned

        # build function arguments
        args_obj = ast.arguments(
            args=[ast.arg(arg=name) for name in args],
            posonlyargs=[],
            kwonlyargs=[ast.arg(arg=name) for name in kwargs],
            kw_defaults=[ast.Constant(value=None) for _ in kwargs],
            defaults=[],
            vararg=None,
            kwarg=None,
        )

        # build return statement
        return_stmt = ast.Return(
            value=ast.Tuple(
                elts=[ast.Name(id=name, ctx=ast.Load()) for name in sorted(returns)],
                ctx=ast.Load(),
            )
        )

        # build function definition, store in `_wrapper_fn_defs`
        fn_name = f"wrapped_{hash(node)}"
        fn_def = ast.FunctionDef(
            name=fn_name,
            args=args_obj,
            body=[node, return_stmt],
            decorator_list=[ast.Name(id="torch.fx.wrap", ctx=ast.Load())],
        )
        self._wrapper_fn_defs.append(fn_def)

        # build call and assignment
        fn_call = ast.Call(
            func=ast.Name(id=fn_name, ctx=ast.Load()),
            args=[ast.Name(id=name, ctx=ast.Load()) for name in args],
            keywords=list(),
        )
        return_tuple = ast.Tuple(
            elts=[ast.Name(id=name, ctx=ast.Store()) for name in sorted(returns)],
            ctx=ast.Store(),
        )
        assign_call = ast.Assign(targets=[return_tuple], value=fn_call)

        # update local names with newly returned values
        self._local_names |= returns

        return assign_call


class ControlFlowAnalyzer(ast.NodeVisitor):
    _contexts: List[Type]
    _is_valid: bool
    _context_types = (ast.For, ast.While, ast.FunctionDef, ast.AsyncFunctionDef)

    def is_valid(self, node: ast.AST) -> bool:
        self._contexts = []
        self._is_valid = True
        self.visit(node)
        return self._is_valid

    def generic_visit(self, node: ast.AST):
        node_type = type(node)
        is_context = node_type in self._context_types

        if is_context:
            self._contexts.append(node_type)

        super().generic_visit(node)

        if is_context:
            self._contexts.pop()

    def visit_Return(self, node: ast.Return):
        if (
            ast.FunctionDef not in self._contexts
            and ast.AsyncFunctionDef not in self._contexts
        ):
            self._is_valid = False
        return super().generic_visit(node)

    def visit_Continue(self, node: ast.Continue):
        if ast.For not in self._contexts and ast.While not in self._contexts:
            self._is_valid = False
        return super().generic_visit(node)

    def visit_Break(self, node: ast.Break):
        if ast.For not in self._contexts and ast.While not in self._contexts:
            self._is_valid = False
        return super().generic_visit(node)

    def visit_Await(self, node: ast.Await):
        if ast.AsyncFunctionDef not in self._contexts:
            self._is_valid = False
        return super().generic_visit(node)

    def visit_Yield(self, node: ast.Yield):
        if ast.FunctionDef not in self._contexts:
            self._is_valid = False
        return super().generic_visit(node)


class NameAnalyzer(ast.NodeVisitor):
    _unbound: Set[str]
    _assigned: Set[str]
    _conditionally_assigned: Set[str]
    _omit: Set[str]

    def __init__(self, omit: Set[str]):
        self._omit = builtins.__dict__.keys() | omit

    def analyze(self, tree: ast.AST) -> Tuple[Set[str], Set[str], Set[str]]:
        self._unbound = set()
        self._assigned = set()
        self._conditionally_assigned = set()
        self.visit(tree)

        return self._unbound, self._assigned, self._conditionally_assigned

    def visit_Name(self, node: ast.Name) -> ast.Name:
        name = node.id

        if isinstance(node.ctx, ast.Load):
            # reading name that has not been written yet
            if name not in self._assigned and name not in self._omit:
                self._unbound.add(node.id)
        else:
            # writing any name that's not omitted
            if name not in self._omit:
                self._assigned.add(name)

        self.generic_visit(node)

    def visit_If(self, node: ast.If):
        self.visit(node.test)

        # collect names from `true` clause
        with patch_attr(self, "_assigned", set()):
            for statement in node.body:
                self.visit(statement)
            true_assigned = self._assigned

        # collect names from `false` clause
        with patch_attr(self, "_assigned", set()):
            for statement in node.orelse:
                self.visit(statement)
            false_assigned = self._assigned

        self._conditionally_assigned |= true_assigned ^ false_assigned
        self._assigned |= true_assigned & false_assigned
