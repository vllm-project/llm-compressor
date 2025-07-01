import ast
from types import FunctionType, MethodType
from typing import Any, Dict, List, Union

from loguru import logger

from .control_flow_analyzer import ControlFlowAnalyzer
from .name_analyzer import NameAnalyzer


class AutoWrapper(ast.NodeTransformer):
    """
    Automatically wraps untracable code according to the following patterns:

    The following patterns are automatically wrapped
    1. If statements whose conditions cannot be statically evaluated
    2. Ignored functions (`_update_causal_mask`)
    3. Starred tuple unpacking
    4. Starred argument unpacking

    See also: https://github.com/vllm-project/llm-compressor/pull/1411
    """

    def __init__(self, namespace: Dict[str, Any], ignore: List[str]):
        self.namespace = namespace
        self.ignore = ignore
        self._wrapper_fn_defs: List[ast.FunctionDef] = list()
        self._local_names = set()

    def auto_wrap(self, tree: ast.Module) -> ast.Module:
        """
        Modify ast by automatically wrapping any untraceable code segments. Segments to
        wrap are determined through analysis of the code and basic pattern matching

        :param tree: module containing a definition to an original forward function
        :return: module with added wrapper function definitions and function calls
        """
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

    def visit_Name(self, node: ast.Name):
        """
        Add any new names in `self._local_names`,
        which are used to determine function wrapper arguments
        """
        if isinstance(node.ctx, ast.Store):
            self._local_names.add(node.id)

        return super().generic_visit(node)

    def visit_Delete(self, node: ast.Delete):
        """
        Remove any deleted names from `self._local_names`,
        which are used to determine function wrapper arguments
        """
        ret = super().generic_visit(node)

        for target in node.targets:
            if isinstance(target, ast.Name):
                self._local_names.remove(target.id)

        return ret

    def visit_If(self, node: ast.If) -> Union[ast.If, ast.Assign]:
        """
        Attempt to statically evaluate the condition of the `if` statement. If the
        condition can not be statically evaluated (1), then attmept to wrap the `if`
        statement

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

    def visit_Tuple(self, node: ast.Tuple) -> Union[ast.Tuple, ast.Call]:
        """
        (3) Wrap any tuples which use starred unpacking
        """
        if any(isinstance(elem, ast.Starred) for elem in node.elts):
            return self._wrap_if_possible(node)

        return super().generic_visit(node)

    def visit_Call(self, node: ast.Call) -> ast.Call:
        """
        Wrap any functions which use (4) variadic arguments or (2) match the ignore list
        """
        # check for variadic starred
        if any(isinstance(elem, ast.Starred) for elem in node.args):
            return self._wrap_if_possible(node)

        # attempt to evaluate caller and check against ignore list
        try:
            caller = self._eval_expr(node.func)

        except Exception:
            caller = None

        finally:
            if (
                isinstance(caller, (FunctionType, MethodType))
                and caller.__name__ in self.ignore
            ):
                return self._wrap_if_possible(node)

        return super().generic_visit(node)

    def _eval_expr(self, node: ast.expr) -> Any:
        """
        Evaluate an expression using evaluation context

        :param node: expression to evaluate
        :return: evaluated value of expression
        """
        if not isinstance(node, ast.expr):
            raise TypeError("Expected an `ast.expr` node")

        expr = ast.Expression(body=node)  # wrap in expression in order to compile
        expr = ast.fix_missing_locations(expr)
        compiled = compile(expr, filename="<_eval_expr>", mode="eval")
        return eval(compiled, self.namespace, {})

    def _can_wrap(self, node: ast.AST) -> bool:
        """
        Some nodes cannot be wrapped because they contain control flow which is invalid
        without its original context. In the future, we can add more checks for module
        calls (see `visit_If`)
        """
        analyzer = ControlFlowAnalyzer()
        return analyzer.is_valid(node)

    def _wrap_if_possible(self, node: ast.AST) -> Union[ast.AST, ast.Assign, ast.Call]:
        """
        Defines a wrapper function containing the wrapped node.

        If a statement is passed, then returns a statement which calls the newly defined
        wrapper function with inputs and assigns the value to the output names

        If an expression is passed, then returns a call to the newly defined wrapper
        function with inputs and no assignment

        The new wrapper function definition is stored in `_wrapper_fn_defs` and is later
        appended to the module ast by `AutoWrapper.auto_wrap`

        :param node: node to be wrapped
        :return: a call to the wrapped function, either being assigned to variable names
            or called as-is
        """
        if not self._can_wrap(node):
            return node

        if isinstance(node, ast.stmt):
            return self._wrap_stmt(node)

        elif isinstance(node, ast.expr):
            return self._wrap_expr(node)

        else:
            raise TypeError(f"Unknown type {type(node)}")

    def _wrap_stmt(self, node: ast.stmt) -> ast.Assign:
        # unbound := names which are read by node before being assigned
        # assigned := names which are assigned by operations in node
        # cond_assigned := names which may be assigned depending on execution
        analyzer = NameAnalyzer(omit=self.namespace.keys())
        unbound, assigned, conditionally_assigned = analyzer.analyze(node)

        # args := names which already existed and are needed for ops or wrapped return
        # kwargs := names which are needed for return but did not already exist
        # returns := names which are assigned or could be assigned
        args = (unbound | conditionally_assigned) & self._local_names
        kwargs = conditionally_assigned - self._local_names
        returns = assigned | conditionally_assigned
        assert "self" not in args, "Cannot trace self, this should be in the namespace"

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

        # build body and return statement
        return_stmt = ast.Return(
            value=ast.Tuple(
                elts=[ast.Name(id=name, ctx=ast.Load()) for name in sorted(returns)],
                ctx=ast.Load(),
            )
        )
        body = [node, return_stmt]

        # build function definition, store in `_wrapper_fn_defs`
        fn_name = f"wrapped_{hash(node)}"
        fn_def = ast.FunctionDef(
            name=fn_name,
            args=args_obj,
            body=body,
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

        # log newly created function definition
        logger.debug("---- Autowrapper ----")
        logger.debug(ast.unparse(ast.fix_missing_locations(fn_def)))
        logger.debug("---------------------")

        return assign_call

    def _wrap_expr(self, node: ast.expr) -> ast.Call:
        return_stmt = ast.Return(value=node)
        wrapped = self._wrap_stmt(return_stmt)
        fn_call = wrapped.value

        return fn_call
