import ast
from types import FunctionType, MethodType
from typing import Any, Dict, List, Union

from .ControlFlowAnalyzer import ControlFlowAnalyzer
from .NameAnalyzer import NameAnalyzer


class AutoWrapper(ast.NodeTransformer):
    def __init__(
        self, globals: Dict[str, Any], locals: Dict[str, Any], ignore: List[str]
    ):
        self.globals = globals
        self.locals = locals
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

    def visit_Tuple(self, node: ast.Tuple) -> Union[ast.Tuple, ast.Call]:
        if any(isinstance(elem, ast.Starred) for elem in node.elts):
            return self._wrap_if_possible(node)

        return super().generic_visit(node)

    def visit_Call(self, node: ast.Call) -> ast.Call:
        # TODO: since self cannot be passed, we may have to add to
        # torch.fx._symbolic_trace._wrapped_methods_to_patch directly and not wrap

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

    def visit_Name(self, node: ast.Name):
        """
        Add any new names in `self._local_names`
        """
        if isinstance(node.ctx, ast.Store):
            self._local_names.add(node.id)

        return super().generic_visit(node)

    def visit_Delete(self, node: ast.Delete):
        """
        Remove any deleted names from `self._local_names`
        """
        ret = super().visit_Delete(node)

        for target in node.targets:
            if isinstance(target, ast.Name):
                self._local_names.remove(target.id)

        return ret

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
        return eval(compiled, self.globals, self.locals)

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
        Defines a wrapper function containing the wrapped node. Returns a statement
        which calls the newly defined wrapper function with required inputs and outputs

        The new wrapper function definition is stored in `_wrapper_fn_defs` and is later
        appended to the module ast by `AutoWrapper.auto_wrap`

        :param node: node to be wrapped
        :return: an Assign statement which calls the wrapper function
        """
        if not self._can_wrap(node):
            return node

        if isinstance(node, ast.stmt):
            return self._wrap_stmt(node)

        elif isinstance(node, ast.expr):
            return self._wrap_expr(node)

        else:
            raise ValueError()

    def _wrap_stmt(self, node: ast.stmt) -> ast.Assign:
        # unbound := names which are read by node before being assigned
        # assigned := names which are assigned by operations in node
        # cond_assigned := names which may be assigned depending on execution
        analyzer = NameAnalyzer(omit=self.globals.keys())
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
        # TODO: when it comes to handling fns with `self`, do one of the following
        # 1. define functions as methods (and patch them in along with the forward),
        #   + use torch.fx._symbolic_trace._wrapped_methods_to_patch
        #   + keep the patches after trace is done? unclear how subgraphs handle fx.wrap
        # 2. expand any self attributes, both in the fn def and in the fn call
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

    def _wrap_expr(self, node: ast.expr) -> ast.Call:
        return_stmt = ast.Return(value=node)
        wrapped = self._wrap_stmt(return_stmt)
        fn_call = wrapped.value

        return fn_call
