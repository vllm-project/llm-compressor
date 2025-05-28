import ast
from typing import List, Type


class ControlFlowAnalyzer(ast.NodeVisitor):
    """
    Used to determine if a piece of code can be wrapped into a function. Includes any
    code which includes `return`, `continue`, `break`, `await`, or `yield`
    without the proper context.

    For example, this code can be wrapped:
    ```python3
    while True:
        if some_condition:
            break
    ```

    Whereas the inner code cannot be wrapped without the `while` context
    ```python3
    def wrapped():
        if some_condition:
            break  # this control statement is now invalid

    while True:
        wrapped()
    ```
    """

    _contexts: List[Type]
    _is_valid: bool
    _context_types = (ast.For, ast.While, ast.FunctionDef, ast.AsyncFunctionDef)

    def is_valid(self, node: ast.AST) -> bool:
        """
        Returns False if a node contains control statements that are not in their
        proper control flow context

        :param node: code to analyze
        :return: True iff the code does not contain invalid control statements
        """
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
        # cannot wrap early returns
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
