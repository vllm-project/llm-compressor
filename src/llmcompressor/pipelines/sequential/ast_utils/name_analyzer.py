import ast
import builtins
from typing import Set, Tuple

from llmcompressor.utils import patch_attr


class NameAnalyzer(ast.NodeVisitor):
    """
    Determines the unbound, assigned, and conditionally assigned names associated with
    a piece of code. This information is used to determine the arguments and return
    values of the wrapper function

    For example, for the following piece of code
    ```python3
    b = a + 1
    if some_condition:
        c = 5
    ```

    `a` is unbound, meaning that it must be an input of wrapper function
    `b` is assigned, meaning that it must be an output of the wrapper function
    `c` is conditionally assigned, meaning that it must be an output of the wrapper
    function, and *might* be an input iff `c` already exists in the namespace

    Note that names which are assigned to before being read are not considered unbound
    ```python3
    a = 2  # no longer unbound
    b = a + 1
    ```
    """

    _unbound: Set[str]
    _assigned: Set[str]
    _conditionally_assigned: Set[str]
    _omit: Set[str]

    def __init__(self, omit: Set[str]):
        self._omit = builtins.__dict__.keys() | omit

    def analyze(self, node: ast.AST) -> Tuple[Set[str], Set[str], Set[str]]:
        """
        Analyzes the use of names in the given piece of code

        :param node: code to analyze
        return: tuple of unbound names, assigned names, and conditionally assigned names
        """
        self._unbound = set()
        self._assigned = set()
        self._conditionally_assigned = set()
        self.visit(node)

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

    def visit_Assign(self, node: ast.Assign):
        # Visit the right side of the assignment first
        self.visit(node.value)

        # Now visit the left side of the assignment
        for target in node.targets:
            self.visit(target)

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
