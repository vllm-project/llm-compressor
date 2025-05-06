import ast
import builtins
from typing import Set, Tuple

from llmcompressor.utils import patch_attr


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
