import contextlib
import inspect
from collections import UserDict, deque
from dataclasses import dataclass
from functools import wraps
from types import FunctionType, MethodType
from typing import TYPE_CHECKING, Any, Callable

import torch
from compressed_tensors.offload import disable_onloading
from compressed_tensors.utils import patch_attr
from compressed_tensors.utils.match import match_named_modules
from loguru import logger
from torch.fx import Graph, GraphModule, Node
from torch.fx.graph import PythonCode
from torch.fx.proxy import Argument
from torch.nn import Module
from transformers import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig

from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.pipelines.sequential.transformers_helpers import HFTracer
from llmcompressor.utils.helpers import calibration_forward_context

from .ast_helpers import append_autowrap_source_on_fail, autowrap_forwards

if TYPE_CHECKING:
    pass

__all__ = [
    "INVOKED_SUBMODULES_META_KEY",
    "trace_subgraphs",
    "Subgraph",
    "SubgraphTrace",
    "handle_sequential_oom",
    "collect_subgraph_modules",
    "find_modules_outside_subgraphs",
    "iter_invoked_submodule_names",
]

# Node.meta key populated by SequentialTracer for modules executed while creating a
# node. Used so Subgraph.submodules() can see calls hidden inside @torch.fx.wrap
# regions (see #3261).
INVOKED_SUBMODULES_META_KEY = "llmcompressor.invoked_submodules"


@dataclass
class Subgraph:
    """
    Dataclass specifying an executable subgraph of a model graph

    :param graph: subgraph of model graph
    :param input_names: argument names of the compiled forward function
    :param consumed_names: argument names which are not used by any subsequent subgraphs
        and can therefore be deleted from the intermediates cache
    """

    graph: Graph
    input_names: set[str]
    consumed_names: set[str]
    _code: PythonCode | None = None

    def _subgraph_module_names(self, model: Module, recurse: bool = True) -> list[str]:
        """
        Qualified names of modules owned by this subgraph.

        Includes ``call_module`` targets and any modules recorded on
        ``node.meta[INVOKED_SUBMODULES_META_KEY]`` while tracing (e.g. leaves executed
        inside ``@torch.fx.wrap`` regions that never appear as ``call_module`` ops).
        """
        ordered_names: list[str] = []
        seen_modules: set[Module] = set()

        def add_tree(qualified_name: str) -> None:
            try:
                module = model.get_submodule(qualified_name)
            except AttributeError:
                logger.warning(
                    "Skipping unknown invoked submodule name {!r} while resolving "
                    "subgraph modules",
                    qualified_name,
                )
                return

            named_modules = module.named_modules() if recurse else [("", module)]
            for relative_name, submodule in named_modules:
                if submodule in seen_modules:
                    continue

                name = qualified_name
                if relative_name:
                    name = f"{qualified_name}.{relative_name}"
                ordered_names.append(name)
                seen_modules.add(submodule)

        # Walk in graph order for deterministic DDP module ordering.
        for node in self.graph.nodes:
            if node.op == "call_module":
                add_tree(node.target)
            for name in node.meta.get(INVOKED_SUBMODULES_META_KEY, ()):
                add_tree(name)

        return ordered_names

    def forward(self, *args, **kwargs) -> dict[str, Any]:
        """
        Execute the operations within the subgraph

        :param \\*args: argument inputs to subgraph forward function
        :param \\**kwargs: keyword inputs to subgraph forward function
        :return keyword outputs of subgraph forward function (non-consumed variables):
        """
        if self._code is None:
            self._code = self.graph.python_code("self")
            exec(self._code.src, self._code.globals)

        forward_fn = self._code.globals.get("forward")

        with append_autowrap_source_on_fail():
            return forward_fn(*args, **kwargs)

    def submodules(self, model: Module, recurse: bool = True) -> list[Module]:
        """
        Modules owned by this subgraph, including leaves invoked inside wrapped
        regions during tracing (#3261).
        """
        return [
            model.get_submodule(name)
            for name in self._subgraph_module_names(model, recurse=recurse)
        ]

    def submodule_dict(self, model: Module, recurse: bool = True) -> dict[str, Module]:
        """Return subgraph modules keyed by their fully qualified model names."""
        return {
            name: model.get_submodule(name)
            for name in self._subgraph_module_names(model, recurse=recurse)
        }


@dataclass
class SubgraphTrace:
    """
    Result of :func:`trace_subgraphs`.

    List-like over ``subgraphs`` so existing callers keep working.
    ``oracle_invoked_names`` is an independent record of modules whose ``forward``
    actually ran during tracing (not derived from graph meta), used to validate
    subgraph ownership (#3261).

    Complementarity:
    - ``call_module`` covers leaves even when HF meta overrides skip real ``forward``
    - meta recording covers wrap-hidden real executions
    - never-invoked modules and sequential ancestors remain excluded
    """

    subgraphs: list[Subgraph]
    oracle_invoked_names: frozenset[str]

    def __iter__(self):
        return iter(self.subgraphs)

    def __len__(self) -> int:
        return len(self.subgraphs)

    def __getitem__(self, index: int | slice) -> Subgraph | list[Subgraph]:
        return self.subgraphs[index]


def collect_subgraph_modules(
    model: Module, subgraphs: list[Subgraph], recurse: bool = True
) -> dict[str, Module]:
    """Return the union of all modules owned by the traced subgraphs."""
    modules: dict[str, Module] = {}
    for subgraph in subgraphs:
        modules.update(subgraph.submodule_dict(model, recurse=recurse))
    return modules


def find_modules_outside_subgraphs(
    model: Module, subgraphs: list[Subgraph], recurse: bool = True
) -> dict[str, Module]:
    """
    Return modules not owned by any subgraph.

    This typically includes sequential ancestors (whose forwards are inlined) and
    modules never invoked on the traced path (e.g. unused towers). Modules executed
    inside ``@torch.fx.wrap`` regions are attributed to subgraphs via trace-time
    recording and therefore are not returned here.
    """
    subgraph_modules = collect_subgraph_modules(model, subgraphs, recurse=recurse)
    return {
        name: module
        for name, module in model.named_modules()
        if name and name not in subgraph_modules
    }


def iter_invoked_submodule_names(subgraphs: list[Subgraph]) -> set[str]:
    """
    Qualified names of modules recorded as invoked while tracing the given subgraphs.

    Combines ``call_module`` targets with ``INVOKED_SUBMODULES_META_KEY`` entries.
    """
    names: set[str] = set()
    for subgraph in subgraphs:
        for node in subgraph.graph.nodes:
            if node.op == "call_module":
                names.add(node.target)
            names.update(node.meta.get(INVOKED_SUBMODULES_META_KEY, ()))
    return names


def trace_subgraphs(
    model: PreTrainedModel,
    sample_input: dict[str, Any],
    sequential_targets: list[str],
    ignore: list[str],
    targets_per_subgraph: int = 1,
) -> SubgraphTrace:
    """
    Trace a model to produce subgraphs, where each sequential target belongs to exactly
    one subgraph and where executing each subgraph in order is equivalent to executing
    the original model

    :param model: model being traced
    :param sample_input: inputs whose values will change during execution but whose
        __len__, __bool__, and __contains__ values are assumed constant across batches
    :param sequential_targets: list of patterns matching sequential targets
    :param ignore: function and method names to skip during tracing
    :param targets_per_subgraph: number of targets to include per subgraph
    :return: SubgraphTrace with subgraphs and an independent forward-invocation oracle
    """
    # find modules
    targets = set(
        module for _, module in match_named_modules(model, sequential_targets)
    )
    ancestors = get_sequential_ancestors(model, targets)

    # initialize arguments
    tracer = SequentialTracer(ancestors, targets=targets)
    concrete_args = populate_concrete_args(model, sample_input)

    with contextlib.ExitStack() as stack:
        # calibration context
        stack.enter_context(calibration_forward_context(model))
        stack.enter_context(HooksMixin.disable_hooks())

        # flags useful for tracing
        # note: eager attention is forced by `calibration_forward_context`
        stack.enter_context(patch_attr(torch.compiler, "_is_compiling_flag", True))

        # autowrap forwards
        stack.enter_context(autowrap_forwards(ancestors, ignore))

        # avoid bug where pytorch cannot handle wrapped root functions
        unwrapped = inspect.unwrap(model.forward).__get__(model)
        stack.enter_context(patch_attr(model, "forward", unwrapped))
        stack.enter_context(patch_attr(type(model), "forward", unwrapped.__func__))
        assert isinstance(model.forward, MethodType)
        assert isinstance(type(model).forward, FunctionType)

        # avoid device movement during tracing
        stack.enter_context(disable_onloading())

        with append_autowrap_source_on_fail():
            graph = GraphModule(
                model,
                tracer.trace(
                    model,
                    dummy_inputs=sample_input,
                    concrete_args=concrete_args,
                    complete_concrete_args_with_inputs_not_in_dummy_inputs=False,
                    # bug in trace throws an error for variadic
                    # args and kwargs in function signature
                ),
            )

    # copy metadata
    graph.config = model.config
    graph.class_for_deserialization = model.__class__
    graph.device = model.device

    # perform subgraph partition
    partitions = topological_partition(graph, targets, targets_per_subgraph)
    subgraphs = partition_graph(model, partitions)
    trace_consumed_names(subgraphs)

    # As currently implemented, `topological_partition` generates an extra subgraph at
    # the beginning which does not contain a target. This adds a little more runtime,
    # and could be folded into the first subgraph in the future
    if len(subgraphs) != len(targets) + 1:
        logger.warning(
            f"Expected {len(targets)} subgraphs, but only traced {len(subgraphs)}. "
            "This is likely due to having wrapped code which calls sequential targets"
        )

    return SubgraphTrace(
        subgraphs=subgraphs,
        oracle_invoked_names=frozenset(tracer.oracle_invoked_names),
    )


class SequentialTracer(HFTracer):
    """
    Get a tracer specialized for the given model. The resulting tracer will not trace
    inside of sequential targets, nor any modules which are not call graph ancestors of
    sequential targets

    During ``create_proxy``, non-ancestor modules that execute (including inside
    ``@torch.fx.wrap`` regions) are recorded on
    ``node.meta[INVOKED_SUBMODULES_META_KEY]`` so subgraph ownership does not
    depend solely on ``call_module`` ops (#3261). Recording is reentrant (stack
    of pending lists) so nested ``create_proxy`` calls cannot wipe parent
    attributions.

    An independent ``oracle_invoked_names`` set records every hooked module whose
    ``forward`` actually ran during ``trace()``, regardless of graph meta, for coverage
    checks.

    :param ancestors: modules which are ancestors of sequential targets
    :param targets: sequential target modules; their descendants are owned via
        ``call_module`` + recurse, so they are not hooked for wrap-hidden discovery
    """

    def __init__(self, ancestors: set[Module], targets: set[Module] | None = None):
        self.ancestors = ancestors
        self.targets = targets if targets is not None else set()
        self._module_to_name: dict[Module, str] = {}
        self._pending_stack: list[list[str]] = []
        self._forward_hook_handles: list[Any] = []
        self.oracle_invoked_names: set[str] = set()

        # skip any mask creation functions not already caught by the autowrapper
        super().__init__(autowrap_functions=_get_autowrap_functions())

    def _modules_under_targets(self) -> set[Module]:
        under: set[Module] = set()
        for target in self.targets:
            under.update(target.modules())
        return under

    def trace(self, root: Module | Callable[..., Any], *args: Any, **kwargs: Any):
        if not isinstance(root, Module):
            return super().trace(root, *args, **kwargs)

        self._module_to_name = {
            module: name for name, module in root.named_modules() if name
        }
        self.oracle_invoked_names = set()
        self._pending_stack = []
        self._forward_hook_handles = []

        try:
            # Only hook modules that can be wrap-hidden: not ancestors (inlined) and not
            # under sequential targets (those are owned by call_module + recurse).
            under_targets = self._modules_under_targets()
            for module, name in self._module_to_name.items():
                if module in self.ancestors or module in under_targets:
                    continue
                # Native autograd hooks (not HooksMixin) so recording still works under
                # HooksMixin.disable_hooks() during calibration tracing.
                self._forward_hook_handles.append(
                    module.register_forward_hook(self._record_invoked_module)
                )
            return super().trace(root, *args, **kwargs)
        finally:
            for handle in self._forward_hook_handles:
                handle.remove()
            self._forward_hook_handles.clear()
            self._pending_stack = []

    def _record_invoked_module(self, module: Module, inputs: Any, output: Any) -> None:
        name = self._module_to_name.get(module)
        if name is None:
            return

        # Oracle: every real forward during trace(), independent of create_proxy meta.
        self.oracle_invoked_names.add(name)

        # Per-node attribution: only while a create_proxy frame is active.
        if self._pending_stack:
            self._pending_stack[-1].append(name)

    def create_proxy(
        self,
        kind: str,
        target: Any,
        args: Any,
        kwargs: Any,
        name: str | None = None,
        type_expr: Any | None = None,
        proxy_factory_fn: Callable[..., Any] | None = None,
    ):
        self._pending_stack.append([])
        try:
            proxy = super().create_proxy(
                kind,
                target,
                args,
                kwargs,
                name=name,
                type_expr=type_expr,
                proxy_factory_fn=proxy_factory_fn,
            )
            pending = self._pending_stack[-1]
            if pending:
                # Preserve first-seen order within this frame.
                seen: set[str] = set()
                ordered: list[str] = []
                for module_name in pending:
                    if module_name in seen:
                        continue
                    seen.add(module_name)
                    ordered.append(module_name)
                proxy.node.meta[INVOKED_SUBMODULES_META_KEY] = ordered
            return proxy
        finally:
            self._pending_stack.pop()

    def create_arg(self, a: Any) -> Argument:
        # special extension allows models which depend on config values to be traced
        if isinstance(a, PretrainedConfig):
            kwargs = {k: self.create_arg(v) for k, v in a.to_dict().items()}
            return self.create_node("call_function", a.__class__, (), kwargs)

        # special extension for supporting `UserDict`s (Gemma4 uses this)
        elif isinstance(a, UserDict):
            return self.create_arg(dict(a))

        else:
            return super().create_arg(a)

    def is_leaf_module(self, module: Module, module_qualified_name: str) -> bool:
        # do not trace non-ancestors; trace sequential ancestors only
        return module not in self.ancestors


def populate_concrete_args(model: Module, sample_input: dict) -> dict:
    """
    Creates concrete args which, unlike the equivalent function provided by
    transformers.utils.fx, creates default values for variadic arguments, which are
    needed by some models.

    :param model: model being traced
    :param sample_input: values used to symbolically trace the model. All arguments
        to the model.forward function which are not in the sample_input are considered
        concrete args
    :return: dictionary mapping concrete argument names to their default values
    """
    sig = inspect.signature(model.forward)

    concrete_args = {}
    for parameter in sig.parameters.values():
        if parameter.name in sample_input:
            continue
        if parameter.kind == inspect._ParameterKind.VAR_POSITIONAL:
            value = list()
        elif parameter.kind == inspect._ParameterKind.VAR_KEYWORD:
            value = dict()
        elif parameter.name == "use_cache":
            value = False
        else:
            value = parameter.default

        concrete_args[parameter.name] = value

    return concrete_args


def find_target_nodes(graph: GraphModule, targets: set[Module]) -> set[Node]:
    """
    Find all nodes whose execution is equivalent to executing the target modules.
    Note that these nodes are guaranteed to be treated as leaf nodes by SequentialTracer

    :param graph: graph containing target nodes
    :param targets: modules whose nodes are being searched for
    :return: set of all nodes which call the target modules
    """
    return set(
        node
        for node in graph.graph.nodes
        if node.op == "call_module" and graph.get_submodule(node.target) in targets
    )


def topological_partition(
    graph: GraphModule, targets: set[Module], targets_per_subgraph: int = 1
) -> list[list[Node]]:
    """
    Partition the graph into partitions such that each `target` belongs to exactly one
    partition and executing each partition depends only on intermediate values produced
    by executing the partitions before it.

    :param graph: graph being partitioned
    :param targets: target modules which will be assigned to disjoint partitions
    :param targets_per_subgraph: number of targets to include per subgraph
    :return: list of partitions, where each partition is a list of nodes belonging to
        that partition
    """
    assert graph_is_well_formed(graph.graph)
    target_nodes = find_target_nodes(graph, targets)

    if targets_per_subgraph <= 0:
        raise ValueError(
            "targets_per_subgraph is required to be greater than or equal to one"
        )

    partitions: list[list[Node]] = [[]]
    node_to_partition: dict[Node, int] = {}
    remaining_indegrees = {
        node: sum(1 for inp in node.all_input_nodes if inp.op != "get_attr")
        for node in graph.graph.nodes
    }
    partition_index = 0  # global counter
    targets_seen = 0  # number of targets encountered so far

    # start with graph input nodes,
    # but delay the `get_attr` nodes as long as possible
    queue = deque(
        node
        for node in graph.graph.nodes
        if remaining_indegrees[node] == 0 and node.op != "get_attr"
    )
    while len(queue) > 0:
        node = queue.popleft()

        is_target = node in target_nodes
        if is_target:
            # put all nodes prior to first target into separate subgraph
            is_head = partition_index == 0 and len(partitions[partition_index]) > 0

            # finish creating subgraph when number of targets has been seen
            is_complete = targets_seen >= targets_per_subgraph

            if is_head or is_complete:
                partition_index += 1
                partitions.append([])
                targets_seen = 0

        # assign to partition
        partitions[partition_index].append(node)
        node_to_partition[node] = partition_index

        # increment after assignment so is_complete fires after the target is placed
        if is_target:
            targets_seen += 1

        # recurse on last indegree only in order to guarantee that
        # the node is assigned to maximal partition
        for user in node.users:
            remaining_indegrees[user] -= 1
            if remaining_indegrees[user] == 0:
                queue.append(user)

    # an ideal implementation would involve implicitly consolidating partition indices
    # so that each node is assigned to the maximum partition possible (in order to delay
    # execution as long as possible), but saving these nodes for last covers the most
    # common and costly case (get_attr)
    for node in graph.graph.find_nodes(op="get_attr"):
        user_partitions = []
        for user in node.users:
            if user in node_to_partition:
                user_partitions.append(node_to_partition[user])

        # workaround
        if len(user_partitions):
            partition_index = min(user_partitions)
            partitions[partition_index].insert(0, node)
            node_to_partition[node] = partition_index

    return partitions


def partition_graph(model: Module, partitions: list[list[Node]]) -> list[Subgraph]:
    """
    Convert each partition into a Subgraph. Each Subgraph returns a dictionary mapping
    of output node names to their computed values. Note that the `consumed_names`
    attribute of each Subgraph remains empty, to be later populated by
    `trace_consumed_names`

    :param model: model which owns the produced Subgraphs
    :param partitions: list of partitions, where each partition is a list of nodes
        belonging to that partition
    :return: list of subgraphs in order of execution
    """
    subgraphs = []

    # create subgraphs
    for partition_nodes in partitions:
        partition_set = set(partition_nodes)

        # create a new graph for the partition
        graph = Graph(model)
        node_map = {}

        # add placeholders for inputs not in this subgraph. use set to deduplicate
        new_input_nodes = {
            input_node
            for node in partition_nodes
            for input_node in node.all_input_nodes
            if input_node not in partition_set and input_node.op
        }
        for input_node in new_input_nodes:
            node_map[input_node] = graph.placeholder(input_node.name)

        # add the nodes to subgraph
        for node in partition_nodes:
            node_map[node] = graph.node_copy(node, lambda n: node_map[n])

        # add an output node to collect all subgraph outputs into a dictionary
        if len(graph.find_nodes(op="output")) <= 0:
            output_dict = {
                node.name: node_map[node]
                for node in partition_nodes
                if any(user not in partition_set for user in node.users.keys())
            }
            graph.output(output_dict)

        # save the subgraph for this partition
        graph.lint()
        input_names = set(node.name for node in graph.nodes if node.op == "placeholder")
        subgraphs.append(
            Subgraph(
                graph=graph,
                input_names=input_names,
                consumed_names=set(),  # populated later
            )
        )

        assert graph_is_well_formed(graph)

    return subgraphs


def trace_consumed_names(subgraphs: list[Subgraph]):
    """
    Populate the `consumed_names` attribute of each Subgraph according to when inputs
    are last used in order to vacate the `intermediates` cache and save memory

    :param subgraphs: list of subgraphs with empty `consumed_names` attributes
    """
    # The first occurrence in reverse order is the final use in execution order.
    seen_names: set[str] = set()
    for subgraph in reversed(subgraphs):
        subgraph.consumed_names.update(subgraph.input_names - seen_names)
        seen_names.update(subgraph.input_names)


def graph_is_well_formed(graph: Graph) -> bool:
    """
    A graph is well formed if and only if
    `nodeA in NodeB.users <=> nodeB in Node.A.all_input_nodes`

    :param graph: graph being checked
    :return: True if the graph is well formed, False otherwise
    """
    for node in graph.nodes:
        for user in node.users:
            if node not in user.all_input_nodes:
                return False

        for input_node in node.all_input_nodes:
            if node not in input_node.users:
                return False

        if len(node.users) != len(set(node.users)) or len(node.all_input_nodes) != len(
            set(node.all_input_nodes)
        ):
            return False

    return True


def add_line_numbers(text: str) -> str:
    lines = text.splitlines()
    numbered_lines = [f"{i + 1} {line}" for i, line in enumerate(lines)]
    return "\n".join(numbered_lines)


def get_sequential_ancestors(model: Module, targets: set[Module]) -> set[Module]:
    """
    Find modules which are call graph ancestors of the given sequential targets

    :param model: model containing sequential targets
    :param targets: sequential targets to find ancestors of
    :return: call graph ancestors of sequential targets
    """
    ancestors = set()

    def is_ancestor(module: Module) -> bool:
        if module in ancestors or module in targets:
            return True

        # eagerly compute list in order to avoid early stopping and :. missing ancestors
        _is_ancestor = any([is_ancestor(child) for child in module.children()])
        if _is_ancestor:
            ancestors.add(module)

        return _is_ancestor

    is_ancestor(model)
    return ancestors


def _get_autowrap_functions() -> tuple[Callable[[Any], Any], ...]:
    try:
        from transformers.masking_utils import LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING

        return tuple(LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING.values())
    except ImportError:
        return tuple()


def handle_sequential_oom(func):
    """Catch ooms and suggest changing sequential targets"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except torch.OutOfMemoryError as e:
            raise torch.OutOfMemoryError(
                "Sequential pipeline ran out of memory. "
                "Please consider choosing a smaller module for `sequential_targets`, "
                "ex. 'Linear' for dense models. "
                "For MoE models with untraceable attention, "
                "use the attention module and individual expert class instead, "
                "ex. sequential_targets=['AttentionClass', 'ExpertClass'] with "
                "sequential_targets_per_subgraph set to batch multiple experts per "
                "subgraph and reduce memory overhead. Choosing a smaller "
                "calibration dataset can also help: reduce `max_seq_length` "
                "(when unset, calibration samples are not truncated, so a few "
                "long samples can dominate memory) or `num_calibration_samples` "
                "(memory scales with sample count for modifiers that cache "
                "activations, such as AWQ)."
            ) from e

    return wrapper
