import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme

from llmcompressor.core import (
    ModuleScopedInitializationFunction,
    PipelineInitializationPlan,
    State,
)
from llmcompressor.modifiers.pruning import SparseGPTModifier
from llmcompressor.modifiers.pruning.constant import ConstantPruningModifier
from llmcompressor.modifiers.quantization import QuantizationModifier


def test_initialization_plan_runs_global_once_and_scopes_incrementally():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    first, second = model
    global_calls = []
    scoped_calls = []

    def initialize_global(context):
        global_calls.append(context.modules)

    def initialize_scope(context):
        scoped_calls.append(context.modules)

    plan = PipelineInitializationPlan(
        [initialize_global, ModuleScopedInitializationFunction(initialize_scope)]
    )

    plan.initialize(State(model=model), model, modules=[first], pipeline="sequential")
    plan.initialize(
        State(model=model), model, modules=[first, second], pipeline="sequential"
    )

    assert global_calls == [None]
    assert scoped_calls == [(first,), (second,)]


def test_eager_initialization_runs_scoped_functions_for_full_model():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    scoped_calls = []
    plan = PipelineInitializationPlan(
        [
            ModuleScopedInitializationFunction(
                lambda context: scoped_calls.append(context.modules)
            )
        ]
    )

    plan.initialize(State(model=model), model, pipeline="sequential")

    assert scoped_calls == [None]


def test_quantization_initialization_is_applied_per_scope():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
    args = QuantizationArgs(
        num_bits=8,
        type="int",
        symmetric=True,
        strategy="tensor",
    )
    modifier = QuantizationModifier(
        config_groups={"group": QuantizationScheme(targets=["Linear"], weights=args)}
    )
    state = State(model=model)
    plan = PipelineInitializationPlan.from_modifiers([modifier], state)

    plan.initialize(state, model, modules=[model[0]], pipeline="sequential")
    assert hasattr(model[0], "quantization_scheme")
    assert not hasattr(model[1], "quantization_scheme")

    plan.initialize(state, model, modules=[model[1]], pipeline="sequential")
    assert hasattr(model[0], "quantization_scheme")
    assert hasattr(model[1], "quantization_scheme")


def test_pruning_initialization_and_start_are_applied_per_scope():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
    model[0].weight.data.zero_()
    modifier = ConstantPruningModifier(targets="Linear", start=-1)
    state = State(model=model)
    plan = PipelineInitializationPlan.from_modifiers([modifier], state)

    plan.initialize(
        state, model, modules=list(model[0].modules()), pipeline="sequential"
    )
    assert torch.count_nonzero(model[0].mask) == 0
    assert not hasattr(model[1], "mask")

    plan.initialize(
        state, model, modules=list(model[1].modules()), pipeline="sequential"
    )
    assert hasattr(model[1], "mask")


def test_owl_sparsegpt_requires_eager_initialization():
    modifier = SparseGPTModifier(
        sparsity=0.5,
        sparsity_profile="owl",
        owl_m=4,
        owl_lmbda=0.1,
    )
    plan = PipelineInitializationPlan.from_modifiers([modifier], State())

    assert plan.requires_eager_initialization
