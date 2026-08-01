import pytest
import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
)

from llmcompressor.observers import mse_quant
from llmcompressor.observers.helpers import flatten_for_calibration
from llmcompressor.observers.mse_quant import _grid_search_eager, _grid_search_mse
from llmcompressor.observers.mse_triton import (
    can_use_triton,
    grid_search_triton,
    neighbors_for_config,
)

GRID = 100.0
MAXSHRINK = 0.20
NORM = 2.4
NO_PATIENCE = 10**6  # larger than the step count, so it never fires

def _args(num_bits=8, qtype="int", **kwargs):
    return QuantizationArgs(
        num_bits=num_bits,
        type=qtype,
        symmetric=True,
        strategy=QuantizationStrategy.GROUP,
        group_size=kwargs.pop("group_size", 128),
        **kwargs,
    )


@pytest.fixture
def requires_kernel_device():
    """Skip unless this box has a device the kernel actually dispatches on.

    A plain cuda.is_available() check is not the same question: production
    also declines ROCm and compute capability below 8.0, so on a T4, a V100
    or a ROCm box these tests would fail while the kernel is correctly not
    being used. Asking can_use_triton keeps the two in step.

    A fixture rather than a module-level marker so collection does not
    initialise CUDA just to decide whether to skip.
    """
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    probe = torch.zeros(1, 2, 2, 32, device="cuda")
    if not can_use_triton(probe, _args(8, "int", group_size=32)):
        pytest.skip(
            "this device is not one the kernel dispatches on "
            "(needs triton, CUDA, compute capability 8.0+)"
        )


CUDA = pytest.mark.usefixtures("requires_kernel_device")


def _eager(observed, args):
    """The reference: full grid, no early stopping."""
    token_args = args.model_copy(
        update={"strategy": QuantizationStrategy.TOKEN}
    )
    min_val = torch.amin(observed, dim=(0, -1))
    max_val = torch.amax(observed, dim=(0, -1))
    return _grid_search_eager(
        observed,
        args,
        token_args,
        min_val,
        max_val,
        torch.full_like(min_val, torch.finfo(min_val.dtype).max),
        min_val.clone(),
        max_val.clone(),
        int(MAXSHRINK * GRID),
        NO_PATIENCE,
        GRID,
        NORM,
    )


@CUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "num_bits,qtype", [(8, "int"), (4, "int"), (8, "float")]
)
def test_matches_eager_full_grid(dtype, num_bits, qtype):
    """The kernel has to pick the same range the eager search picks.

    Not "close to": the same. Everything the kernel is allowed to run on is
    a format where it reproduces eager exactly, and can_use_triton exists to
    keep it off everything else.
    """
    torch.manual_seed(0)
    observed = torch.randn(1, 64, 8, 128, device="cuda", dtype=dtype)
    args = _args(num_bits, qtype)

    assert can_use_triton(observed, args)
    got_min, got_max = grid_search_triton(observed, args, MAXSHRINK, GRID, NORM)
    want_min, want_max = _eager(observed, args)

    assert torch.equal(got_min, want_min)
    assert torch.equal(got_max, want_max)


@CUDA
def test_matches_eager_at_channel_granularity():
    """CHANNEL flattens to one group per row, which is still 4d."""
    torch.manual_seed(0)
    observed = torch.randn(1, 32, 1, 256, device="cuda")
    args = QuantizationArgs(
        num_bits=8, type="int", symmetric=True,
        strategy=QuantizationStrategy.CHANNEL,
    )
    assert can_use_triton(observed, args)
    got = grid_search_triton(observed, args, MAXSHRINK, GRID, NORM)
    want = _eager(observed, args)
    assert torch.equal(got[0], want[0])
    assert torch.equal(got[1], want[1])


@CUDA
def test_subnormal_scales_still_match():
    """Scales below the smallest normal void the neighbor bound.

    The spacing there is absolute rather than relative, so a scale can halve
    between steps; enumerating every representable half scale finds a worst
    ratio of 0.5, which no small neighbor count covers. The kernel is
    supposed to notice and run its full search for those steps.
    """
    torch.manual_seed(0)
    observed = (
        torch.randn(1, 32, 4, 128, device="cuda") * 1e-6
    ).to(torch.float16)
    args = _args(4, "int")
    scale = observed.abs().amax(dim=(0, -1)) / 7.5
    assert (scale < torch.finfo(torch.float16).tiny).any(), (
        "test data does not actually reach the subnormal range"
    )

    got = grid_search_triton(observed, args, MAXSHRINK, GRID, NORM)
    want = _eager(observed, args)
    assert torch.equal(got[0], want[0])
    assert torch.equal(got[1], want[1])


@pytest.mark.parametrize(
    "observed,args,why",
    [
        (
            torch.zeros(1, 1, 8192),
            QuantizationArgs(
                num_bits=8, type="int", symmetric=True,
                strategy=QuantizationStrategy.TENSOR,
            ),
            "TENSOR flattens to 3d and the launcher reads four dims",
        ),
        (
            torch.zeros(1, 8, 4, 32),
            QuantizationArgs(
                num_bits=8, type="int", symmetric=False,
                strategy=QuantizationStrategy.GROUP, group_size=32,
            ),
            "asymmetric needs a zero point the codebook search has none of",
        ),
        (
            torch.zeros(1, 8, 4, 32),
            QuantizationArgs(
                num_bits=4, type="float", symmetric=True,
                strategy=QuantizationStrategy.GROUP, group_size=32,
            ),
            "float4 rounds through bfloat16 in compressed_tensors",
        ),
        (
            torch.zeros(1, 8, 4, 32, dtype=torch.float64),
            QuantizationArgs(
                num_bits=8, type="int", symmetric=True,
                strategy=QuantizationStrategy.GROUP, group_size=32,
            ),
            "float64 is not one of the dtypes the error path reproduces",
        ),
        (
            torch.zeros(1, 8, 4, 32),
            QuantizationArgs(
                num_bits=16, type="int", symmetric=True,
                strategy=QuantizationStrategy.GROUP, group_size=32,
            ),
            "int16 builds 65536 codes and was never validated",
        ),
        (
            torch.zeros(1, 8, 4, 32),
            QuantizationArgs(
                num_bits=8, type="float", symmetric=True,
                strategy=QuantizationStrategy.GROUP, group_size=32,
                scale_dtype=torch.float8_e4m3fn,
            ),
            "a rounded scale is a different scale path",
        ),
        (
            torch.zeros(1, 8, 1, 32),
            QuantizationArgs(
                num_bits=8, type="float", symmetric=True,
                strategy=QuantizationStrategy.GROUP, group_size=32,
                scale_dtype=torch.uint8,
            ),
            "MX derives the scale as an exponent, not by the symmetric formula",
        ),
        (
            torch.zeros(1, 8, 1, 4096),
            QuantizationArgs(
                num_bits=8, type="int", symmetric=True,
                strategy=QuantizationStrategy.CHANNEL,
            ),
            "a 4096 wide group would ask one program to hold 4096 elements",
        ),
    ],
)
def test_declines_what_it_cannot_reproduce(observed, args, why):
    """Each of these must fall back rather than silently differ."""
    if torch.cuda.is_available():
        observed = observed.cuda()
    assert not can_use_triton(observed, args), why


def test_declines_without_cuda():
    """A CPU tensor keeps the eager path even where triton is installed."""
    assert not can_use_triton(torch.zeros(1, 8, 4, 32), _args(8, "int", group_size=32))


@pytest.mark.parametrize(
    "dtype,expect_at_least",
    [(torch.float32, 2), (torch.bfloat16, 3), (torch.float16, 2)],
)
def test_bound_widens_for_coarser_scales(dtype, expect_at_least):
    """Rounding the scale makes steps move unevenly, so the bound must grow.

    int8 at grid 100 / maxshrink 0.2 needs 2 neighbors when the scale is
    exact. Rounding it to bf16 lets two consecutive scales drift further
    apart than 1/grid, and the measured per-step code index jump goes to 3.
    """
    n = neighbors_for_config(_args(8, "int"), GRID, MAXSHRINK, dtype)
    assert n >= expect_at_least


def test_bound_is_config_only():
    """N must not depend on the data.

    NUM_NEIGHBORS is a tl.constexpr, so a data-dependent bound would compile
    a separate kernel per layer, and reading a per-step ratio back to the
    host to compute it would sync inside the launcher.
    """
    args = _args(8, "int")
    first = neighbors_for_config(args, GRID, MAXSHRINK, torch.bfloat16)
    second = neighbors_for_config(args, GRID, MAXSHRINK, torch.bfloat16)
    assert first == second
    assert isinstance(first, int)


@CUDA
def test_grid_search_mse_actually_dispatches(monkeypatch):
    """The public entry point has to reach the kernel, not just the kernel.

    Testing grid_search_triton directly says nothing about whether
    _grid_search_mse ever calls it; a wrong condition in can_use_triton would
    leave every one of those tests green while the kernel never ran.
    """
    torch.manual_seed(0)
    observed = torch.randn(1, 32, 4, 128, device="cuda", dtype=torch.bfloat16)
    args = _args(8, "int")
    token_args = args.model_copy(
        update={"strategy": QuantizationStrategy.TOKEN}
    )

    calls = []
    real = mse_quant.grid_search_triton
    monkeypatch.setattr(
        mse_quant,
        "grid_search_triton",
        lambda *a, **k: (calls.append(1), real(*a, **k))[1],
    )

    got = _grid_search_mse(
        observed, args, token_args, MAXSHRINK, 5, GRID, NORM, 5
    )
    assert calls, "_grid_search_mse did not take the triton path"

    want = _eager(observed, args)
    assert torch.equal(got[0], want[0])
    assert torch.equal(got[1], want[1])


def test_grid_search_mse_falls_back_off_gpu(monkeypatch):
    """A CPU tensor must not reach the kernel at all.

    Comparing results would pass even if the kernel ran and happened to
    agree, so the kernel is replaced with something that raises: the only
    way this completes is by not calling it.
    """
    def _explode(*args, **kwargs):
        raise AssertionError("the kernel was called for a CPU tensor")

    monkeypatch.setattr(mse_quant, "grid_search_triton", _explode)

    torch.manual_seed(0)
    observed = torch.randn(1, 8, 2, 32)
    args = _args(8, "int", group_size=32)
    token_args = args.model_copy(
        update={"strategy": QuantizationStrategy.TOKEN}
    )
    got = _grid_search_mse(
        observed, args, token_args, MAXSHRINK, 10**6, GRID, NORM, 5
    )
    want = _eager(observed, args)
    assert torch.equal(got[0], want[0])
    assert torch.equal(got[1], want[1])


@CUDA
@pytest.mark.parametrize("maxshrink,grid", [(0.0, 100.0), (0.001, 100.0), (0.2, 0.0)])
def test_degenerate_grid_returns_the_unshrunk_range(maxshrink, grid):
    """int(maxshrink * grid) can be zero, and grid can be zero.

    With no steps to search the loop that fills the result never runs, so
    starting it from empty_like would return uninitialised memory, and
    1.0 / grid would raise. Eager returns the unshrunk range here.
    """
    torch.manual_seed(0)
    observed = torch.randn(1, 8, 2, 32, device="cuda")
    args = _args(8, "int", group_size=32)

    got_min, got_max = grid_search_triton(observed, args, maxshrink, grid, NORM)
    assert torch.equal(got_min, torch.amin(observed, dim=(0, -1)))
    assert torch.equal(got_max, torch.amax(observed, dim=(0, -1)))


@CUDA
def test_declines_stacked_observations():
    """The kernel reads one observation; eager reduces over all of them."""
    observed = torch.zeros(4, 8, 2, 32, device="cuda")
    assert not can_use_triton(observed, _args(8, "int", group_size=32))


@CUDA
@pytest.mark.parametrize(
    "strategy_args",
    [
        dict(strategy=QuantizationStrategy.TENSOR_GROUP, group_size=32),
        dict(strategy=QuantizationStrategy.BLOCK, block_structure=[8, 32]),
    ],
    ids=["tensor_group", "block"],
)
def test_matches_eager_on_other_dispatched_layouts(strategy_args):
    """Layouts the gate lets through but the parity tests did not cover.

    can_use_triton keys off the flattened rank and group width, not the
    strategy name, so anything that lands 4d with a small enough group is
    dispatched. These two do; they should be held to the same bar as GROUP.
    """
    torch.manual_seed(0)
    weight = torch.randn(64, 128, device="cuda")
    args = QuantizationArgs(
        num_bits=8, type="int", symmetric=True, **strategy_args
    )
    observed = flatten_for_calibration(weight, "weight", args)

    assert can_use_triton(observed, args), (
        f"{strategy_args} no longer dispatches; the parity claim below would "
        "be vacuous"
    )
    got = grid_search_triton(observed, args, MAXSHRINK, GRID, NORM)
    want = _eager(observed, args)
    assert torch.equal(got[0], want[0])
    assert torch.equal(got[1], want[1])
