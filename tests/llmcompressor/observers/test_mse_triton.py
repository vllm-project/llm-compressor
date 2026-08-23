import pytest
import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
)
from compressed_tensors.quantization.lifecycle import fake_quantize
from compressed_tensors.quantization.utils import calculate_qparams

from llmcompressor.observers import mse_quant
from llmcompressor.observers.helpers import flatten_for_calibration
from llmcompressor.observers.mse_quant import _grid_search_eager, _grid_search_mse
from llmcompressor.observers.mse_triton import (
    _codebook,
    can_use_triton,
    grid_search_triton,
    neighbors_for_config,
)
from tests.testing_utils import get_accelerator_type

GRID = 100.0
MAXSHRINK = 0.20
NORM = 2.4
NO_PATIENCE = 10**6  # larger than the step count, so it never fires


def _args(num_bits=8, qtype="int", **kwargs):
    return QuantizationArgs(
        num_bits=num_bits,
        type=qtype,
        symmetric=kwargs.pop("symmetric", True),
        strategy=kwargs.pop("strategy", QuantizationStrategy.GROUP),
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
    if not torch.accelerator.is_available() or get_accelerator_type() != "cuda":
        pytest.skip("no CUDA device")
    probe = torch.zeros(1, 2, 2, 32, device="cuda")
    if not can_use_triton(probe, _args(8, "int", group_size=32)):
        pytest.skip(
            "this device is not one the kernel dispatches on "
            "(needs triton, CUDA, compute capability 8.0+)"
        )


CUDA = pytest.mark.usefixtures("requires_kernel_device")


def _reference_search(observed, args, patience=NO_PATIENCE, expand=1.0):
    """Brute-force the normalized objective implemented by the kernel.

    This shares the kernel's math on purpose - it exists to catch
    implementation bugs (indexing, the binary search, the patience counter),
    not design ones. A term missing from both sides cancels here, which is
    how the zero point went unnoticed once; the fake_quantize-scored test
    below is the one that cannot cancel.
    """
    min_val = torch.amin(observed, dim=(0, -1)) * expand
    max_val = torch.amax(observed, dim=(0, -1)) * expand
    scale, zero_point = calculate_qparams(
        min_vals=min_val,
        max_vals=max_val,
        quantization_args=args,
        global_scale=None,
    )
    scale = scale.float().unsqueeze(0).unsqueeze(-1)
    zero_point = zero_point.float().unsqueeze(0).unsqueeze(-1)
    codes = _codebook(args).to(observed.device)
    best_error = torch.full_like(min_val, float("inf"), dtype=torch.float32)
    best_step = torch.zeros_like(min_val, dtype=torch.long)
    stalled = torch.zeros_like(min_val, dtype=torch.long)
    patience = max(int(patience), 1)
    total_steps = int(MAXSHRINK * GRID)
    errors = []

    for step in range(total_steps):
        active = stalled < patience
        p = 1.0 - step / GRID
        step_scale = scale * p
        selected = observed.float() / step_scale + zero_point
        distance = (selected.unsqueeze(-1) - codes).abs().amin(-1)
        error = (distance * step_scale).pow(NORM).sum(dim=(0, -1))
        errors.append(error)
        improved = active & (error < best_error)
        best_error = torch.where(improved, error, best_error)
        best_step = torch.where(improved, step, best_step)
        stalled = torch.where(active, torch.where(improved, 0, stalled + 1), stalled)

    ps = torch.tensor(
        [1.0 - step / GRID for step in range(total_steps)],
        dtype=min_val.dtype,
        device=min_val.device,
    )
    best_p = ps[best_step]
    return min_val * best_p, max_val * best_p, best_step, torch.stack(errors)


def _returned_objective(observed, chosen, errors, expand=1.0):
    """Best reference objective among steps represented by this range."""
    max_val = torch.amax(observed, dim=(0, -1)) * expand
    ps = torch.tensor(
        [1.0 - step / GRID for step in range(int(MAXSHRINK * GRID))],
        dtype=max_val.dtype,
        device=max_val.device,
    )
    candidates = ps[:, None, None] * max_val[None]
    compatible = candidates == chosen[1][None]
    assert bool(compatible.any(0).all())
    return errors.masked_fill(~compatible, float("inf")).amin(0)


def _eager(observed, args, patience=NO_PATIENCE):
    """The reference implementation; full grid unless told otherwise."""
    token_args = args.model_copy(update={"strategy": QuantizationStrategy.TOKEN})
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
        patience,
        GRID,
        NORM,
    )


@CUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "num_bits,qtype,symmetric,scale_dtype",
    [
        pytest.param(8, "int", True, None, id="int8-symmetric"),
        pytest.param(4, "int", True, None, id="int4-symmetric"),
        pytest.param(8, "int", False, None, id="int8-asymmetric"),
        pytest.param(4, "int", False, None, id="int4-asymmetric"),
        pytest.param(8, "float", True, None, id="fp8"),
        pytest.param(4, "float", True, torch.float8_e4m3fn, id="fp4-e2m1"),
    ],
)
def test_matches_normalized_reference(dtype, num_bits, qtype, symmetric, scale_dtype):
    torch.manual_seed(0)
    observed = torch.randn(1, 16, 4, 128, device="cuda", dtype=dtype)
    args = _args(
        num_bits,
        qtype,
        symmetric=symmetric,
        scale_dtype=scale_dtype,
    )

    assert can_use_triton(observed, args)
    got = grid_search_triton(observed, args, MAXSHRINK, NO_PATIENCE, GRID, NORM)
    want = _reference_search(observed, args)
    got_error = _returned_objective(observed, got, want[3])
    torch.testing.assert_close(got_error, want[3].amin(0))


def _true_objective(observed, args, min_val, max_val):
    """The error the model actually sees for a range, scored in double.

    Unlike _reference_search this goes through fake_quantize with the
    qparams the range produces, so nothing the kernel and the reference
    might both omit can cancel out of it.
    """
    token_args = args.model_copy(update={"strategy": QuantizationStrategy.TOKEN})
    scale, zp = calculate_qparams(
        min_vals=min_val,
        max_vals=max_val,
        quantization_args=args,
        global_scale=None,
    )
    q = fake_quantize(observed, scale.unsqueeze(-1), zp.unsqueeze(-1), token_args)
    return (q.double() - observed.double()).abs().pow(NORM).sum(dim=(0, -1))


@CUDA
@pytest.mark.parametrize("patience", [5, NO_PATIENCE], ids=["p5", "full"])
@pytest.mark.parametrize(
    "num_bits,qtype,symmetric,scale_dtype",
    [
        pytest.param(8, "int", False, None, id="int8-asym"),
        pytest.param(4, "int", False, None, id="int4-asym"),
        pytest.param(8, "int", True, None, id="int8-sym"),
        pytest.param(8, "float", True, None, id="fp8"),
        pytest.param(4, "float", True, torch.float8_e4m3fn, id="fp4-e2m1"),
    ],
)
def test_range_survives_the_real_objective(
    num_bits, qtype, symmetric, scale_dtype, patience
):
    """The chosen range has to hold up under fake_quantize, not just the
    kernel's own metric.

    The reference search shares the kernel's arithmetic, so a term missing
    from both - the zero point was one - cancels there and every comparison
    stays green while the model quality craters. Before the zp shift this
    ratio was 179x on exactly this shape; the bar is far below that and far
    above the fp32-vs-dtype noise the normalized form legitimately carries.

    Every codebook the gate accepts appears here, not only the int ones:
    fp4's midpoint rounding and fp8-scale staircase are exactly the kind of
    shared term the reference cannot see, so they need the fake_quantize
    scoring most. The p5 rows hold the production default patience to the
    same bar; the reference stays eager full grid either way, since early
    stopping is only allowed to cost, not to redefine the target.
    """
    torch.manual_seed(0)
    observed = torch.randn(1, 32, 8, 128, device="cuda", dtype=torch.bfloat16)
    args = _args(num_bits, qtype, symmetric=symmetric, scale_dtype=scale_dtype)

    assert can_use_triton(observed, args)
    got = grid_search_triton(observed, args, MAXSHRINK, patience, GRID, NORM)
    want = _eager(observed, args)

    ours = float(_true_objective(observed, args, *got).sum())
    theirs = float(_true_objective(observed, args, *want).sum())
    assert (
        ours <= 1.5 * theirs
    ), f"kernel range scores {ours / theirs:.2f}x eager under fake_quantize"


@CUDA
@pytest.mark.parametrize("patience", [1, 5, 10, 15])
def test_patience_matches_reference(patience):
    torch.manual_seed(0)
    observed = torch.randn(1, 16, 4, 128, device="cuda", dtype=torch.bfloat16)
    args = _args(4, "int")

    got = grid_search_triton(observed, args, MAXSHRINK, patience, GRID, NORM)
    want = _reference_search(observed, args, patience)
    assert torch.equal(got[0], want[0])
    assert torch.equal(got[1], want[1])


@CUDA
def test_expand_widens_the_searched_range():
    """expand scales the starting range the way eager does.

    The shrink grid is unchanged, it just runs from a wider range, so the
    whole kernel is untouched and only the launcher's min/max move. Checked
    against the reference at the same expand, and against itself at
    expand=1.0 to make sure the argument is not being ignored.
    """
    torch.manual_seed(0)
    observed = torch.randn(1, 16, 4, 128, device="cuda", dtype=torch.bfloat16)
    args = _args(4, "float", scale_dtype=torch.float8_e4m3fn)

    plain = grid_search_triton(observed, args, MAXSHRINK, NO_PATIENCE, GRID, NORM)
    wide = grid_search_triton(
        observed, args, MAXSHRINK, NO_PATIENCE, GRID, NORM, expand=1.5
    )
    assert not torch.equal(plain[1], wide[1]), "expand changed nothing"

    want = _reference_search(observed, args, expand=1.5)
    got_error = _returned_objective(observed, wide, want[3], expand=1.5)
    torch.testing.assert_close(got_error, want[3].amin(0))


@CUDA
def test_patience_actually_fires():
    """A patience low enough to bite has to change the answer.

    Without this the invariant above is satisfied by a kernel that ignores
    the argument entirely.
    """
    torch.manual_seed(0)
    observed = torch.randn(1, 32, 8, 128, device="cuda", dtype=torch.bfloat16)
    args = _args(4, "int")

    full = grid_search_triton(observed, args, MAXSHRINK, NO_PATIENCE, GRID, NORM)
    early = grid_search_triton(observed, args, MAXSHRINK, 1, GRID, NORM)
    assert not torch.equal(
        full[0], early[0]
    ), "patience=1 chose the same range everywhere as the full grid"


@CUDA
def test_dead_groups_do_not_produce_nan():
    """An all-zero group divides by the eps calculate_qparams substitutes.

    The kernel has no zero-scale guard of its own, on purpose: the host
    qparam call already replaces an exactly zero scale with the dtype eps. If
    that ever stopped being true this would divide by zero, every step would
    score NaN, and NaN < inf is false, so the group would silently keep step
    0 rather than fail. Pruned weights make this a real input, not a corner.
    """
    torch.manual_seed(0)
    observed = torch.randn(1, 8, 4, 128, device="cuda", dtype=torch.bfloat16)
    observed[:, ::2] = 0.0
    args = _args(8, "int")

    got_min, got_max = grid_search_triton(
        observed, args, MAXSHRINK, NO_PATIENCE, GRID, NORM
    )
    assert torch.isfinite(got_min).all() and torch.isfinite(got_max).all()
    # a dead group has one range, and shrinking zero leaves it there
    assert bool((got_min[::2] == 0).all() and (got_max[::2] == 0).all())
    want = _eager(observed, args)
    assert torch.equal(got_min[::2], want[0][::2])
    assert torch.equal(got_max[::2], want[1][::2])


@CUDA
def test_zero_patience_means_one_not_none():
    """patience=0 has to keep searching, on both paths.

    Eager increments its counter before testing it, so the value is at least
    1 by then and 0 behaves exactly like 1 - it still walks the grid. The
    kernel tests its counter before evaluating, so an unclamped 0 would
    search nothing and hand back the unshrunk range while eager handed back
    a searched one. The observer kwarg accepts 0, so users do pass it.
    """
    torch.manual_seed(0)
    observed = torch.randn(1, 8, 4, 128, device="cuda", dtype=torch.bfloat16)
    args = _args(8, "int")

    got = grid_search_triton(observed, args, MAXSHRINK, 0, GRID, NORM)
    one = grid_search_triton(observed, args, MAXSHRINK, 1, GRID, NORM)
    assert torch.equal(got[0], one[0]) and torch.equal(got[1], one[1])

    # and it is not the degenerate answer
    unshrunk = torch.amin(observed, dim=(0, -1))
    assert not torch.equal(got[0], unshrunk), "patience=0 searched nothing"

    # eager agrees that 0 and 1 are the same setting
    e0, e1 = _eager(observed, args, patience=0), _eager(observed, args, 1)
    assert torch.equal(e0[0], e1[0]) and torch.equal(e0[1], e1[1])


@CUDA
def test_repeated_calls_agree():
    """Nothing in the launcher may return uninitialised memory.

    best_step is zeroed and _restore_range is seeded rather than allocated
    empty, but both are easy to regress into an empty_like, and the result
    would still look plausible on a single run.
    """
    torch.manual_seed(0)
    observed = torch.randn(1, 16, 4, 128, device="cuda", dtype=torch.bfloat16)
    args = _args(4, "int")
    first = grid_search_triton(observed, args, MAXSHRINK, 5, GRID, NORM)
    for _ in range(3):
        again = grid_search_triton(observed, args, MAXSHRINK, 5, GRID, NORM)
        assert torch.equal(first[0], again[0])
        assert torch.equal(first[1], again[1])


@CUDA
def test_masked_lanes_do_not_leak_into_the_error():
    """group_size 96 pads BLOCK_G to 128, so 32 lanes run masked.

    Every other test uses power-of-two groups, where the mask is all true,
    which leaves the masked load's ``other=0.0`` and the masked sum as the
    one kernel path with no coverage. A leak there would add phantom
    zero-valued elements to every non-power-of-two group's error.
    """
    torch.manual_seed(0)
    observed = torch.randn(1, 16, 2, 96, device="cuda", dtype=torch.bfloat16)
    args = _args(8, "int", group_size=96)

    assert can_use_triton(observed, args)
    got = grid_search_triton(observed, args, MAXSHRINK, NO_PATIENCE, GRID, NORM)
    want = _reference_search(observed, args)
    assert torch.equal(got[0], want[0])
    assert torch.equal(got[1], want[1])


@CUDA
def test_channel_layout_matches_reference():
    """CHANNEL flattens to one group per row, which is still 4d."""
    torch.manual_seed(0)
    observed = torch.randn(1, 32, 1, 256, device="cuda")
    args = QuantizationArgs(
        num_bits=8,
        type="int",
        symmetric=True,
        strategy=QuantizationStrategy.CHANNEL,
    )
    assert can_use_triton(observed, args)
    got = grid_search_triton(observed, args, MAXSHRINK, NO_PATIENCE, GRID, NORM)
    want = _reference_search(observed, args)
    assert torch.equal(got[0], want[0])
    assert torch.equal(got[1], want[1])


@CUDA
def test_tiny_values_match_reference():
    torch.manual_seed(0)
    observed = (torch.randn(1, 32, 4, 128, device="cuda") * 1e-6).to(torch.float16)
    args = _args(4, "int")
    scale = observed.abs().amax(dim=(0, -1)) / 7.5
    assert (
        scale < torch.finfo(torch.float16).tiny
    ).any(), "test data does not actually reach the subnormal range"

    got = grid_search_triton(observed, args, MAXSHRINK, NO_PATIENCE, GRID, NORM)
    want = _reference_search(observed, args)
    assert torch.equal(got[0], want[0])
    assert torch.equal(got[1], want[1])


@pytest.mark.parametrize(
    "observed,args,why",
    [
        (
            torch.zeros(1, 1, 8192),
            QuantizationArgs(
                num_bits=8,
                type="int",
                symmetric=True,
                strategy=QuantizationStrategy.TENSOR,
            ),
            "TENSOR flattens to 3d and the launcher reads four dims",
        ),
        (
            torch.zeros(1, 8, 4, 32, dtype=torch.float64),
            QuantizationArgs(
                num_bits=8,
                type="int",
                symmetric=True,
                strategy=QuantizationStrategy.GROUP,
                group_size=32,
            ),
            "float64 is not one of the dtypes this has been run on",
        ),
        (
            torch.zeros(1, 8, 4, 32),
            QuantizationArgs(
                num_bits=16,
                type="int",
                symmetric=True,
                strategy=QuantizationStrategy.GROUP,
                group_size=32,
            ),
            "int16 builds 65536 codes and was never validated",
        ),
        (
            torch.zeros(1, 8, 1, 32),
            QuantizationArgs(
                num_bits=8,
                type="float",
                symmetric=True,
                strategy=QuantizationStrategy.GROUP,
                group_size=32,
                scale_dtype=torch.uint8,
            ),
            "MX derives the scale as an exponent, not by the symmetric formula",
        ),
        (
            torch.zeros(1, 8, 4, 32),
            QuantizationArgs(
                num_bits=8,
                type="int",
                symmetric=True,
                strategy=QuantizationStrategy.GROUP,
                group_size=32,
                scale_dtype=torch.float8_e4m3fn,
            ),
            "an fp8 scale on an int codebook answers to no scheme, and the "
            "rounded per-step scale is a staircase scale_base * p is not",
        ),
        (
            torch.zeros(1, 8, 4, 32),
            QuantizationArgs(
                num_bits=8,
                type="float",
                symmetric=True,
                strategy=QuantizationStrategy.GROUP,
                group_size=32,
                scale_dtype=torch.float8_e4m3fn,
            ),
            "fp8 scale storage exists for fp4 schemes, not fp8 codebooks",
        ),
        (
            torch.zeros(1, 8, 4, 32),
            QuantizationArgs(
                num_bits=8,
                type="float",
                symmetric=False,
                strategy=QuantizationStrategy.GROUP,
                group_size=32,
            ),
            "no scheme quantizes float asymmetrically; fp4-asym raises "
            "inside calculate_qparams and nothing exercises fp8-asym",
        ),
        (
            torch.zeros(1, 8, 1, 4096),
            QuantizationArgs(
                num_bits=8,
                type="int",
                symmetric=True,
                strategy=QuantizationStrategy.CHANNEL,
            ),
            "a 4096 wide group would ask one program to hold 4096 elements",
        ),
    ],
)
def test_declines_what_it_cannot_reproduce(observed, args, why):
    """Each of these must fall back rather than silently differ."""
    if torch.accelerator.is_available():
        observed = observed.to(torch.accelerator.current_accelerator())
    assert not can_use_triton(observed, args), why


def test_declines_without_cuda():
    """A CPU tensor keeps the eager path even where triton is installed."""
    assert not can_use_triton(torch.zeros(1, 8, 4, 32), _args(8, "int", group_size=32))


def test_codebooks_cover_symmetric_asymmetric_and_fp4():
    symmetric = _codebook(_args(8, "int"))
    asymmetric = _codebook(_args(8, "int", symmetric=False))
    fp4 = _codebook(_args(4, "float", scale_dtype=torch.float8_e4m3fn))

    assert (symmetric[0].item(), symmetric[-1].item()) == (-128.0, 127.0)
    assert (asymmetric[0].item(), asymmetric[-1].item()) == (-128.0, 127.0)
    assert fp4.tolist() == [
        -6.0,
        -4.0,
        -3.0,
        -2.0,
        -1.5,
        -1.0,
        -0.5,
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
    ]


@pytest.mark.parametrize(
    "num_bits,qtype,symmetric,scale_dtype",
    [
        (8, "int", True, None),
        (4, "int", True, None),
        (8, "int", False, None),
        (4, "int", False, None),
        (8, "float", True, None),
        (4, "float", True, torch.float8_e4m3fn),
    ],
)
def test_bound_covers_the_jump_the_data_makes(num_bits, qtype, symmetric, scale_dtype):
    """N has to be at least the largest step-to-step move that occurs.

    The bound is derived from cutoff ratios rather than measured, so this
    walks the grid in torch and checks the derivation against what actually
    happens. A bound one short is invisible in a timing run and shows up
    only as a group that quietly picked a worse range.
    """
    args = _args(
        num_bits,
        qtype,
        symmetric=symmetric,
        scale_dtype=scale_dtype,
    )
    codes = _codebook(args)
    n = neighbors_for_config(args, GRID, MAXSHRINK)

    # the drift a value experiences is proportional to its distance from the
    # zero point, so the walk has to try the positions that maximise it -
    # both ends of the code range - not only the symmetric zp of 0
    zps = [0.0] if symmetric else [float(codes[0]), 0.0, float(codes[-1])]

    worst = 0
    for zp in zps:
        u = torch.linspace(float(codes[0]) - zp - 1, float(codes[-1]) - zp + 1, 8192)
        prev = None
        for i in range(int(MAXSHRINK * GRID)):
            p = 1.0 - i / GRID
            sel = u.unsqueeze(-1) / p + zp
            idx = (sel - codes).abs().argmin(-1)
            if prev is not None:
                worst = max(worst, int((idx - prev).abs().max()))
            prev = idx
    assert n >= worst, f"bound {n} but the data moved {worst}"


def test_bound_is_config_only():
    """N must not depend on the data.

    NUM_NEIGHBORS is a tl.constexpr, so a data-dependent bound would compile
    a separate kernel per layer, and reading a per-step ratio back to the
    host to compute it would sync inside the launcher.
    """
    args = _args(8, "int")
    first = neighbors_for_config(args, GRID, MAXSHRINK)
    second = neighbors_for_config(args, GRID, MAXSHRINK)
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
    token_args = args.model_copy(update={"strategy": QuantizationStrategy.TOKEN})

    calls = []
    real = mse_quant.grid_search_triton
    monkeypatch.setattr(
        mse_quant,
        "grid_search_triton",
        lambda *a, **k: (calls.append((a, k)), real(*a, **k))[1],
    )

    got = _grid_search_mse(
        observed, args, token_args, MAXSHRINK, 5, GRID, NORM, 5, expand=1.25
    )
    assert calls, "_grid_search_mse did not take the triton path"
    a, k = calls[0]
    assert a[3] == 5, "patience was not forwarded to the kernel"
    # the kernel computes its own min/max, so a dispatch that dropped expand
    # would silently search the unexpanded range while eager searched the
    # expanded one
    assert k.get("expand") == 1.25, "expand was not forwarded to the kernel"
    want = _reference_search(observed, args, patience=5, expand=1.25)
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
    token_args = args.model_copy(update={"strategy": QuantizationStrategy.TOKEN})
    got = _grid_search_mse(
        observed, args, token_args, MAXSHRINK, NO_PATIENCE, GRID, NORM, 5
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

    got_min, got_max = grid_search_triton(
        observed, args, maxshrink, NO_PATIENCE, grid, NORM
    )
    assert torch.equal(got_min, torch.amin(observed, dim=(0, -1)))
    assert torch.equal(got_max, torch.amax(observed, dim=(0, -1)))


@CUDA
def test_declines_stacked_observations():
    """The kernel reads one observation; eager reduces over all of them."""
    observed = torch.zeros(4, 8, 2, 32, device="cuda")
    assert not can_use_triton(observed, _args(8, "int", group_size=32))


@CUDA
@pytest.mark.parametrize("symmetric", [True, False], ids=["sym", "asym"])
@pytest.mark.parametrize(
    "strategy_args",
    [
        dict(strategy=QuantizationStrategy.TENSOR_GROUP, group_size=32),
        dict(strategy=QuantizationStrategy.BLOCK, block_structure=[8, 32]),
    ],
    ids=["tensor_group", "block"],
)
def test_other_dispatched_layouts_match_reference(strategy_args, symmetric):
    """Layouts the gate lets through but the main test did not cover.

    can_use_triton keys off the flattened rank and group width, not the
    strategy name, so anything that lands 4d with a small enough group is
    dispatched. These should be held to the same bar as GROUP, and in both
    symmetries: symmetric pins the zero point to zero, so only the
    asymmetric rows exercise the kernel's zp gather under these layouts'
    flattening - a stride mismatch there would read zeros and pass every
    symmetric test.
    """
    torch.manual_seed(0)
    weight = torch.randn(64, 128, device="cuda")
    args = QuantizationArgs(
        num_bits=8, type="int", symmetric=symmetric, **strategy_args
    )
    observed = flatten_for_calibration(weight, "weight", args)

    assert can_use_triton(observed, args), (
        f"{strategy_args} no longer dispatches; the claim below would " "be vacuous"
    )
    got = grid_search_triton(observed, args, MAXSHRINK, NO_PATIENCE, GRID, NORM)
    want = _reference_search(observed, args)
    assert torch.equal(got[0], want[0])
    assert torch.equal(got[1], want[1])
