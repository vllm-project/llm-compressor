"""
Utilities for the HOPE (Higher-Order Pruning of Experts) modifier: solving
the per-layer prune-set quadratic program over the interaction matrix.
"""

import numpy as np
import torch
from loguru import logger

try:
    from scipy.optimize import minimize as scipy_minimize
except ImportError:
    scipy_minimize = None

__all__ = ["solve_hope_qp"]

# solver settings for the continuous relaxation, following Appendix D.2 of
# the paper: bounds of [0, 1] per variable, a hard equality constraint on
# the sum, a uniform initial point, a tolerance of 1e-12 and a maximum of
# 1000 iterations
_SLSQP_MAX_ITERATIONS = 1000
_SLSQP_TOLERANCE = 1e-12


def _require_scipy():
    """
    Raise an ImportError when scipy, which the HOPE quadratic-program solver
    is implemented with, is not installed.
    """
    if scipy_minimize is None:
        raise ImportError(
            "HOPEPruningModifier solves its prune-set quadratic program with "
            "scipy, which is not installed. Install with: "
            "pip install llmcompressor[hope]"
        )


def solve_hope_qp(
    interaction_matrix: torch.Tensor,
    num_to_drop: int,
    group_sizes: list[int] | None = None,
) -> list[int]:
    """
    Select the experts to prune by minimizing the HOPE objective.

    Finds an (approximately) optimal prune-set ``P`` for the binary quadratic
    program

        ``min_p p^T F p  s.t.  p in {0, 1}^E, sum_k p_k = |P|``

    where ``F`` is the layer's interaction matrix and ``p_k = 1`` iff expert
    ``k`` is pruned, so that ``p^T F p`` is the interaction mass the
    prune-set removes from the layer. Following the paper, the binary program
    is solved through a continuous relaxation (``p in [0, 1]^E`` with a hard
    equality constraint on the sum, solved with SciPy's SLSQP from a uniform
    initial point) which is rounded to binary by pruning the ``num_to_drop``
    entries with the highest relaxed values.

    :param interaction_matrix: the ``[num_experts, num_experts]`` interaction
        matrix ``F`` of the layer, with per-expert self-importance on the
        diagonal and pairwise co-selection contributions off the diagonal
    :param num_to_drop: number of experts to prune: from the whole layer when
        ``group_sizes`` is None, else from each group individually
        (group-limited routers require an equal number of drops per group)
    :param group_sizes: sizes of the router's expert groups, which must
        partition the layer's experts, or None when the router is not
        group-limited
    :return: the pruned expert indices, in ascending order
    """
    num_experts = interaction_matrix.shape[0]
    if group_sizes is None:
        group_sizes = [num_experts]
    if sum(group_sizes) != num_experts:
        raise ValueError(
            f"group sizes {group_sizes} must partition all "
            f"{num_experts} experts of the layer"
        )
    if any(num_to_drop >= size for size in group_sizes):
        raise ValueError(
            f"a prune budget of {num_to_drop} leaves no expert in every one "
            f"of the groups {group_sizes}"
        )

    # with a zero budget the relaxed solution is identically zero, and no
    # solver is needed
    if num_to_drop == 0:
        return []

    _require_scipy()

    matrix = interaction_matrix.detach().to(torch.float64).cpu().numpy()

    # uniform initial point and one equality constraint per group, each
    # holding that group's drop budget
    initial: np.ndarray = np.zeros(num_experts)
    constraints = []
    start = 0
    for size in group_sizes:
        end = start + size
        initial[start:end] = num_to_drop / size
        constraints.append(_group_sum_constraint(start, end, num_to_drop))
        start = end

    result = scipy_minimize(
        lambda values: values @ matrix @ values,
        initial,
        jac=lambda values: 2.0 * (matrix @ values),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * num_experts,
        constraints=constraints,
        options={"maxiter": _SLSQP_MAX_ITERATIONS, "ftol": _SLSQP_TOLERANCE},
    )

    relaxed = result.x
    if not np.all(np.isfinite(relaxed)):
        logger.warning(
            "HOPE quadratic-program solver produced a non-finite solution; "
            "falling back to ranking experts by the diagonal of the "
            "interaction matrix"
        )
        # prune the experts with the least self-importance
        relaxed = np.negative(np.diagonal(matrix))
    elif not result.success:
        logger.warning(
            f"HOPE quadratic-program solver did not converge "
            f"({result.message}); rounding the best solution found"
        )

    # round the continuous solution: the highest-valued num_to_drop entries
    # of each group are pruned
    dropped: list[int] = []
    start = 0
    for size in group_sizes:
        end = start + size
        order = np.argsort(-relaxed[start:end], kind="stable")
        dropped.extend(start + int(index) for index in order[:num_to_drop])
        start = end

    return sorted(dropped)


def _group_sum_constraint(start: int, end: int, budget: int) -> dict:
    """
    An SLSQP equality constraint holding the prune budget of the expert
    group ``[start, end)``.

    :param start: first expert index of the group
    :param end: one past the last expert index of the group
    :param budget: number of experts the group must contribute to the
        prune-set
    :return: the constraint, with its analytic jacobian
    """

    def constraint(values: np.ndarray) -> float:
        return values[start:end].sum() - budget

    def jacobian(values: np.ndarray) -> np.ndarray:
        gradient = np.zeros_like(values)
        gradient[start:end] = 1.0
        return gradient

    return {"type": "eq", "fun": constraint, "jac": jacobian}
