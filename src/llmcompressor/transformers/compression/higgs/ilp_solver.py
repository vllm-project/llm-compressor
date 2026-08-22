"""
ILP-based optimization for mixed-precision quantization scheme selection.

This module formulates and solves an Integer Linear Programming problem
to select the optimal quantization scheme for each layer, minimizing
the weighted sum of MSE values.
"""

from typing import Dict, List, Optional

from loguru import logger

try:
    import pulp
except ImportError:
    pulp = None


__all__ = ["solve_ilp_mixed_precision"]


def solve_ilp_mixed_precision(
    mse_matrix: Dict[str, Dict[str, float]],
    alphas: Dict[str, float],
    candidate_schemes: List[str],
    fused_groups: List[List[str]] = None,
    target_avg_bitwidth: Optional[float] = None,
    layer_param_counts: Optional[Dict[str, int]] = None,
    scheme_bitwidths: Optional[Dict[str, float]] = None,
    target_avg_act_bitwidth: Optional[float] = None,
    scheme_act_bitwidths: Optional[Dict[str, float]] = None,
) -> Dict[str, str]:
    """
    Solve ILP to select optimal quantization scheme for each layer.

    Formulation:
        Decision variables: x[layer][scheme] ∈ {0, 1}

        Objective:
            minimize: Σ_layer Σ_scheme (MSE[layer][scheme] * alpha[layer] * x[layer][scheme])

        Constraints:
            1. Each layer gets exactly one scheme:
               Σ_scheme x[layer][scheme] = 1  ∀ layer

            2. Fused layers use same scheme (if fused_groups provided):
               x[layer_i][scheme] = x[layer_j][scheme]  ∀ scheme, ∀ (layer_i, layer_j) in same group

            3. Optional: Average weight bitwidth constraint (if target_avg_bitwidth provided):
               Σ_layer (bitwidth[scheme] * params[layer] * x[layer][scheme])
               / Σ_layer params[layer] <= target_avg_bitwidth

            4. Optional: Average activation bitwidth constraint (if target_avg_act_bitwidth provided):
               Σ_layer (act_bitwidth[scheme] * params[layer] * x[layer][scheme])
               / Σ_layer params[layer] <= target_avg_act_bitwidth

    Args:
        mse_matrix: Nested dict mapping layer_name -> scheme_name -> mse_value
        alphas: Dict mapping layer_name -> importance weight
        candidate_schemes: List of scheme names (must match keys in mse_matrix)
        fused_groups: Optional list of layer name groups that must use same scheme
        target_avg_bitwidth: Optional constraint on weighted average weight bitwidth
        layer_param_counts: Required if target_avg_bitwidth or target_avg_act_bitwidth is set
        scheme_bitwidths: Required if target_avg_bitwidth is set
        target_avg_act_bitwidth: Optional constraint on weighted average activation bitwidth
        scheme_act_bitwidths: Required if target_avg_act_bitwidth is set

    Returns:
        Dict mapping layer_name -> selected_scheme_name

    Raises:
        ValueError: If ILP is infeasible or unbounded
    """
    if pulp is None:
        raise ImportError(
            "HIGGS ILP solver requires the 'pulp' package. "
            "Install it with: pip install pulp"
        )

    if fused_groups is None:
        fused_groups = []

    # Validate inputs
    if target_avg_bitwidth is not None:
        if layer_param_counts is None or scheme_bitwidths is None:
            raise ValueError(
                "target_avg_bitwidth requires layer_param_counts and scheme_bitwidths"
            )
    if target_avg_act_bitwidth is not None:
        if layer_param_counts is None or scheme_act_bitwidths is None:
            raise ValueError(
                "target_avg_act_bitwidth requires layer_param_counts and scheme_act_bitwidths"
            )

    # Create ILP problem
    prob = pulp.LpProblem("MixedPrecisionQuantization", pulp.LpMinimize)

    # Decision variables: x[layer][scheme] ∈ {0, 1}
    x = {}
    for layer in mse_matrix:
        x[layer] = {}
        for scheme in candidate_schemes:
            x[layer][scheme] = pulp.LpVariable(
                f"x_{_sanitize_name(layer)}_{scheme}",
                cat=pulp.LpBinary,
            )

    # Objective: minimize weighted MSE
    objective_terms = []
    for layer in mse_matrix:
        alpha = alphas.get(layer, 1.0)  # Default alpha = 1.0 if missing
        for scheme in candidate_schemes:
            mse = mse_matrix[layer].get(scheme, float('inf'))
            mse = max(min(1e10, mse), 0) # clamp between 0 and 1e10
            objective_terms.append(mse * alpha * x[layer][scheme])

    prob += pulp.lpSum(objective_terms), "WeightedMSE"

    # Constraint 1: Each layer gets exactly one scheme
    for layer in mse_matrix:
        prob += (
            pulp.lpSum([x[layer][scheme] for scheme in candidate_schemes]) == 1,
            f"OneSchemePerLayer_{_sanitize_name(layer)}",
        )

    # Constraint 2: Fused layers use same scheme
    for group_idx, group in enumerate(fused_groups):
        if len(group) < 2:
            continue

        base_layer = group[0]
        if base_layer not in mse_matrix:
            logger.warning(f"Fused group layer {base_layer} not in MSE matrix, skipping")
            continue

        for other_layer in group[1:]:
            if other_layer not in mse_matrix:
                logger.warning(f"Fused group layer {other_layer} not in MSE matrix, skipping")
                continue

            for scheme in candidate_schemes:
                prob += (
                    x[base_layer][scheme] == x[other_layer][scheme],
                    f"FusedGroup{group_idx}_{_sanitize_name(base_layer)}_{_sanitize_name(other_layer)}_{scheme}",
                )

    # Constraint 3: Optional average bitwidth constraint
    if target_avg_bitwidth is not None:
        total_params = sum(layer_param_counts.get(layer, 0) for layer in mse_matrix)

        if total_params == 0:
            logger.warning("Total parameter count is 0, skipping bitwidth constraint")
        else:
            weighted_bitwidth_terms = []
            for layer in mse_matrix:
                params = layer_param_counts.get(layer, 0)
                for scheme in candidate_schemes:
                    bitwidth = scheme_bitwidths.get(scheme, 0)
                    weighted_bitwidth_terms.append(
                        bitwidth * params * x[layer][scheme]
                    )

            # Average bitwidth = sum(bitwidth * params * x) / total_params
            prob += (
                pulp.lpSum(weighted_bitwidth_terms) <= target_avg_bitwidth * total_params,
                "AverageWeightBitwidthConstraint",
            )

    # Constraint 4: Optional average activation bitwidth constraint
    # NOTE: actual number of activation elements will be proportional to in_features
    # rather than in_features * out_features = params. However we should probably
    # just use actual performance numbers at the end of the day.
    if target_avg_act_bitwidth is not None:
        total_params = sum(layer_param_counts.get(layer, 0) for layer in mse_matrix)

        if total_params == 0:
            logger.warning("Total parameter count is 0, skipping activation bitwidth constraint")
        else:
            weighted_act_bitwidth_terms = []
            for layer in mse_matrix:
                params = layer_param_counts.get(layer, 0)
                for scheme in candidate_schemes:
                    act_bitwidth = scheme_act_bitwidths.get(scheme, 16.0)
                    weighted_act_bitwidth_terms.append(
                        act_bitwidth * params * x[layer][scheme]
                    )

            prob += (
                pulp.lpSum(weighted_act_bitwidth_terms) <= target_avg_act_bitwidth * total_params,
                "AverageActBitwidthConstraint",
            )

    # Solve the ILP
    logger.info(f"Solving ILP with {len(mse_matrix)} layers and {len(candidate_schemes)} schemes")
    logger.info(f"  Variables: {sum(len(x[layer]) for layer in x)}")
    logger.info(f"  Fused groups: {len([g for g in fused_groups if len(g) >= 2])}")

    # Use CBC solver (included with PuLP)
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=300)  # 5 minute timeout
    status = prob.solve(solver)

    # Check solution status
    status_name = pulp.LpStatus[status]
    logger.info(f"ILP solver status: {status_name}")

    if status != pulp.LpStatusOptimal:
        raise ValueError(
            f"ILP solver failed with status: {status_name}. "
            f"Problem may be infeasible or unbounded."
        )

    # Extract solution
    solution = {}
    for layer in mse_matrix:
        for scheme in candidate_schemes:
            if pulp.value(x[layer][scheme]) > 0.5:  # Binary variable = 1
                solution[layer] = scheme
                break

        # Sanity check: every layer should have a scheme
        if layer not in solution:
            logger.error(f"Layer {layer} has no scheme assigned in ILP solution!")
            # Assign first scheme as fallback
            solution[layer] = candidate_schemes[0]

    # Log objective value
    objective_value = pulp.value(prob.objective)
    logger.info(f"ILP optimal objective value: {objective_value:.6f}")

    # Log scheme distribution
    scheme_counts = {}
    for scheme in solution.values():
        scheme_counts[scheme] = scheme_counts.get(scheme, 0) + 1
    logger.info(f"Scheme distribution: {scheme_counts}")

    return solution


def _sanitize_name(name: str) -> str:
    """
    Sanitize layer name for use in PuLP variable names.

    PuLP variable names can't contain certain characters like dots.
    """
    return name.replace(".", "_").replace("[", "_").replace("]", "_")
