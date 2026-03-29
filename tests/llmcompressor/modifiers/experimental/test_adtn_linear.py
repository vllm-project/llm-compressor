import pytest
import torch
import torch.nn.functional as F

from llmcompressor.modifiers.experimental.adtn_linear import ADTNLinear, ADTNSublayer


def test_adtn_sublayer_forward():
    """Test that ADTNSublayer forward pass works correctly with permutation."""
    in_features = 128
    out_features = 64
    group_size = 32
    num_groups = in_features // group_size

    # Create group linear layers
    group_linears = []
    for _ in range(num_groups):
        linear = torch.nn.Linear(group_size, out_features, bias=False)
        group_linears.append(linear)

    # Create a simple permutation (just reverse for testing)
    input_perm = torch.arange(in_features).flip(0)

    # Create sublayer
    sublayer = ADTNSublayer(
        in_features=in_features,
        out_features=out_features,
        linears=group_linears,
        input_perm=input_perm,
    )

    # Test with 2D input
    batch_size = 8
    x = torch.randn(batch_size, in_features)
    output = sublayer(x)

    assert output.shape == (batch_size, out_features), f"Expected shape {(batch_size, out_features)}, got {output.shape}"

    # Test with 3D input
    seq_len = 16
    x_3d = torch.randn(batch_size, seq_len, in_features)
    output_3d = sublayer(x_3d)

    assert output_3d.shape == (batch_size, seq_len, out_features), f"Expected shape {(batch_size, seq_len, out_features)}, got {output_3d.shape}"


def test_adtn_linear_approximates_linear():
    """Test that ADTNLinear with residual fitting can approximate a linear layer."""
    in_features = 256
    out_features = 128
    batch_size = 32
    num_samples = 1000

    # Create a linear layer
    linear = torch.nn.Linear(in_features, out_features, bias=False)

    # Generate synthetic input/output activations
    input_activations = torch.randn(num_samples, in_features)
    with torch.no_grad():
        output_activations = linear(input_activations)

    # Manually create ADTN with one sublayer using residual-based OLS
    group_size = 64
    input_perm = ADTNLinear._spectral_reordering(input_activations, group_size)
    input_permuted = input_activations[:, input_perm]

    num_groups = in_features // group_size
    group_linears = []
    current_residual = output_activations.clone()

    for group_idx in range(num_groups):
        start_idx = group_idx * group_size
        end_idx = (group_idx + 1) * group_size

        X_group = input_permuted[:, start_idx:end_idx]
        Y_group = current_residual  # Fit to residual

        # OLS solution
        solution = torch.linalg.lstsq(X_group, Y_group).solution

        group_linear = torch.nn.Linear(group_size, out_features, bias=False)
        group_linear.weight.data = solution.T
        group_linears.append(group_linear)

        # Update residual
        with torch.no_grad():
            group_output = group_linear(X_group)
            current_residual = current_residual - group_output

    sublayer = ADTNSublayer(
        in_features=in_features,
        out_features=out_features,
        linears=group_linears,
        input_perm=input_perm,
    )

    adtn = ADTNLinear(
        in_features=in_features,
        out_features=out_features,
        sublayers=[sublayer],
    )

    # Test on training data first (should fit well)
    with torch.no_grad():
        train_output = adtn(input_activations)

    train_similarity = F.cosine_similarity(
        output_activations.reshape(-1),
        train_output.reshape(-1),
        dim=0
    )
    assert train_similarity > 0.95, f"Training fit poor: cosine similarity = {train_similarity:.3f}"

    # Test on new batch (generalization)
    test_input = torch.randn(batch_size, in_features)
    with torch.no_grad():
        original_output = linear(test_input)
        adtn_output = adtn(test_input)

    # Check that shapes match
    assert adtn_output.shape == original_output.shape

    # Check that outputs are reasonably similar (cosine similarity)
    similarity = F.cosine_similarity(
        original_output.reshape(-1),
        adtn_output.reshape(-1),
        dim=0
    )

    # ADTN with single sublayer should have reasonable approximation
    assert similarity > 0.7, f"ADTN approximation too poor: cosine similarity = {similarity:.3f}"


def test_adtn_stacking_improves_fit():
    """Test that stacking multiple ADTN sublayers improves approximation (additive)."""
    in_features = 128
    out_features = 64
    num_samples = 500

    # Create a linear layer
    linear = torch.nn.Linear(in_features, out_features, bias=False)

    # Generate synthetic data
    input_activations = torch.randn(num_samples, in_features)
    with torch.no_grad():
        output_activations = linear(input_activations)

    # Create ADTN with 2 sublayers using additive strategy
    adtn = ADTNLinear(
        in_features=in_features,
        out_features=out_features,
        sublayers=[],
    )

    group_size = 32

    for sublayer_idx in range(2):
        # Compute global residual
        with torch.no_grad():
            if sublayer_idx == 0:
                global_residual = output_activations.clone()
            else:
                current_approx = adtn(input_activations)
                global_residual = output_activations - current_approx

        # Always use original inputs for permutation
        input_perm = ADTNLinear._spectral_reordering(input_activations, group_size)
        input_permuted = input_activations[:, input_perm]

        num_groups = in_features // group_size
        group_linears = []
        group_residual = global_residual.clone()

        for group_idx in range(num_groups):
            start_idx = group_idx * group_size
            end_idx = (group_idx + 1) * group_size

            X_group = input_permuted[:, start_idx:end_idx]
            Y_group = group_residual

            solution = torch.linalg.lstsq(X_group, Y_group).solution

            group_linear = torch.nn.Linear(group_size, out_features, bias=False)
            group_linear.weight.data = solution.T
            group_linears.append(group_linear)

            with torch.no_grad():
                group_output = group_linear(X_group)
                group_residual = group_residual - group_output

        sublayer = ADTNSublayer(
            in_features=in_features,
            out_features=out_features,
            linears=group_linears,
            input_perm=input_perm,
        )

        adtn.append_sublayer(sublayer)

    # Check that ADTN approximates the linear layer
    with torch.no_grad():
        adtn_output = adtn(input_activations)

    similarity = F.cosine_similarity(
        output_activations.reshape(-1),
        adtn_output.reshape(-1),
        dim=0
    )

    # With 2 sublayers, should have good approximation
    assert similarity > 0.9, f"ADTN with 2 sublayers poor: cosine similarity = {similarity:.3f}"
