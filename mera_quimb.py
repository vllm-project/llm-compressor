"""MERA compression using quimb tensor network library.

Clean implementation following quimb best practices.
Data: [seq_len, hidden_dim] flattened to dense state vector.
"""

import torch
import numpy as np
import quimb as qu
import quimb.tensor as qtn
from generate_tree_dataset import generate_power_law_tree_dataset


def data_to_vector(data_sample):
    """Convert data [seq_len, hidden_dim] to dense state vector.

    Simply flatten and normalize.
    """
    vec = data_sample.flatten()
    norm = np.linalg.norm(vec)
    if norm > 1e-10:
        vec = vec / norm
    return vec


def vector_to_data(vec, seq_len, hidden_dim):
    """Convert dense state vector back to [seq_len, hidden_dim]."""
    return vec.reshape(seq_len, hidden_dim)


def train_mera_on_data(data, max_bond=16, num_steps=100, optimizer='L-BFGS-B', backend='torch'):
    """Train MERA to compress activation data.

    Args:
        data: Tensor [batch, seq_len, hidden_dim]
        max_bond: Maximum bond dimension (chi)
        num_steps: Optimization steps
        optimizer: Optimizer name ('L-BFGS-B', 'adam', etc.)
        backend: Autodiff backend ('torch', 'jax', 'tensorflow')

    Returns:
        Optimized MERA tensor network
    """
    batch_size, seq_len, hidden_dim = data.shape

    # Verify seq_len is power of 2
    assert (seq_len & (seq_len - 1)) == 0, "seq_len must be power of 2"

    L = seq_len
    phys_dim = hidden_dim

    print(f"="*70)
    print(f"MERA TRAINING")
    print(f"="*70)
    print(f"  Sites (L): {L}")
    print(f"  Physical dim: {phys_dim}")
    print(f"  Max bond (D): {max_bond}")
    print(f"  Total state dimension: {phys_dim**L}")
    print(f"  Optimizer: {optimizer}")
    print(f"  Backend: {backend}")

    # Convert first sample to target state vector
    # Move to CPU for numpy conversion, but keep on GPU if using torch backend
    if backend == 'torch':
        target_data = data[0]  # Keep as torch tensor on GPU
        target_state = target_data.flatten()
        target_state = target_state / torch.linalg.norm(target_state)
    else:
        target_data = data[0].cpu().numpy()
        target_state = data_to_vector(target_data)

    print(f"\nTarget state:")
    print(f"  Shape: {target_state.shape}")
    if backend == 'torch':
        print(f"  Norm: {torch.linalg.norm(target_state).item():.6f}")
        print(f"  Device: {target_state.device}")
    else:
        print(f"  Norm: {np.linalg.norm(target_state):.6f}")

    # Initialize MERA
    print(f"\nInitializing random MERA...")
    mera = qtn.MERA.rand(L=L, phys_dim=phys_dim, D=max_bond, dtype='float64')
    print(f"  Number of tensors: {len(mera.tensor_map)}")

    # Loss function: maximize overlap with target state
    def loss_fn(mera_tn):
        # Contract MERA to dense vector and compute overlap
        overlap = mera_tn.to_dense() | target_state
        # Loss: 1 - |overlap|²
        loss = 1.0 - abs(overlap)**2
        return loss

    # Norm function: maintain MERA normalization
    def norm_fn(mera_tn):
        return mera_tn.norm()

    # Extract variables for optimization
    x0, tags, guess_structure = mera.get_variables(tag_only=True)
    print(f"\nOptimization variables:")
    print(f"  Number of parameters: {len(x0)}")

    # Create optimizer
    print(f"\nSetting up TNOptimizer...")
    tnopt = qtn.TNOptimizer(
        mera,
        loss_fn=loss_fn,
        norm_fn=norm_fn,
        x0=x0,
        tags=tags,
        guess_structure=guess_structure,
        optimizer=optimizer,
        autodiff_backend=backend
    )

    # Run optimization
    print(f"\nOptimizing for {num_steps} steps...")
    initial_loss = loss_fn(mera)
    print(f"  Initial loss: {initial_loss:.6f}")

    optimized_params = tnopt.optimize(n=num_steps)

    # Get optimized MERA
    optimized_mera = tnopt.get_compressed_tn()
    final_loss = loss_fn(optimized_mera)

    print(f"\nOptimization complete!")
    print(f"  Final loss: {final_loss:.6f}")
    print(f"  Loss reduction: {initial_loss - final_loss:.6f}")

    # Test reconstruction
    print(f"\nTesting reconstruction...")
    reconstructed_vec = optimized_mera.to_dense()

    if backend == 'torch':
        reconstructed_data = reconstructed_vec.reshape(seq_len, hidden_dim)
        reconstruction_error = torch.linalg.norm(reconstructed_data - target_data).item()
    else:
        reconstructed_data = vector_to_data(reconstructed_vec, seq_len, hidden_dim)
        reconstruction_error = np.linalg.norm(reconstructed_data - target_data)

    print(f"  Reconstruction error: {reconstruction_error:.6f}")

    # Compute compression ratio
    # MERA bond dimension vs full representation
    original_size = seq_len * hidden_dim
    compressed_size = max_bond * max_bond * seq_len  # Rough estimate
    compression_ratio = original_size / compressed_size
    print(f"  Compression ratio: {compression_ratio:.2f}x")

    return optimized_mera


if __name__ == "__main__":
    # Generate tree-structured test data
    print("\nGenerating tree-structured dataset (α=1.0)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data = generate_power_law_tree_dataset(
        batch_size=4,
        seq_len=16,      # Start very small for testing
        hidden_dim=4,    # Small dimension
        alpha=1.0,
        device=device
    )

    print(f"Generated data shape: {data.shape}")

    # Train MERA
    mera_opt = train_mera_on_data(
        data,
        max_bond=8,
        num_steps=50,
        optimizer='L-BFGS-B',
        backend='torch'
    )

    print("\n" + "="*70)
    print("DONE")
    print("="*70)
