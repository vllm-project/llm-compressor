"""MERA compression using quimb tensor network library.

Correct approach: Treat data [seq_len, hidden_dim] as a product-state MPS
where each site has physical dimension = hidden_dim.

NO FLATTENING - preserves structure and avoids exponential blowup.
"""

import numpy as np
import autoray
import quimb.tensor as qtn
from generate_tree_dataset import generate_power_law_tree_dataset


def data_to_tensor_network(data_matrix):
    """Convert data [seq_len, hidden_dim] to tensor network (product-state MPS).

    Each site i gets a tensor with physical index k{i} of dimension hidden_dim.
    Bond dimension between sites = 1 (product state, no entanglement).

    Args:
        data_matrix: Array [seq_len, hidden_dim]

    Returns:
        TensorNetwork representing the data
    """
    # Ensure numpy array
    if not isinstance(data_matrix, np.ndarray):
        data_matrix = np.asarray(data_matrix)

    seq_len, hidden_dim = data_matrix.shape

    data_tensors = []
    for i in range(seq_len):
        # Local token vector of shape (hidden_dim,)
        vec = data_matrix[i]

        # Ensure it's a numpy array
        if not isinstance(vec, np.ndarray):
            vec = np.asarray(vec)

        # Create tensor for this site with physical index k{i}
        t = qtn.Tensor(data=vec, inds=(f"k{i}",), tags={f"I{i}", "DATA"})
        data_tensors.append(t)

    # Combine into unified tensor network
    data_tn = qtn.TensorNetwork(data_tensors)

    return data_tn


def initialize_mera(seq_len, hidden_dim, max_bond):
    """Initialize MERA tensor network.

    Args:
        seq_len: Sequence length (must be power of 2)
        hidden_dim: Physical dimension per site
        max_bond: Maximum bond dimension

    Returns:
        MERA tensor network with numpy arrays
    """
    assert (seq_len & (seq_len - 1)) == 0, "seq_len must be power of 2"

    print(f"\nInitializing MERA...")
    print(f"  Sites: {seq_len}")
    print(f"  Physical dim: {hidden_dim}")
    print(f"  Max bond: {max_bond}")

    mera = qtn.MERA.rand(L=seq_len, phys_dim=hidden_dim, D=max_bond, dtype="float64")

    # Ensure all tensors are numpy arrays
    for tid, tensor in mera.tensor_map.items():
        if not isinstance(tensor.data, np.ndarray):
            tensor.modify(data=np.asarray(tensor.data))
        mera.tensor_map[tid] = tensor

    print(f"  Number of tensors: {len(mera.tensor_map)}")
    print(f"  Types: {[type(a) for a in mera.arrays]}")

    return mera


def train_mera_on_data(mera, data, num_steps=100, optimizer="adam"):
    """Train MERA to compress activation data.

    Args:
        mera: Initialized MERA tensor network
        data: Array [batch, seq_len, hidden_dim]
        num_steps: Optimization steps
        optimizer: Optimizer name
        backend: Autodiff backend

    Returns:
        Optimized MERA tensor network
    """
    batch_size, seq_len, hidden_dim = data.shape

    print(f"=" * 70)
    print(f"MERA TRAINING")
    print(f"=" * 70)
    print(f"  Data: [{batch_size}, {seq_len}, {hidden_dim}]")
    print(f"  Optimizer: {optimizer}")
    print(f"  Steps: {num_steps}")

    # Extract single sample and normalize
    target_matrix = np.asarray(data[0], dtype=np.float64)
    target_matrix = target_matrix / np.linalg.norm(target_matrix)

    print(f"\nTarget data:")
    print(f"  Shape: {target_matrix.shape}")
    print(f"  Norm: {np.linalg.norm(target_matrix):.6f}")

    # Convert to tensor network
    print(f"\nCreating data tensor network...")
    data_tn = data_to_tensor_network(target_matrix)
    print(f"  Number of tensors: {len(data_tn.tensor_map)}")

    # Loss function: tensor network contraction for overlap
    def loss_fn(mera_tn):
        """Compute overlap via tensor network contraction.

        Both MERA and data TN share physical indices k0, k1, ..., k{L-1}.
        Contracting these gives the inner product ⟨MERA|data⟩.
        """
        # Contract MERA with data TN
        overlap = mera_tn @ data_tn

        # Loss: 1 - |overlap|²
        loss = 1.0 - abs(overlap) ** 2

        return loss

    # Norm function: skip normalization to avoid expensive contraction
    # (normalization can be done periodically instead of every step)
    def norm_fn(mera_tn):
        return mera_tn  # Identity - no normalization

    # Create optimizer - TNOptimizer will extract variables automatically
    print(f"\nSetting up TNOptimizer...")
    tnopt = qtn.TNOptimizer(
        mera,
        loss_fn=loss_fn,
        norm_fn=norm_fn,
        optimizer=optimizer,
        autodiff_backend="jax",
    )

    print(f"  Number of parameters: {sum(t.size for t in mera.tensor_map.values())}")

    # Run optimization
    print(f"\nOptimizing for {num_steps} steps...")
    initial_loss = loss_fn(mera)
    print(f"  Initial loss: {initial_loss:.6f}")

    optimized_params = tnopt.optimize(n=num_steps)

    # Get optimized MERA - it's stored in tnopt.tn
    optimized_mera = tnopt.get_tn_opt()
    final_loss = loss_fn(optimized_mera)

    print(f"\nOptimization complete!")
    print(f"  Final loss: {final_loss:.6f}")
    print(f"  Loss reduction: {initial_loss - final_loss:.6f}")
    print(f"  Final overlap: {np.sqrt(1.0 - final_loss):.6f}")

    # Compute compression ratio
    original_size = seq_len * hidden_dim
    mera_params = sum(t.size for t in optimized_mera.tensors)
    compression_ratio = original_size / mera_params

    print(f"\nCompression analysis:")
    print(f"  Original parameters: {original_size}")
    print(f"  MERA parameters: {mera_params}")
    print(f"  Compression ratio: {compression_ratio:.2f}x")

    return optimized_mera


if __name__ == "__main__":
    # Generate tree-structured test data
    print("\nGenerating tree-structured dataset (α=1.0)...")

    data = generate_power_law_tree_dataset(
        batch_size=2,
        seq_len=8,  # Very small to reduce memory
        hidden_dim=4,  # Very small dimension
        alpha=1.0,
        device="cpu",
    )

    print(f"Generated data shape: {data.shape}")

    # Initialize MERA
    mera = initialize_mera(seq_len=data.shape[1], hidden_dim=data.shape[2], max_bond=4)

    # Train MERA
    mera_opt = train_mera_on_data(
        mera=mera,
        data=data,
        num_steps=10,
        optimizer="adam",
    )

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
