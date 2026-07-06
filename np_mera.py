import numpy as np
from generate_tree_dataset import generate_power_law_tree_dataset


import numpy as np


class AdvancedMeraSvdCompressor:
    def __init__(self, seq_len, hidden_dim, target_latent_dim):
        """
        True MERA Compressor using shifted/overlapping boundaries to
        actively alter the singular value spectrum across blocks.
        """
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.target_latent_dim = target_latent_dim
        self.layers = []  # Stores (U, W, orig_seq, orig_dim, next_dim)

    def fit_encode(self, X):
        batch_size, current_seq, current_dim = X.shape
        current_state = X.copy()
        self.layers = []

        while (current_seq * current_dim) > self.target_latent_dim:
            half_seq = current_seq // 2

            # 1. --- TRUE OVERLAPPING DISENTANGLING STAGE ---
            # Roll the sequence axis by 1 to force the blocks to span across
            # the boundaries of the upcoming isometry blocks
            shifted_state = np.roll(current_state, shift=1, axis=1)
            flat_blocks_shifted = shifted_state.reshape(-1, 2 * current_dim)

            # Compute Disentangler U on the boundary-spanning features
            covariance_shifted = flat_blocks_shifted.T @ flat_blocks_shifted
            U_mat, _, Vt_mat = np.linalg.svd(covariance_shifted, full_matrices=False)
            U = U_mat @ Vt_mat  # Square Unitary [2*Dim, 2*Dim]

            # Apply U and shift back to restore alignment for the Isometry step
            disentangled_shifted = flat_blocks_shifted @ U
            disentangled_state = disentangled_shifted.reshape(
                batch_size, current_seq, current_dim
            )
            current_state = np.roll(disentangled_state, shift=-1, axis=1)

            # 2. --- COARSE-GRAINING ISOMETRY STAGE ---
            # Now the isometry compresses blocks whose internal boundary correlations
            # have been actively minimized by U
            paired = current_state.reshape(batch_size, half_seq, 2 * current_dim)
            block_matrix = paired.reshape(-1, 2 * current_dim)

            _, _, Vt = np.linalg.svd(block_matrix, full_matrices=False)

            next_dim = max(1, (self.target_latent_dim // half_seq))
            next_dim = min(next_dim, 2 * current_dim)

            W = Vt[:next_dim, :].T  # Isometry [2*Dim, Next_Dim]
            self.layers.append((U, W, current_seq, current_dim, next_dim))

            # Project down
            compressed_blocks = block_matrix @ W
            current_state = compressed_blocks.reshape(batch_size, half_seq, next_dim)

            current_seq = half_seq
            current_dim = next_dim

            if (current_seq * current_dim) == self.target_latent_dim:
                break

        return current_state.reshape(batch_size, -1)

    def decode(self, latent_state):
        batch_size = latent_state.shape[0]
        current_state = latent_state.copy()

        for U, W, orig_seq, orig_dim, next_dim in reversed(self.layers):
            half_seq = orig_seq // 2

            # 1. Reverse Isometry Expansion (W^T)
            block_matrix = current_state.reshape(-1, next_dim)
            expanded_isom = block_matrix @ W.T
            expanded_state = expanded_isom.reshape(batch_size, orig_seq, orig_dim)

            # 2. Reverse Shifted Disentangler Rotation (U^T)
            # Shift forward, apply U.T, and shift backward to perfectly invert the pass
            shifted_expanded = np.roll(expanded_state, shift=1, axis=1)
            flat_shifted = shifted_expanded.reshape(-1, 2 * orig_dim)

            reconstructed_flat = flat_shifted @ U.T
            reconstructed_state = reconstructed_flat.reshape(
                batch_size, orig_seq, orig_dim
            )
            current_state = np.roll(reconstructed_state, shift=-1, axis=1)

        return current_state


# --- VERIFICATION EXECUTION ON HIGH-ENTROPY DATA ---
if __name__ == "__main__":
    batch_size = 16
    seq_len = 4096
    hidden_dim = 32
    target_latent_dim = 65536

    print(f"Testing MERA SVD Compressor with alpha=0.7 Profile Simulation")
    print(f"Original Vector Dimension: {seq_len * hidden_dim} values per batch.")
    print(f"Latent Bottleneck Target:  {target_latent_dim} values per batch.\n")

    # Simulating a highly distributed, difficult dataset (Alpha = 0.7 layout)
    dataset = (
        generate_power_law_tree_dataset(batch_size, seq_len, hidden_dim, alpha=1.7)
        .detach()
        .numpy()
    )

    # Run Compressor
    compressor = AdvancedMeraSvdCompressor(seq_len, hidden_dim, target_latent_dim)
    latent = compressor.fit_encode(dataset)
    reconstructed = compressor.decode(latent)

    # Calculate performance metrics
    error_vector = dataset - reconstructed
    l2_error = np.linalg.norm(error_vector)
    signal_power = np.mean(dataset**2)
    noise_power = np.mean(error_vector**2)
    snr_db = (
        10 * np.log10(signal_power / noise_power)
        if noise_power > 1e-20
        else float("inf")
    )

    print("------------- REFINED MERA LOGS -------------")
    print(f"Latent State Shape:      {latent.shape}")
    print(f"Reconstructed Shape:     {reconstructed.shape}")
    print(f"L2 Reconstruction Error: {l2_error:.6e}")
    print(f"Signal-to-Noise Ratio:   {snr_db:.2f} dB")
    print("---------------------------------------------")
