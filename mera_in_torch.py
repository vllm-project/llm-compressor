import torch
import quimb.tensor as qtn
from torch.utils.data import TensorDataset, DataLoader
from generate_tree_dataset import generate_power_law_tree_dataset


class MeraLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.u = torch.nn.Parameter(
            torch.eye(in_dim**2).view(in_dim, in_dim, in_dim, in_dim)
        )
        self.w = torch.nn.Parameter(torch.randn(in_dim, in_dim, out_dim) * 0.1)
        self.project_to_stiefel()

    @torch.no_grad()
    def project_to_stiefel(self):
        shape_u = self.u.shape
        u_mat = self.u.view(shape_u[0] * shape_u[1], -1)
        U, _, Vh = torch.linalg.svd(u_mat, full_matrices=False)
        self.u.copy_((U @ Vh).view(shape_u))

        shape_w = self.w.shape
        w_mat = self.w.view(shape_w[0] * shape_w[1], -1)
        U, _, Vh = torch.linalg.svd(w_mat, full_matrices=False)
        self.w.copy_((U @ Vh).view(shape_w))


class ClassicalMeraNetwork(torch.nn.Module):
    def __init__(self, seq_len, hidden_dim, max_bond):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.max_bond = max_bond
        self.layer1 = MeraLayer(hidden_dim, max_bond)

    def forward(self, x):
        """Encode: compress input to latent representation using pure PyTorch.

        Args:
            x: [batch_size, seq_len, hidden_dim]

        Returns:
            compressed: [batch_size, seq_len//2, max_bond]
        """
        batch_size, seq_len, hidden_dim = x.shape

        # 1. Apply disentanglers to pairs
        # u: [hidden_dim, hidden_dim, hidden_dim, hidden_dim]
        # Take pairs (i, i+1) and apply u
        disentangled = []
        for i in range(0, seq_len, 2):
            # x[:, i]: [batch_size, hidden_dim]
            # x[:, i+1]: [batch_size, hidden_dim]
            # We need to contract: u[a,b,c,d] * x[i,a] * x[i+1,b] -> result[c,d]
            pair = torch.einsum('abcd,ba,bb->bcd',
                              self.layer1.u,
                              x[:, i],
                              x[:, i+1])
            disentangled.append(pair)

        # disentangled: list of [batch_size, hidden_dim, hidden_dim]
        disentangled = torch.stack(disentangled, dim=1)  # [batch_size, seq_len//2, hidden_dim, hidden_dim]

        # 2. Apply isometries
        # w: [hidden_dim, hidden_dim, max_bond]
        # Contract: w[a,b,c] * disentangled[..., a, b] -> compressed[..., c]
        compressed = torch.einsum('abc,bxyab->bxyc',
                                 self.layer1.w,
                                 disentangled)

        # compressed: [batch_size, seq_len//2, max_bond]
        return compressed

    def decode(self, compressed):
        """Decode: reconstruct from latent representation.

        Args:
            compressed: [batch_size, seq_len//2, max_bond]

        Returns:
            reconstructed: [batch_size, seq_len, hidden_dim]
        """
        batch_size = compressed.shape[0]
        batch_reconstructed = []

        for b in range(batch_size):
            # Start with scale1 indices
            tensors = []
            for j in range(compressed.shape[1]):
                tensors.append(
                    qtn.Tensor(
                        data=compressed[b, j],
                        inds=(f"scale1_{j}",),
                        tags={"COMPRESSED"},
                    )
                )
            tn = qtn.TensorNetwork(tensors)

            # 1. Apply inverse isometries (w^†)
            # w: [hidden_dim, hidden_dim, max_bond], contract on last index
            # w^†: transpose to expand from scale1 back to (u1_i, u1_{i+1})
            w_dagger = self.layer1.w.permute(
                2, 0, 1
            )  # [max_bond, hidden_dim, hidden_dim]

            for i in range(0, self.seq_len, 2):
                tn &= qtn.Tensor(
                    data=w_dagger,
                    inds=(f"scale1_{i//2}", f"u1_{i}", f"u1_{(i+1)%self.seq_len}"),
                    tags={"L1_W_INV"},
                )

            # 2. Apply inverse disentanglers (u^†)
            # u: [hidden_dim, hidden_dim, hidden_dim, hidden_dim]
            # Permute to reverse the contraction
            u_dagger = self.layer1.u.permute(2, 3, 0, 1)

            for i in range(0, self.seq_len, 2):
                next_i = (i + 1) % self.seq_len
                tn &= qtn.Tensor(
                    data=u_dagger,
                    inds=(f"u1_{i}", f"u1_{next_i}", f"phys_{i}", f"phys_{next_i}"),
                    tags={"L1_U_INV"},
                )

            # 3. Extract reconstructed physical indices
            reconstructed_vecs = []
            for i in range(self.seq_len):
                vec = tn.contract(output_inds=[f"phys_{i}"], optimize="greedy")
                reconstructed_vecs.append(vec.data)

            reconstructed_sample = torch.stack(
                reconstructed_vecs
            )  # [seq_len, hidden_dim]
            batch_reconstructed.append(reconstructed_sample)

        reconstructed = torch.stack(
            batch_reconstructed
        )  # [batch_size, seq_len, hidden_dim]
        return reconstructed


# ==========================================
# Training Loop Verification
# ==========================================
if __name__ == "__main__":
    total_samples = 1
    batch_size = 1
    seq_len = 128
    hidden_dim = 32
    max_bond = 16
    epochs = 50

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Generating Dataset...")
    raw_data = generate_power_law_tree_dataset(
        total_samples, seq_len, hidden_dim, alpha=0.7, device=device
    )

    dataset = TensorDataset(raw_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ClassicalMeraNetwork(seq_len, hidden_dim, max_bond).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    def overlap_loss(reconstructed, target):
        """Compute 1 - |⟨target|reconstructed⟩|² loss.

        This is the standard loss for quantum state/tensor network reconstruction.
        """
        # Flatten to vectors for inner product
        recon_flat = reconstructed.reshape(reconstructed.shape[0], -1)
        target_flat = target.reshape(target.shape[0], -1)

        # Normalize
        recon_norm = recon_flat / (torch.norm(recon_flat, dim=1, keepdim=True) + 1e-10)
        target_norm = target_flat / (
            torch.norm(target_flat, dim=1, keepdim=True) + 1e-10
        )

        # Compute overlap (inner product)
        overlap = torch.sum(recon_norm * target_norm, dim=1)

        # Loss: 1 - |overlap|²
        loss = 1.0 - overlap**2

        return loss.mean()

    # Calculate compression ratio
    original_size = seq_len * hidden_dim
    mera_params = sum(p.numel() for p in model.parameters())
    compression_ratio = original_size / mera_params

    print("\n" + "=" * 50)
    print("STARTING REAL MERA TRAINING LOOP")
    print("=" * 50)
    print(f"Original data size: {original_size:,} parameters")
    print(f"MERA model size: {mera_params:,} parameters")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    print(f"Raw data range: [{raw_data.min().item():.3e}, {raw_data.max().item():.3e}]")
    print("=" * 50 + "\n")

    for epoch in range(epochs):
        epoch_loss = 0.0
        all_reconstructed = []
        all_data = []

        for batch in dataloader:
            batch_x = batch[0]
            optimizer.zero_grad()

            # Encode-decode pass
            compressed = model(batch_x)  # Forward = encode
            reconstructed = model.decode(compressed)  # Decode back to original space

            loss = overlap_loss(reconstructed, batch_x)
            loss.backward()
            optimizer.step()

            # Enforce the multi-scale physics manifold mapping
            model.layer1.project_to_stiefel()

            epoch_loss += loss.item() * batch_x.size(0)
            all_reconstructed.append(reconstructed.detach())
            all_data.append(batch_x.detach())

        # Calculate metrics
        total_epoch_loss = epoch_loss / total_samples

        all_reconstructed = torch.cat(all_reconstructed)
        all_data = torch.cat(all_data)

        # SNR = 10 * log10(signal_power / noise_power)
        signal_power = torch.mean(all_data**2)
        noise_power = torch.mean((all_data - all_reconstructed) ** 2)
        snr_db = 10 * torch.log10(signal_power / (noise_power + 1e-10))

        # Also compute normalized SNR (signal variance / noise variance)
        signal_var = torch.var(all_data)
        noise_var = torch.var(all_data - all_reconstructed)
        normalized_snr_db = 10 * torch.log10(signal_var / (noise_var + 1e-10))

        if epoch == 0:
            print(
                f"  Signal power: {signal_power.item():.3e}, Noise power: {noise_power.item():.3e}"
            )
            # Check gradients
            total_grad = sum(
                p.grad.abs().sum().item()
                for p in model.parameters()
                if p.grad is not None
            )
            print(f"  Total gradient magnitude: {total_grad:.3e}")

        print(
            f"Epoch {epoch+1:02d}/{epochs} | Loss: {total_epoch_loss:.3e} | "
            f"SNR: {snr_db.item():.2f} dB | Norm-SNR: {normalized_snr_db.item():.2f} dB | "
            f"Compression: {compression_ratio:.2f}x"
        )

    print("\nTRAINING COMPLETE!")
