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
        self.layer1 = MeraLayer(hidden_dim, max_bond)

        # The output features will be exactly equal to max_bond (8 channels)
        self.total_flat_features = max_bond
        self.regression_head = torch.nn.Linear(self.total_flat_features, 1)

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        batch_outputs = []
        num_output_legs = seq_len // 2

        for b in range(batch_size):
            tensors = []
            for i in range(seq_len):
                tensors.append(
                    qtn.Tensor(data=x[b, i], inds=(f"phys_{i}",), tags={f"INPUT_{i}"})
                )
            tn = qtn.TensorNetwork(tensors)

            # 1. Apply Layer 1 Disentanglers
            for i in range(0, seq_len, 2):
                next_i = (i + 1) % seq_len
                tn &= qtn.Tensor(
                    data=self.layer1.u,
                    inds=(f"phys_{i}", f"phys_{next_i}", f"u1_{i}", f"u1_{next_i}"),
                    tags={"L1_U"},
                )

            # 2. Apply Layer 1 Isometries
            for i in range(0, seq_len, 2):
                next_i = (i + 1) % seq_len
                tn &= qtn.Tensor(
                    data=self.layer1.w,
                    inds=(f"u1_{i}", f"u1_{next_i}", f"scale1_{i//2}"),
                    tags={"L1_W"},
                )

            # 3. FIXED: Spatial Identity Average Pooling Matrix
            # Create an [8, 8] identity matrix scaled by 1 / num_legs
            # This routes channel i of site j directly to channel i of the global feature
            pool_matrix = (
                torch.eye(self.layer1.w.shape[-1], device=x.device) / num_output_legs
            )

            for j in range(num_output_legs):
                tn &= qtn.Tensor(
                    data=pool_matrix,
                    inds=(f"scale1_{j}", "global_feature"),
                    tags={"MEAN_POOL"},
                )

            # 4. Contract down to the 8-dimensional global feature channel index
            sample_compressed = tn.contract(
                output_inds=["global_feature"], optimize="greedy"
            )

            # sample_compressed.data now has the perfect shape: torch.Size([8])
            batch_outputs.append(sample_compressed.data)

        feature_matrix = torch.stack(batch_outputs)  # Shape: [Batch, max_bond]
        predictions = self.regression_head(feature_matrix)
        return predictions.squeeze(-1)


# ==========================================
# Training Loop Verification
# ==========================================
if __name__ == "__main__":
    total_samples = 1
    batch_size = 1
    seq_len = 4096
    hidden_dim = 32
    max_bond = 8
    epochs = 50

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Generating Dataset...")
    raw_data = generate_power_law_tree_dataset(
        total_samples, seq_len, hidden_dim, alpha=0.7, device=device
    )

    # Scale up target magnitude so the loss values are human-readable
    mock_targets = torch.mean(raw_data, dim=[1, 2]) * 100.0

    dataset = TensorDataset(raw_data, mock_targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ClassicalMeraNetwork(seq_len, hidden_dim, max_bond).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    print("\n" + "=" * 50)
    print("STARTING REAL MERA TRAINING LOOP")
    print("=" * 50)

    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()

            # Forward pass through MERA + Linear Head
            predictions = model(batch_x)

            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()

            # Enforce the multi-scale physics manifold mapping
            model.layer1.project_to_stiefel()

            epoch_loss += loss.item() * batch_x.size(0)

        total_epoch_loss = epoch_loss / total_samples
        print(
            f"Epoch {epoch+1:02d}/{epochs} | Avg Training Loss: {total_epoch_loss:.3e}"
        )

    print("\nTRAINING COMPLETE!")
