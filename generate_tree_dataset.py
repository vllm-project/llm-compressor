import torch


def generate_power_law_tree_dataset(
    num_samples, seq_len, hidden_dim, alpha=0.7, device="cpu"
):
    """Generates synthetic sequence data with cross-site correlations that decay

    according to a power-law spectrum, optimized to run natively on the target hardware device.
    """
    # 1. Allocate white noise directly on the target device (e.g., 'cuda')
    noise = torch.randn(
        num_samples, hidden_dim, seq_len, dtype=torch.float32, device=device
    )

    # 2. Build the filter spectrum on the exact same device
    frequencies = torch.fft.fftfreq(seq_len, device=device)

    # Avoid division by zero at the DC component safely
    frequencies_clamped = torch.where(frequencies == 0, frequencies[1], frequencies)

    amplitude_filter = 1.0 / (torch.abs(frequencies_clamped) ** (alpha / 2.0))

    # Use torch.where instead of in-place assignment to preserve the autograd graph if needed
    amplitude_filter = torch.where(
        frequencies == 0, torch.tensor(0.0, device=device), amplitude_filter
    )
    amplitude_filter = amplitude_filter.view(1, 1, seq_len)

    # 3. Fast Fourier Transform
    noise_fft = torch.fft.fft(noise, dim=-1)
    filtered_fft = noise_fft * amplitude_filter

    # 4. Inverse Transform back to spatial sequence
    spatial_data = torch.fft.ifft(filtered_fft, dim=-1).real

    # 5. Permute to [Batch, Seq_Len, Hidden_Dim]
    return spatial_data.permute(0, 2, 1).contiguous()
