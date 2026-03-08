import torch
from glq.codebook import E8ShellCodebook
from glq.ldlq import quantize_ldlq_codebook, quantize_ldlq_codebook_2stage
from glq.rht import RHT

GLQ_PRECISION = torch.float32

__all__ = ["glq_quantize_weight"]


def glq_quantize_weight(
    W: torch.Tensor,
    H: torch.Tensor,
    bits: int,
    codebook: E8ShellCodebook,
    codebook_small: E8ShellCodebook | None = None,
    dampening_frac: float = 0.01,
    tune_iters: int = 0,
) -> tuple[torch.Tensor, float]:
    """
    Quantize a weight matrix using GLQ (E8 codebook + RHT + LDLQ).

    Args:
        W: (m, n) weight matrix
        H: (n, n) Hessian (already divided by num_samples)
        bits: 2, 3, or 4 bpw
        codebook: E8ShellCodebook (65536 entries)
        codebook_small: E8ShellCodebook (256 entries) for 3bpw, or None
        dampening_frac: Hessian damping fraction
        tune_iters: LDLQ refinement iterations

    Returns:
        (W_hat, proxy_loss): dequantized weight and proxy loss
    """
    m, n = W.shape
    device = W.device
    dtype = W.dtype

    W = W.to(dtype=GLQ_PRECISION)
    H = H.to(dtype=GLQ_PRECISION, device=device)

    # Dampen Hessian diagonal for numerical stability
    if dampening_frac > 0.0:
        damp = dampening_frac * torch.mean(torch.diag(H))
        diag = torch.arange(H.shape[-1], device=H.device)
        H[diag, diag] += damp

    # Apply RHT incoherence transform
    rht = RHT(m, n, device=device)
    W_pad = rht.transform_weights(W)
    H_pad = rht.transform_hessian(H)

    # Move codebook to device
    if codebook.device != device:
        codebook = codebook.to(device)
    if codebook_small is not None and codebook_small.device != device:
        codebook_small = codebook_small.to(device)

    # Dispatch on bits
    if bits == 2:
        result = quantize_ldlq_codebook(
            W_pad,
            H_pad,
            codebook,
            tune_iters=tune_iters,
        )
    elif bits == 3:
        if codebook_small is None:
            raise ValueError("3bpw requires codebook_small (256 entries)")
        result = quantize_ldlq_codebook_2stage(
            W_pad,
            H_pad,
            codebook,
            codebook_small,
            resid_scale=codebook.resid_scale,
            tune_iters=tune_iters,
        )
    elif bits == 4:
        result = quantize_ldlq_codebook_2stage(
            W_pad,
            H_pad,
            codebook,
            codebook,
            resid_scale=codebook.resid_scale,
            tune_iters=tune_iters,
        )
    else:
        raise ValueError(f"Unsupported bits={bits}, must be 2, 3, or 4")

    W_hat_pad = result["W_hat"]
    proxy_loss = result["proxy_loss"]

    # Inverse RHT to get back to original domain
    W_hat = rht.inverse_transform_weights(W_hat_pad)

    return W_hat.to(dtype=dtype), proxy_loss
