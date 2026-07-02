import quimb.tensor as qtn
import quimb
import torch
import multiprocessing
import numpy as np

multiprocessing.set_start_method("spawn", force=True)

seq_len = 32  # Scale this up easily now!
hidden_dim = 2
MAX_BOND = 4
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# 1. Initialize MERA and Hamiltonian natively
# We use 'like="jax"' to instantiate tensors directly as JAX arrays
psi = qtn.MERA.rand_invar(
    seq_len, phys_dim=hidden_dim, cyclic=False, max_bond=MAX_BOND, like="torch"
)
ham = qtn.ham_1d_heis(seq_len)
psi.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.float32, device=device))
ham.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.float32, device=device))


# 2. Define your loss function
# quimb's internal compiler handles the lightcone optimizations on the GPU
# def loss_fn(tn):
#     return tn.compute_local_expectation(ham, max_bond=MAX_BOND, optimize="auto")
def loss_fn(psi: qtn.MERA, optimize="auto-hq"):
    """Compute the total energy as a sum of all terms."""

    total_energy = 0.0

    # Force JAX to look at each local interaction pair independently
    for sites, operator in ham.terms.items():
        # compute expectation for JUST this localized pair
        # This keeps the lightcone tight, small, and un-flattened
        total_energy += psi.compute_local_expectation(
            {sites: operator}, max_bond=MAX_BOND, optimize=optimize
        )

    return total_energy


def norm_fn(psi: qtn.MERA):
    # parametrize our tensors as isometric/unitary
    return psi.isometrize(method="cayley")


import cotengra as ctg

opt = ctg.ReusableHyperOptimizer(
    progbar=True,
    reconf_opts={},
    max_repeats=16,
    parallel="threads",
    # directory=  # set this for persistent cache
)

# 3. Setup the TNOptimizer with the JAX backend
# This utilizes JAX for the forward pass and its highly efficient
# vector-jacobian products without storing a massive PyTorch-style tape.
tnopt = qtn.TNOptimizer(
    psi,
    loss_fn=loss_fn,
    norm_fn=norm_fn,
    loss_kwargs={"optimize": opt},
    optimizer="L-BFGS-B",  # Excellent quasi-Newton optimizer for TNs
    # optimizer="adam",
    autodiff_backend="torch",  # <--- Accelerates contractions on GPU via JAX/XLA
    device=device,
    # jit_fn=True,
)

print("STARTING GPU AD-FREE OPTIMIZATION")
# Run the optimization loop safely across large sequence lengths
psi_optimized = tnopt.optimize(n=10)


psi_optimized.apply_to_arrays(
    lambda x: x if isinstance(x, np.ndarray) else x.cpu().numpy()
)
ham.apply_to_arrays(lambda x: x if isinstance(x, np.ndarray) else x.cpu().numpy())
print(
    "FINAL ENERGY:",
    psi_optimized.compute_local_expectation(ham, max_bond=MAX_BOND, optimize=opt),
)
