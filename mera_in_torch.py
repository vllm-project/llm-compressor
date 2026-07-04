import quimb.tensor as qtn
import quimb
import torch
import tqdm
import warnings
import cotengra as ctg

# our ansatz and hamiltonian
seq_len = 128
hidden_dim = 2
MAX_BOND = 2
N_EPOCHS = 40


def norm_fn(psi: qtn.MERA):
    # parametrize our tensors as isometric/unitary
    # return psi.isometrize(method="cayley")
    return psi.isometrize(method="exp")


# def loss_fn(psi: qtn.MERA, ham: qtn.LocalHam1D, optimize="auto-hq"):
#     # compute the total energy, here quimb handles constructing
#     # and contracting all the appropriate lightcones
#     return psi.compute_local_expectation(ham, max_bond=MAX_BOND, optimize=optimize)


# def loss_fn(mera, terms, **kwargs):
#     """Compute the total energy as a sum of all terms."""

#     def local_expectation(mera, terms, where, optimize="auto-hq"):
#         """Compute the energy for a single local term."""
#         # get the lightcone for `where`
#         tags = [mera.site_tag(coo) for coo in where]
#         mera_ij = mera.select(tags, "any")

#         # apply the local gate
#         G = terms[where]
#         mera_ij_G = mera_ij.gate(terms[where], where)

#         # compute the overlap - this is where the real computation happens
#         mera_ij_ex = mera_ij_G & mera_ij.H
#         return mera_ij_ex.contract(all, optimize=optimize)

#     return sum(local_expectation(mera, terms, where, **kwargs) for where in terms)


def loss_fn(psi: qtn.MERA, ham: qtn.LocalHam1D, optimize="auto-hq"):
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


class MeraModel(torch.nn.Module):
    def __init__(self, tn):
        super().__init__()
        # extract the raw arrays and a skeleton of the TN
        params, self.skeleton = qtn.pack(tn)
        # n.b. you might want to do extra processing here to e.g. store each
        # parameter as a reshaped matrix (from left_inds -> right_inds), for
        # some optimizers, and for some torch parametrizations
        self.torch_params = torch.nn.ParameterDict(
            {
                # torch requires strings as keys
                str(i): torch.nn.Parameter(initial)
                for i, initial in params.items()
            }
        )

    def get_psi(self):
        # Reconstruct the current Tensor Network state
        params = {int(i): p for i, p in self.torch_params.items()}
        psi = qtn.unpack(params, self.skeleton)
        return norm_fn(psi)


if __name__ == "__main__":
    warnings.filterwarnings(
        action="ignore", message=".*The contraction tree is not a compressed one.*"
    )
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    psi: qtn.MERA = qtn.MERA.rand_invar(
        seq_len, phys_dim=hidden_dim, max_bond=MAX_BOND, seed=42, cyclic=False
    )
    ham: qtn.LocalHam1D = qtn.ham_1d_heis(seq_len)

    psi.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.float32).to(device))
    ham.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.float32).to(device))

    # psi.draw(
    #     color=["UNI", "ISO"],
    #     fix={psi.site_ind(i): (i, 0) for i in range(seq_len)},
    # )

    print("CREATING MODEL")
    model = MeraModel(psi)

    print("CREATING HYPEROPTIMIZER")
    # opt = "branch-2"
    opt = ctg.ReusableHyperOptimizer(
        progbar=True,
        reconf_opts={},
        max_repeats=16,
        parallel="threads",
        # directory=  # set this for persistent cache
    )

    with torch.no_grad():
        # print("INITIAL LOSS", loss_fn(psi, ham, optimize=opt))
        print("INITIAL OVERLAP", psi.H @ psi)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print("TRAINING")
    for epoch in (pbar := tqdm.tqdm(range(N_EPOCHS))):
        optimizer.zero_grad()
        epoch_loss = 0.0

        for sites, operator in ham.terms.items():
            # 1. Regenerate a clean, isolated forward graph for this step
            current_psi = model.get_psi()

            # 2. Extract ONLY the tensors inside the causal cone for these sites
            site_tags = [current_psi.site_tag(s) for s in sites]
            causal_cone = current_psi.select(site_tags, which="any")

            # 3. Apply the local Hamiltonian gate matrix to the cone
            # (By default, quimb handles index matching for gates)
            causal_cone_with_gate = causal_cone.gate(operator, sites)

            # 4. Form the lazy inner product network: <psi_cone | H_local | psi_cone>
            # The .H operator cleanly flips the bra/ket indices
            local_expectation_network = causal_cone_with_gate & causal_cone.H

            # 5. Contract the entire block in one perfectly-optimized pairwise sequence.
            # Because we contract 'all', 'greedy' can smoothly zip up matching legs
            # layer-by-layer without ever crossing PyTorch's 25-dimension threshold!
            local_loss = local_expectation_network.contract(all, optimize=opt)

            # 6. Backpropagate and clear the VRAM for this term immediately
            local_loss.backward()

            epoch_loss += local_loss.item()

        optimizer.step()
        pbar.set_description(f"Energy (Loss): {epoch_loss:.6f}")

    # for _ in (pbar := tqdm.tqdm(range(N_EPOCHS))):
    #     optimizer.zero_grad()
    #     epoch_loss = 0.0

    #     # 4. CRITICAL: Accumulate gradients term-by-term
    #     for sites, operator in ham.terms.items():
    #         current_psi = model.get_psi()
    #         # Compute expectation value for ONLY this local pair
    #         local_loss = current_psi.compute_local_expectation(
    #             {sites: operator}, max_bond=MAX_BOND, optimize=opt
    #         )

    #         # Backpropagate this isolated term immediately
    #         local_loss.backward()

    #         # Track the total energy scalar
    #         epoch_loss += local_loss.item()

    #     # 5. Take an optimization step once all terms have accumulated
    #     optimizer.step()

    #     pbar.set_description(f"Energy (Loss): {epoch_loss:.6f}")

    # for _ in pbar:
    #     optimizer.zero_grad()
    #     loss = model(ham)
    #     loss.backward()
    #     optimizer.step()
    #     pbar.set_description(f"Loss: {loss}")

    mera_opt = psi.copy()
    params = {
        int(i): model.torch_params.get_parameter(str(i)).detach()
        for i in mera_opt.get_params()
    }
    mera_opt.set_params(params)

    # then we want the constrained form
    mera_opt = norm_fn(mera_opt)

    # compute the energy
    with torch.no_grad():
        # print("FINAL LOSS", loss_fn(mera_opt, ham, optimize=opt))
        print("FINAL OVERLAP", mera_opt.H @ mera_opt)
