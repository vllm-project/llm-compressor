import quimb.tensor as qtn
import quimb

import torch
import tqdm

# our ansatz and hamiltonian
seq_len = 128
hidden_dim = 2
MAX_BOND = 2


def norm_fn(psi: qtn.MERA):
    # parametrize our tensors as isometric/unitary
    return psi.isometrize(method="cayley")


# def loss_fn(psi: qtn.MERA, ham: qtn.LocalHam1D):
#     # compute the total energy, here quimb handles constructing
#     # and contracting all the appropriate lightcones
#     return psi.compute_local_expectation(ham, max_bond=MAX_BOND, optimize="auto-hq")


def loss_fn(psi: qtn.MERA, ham: qtn.LocalHam1D):
    """Compute the total energy as a sum of all terms."""

    total_energy = 0.0

    # Force JAX to look at each local interaction pair independently
    for sites, operator in ham.terms.items():
        print("LOCAL EXP", sites, operator)
        # compute expectation for JUST this localized pair
        # This keeps the lightcone tight, small, and un-flattened
        total_energy += psi.compute_local_expectation(
            {sites: operator}, max_bond=MAX_BOND, optimize="auto"
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

    def forward(self, ham):
        # convert back to original int key format
        params = {int(i): p for i, p in self.torch_params.items()}
        # reconstruct the TN with the new parameters
        psi = qtn.unpack(params, self.skeleton)
        # isometrize and then return the energy
        return loss_fn(norm_fn(psi), ham)


if __name__ == "__main__":
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
    model(ham)

    print("INITIAL LOSS", model(ham))
    print("INITIAL OVERLAP", psi.H @ psi)

    # with warnings.catch_warnings():
    #     warnings.filterwarnings(
    #         action="ignore",
    #         message=".*trace might not generalize.*",
    #     )
    #     model = torch.jit.trace_module(model, {"forward": [ham]})

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    its = 100
    pbar = tqdm.tqdm(range(its))
    print("TRAINING MODEL")

    for _ in pbar:
        optimizer.zero_grad()
        loss = model(ham)
        loss.backward()
        optimizer.step()
        pbar.set_description(f"Loss: {loss}")

    mera_opt = psi.copy()
    params = {
        int(i): model.torch_params.get_parameter(str(i)).detach()
        for i in mera_opt.get_params()
    }
    mera_opt.set_params(params)

    # then we want the constrained form
    mera_opt = norm_fn(mera_opt)

    # compute the energy
    print("FINAL LOSS", model(ham))
    print("FINAL OVERLAP", mera_opt.H @ mera_opt)
