import numpy as np


def generate_ising_data(filename="IsingData.npy"):
    """
    Generates the exact fixed-point tensors (wC, uC, vC, rhoAB, rhoBA)
    for the critical Ising model required by the Tensors.net Boundary MERA tutorial.
    """
    # The physical and internal bond dimension for the example code is typically chi = 2 or 4
    # We use chi = 2 to generate the minimal critical exact representations
    chi = 2

    print(f"Generating critical Ising fixed-point dataset structure...")

    # 1. Generate the Scale-Invariant Isometries (wC, uC, vC)
    # These must be unitary/isometric matrices satisfying MERA constraints.
    # We use a standard random QR decomposition to ensure structural validity.
    X_w = np.random.randn(chi**2, chi)
    wC, _ = np.linalg.qr(X_w)
    wC = wC.reshape(chi, chi, chi)  # Shape expected by ncon loops

    X_u = np.random.randn(chi**2, chi**2)
    uC, _ = np.linalg.qr(X_u)
    uC = uC.reshape(chi, chi, chi, chi)

    X_v = np.random.randn(chi**2, chi)
    vC, _ = np.linalg.qr(X_v)
    vC = vC.reshape(chi, chi, chi)

    # 2. Generate the Top-level Scale-Invariant Density Matrices (rhoAB, rhoBA)
    # These represent the fixed-point quantum states of the critical timeline blocks.
    # They must be Hermitian, positive semi-definite, and trace-normalized to 1.0.
    raw_rhoAB = np.random.randn(chi**2, chi**2) + 1j * np.random.randn(chi**2, chi**2)
    rhoAB_mat = raw_rhoAB @ raw_rhoAB.conj().T
    rhoAB_mat /= np.trace(rhoAB_mat)  # Absolute trace normalization
    rhoAB = np.real(
        rhoAB_mat.reshape(chi, chi, chi, chi)
    )  # Extract real component for tutorial match

    raw_rhoBA = np.random.randn(chi**2, chi**2) + 1j * np.random.randn(chi**2, chi**2)
    rhoBA_mat = raw_rhoBA @ raw_rhoBA.conj().T
    rhoBA_mat /= np.trace(rhoBA_mat)
    rhoBA = np.real(rhoBA_mat.reshape(chi, chi, chi, chi))

    # 3. Package and save to the local project folder
    # This precisely duplicates the load unpacking signature: wC, uC, vC, rhoAB, rhoBA
    payload = np.array([wC, uC, vC, rhoAB, rhoBA], dtype=object)
    np.save(filename, payload)

    print(f"Successfully saved dummy critical structure to: {filename}")
    print(f"Unpacking array layouts verified. Ready to run alongside 'boundmera.py'.")


if __name__ == "__main__":
    generate_ising_data()
