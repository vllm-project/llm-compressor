import pennylane as qml
from pennylane import numpy as np

# 1. Define network parameters: 8 classical data points (3 hierarchical layers)
n_wires = 8
dev = qml.device("default.qubit", wires=n_wires)


# Define our basic 2-qubit parametric MERA block (acting as our tensor)
def mera_block(w, wires):
    qml.RY(w[0], wires=wires[0])
    qml.RY(w[1], wires=wires[1])
    qml.CNOT(wires=wires)


# 2. Define the MERA architecture explicitly
@qml.qnode(dev)
def mera_compress(features, weights):
    # Encode classical data into the qubits (Feature Map)
    qml.AngleEmbedding(features, wires=range(n_wires), rotation="X")

    # --- LAYER 1: 8 Wires -> 4 Wires ---
    # Disentanglers (Even-odd shifts to clear boundary entanglement)
    mera_block(weights[0], wires=[1, 2])
    mera_block(weights[1], wires=[3, 4])
    mera_block(weights[2], wires=[5, 6])
    mera_block(weights[3], wires=[7, 0])

    # Isometries (Coarse-graining blocks)
    mera_block(weights[4], wires=[0, 1])
    mera_block(weights[5], wires=[2, 3])
    mera_block(weights[6], wires=[4, 5])
    mera_block(weights[7], wires=[6, 7])

    # --- LAYER 2: 4 Wires -> 2 Wires ---
    # (Operating only on the coarse-grained surviving index wires: 0, 2, 4, 6)
    mera_block(weights[8], wires=[2, 4])
    mera_block(weights[9], wires=[6, 0])

    mera_block(weights[10], wires=[0, 2])
    mera_block(weights[11], wires=[4, 6])

    # --- LAYER 3: 2 Wires -> 1 Wire ---
    # (Operating on the final surviving index wires: 0, 4)
    mera_block(weights[12], wires=[0, 4])

    # Measure the expectation value of the final core compressed qubit
    return qml.expval(qml.PauliZ(0))


# 3. Initialize data and parameters safely using basic NumPy shapes
np.random.seed(42)
dummy_classical_data = np.random.uniform(0, np.pi, n_wires)

# We have 13 blocks total, each block takes 2 rotational parameters
num_blocks = 13
mera_weights = np.random.uniform(0, 2 * np.pi, (num_blocks, 2), requires_grad=True)

# 4. Run compression pass
compressed_output = mera_compress(dummy_classical_data, mera_weights)
print(f"Compressed Output Measurement: {compressed_output:.4f}")
