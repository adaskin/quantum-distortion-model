import pennylane as qml
from pennylane import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
class VQEHamiltonian:
    """Analyzes Hamiltonian properties."""
    
    @staticmethod
    def create_hamiltonian(n_qubits: int, params: Dict[str, Any]) -> qml.Hamiltonian:
        """Creates a Hamiltonian based on configuration parameters."""
        coeffs, obs = [], []
        local_strength = float(params.get("local_strength", -1.2))
        
        # Local terms
        for i in range(n_qubits):
            coeffs.append(local_strength)
            obs.append(qml.PauliZ(i))
        
        # Nearest-neighbor interactions
        interaction_strengths = params.get("interaction_strengths", [0.8, 0.5, 0.3])
        for i in range(n_qubits - 1):
            coeffs.extend(interaction_strengths)
            obs.extend([
                qml.PauliX(i) @ qml.PauliX(i + 1),
                qml.PauliY(i) @ qml.PauliY(i + 1),
                qml.PauliZ(i) @ qml.PauliZ(i + 1),
            ])
        
        # Next-nearest neighbor terms
        if params.get("include_next_nearest", True):
            next_strength = float(params.get("next_nearest_strength", 0.4))
            for i in range(n_qubits - 2):
                coeffs.append(next_strength)
                obs.append(qml.PauliZ(i) @ qml.PauliZ(i + 2))
        
        # Random terms
        if params.get("add_random_terms", True):
            np.random.seed(int(params.get("random_seed", 42)))
            for i in range(n_qubits):
                if np.random.random() < 0.3:
                    coeffs.append(float(np.random.uniform(-0.5, 0.5)))
                    pauli_choice = np.random.choice([qml.PauliX, qml.PauliY, qml.PauliZ])
                    obs.append(pauli_choice(i))
        
        return qml.Hamiltonian(coeffs, obs)
    
    @staticmethod
    def get_eigenvalues(hamiltonian: qml.Hamiltonian, n_qubits: int, method: str = "matrix") -> np.ndarray:
        """Computes Hamiltonian eigenvalues."""
        if method == "matrix":
            H_matrix = qml.matrix(hamiltonian, wire_order=range(n_qubits))
            eigenvalues = np.linalg.eigvalsh(H_matrix)
        elif method == "sparse":
            from scipy.sparse.linalg import eigsh
            H_sparse = qml.SparseHamiltonian(hamiltonian, wires=range(n_qubits))
            eigenvalues, _ = eigsh(
                H_sparse.sparse_matrix, k=min(8, 2**n_qubits), which="SA"
            )
        elif method == "eigvals":
            eigenvalues = np.linalg.eigvalsh(
                qml.utils.sparse_hamiltonian(hamiltonian).toarray()
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return np.sort(eigenvalues)
    
    @staticmethod
    def analyze(hamiltonian: qml.Hamiltonian, n_qubits: int) -> Dict[str, Any]:
        """Performs comprehensive Hamiltonian analysis."""
        eigenvalues = VQEHamiltonian.get_eigenvalues(hamiltonian, n_qubits)
        ground_energy = eigenvalues[0]
        spectral_gap = eigenvalues[1] - eigenvalues[0] if len(eigenvalues) > 1 else 0
        
        return {
            'eigenvalues': eigenvalues,
            'ground_energy': ground_energy,
            'spectral_gap': spectral_gap,
            'spectral_range': eigenvalues[-1] - eigenvalues[0],
            'mean_energy': np.mean(eigenvalues),
        }



class VQECircuit:
    """Manages the quantum circuit for VQE."""
    
    def __init__(self, n_qubits: int, layers: int, hamiltonian: qml.Hamiltonian):
        self.n_qubits = n_qubits
        self.layers = layers
        self.H = hamiltonian
        self.H_matrix = qml.matrix(self.H, wire_order=range(self.n_qubits))
        
        # Create devices
        # We use the same device, just in case
        self.device_noisy = qml.device("default.mixed", wires=self.n_qubits)
        # self.device_noiseless = qml.device("default.mixed", wires=self.n_qubits)
        
        # Create QNodes
        self.qnode = qml.QNode(self._circuit_template, self.device_noisy)
    
    def _circuit_template(self, params,
                          noisy_circuit=False, noise_level=0.0, 
                          noise_type="combined",circuit_out_type="energy") -> Any:
        """Template circuit that can be used with different noise settings."""
        
        # Ensure params is a numpy array
        if not isinstance(params, np.ndarray):
            params = np.array(params)
        
        p = params.reshape(self.layers, self.n_qubits, 3)

        
        # Initial Hadamard layer
        for q in range(self.n_qubits):
            qml.Hadamard(wires=q)
        
        # Main circuit layers
        for layer in range(self.layers):
            # Single-qubit rotations
            for q in range(self.n_qubits):
                qml.RX(p[layer, q, 0], wires=q)
                qml.RY(p[layer, q, 1], wires=q)
                qml.RZ(p[layer, q, 2], wires=q)
                # noise on this qubit
                if noisy_circuit:
                    self._apply_noise(
                        noise_level=noise_level,
                        wires=q,
                        noise_type=noise_type)
            
            # Entangling layers
            if layer % 2 == 0:
                for q in range(self.n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
                    # noise on these two qubits
                    if noisy_circuit:
                        self._apply_noise(
                            noise_level=noise_level,
                            wires=q,
                            noise_type=noise_type)
                        self._apply_noise(
                            noise_level=noise_level,
                            wires=q+1,
                            noise_type=noise_type,)
            else:
                for q in range(0, self.n_qubits - 2, 2):
                    if q + 2 < self.n_qubits:
                        qml.CNOT(wires=[q, q + 2])
                        # noise on these two qubits
                        if noisy_circuit:
                            self._apply_noise(
                                noise_level=noise_level,
                                wires=q,
                                noise_type=noise_type)
                            self._apply_noise(
                                noise_level=noise_level,
                                wires=q+2,
                                noise_type=noise_type,)
        
        # Measurement
        if circuit_out_type == "probs":
            return qml.probs(wires=range(self.n_qubits))
        elif circuit_out_type == "energy":
            return qml.expval(self.H)
        else:
            raise ValueError(f"Unknown circuit_out_type: {circuit_out_type}")
    
    def _apply_noise(self, noise_level: float=0.1,
                                wires: int=0,
                                noise_type = "combined"
                                    ):
        """Apply a single-qubit noise channel to `wire`."""

        if noise_level <= 0:
            return
        
        if noise_type == "depolarizing":
            qml.DepolarizingChannel(noise_level, wires=wires)
        elif noise_type == "amplitude_damping":
            qml.AmplitudeDamping(noise_level, wires=wires)
        elif noise_type == "phase_damping":
            qml.PhaseDamping(noise_level, wires=wires)
        elif noise_type == "bit_flip":
            qml.BitFlip(noise_level, wires=wires)
        else:
            # apply all if unknown
            qml.BitFlip(p = noise_level, wires=wires)
            qml.PhaseDamping(noise_level, wires=wires)
            qml.AmplitudeDamping(noise_level, wires=wires)
            qml.DepolarizingChannel(noise_level, wires=wires)
