# adaptive_vqe_refactored.py

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from vqe_circuit_hamiltonian import VQECircuit, VQEHamiltonian
from distortion_model import CompositeDistortionModel
from failure_model import DynamicThresholdModel
from recovery_strategy import RecoveryManager

warnings.filterwarnings("ignore")


@dataclass
class VQEConfig:
    """Configuration for Adaptive VQE simulation."""

    n_qubits: int = 4
    layers: int = 2
    max_iterations: int = 200
    hamiltonian_params: Dict[str, Any] = field(default_factory=dict)
    noise_params: Dict[str, Any] = field(default_factory=dict)
    noise_schedule: Dict[str, Any] = field(default_factory=dict)
    distortion_params: Dict[str, Any] = field(default_factory=dict)
    threshold_params: Dict[str, Any] = field(default_factory=dict)
    recovery_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VQEResults:
    """Container for VQE experiment results."""

    energies: List[float] = field(default_factory=list)
    fidelities: List[float] = field(default_factory=list)
    params_history: List[np.ndarray] = field(default_factory=list)
    distortions: List[float] = field(default_factory=list)
    component_distortions: List[Tuple] = field(default_factory=list)
    thresholds: List[float] = field(default_factory=list)
    recoveries: List[bool] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    phases: List[str] = field(default_factory=list)
    noise_levels: List[float] = field(default_factory=list)
    iterations: List[int] = field(default_factory=list)

    def add_iteration(self, **kwargs) -> None:
        """Add data for a single iteration."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                getattr(self, key).append(value)


class NoiseSchedule:
    """
    Simple noise schedule helper.
    - noise_intervals: list of tuples (start, end, noise_level, noise_prob)
    - base_noise_level and base_noise_prob are used outside intervals
                'noise_intervals': [
                (10, 30, 0.15),   # noise level 0.15,
                (40, 60, 0.10),   # noise level 0.10,
                (80, 100, 0.20),  # noise level 0.20,
                (120, 140, 0.05), # noise level 0.05,
            ]
    """

    def __init__(self, noise_intervals=[], base_noise_level=0.0):
        self.noise_intervals = noise_intervals
        self.base_noise_level = base_noise_level

    def get_noise_level(self, iteration: int):
        """Return (noise_level, noise_prob) for the given iteration."""
        for start, end, noise_level in self.noise_intervals:
            if start <= iteration <= end:
                return float(noise_level)
        return self.base_noise_level

    def should_apply_noise(self, iteration: int) -> bool:
        """Randomized decision whether noise is applied this iteration."""
        noise_level = self.get_noise_level(iteration)
        if noise_level <= 0:
            return False
        return True

    def get_current_phase(self, iteration: int) -> str:
        """Return 'NOISY' or 'CLEAN' for the given iteration (uses should_apply_noise)."""
        return "NOISY" if self.should_apply_noise(iteration) else "CLEAN"

    def get_effective_noise(self, iteration: int) -> float:
        """Return the effective noise level (0 if not applied)."""
        if self.should_apply_noise(iteration):
            noise_level = self.get_noise_level(iteration)
            return float(noise_level)
        return 0.0


class AdaptiveVQESimulator:
    """
    Adaptive VQE simulator with modular components.
    """

    def __init__(self, config: VQEConfig):
        self.config = config
        self._initialize_models(config)
        self._initialize_circuit(config)
        self.optimizer_step_count = 0

    def _initialize_models(self, config: VQEConfig) -> None:
        """Initialize all models from configuration."""
        # Distortion model
        dist_cfg = config.distortion_params
        self.distortion_model = CompositeDistortionModel(**dist_cfg)

        # Threshold model
        thr_cfg = config.threshold_params
        self.threshold_model = DynamicThresholdModel(**thr_cfg)

        # Recovery manager
        rec_cfg = config.recovery_params
        self.recovery_manager = RecoveryManager(**rec_cfg)

        noise_sch_cfg = config.noise_schedule
        # Noise schedule
        self.noise_schedule = NoiseSchedule(**noise_sch_cfg)

        # Noise parameters
        noise_cfg = config.noise_params
        self.noise_type = noise_cfg.get("noise_type")
        self.prob_for_each_noise = float(noise_cfg.get("prob_for_each_noise", 0.1))

    def _initialize_circuit(self, config: VQEConfig) -> None:
        """Initialize quantum circuit and Hamiltonian."""
        # Create Hamiltonian
        self.H = VQEHamiltonian.create_hamiltonian(
            config.n_qubits, config.hamiltonian_params
        )

        # Create circuit manager
        self.circuit_manager = VQECircuit(config.n_qubits, config.layers, self.H)
        self.n_params = config.layers * config.n_qubits * 3

    def run_experiment(
        self,
        initial_params: np.ndarray,
        learning_rate: float,
        with_recovery=True,
        use_noise: bool = True,
        use_noisy_device_for_recovery=True,
    ) -> VQEResults:
        """Run a VQE experiment WITH recovery mechanism."""
        results = VQEResults()
        params = initial_params.copy()
        opt = qml.AdamOptimizer(stepsize=learning_rate)
        recovery_count = 0
        max_recoveries = self.recovery_manager.max_recoveries

        prev_probs = None
        current_qnode = self.circuit_manager.qnode

        for iteration in range(self.config.max_iterations):
            # Determine noise settings for this iteration
            noise_level, phase, _ = self._get_iteration_settings(iteration, use_noise)

            # Store iteration metadata
            results.add_iteration(
                phases=phase,
                noise_levels=noise_level,
                iterations=iteration,
                learning_rates=learning_rate,
            )

            # Compute energy and fidelity
            energy, current_probs = self._compute_metrics(
                params,
                noise_level=noise_level,
                noisy_circuit=(phase != "CLEAN"),
                qnode=current_qnode,
            )

            # Update fidelity history (with fix for first iteration)
            if iteration == 0:
                results.fidelities.append(current_probs[0])
            elif prev_probs is not None:
                fidelity = np.inner(current_probs, prev_probs)
                results.fidelities.append(fidelity)

            results.energies.append(float(energy))
            results.params_history.append(params.copy())

            # Compute distortion
            prev_params = (
                results.params_history[-2] if len(results.params_history) >= 2 else None
            )
            total_distortion, comp_tuple = self._compute_distortion(
                results, params, prev_params
            )

            results.distortions.append(float(total_distortion))
            results.component_distortions.append(comp_tuple)

            # Compute threshold

            current_threshold = self.threshold_model.compute_threshold(
                iteration, noise_level
            )
            results.thresholds.append(float(current_threshold))

            # Check for recovery
            did_recover = False
            if with_recovery:
                # Only trigger recovery for no energy improvement and positive (bad) distortion
                if (
                    iteration > 2
                    and results.energies[-1] > results.energies[-2]
                    and total_distortion > current_threshold
                    and recovery_count < max_recoveries
                ):
                    # Consider recovery
                    did_recover, params, learning_rate, opt = self._attempt_recovery(
                        iteration,
                        results,
                        params,
                        learning_rate,
                        opt,
                        total_distortion,
                        current_threshold,
                    )
                    if did_recover:
                        recovery_count += 1
                        # Recompute energy with recovered parameters
                        if use_noisy_device_for_recovery:

                            energy, current_probs = self._compute_metrics(
                                params,
                                noise_level=noise_level,
                                noisy_circuit=True,
                                qnode=current_qnode,
                            )

                        else:
                            energy, current_probs = self._compute_metrics(
                                params,
                                noise_level=0.0,
                                noisy_circuit=False,
                                qnode=current_qnode,
                            )
                        results.energies[-1] = float(energy)
                        results.fidelities[-1] = np.inner(current_probs, prev_probs)
                        results.params_history[-1] = params.copy()
                        # Update recovery effectiveness in manager
                        self.recovery_manager.update_effectiveness(
                            energy_before=(
                                results.energies[-2]
                                if len(results.energies) >= 2
                                else results.energies[-1]
                            ),
                            energy_after=energy,
                        )

            results.recoveries.append(did_recover)

            # Optimization step (skip if we just recovered and want to evaluate recovery first)
            if not did_recover:
                params = opt.step(
                    current_qnode,
                    params,
                    noisy_circuit=(phase != "CLEAN"),
                    noise_level=noise_level,
                    noise_type=self.noise_type,
                    circuit_out_type="energy",
                )

                self.optimizer_step_count += 1

            prev_probs = current_probs
            # Periodic logging
            if (
                iteration % 20 == 0
                or did_recover
                or iteration == self.config.max_iterations - 1
            ):
                self._log_iteration(
                    iteration,
                    energy,
                    total_distortion,
                    current_threshold,
                    phase,
                    noise_level,
                    learning_rate,
                    did_recover,
                )

        return results

    def _get_iteration_settings(self, iteration: int, use_noise: bool) -> Tuple:
        """Get noise settings for current iteration."""
        if use_noise:
            noise_level = self.noise_schedule.get_effective_noise(iteration)
            phase = self.noise_schedule.get_current_phase(iteration)
            current_qnode = self.circuit_manager.qnode
        else:
            noise_level = 0.0
            phase = "CLEAN"
            current_qnode = self.circuit_manager.qnode

        return noise_level, phase, current_qnode

    def _compute_metrics(
        self,
        params: np.ndarray,
        noise_level: float,
        noisy_circuit: bool,
        qnode: Callable,
    ) -> Tuple[float, np.ndarray]:
        """Compute energy and probabilities for current parameters."""
        energy = qnode(
            params,
            noisy_circuit=noisy_circuit,
            noise_level=noise_level,
            noise_type=self.noise_type,
            circuit_out_type="energy",
        )
        probs = qnode(
            params,
            noisy_circuit=noisy_circuit,
            noise_level=noise_level,
            noise_type=self.noise_type,
            circuit_out_type="probs",
        )

        return energy, probs

    def _compute_distortion(
        self, results: VQEResults, params: np.ndarray, prev_params: Optional[np.ndarray]
    ) -> Tuple:
        """Compute distortion metrics."""
        comp_tuple = self.distortion_model.compute_component_distortions(
            energy_history=results.energies,
            fidelity_history=results.fidelities,
            params_current=params,
            params_previous=prev_params,
        )
        distortion = self.distortion_model.compute_total_distortion(comp_tuple)

        return distortion, comp_tuple

    def _attempt_recovery(
        self,
        iteration: int,
        results: VQEResults,
        params: np.ndarray,
        learning_rate: float,
        opt,
        total_distortion: float,
        threshold: float,
    ) -> Tuple:
        """Attempt recovery if conditions are met."""
        # First check if recovery should be triggered
        trigger = self.threshold_model.should_trigger_recovery(
            current_distortion=total_distortion,
            threshold=threshold,
            distortion_history=results.distortions,
            threshold_history=results.thresholds,
            energy_history=results.energies,
            current_iteration=iteration,
        )

        # Then check if recovery is allowed (cooldown, max recoveries, etc.)
        if not (trigger and self.recovery_manager.can_recover(iteration)):
            return False, params, learning_rate, opt

        # Attempt recovery using the recovery manager
        recovered_params, new_lr, success, rtype = (
            self.recovery_manager.attempt_recovery(
                current_iteration=iteration,
                energies=results.energies,
                params_history=results.params_history,
                current_lr=learning_rate,
                current_energy=results.energies[-1],
                current_distortion=total_distortion,
                threshold=threshold,
            )
        )

        if success:
            # Update parameters and optimizer
            params = recovered_params
            # learning_rate = new_lr  # Keep the same learning rate
            opt = qml.AdamOptimizer(stepsize=learning_rate)
            return True, params, learning_rate, opt

        return False, params, learning_rate, opt

    def _log_iteration(
        self,
        iteration: int,
        energy: float,
        distortion: float,
        threshold: float,
        phase: str,
        noise_level: float,
        learning_rate: float,
        recovered: bool,
    ) -> None:
        """Log iteration progress."""
        recovery_str = "RECOVERED" if recovered else ""
        print(
            f"Iter {iteration:4d} | E = {energy:9.6f} | Dist = {distortion:6.4f} | "
            f"Thr = {threshold:6.4f} | Phase = {phase:6} | Noise = {noise_level:.4f} "
            f"LR = {learning_rate:.4f} {recovery_str}"
        )

    def analyze_hamiltonian(self) -> Dict[str, Any]:
        """Analyze the Hamiltonian properties."""
        analysis = VQEHamiltonian.analyze(self.H, self.config.n_qubits)

        print("\n" + "=" * 60)
        print("HAMILTONIAN ANALYSIS")
        print("=" * 60)
        print(f"Number of qubits: {self.config.n_qubits}")
        print(f"Hilbert space dimension: {2**self.config.n_qubits}")
        print(f"Ground state energy: {analysis['ground_energy']:.8f}")

        print("\nEigenvalue spectrum:")
        print(f"  Minimum: {analysis['eigenvalues'][0]:.6f}")
        print(f"  Maximum: {analysis['eigenvalues'][-1]:.6f}")
        print(f"  Range: {analysis['spectral_range']:.6f}")
        print(f"  Mean: {analysis['mean_energy']:.6f}")
        print(f"  Ground-first excited gap: {analysis['spectral_gap']:.6f}")

        print("\nFirst 10 eigenvalues:")
        for i, eigval in enumerate(analysis["eigenvalues"][:10]):
            print(f"  E{i} = {eigval:.6f}")

        print(f"\nDifficulty assessment:")
        print(f"  Spectral gap: {analysis['spectral_gap']:.6f}")
        print(f"  Spectral range: {analysis['spectral_range']:.6f}")

        if analysis["spectral_gap"] < 0.1:
            print("  ⚠️  Small spectral gap - may be challenging for VQE")
        else:
            print("  ✅ Reasonable spectral gap")

        if analysis["spectral_range"] > 5.0:
            print("  ⚠️  Large spectral range - optimization landscape may be rough")
        else:
            print("  ✅ Moderate spectral range")

        return analysis

    def run_four_case_comparison(
        self, learning_rate: float = 0.01, use_noisy_device_for_recovery=True
    ) -> Dict[str, VQEResults]:
        """Run all four comparison cases with the same initial parameters."""
        np.random.seed(42)
        initial_params = np.random.normal(0, 0.5, self.n_params)

        print("=" * 80)
        print("FOUR-CASE VQE COMPARISON SIMULATION")
        print("=" * 80)
        print(
            f"Configuration: {self.config.n_qubits} qubits, {self.config.layers} layers, "
            f"{self.config.max_iterations} iterations"
        )
        print(f"Learning rate: {learning_rate}")
        print()

        cases = {
            "ideal": self.run_experiment(
                initial_params.copy(),
                learning_rate,
                with_recovery=False,
                use_noise=False,
                use_noisy_device_for_recovery=use_noisy_device_for_recovery,
            ),
            "ideal_recovery": self.run_experiment(
                initial_params.copy(),
                learning_rate,
                use_noise=False,
                with_recovery=True,
                use_noisy_device_for_recovery=use_noisy_device_for_recovery,
            ),
            "noisy": self.run_experiment(
                initial_params.copy(),
                learning_rate,
                use_noise=True,
                with_recovery=False,
                use_noisy_device_for_recovery=use_noisy_device_for_recovery,
            ),
            "noisy_recovery": self.run_experiment(
                initial_params.copy(),
                learning_rate,
                use_noise=True,
                with_recovery=True,
                use_noisy_device_for_recovery=use_noisy_device_for_recovery,
            ),
        }

        return cases

    def print_four_case_results(
        self, histories: Dict[str, VQEResults], ground_energy: Optional[float] = None
    ) -> None:
        """Print comprehensive results analysis for all four cases."""
        print("\n" + "=" * 90)
        print("FOUR-CASE VQE COMPARISON - COMPREHENSIVE RESULTS")
        print("=" * 90)

        case_data = [
            ("Ideal (no noise)", histories["ideal"]),
            ("Ideal + Recovery", histories["ideal_recovery"]),
            ("Noisy (no recovery)", histories["noisy"]),
            ("Noisy + Recovery", histories["noisy_recovery"]),
        ]

        print(
            f"{'Case':<25} {'Best Energy':<15} {'Error':<15} {'Recoveries':<12} {'Final LR':<10}"
        )
        print("-" * 90)

        for case_name, hist in case_data:
            best_energy = np.min(hist.energies)
            error = abs(best_energy - ground_energy) if ground_energy else 0
            recoveries = sum(hist.recoveries)
            final_lr = hist.learning_rates[-1] if hist.learning_rates else "N/A"

            print(
                f"{case_name:<25} {best_energy:<15.8f} {error:<15.8f} "
                f"{recoveries:<12} {final_lr:<10.6f}"
            )

        print("-" * 90)

        # Calculate improvements
        ideal_best = np.min(histories["ideal"].energies)
        noisy_best = np.min(histories["noisy"].energies)
        noisy_recovery_best = np.min(histories["noisy_recovery"].energies)

        improvement_noisy_vs_ideal = noisy_best - ideal_best
        improvement_recovery_vs_noisy = noisy_best - noisy_recovery_best
        recovery_efficiency = (
            improvement_recovery_vs_noisy / improvement_noisy_vs_ideal
            if improvement_noisy_vs_ideal != 0
            else 0
        )

        print(f"\n📊 KEY INSIGHTS:")
        print(f"  • Noise degrades performance by: {improvement_noisy_vs_ideal:.6f}")
        print(
            f"  • Recovery improves noisy case by: {improvement_recovery_vs_noisy:.6f}"
        )
        print(
            f"  • Recovery recovers {recovery_efficiency*100:.1f}% of noise degradation"
        )

        if ground_energy:
            ideal_error = abs(ideal_best - ground_energy)
            noisy_recovery_error = abs(noisy_recovery_best - ground_energy)
            accuracy_preservation = (
                (1 - noisy_recovery_error / ideal_error) * 100
                if ideal_error != 0
                else 0
            )
            print(
                f"  • Accuracy preservation: {accuracy_preservation:.1f}% of ideal accuracy"
            )

        print("=" * 90)


def create_demonstration_config():
    """Create configuration with mixed noise/noise-free intervals"""
    import numpy as np

    num_of_iterations = 220
    noisy_iter = 2  # short: 2, half:6, long:9
    config = {
        "n_qubits": 7,
        "layers": 3,
        "max_iterations": num_of_iterations,
        "random_seed": 42,
        # Mixed noise schedule with probabilities
        "noise_schedule": {
            "noise_intervals": [
                (i, i + noisy_iter, np.random.rand() * 0.01)
                for i in range(1, num_of_iterations, 10)
            ],
        },
        # Recovery parameters
        "recovery_params": {
            "cooldown": 3,  # cool down for recoveries used in adaptive cool down
            "max_recoveries": num_of_iterations,
            "min_spacing": 3,  # space between recoveries
            "adaptive_cooldown": True,
        },
        "distortion_params": {
            "energy_window": 10,
            "weight_energy": 0.85,
            "weight_fidelity": 0.05,
            "weight_parameter": 0.05,
            "weight_convergence": 0.05,
        },
        # Threshold model
        "threshold_params": {
            "tau_min": 0.04,
            "tau_max": 0.35,
            "decay_rate": 0.025,
            "failure_tolerance": 0.15,
            "kappa": 0.9,
            "derivative_threshold": 0.0015,
            "degradation_threshold": 0.07,
            "noisy_multiplier": 1.3,
        },
        # Hamiltonian
        "hamiltonian_params": {
            "local_strength": -1.8,
            "interaction_strengths": [0.9, 0.6, 0.4],
            "include_next_nearest": True,
            "next_nearest_strength": 0.5,
            "add_random_terms": True,
        },
    }
    return config


if __name__ == "__main__":
    """Main function with Hamiltonian analysis and four-case comparison."""
    print("🧪 QUANTUM DISTORTION-AWARE VQE: FOUR-CASE COMPARISON")
    print("=" * 80)

    # Create configuration

    from vqe_plotting import VQEPlotter

    config_dict = create_demonstration_config()
    config = VQEConfig(
        n_qubits=int(config_dict.get("n_qubits", 4)),
        layers=int(config_dict.get("layers", 2)),
        max_iterations=int(config_dict.get("max_iterations", 200)),
        hamiltonian_params=config_dict.get("hamiltonian_params", {}),
        noise_params=config_dict.get("noise_params", {}),
        noise_schedule=config_dict.get("noise_schedule", {}),
        distortion_params=config_dict.get("distortion_params", {}),
        threshold_params=config_dict.get("threshold_params", {}),
        recovery_params=config_dict.get("recovery_params", {}),
    )

    # Create simulator
    simulator = AdaptiveVQESimulator(config)

    # Analyze Hamiltonian before running VQE
    analysis = simulator.analyze_hamiltonian()
    ground_energy = analysis["ground_energy"]

    print("\n" + "=" * 80)
    print("STARTING FOUR-CASE VQE OPTIMIZATION")
    print("=" * 80)

    # Run four-case comparison
    histories = simulator.run_four_case_comparison(
        learning_rate=0.01, use_noisy_device_for_recovery=False
    )

    # Create plotter and plot results
    plotter = VQEPlotter(
        simulator.noise_schedule,
        simulator.config.max_iterations,
        simulator.config.n_qubits,
    )
    plotter.plot_four_case_results(histories, ground_energy)

    # Print detailed results
    simulator.print_four_case_results(histories, ground_energy)
