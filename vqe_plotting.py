
import matplotlib.pyplot as plt
from vqe_simulator import NoiseSchedule,VQEResults
from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np

class VQEPlotter:
    """Handles plotting for VQE results."""
    
    def __init__(self, noise_schedule: NoiseSchedule, max_iterations: int):
        self.noise_schedule = noise_schedule
        self.max_iterations = max_iterations
    
    def plot_four_case_results(self, histories: Dict[str, VQEResults], ground_energy: float) -> None:
        """Create comprehensive visualization for all four cases."""
        BIGGER_SIZE = 24
        plt.rc("font", size=BIGGER_SIZE)
        plt.rc("axes", titlesize=BIGGER_SIZE)
        plt.rc("axes", labelsize=BIGGER_SIZE)
        plt.rc("xtick", labelsize=BIGGER_SIZE - 2)
        plt.rc("ytick", labelsize=BIGGER_SIZE - 2)
        plt.rc("legend", fontsize=BIGGER_SIZE - 2)
        plt.rc("figure", titlesize=BIGGER_SIZE)
        
        fig, axes = plt.subplots(2, 3, figsize=(24, 14))
        fig.suptitle(
            "Four-Case VQE Comparison: Quantum Distortion Framework",
            fontsize=20,
            fontweight="bold",
            y=1.02,
        )
        
        # Plot 1: Energy convergence comparison
        self._plot_four_case_energy(axes[0, 0], histories, ground_energy)
        
        # Plot 2: Best energy comparison
        self._plot_best_energy_comparison(axes[0, 1], histories, ground_energy)
        
        # Plot 3: Performance gap analysis
        self._plot_performance_gaps(axes[0, 2], histories)
        
        # Plot 4: Distortion timeline for recovery cases
        self._plot_distortion_timelines(axes[1, 0], histories)
        
        # Plot 5: Recovery events analysis
        self._plot_recovery_analysis(axes[1, 1], histories)
        
        # Plot 6: Noise schedule visualization
        self._plot_noise_schedule(axes[1, 2])
        
        plt.tight_layout()
        plt.show()
    
    def _plot_four_case_energy(self, ax, histories: Dict[str, VQEResults], 
                               ground_energy: Optional[float] = None) -> None:
        """Plot energy convergence for all four cases."""
        iterations = range(len(histories["ideal"].energies))
        
        # Plot all four energy trajectories
        ax.plot(iterations, histories["ideal"].energies, "b-", 
                label="Ideal (no noise)", linewidth=2.5, alpha=0.9)
        ax.plot(iterations, histories["ideal_recovery"].energies, "c--", 
                label="Ideal + Recovery", linewidth=2, alpha=0.8)
        ax.plot(iterations, histories["noisy"].energies, "r-", 
                label="Noisy (no recovery)", linewidth=2.5, alpha=0.7)
        ax.plot(iterations, histories["noisy_recovery"].energies, "g-", 
                label="Noisy + Recovery", linewidth=2, alpha=0.9)
        
        # Add exact ground state line
        if ground_energy is not None:
            ax.axhline(y=ground_energy, color="black", linestyle=":", 
                      linewidth=3, alpha=0.8, label="Exact Ground State")
        
        # Highlight noise intervals with proper labeling
        for idx, (start, end, noise_level) in enumerate(self.noise_schedule.noise_intervals):
            if noise_level > 0:
                label = "Noise Interval" if idx == 0 else ""
                ax.axvspan(start, end, alpha=0.15, color="red", label=label)
        
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Energy")
        ax.set_title("Energy Convergence: Four-Case Comparison")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
    
    def _plot_best_energy_comparison(self, ax, histories: Dict[str, VQEResults], 
                                     ground_energy: Optional[float] = None) -> None:
        """Plot bar chart of best energies."""
        cases = ["Ideal", "Ideal+Recovery", "Noisy", "Noisy+Recovery"]
        best_energies = [
            np.min(histories["ideal"].energies),
            np.min(histories["ideal_recovery"].energies),
            np.min(histories["noisy"].energies),
            np.min(histories["noisy_recovery"].energies),
        ]
        
        colors = ["blue", "cyan", "red", "green"]
        bars = ax.bar(cases, best_energies, color=colors, alpha=0.7, 
                     edgecolor="black", linewidth=2)
        
        if ground_energy is not None:
            ax.axhline(y=ground_energy, color="black", linestyle=":", 
                      linewidth=2, alpha=0.8, label="Exact Ground State")
            ax.legend()
        
        ax.set_ylabel("Best Minimum Energy")
        ax.set_title("Best Energy Achieved")
        ax.grid(True, alpha=0.3, axis="y")
        
        # Add value labels
        for bar, energy in zip(bars, best_energies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height,
                   f"{energy:.4f}", ha="center", va="bottom",
                   fontweight="bold", fontsize=16)
    
    def _plot_performance_gaps(self, ax, histories: Dict[str, VQEResults]) -> None:
        """Plot performance gaps from ideal case."""
        iterations = range(len(histories["ideal"].energies))
        
        # Calculate gaps correctly
        ideal_energies = histories["ideal"].energies
        gaps = {
            "Ideal+Recovery": [
                ir - i for ir, i in zip(
                    histories["ideal_recovery"].energies,
                    ideal_energies
                )
            ],
            "Noisy": [
                n - i for n, i in zip(
                    histories["noisy"].energies,
                    ideal_energies
                )
            ],
            "Noisy+Recovery": [
                nr - i for nr, i in zip(
                    histories["noisy_recovery"].energies,
                    ideal_energies
                )
            ],
        }
        
        colors = {"Ideal+Recovery": "cyan", "Noisy": "red", "Noisy+Recovery": "green"}
        
        for name, gap_values in gaps.items():
            ax.plot(iterations, gap_values, color=colors[name], 
                   label=name, linewidth=2, alpha=0.8)
        
        ax.axhline(y=0, color="blue", linestyle="--", alpha=0.5, label="Ideal Baseline")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Energy Gap (vs Ideal)")
        ax.set_title("Performance Gaps Relative to Ideal VQE")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_distortion_timelines(self, ax, histories: Dict[str, VQEResults]) -> None:
        """Plot distortion monitoring for recovery cases."""
        iterations = range(len(histories["noisy_recovery"].energies))
        
        # Plot ideal+recovery distortions if available
        if histories["ideal_recovery"].distortions and len(histories["ideal_recovery"].distortions) == len(iterations):
            ax.plot(iterations, histories["ideal_recovery"].distortions,
                   "c--", label="Ideal+Recovery Distortion", linewidth=2, alpha=0.8)
        
        # Plot noisy+recovery distortions and thresholds
        if histories["noisy_recovery"].distortions and len(histories["noisy_recovery"].distortions) == len(iterations):
            ax.plot(iterations, histories["noisy_recovery"].distortions,
                   "g-", label="Noisy+Recovery Distortion", linewidth=2.5, alpha=0.9)
            
            if histories["noisy_recovery"].thresholds and len(histories["noisy_recovery"].thresholds) == len(iterations):
                ax.plot(iterations, histories["noisy_recovery"].thresholds,
                       "r:", label="Noisy Threshold", linewidth=2, alpha=0.7)
        
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Distortion")
        ax.set_title("Distortion Monitoring in Recovery Cases")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_recovery_analysis(self, ax, histories: Dict[str, VQEResults]) -> None:
        """Plot recovery events for both ideal and noisy recovery cases."""
        ideal_hist = histories["ideal_recovery"]
        noisy_hist = histories["noisy_recovery"]
        
        ideal_recoveries = [i for i, r in enumerate(ideal_hist.recoveries) if r]
        noisy_recoveries = [i for i, r in enumerate(noisy_hist.recoveries) if r]
        
        # Create proper visualization
        y_positions = []
        colors = []
        labels = []
        
        if ideal_recoveries:
            y_positions.extend([0.5] * len(ideal_recoveries))
            colors.extend(['cyan'] * len(ideal_recoveries))
            labels.extend(['Ideal+Recovery'] * len(ideal_recoveries))
        
        if noisy_recoveries:
            y_positions.extend([0.7] * len(noisy_recoveries))
            colors.extend(['green'] * len(noisy_recoveries))
            labels.extend(['Noisy+Recovery'] * len(noisy_recoveries))
        
        if not (ideal_recoveries or noisy_recoveries):
            ax.text(0.5, 0.5, "No Recovery Events", ha="center", va="center",
                   transform=ax.transAxes, fontsize=16)
            ax.set_title("Recovery Events")
            return
        
        # Plot recovery events
        all_events = ideal_recoveries + noisy_recoveries
        ax.scatter(all_events, y_positions, c=colors, s=100, alpha=0.8, edgecolors='black')
        
        # Add vertical lines for events
        for event in ideal_recoveries:
            ax.axvline(x=event, color='cyan', linestyle='--', alpha=0.3, linewidth=1)
        for event in noisy_recoveries:
            ax.axvline(x=event, color='green', linestyle='--', alpha=0.3, linewidth=1)
        
        ax.set_xlabel("Iteration")
        ax.set_yticks([0.5, 0.7])
        ax.set_yticklabels(["Ideal+Rec", "Noisy+Rec"])
        ax.set_ylim([0, 1])
        ax.set_title(f"Recovery Events\nTotal: {len(all_events)}")
        ax.grid(True, alpha=0.3)
    
    def _plot_noise_schedule(self, ax) -> None:
        """Visualize the noise schedule."""
        iterations = range(self.max_iterations)
        noise_levels = [self.noise_schedule.get_effective_noise(i) for i in iterations]
        
        ax.plot(iterations, noise_levels, "b-", linewidth=2, 
               label="Noise Level", alpha=0.8)
        
        # Shade areas where noise is applied
        for start, end, noise_level in self.noise_schedule.noise_intervals:
            if noise_level > 0:
                ax.axvspan(start, end, alpha=0.1, color="red", label=f"Noise Interval" if start == self.noise_schedule.noise_intervals[0][0] else "")
        
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Noise Level")
        ax.set_title("Noise Schedule Visualization")
        ax.legend()
        ax.grid(True, alpha=0.3)