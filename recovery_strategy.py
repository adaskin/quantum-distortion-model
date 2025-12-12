import numpy as np
from typing import List, Tuple, Optional


class RecoveryStrategy:
    """
    Recovery strategies based on distortion monitoring and optimization state.

    Recovery logic:
    1. If distortion is severe (normalized > 2.0): Reset to best recent parameters
    2. If distortion is moderate (normalized > 1.5): Blend best and smoothed parameters
    3. If distortion is mild (normalized > 1.0): Small adjustment toward best
    4. Energy-based fallback: If current energy is significantly worse than best
    """

    def __init__(
        self,
        recent_lookback: int = 15,
        energy_threshold: float = 0.1,
        lr_reduction: float = 1,
        max_param_change: float = 0.2,
        min_history: int = 3,
        use_gradient: bool = False,  # Option to use gradient-based recovery
    ):
        self.recent_lookback = recent_lookback
        self.energy_threshold = energy_threshold
        self.lr_reduction = lr_reduction
        self.max_param_change = max_param_change
        self.min_history = min_history
        self.use_gradient = use_gradient

    def compute_recovery(
        self,
        energies: List[float],
        params_history: List[np.ndarray],
        current_lr: float,
        current_energy: float,
        current_distortion: float,
        threshold: float,
    ) -> Tuple[Optional[np.ndarray], float, bool, str]:
        """Decide and produce recovery parameters."""

        # Basic checks - need enough history to make meaningful recovery
        if (
            not energies
            or not params_history
            or len(energies) < self.min_history
            or len(params_history) < self.min_history
        ):
            return (
                params_history[-1] if params_history else None,
                current_lr,
                False,
                "insufficient_history",
            )

        current_params = params_history[-1]

        # Get candidate parameters
        best_params, best_energy = self._best_recent(energies, params_history)
        smoothed_params = self._smoothed_params(params_history)

        # Initialize with no recovery
        recovered = current_params
        new_lr = current_lr
        recovery_type = "no_recovery"

        # Normalize distortion relative to threshold
        normalized_distortion = current_distortion - threshold

        # Recovery decision based on normalized distortion
        if normalized_distortion > 0.5:
            # Severe distortion: strong reset toward best parameters
            if best_params is not None:
                recovered = self._blend(current_params, best_params, alpha=0.5)
                new_lr = current_lr * (self.lr_reduction**2)
                recovery_type = "severe_reset"

        elif normalized_distortion > 0.3:
            # Moderate distortion: blend best and smoothed parameters
            if best_params is not None and smoothed_params is not None:
                recovered = self._blend(smoothed_params, best_params, alpha=0.5)
                new_lr = current_lr * self.lr_reduction
                recovery_type = "moderate_blend"

        elif normalized_distortion > 0.1:
            # Mild distortion: small adjustment toward best
            if best_params is not None:
                recovered = self._blend(current_params, best_params, alpha=0.9)
                new_lr = current_lr * 0.8
                recovery_type = "mild_adjust"

        elif normalized_distortion > 0.0:
            # Mild distortion: small adjustment toward best
            if best_params is not None:
                recovered = self._blend(current_params, best_params, alpha=0.95)
                new_lr = current_lr * 0.8
                recovery_type = "very_mild_adjust"

        # Optionally use gradient-based recovery for mild cases
        if (
            self.use_gradient
            and recovery_type in ["very_mild_adjust", "mild_adjust"]
            and normalized_distortion > 0.0
        ):
            grad_params = self._gradient_recovery(energies, params_history)
            if grad_params is not None:
                # Blend gradient direction with current recovery
                recovered = self._blend(recovered, grad_params, alpha=0.3)
                recovery_type = f"{recovery_type}_gradient"

        # Bound parameter changes to avoid extreme jumps
        if recovered is not None and current_params is not None:
            recovered = self._bound_change(
                recovered, current_params, self.max_param_change
            )

        applied = recovery_type != "no_recovery"
        return recovered, new_lr, applied, recovery_type

    def _best_recent(
        self, energies: List[float], params_history: List[np.ndarray]
    ) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """Find best parameters from recent history."""
        if not energies or not params_history:
            return None, None

        lookback = min(self.recent_lookback, len(energies), len(params_history))
        if lookback == 0:
            return None, None

        start_idx = len(energies) - lookback
        recent_energies = energies[start_idx:]
        recent_params = params_history[start_idx:]

        if not recent_energies or not recent_params:
            return None, None

        best_idx = int(np.argmin(recent_energies))
        return recent_params[best_idx], float(recent_energies[best_idx])

    def _smoothed_params(
        self, params_history: List[np.ndarray], window: int = 10
    ) -> Optional[np.ndarray]:
        """Compute exponentially weighted moving average of recent parameters."""
        if not params_history:
            return None

        window = min(window, len(params_history))
        recent = params_history[-window:]

        # Exponential weights (more recent = higher weight)
        weights = np.exp(np.linspace(-1, 0, window))
        weights = weights / weights.sum()

        smoothed = np.zeros_like(recent[0], dtype=float)
        for i, p in enumerate(recent):
            smoothed += weights[i] * p

        return smoothed

    def _gradient_recovery(
        self, energies: List[float], params_history: List[np.ndarray]
    ) -> Optional[np.ndarray]:
        """Compute recovery based on recent gradient direction."""
        if len(energies) < 3 or len(params_history) < 3:
            return None

        # Look at last k steps
        k = min(5, len(energies), len(params_history))
        if k < 2:
            return None

        recent_e = np.array(energies[-k:])
        recent_p = params_history[-k:]

        # Find the step that gave the best energy improvement
        energy_diffs = np.diff(recent_e)
        param_diffs = [recent_p[i + 1] - recent_p[i] for i in range(len(recent_p) - 1)]

        if len(energy_diffs) == 0 or len(param_diffs) == 0:
            return None

        # Find step with maximum negative energy change (best improvement)
        best_idx = int(np.argmin(energy_diffs))
        if best_idx >= len(param_diffs):
            return None

        best_direction = param_diffs[best_idx]
        direction_norm = np.linalg.norm(best_direction)

        if direction_norm > 1e-8:
            # Small step in the direction that worked best
            step_size = 0.05 * direction_norm
            unit_direction = best_direction / direction_norm
            return recent_p[-1] + unit_direction * step_size

        return None

    def _blend(self, a: np.ndarray, b: np.ndarray, alpha: float = 0.7) -> np.ndarray:
        """Blend two parameter vectors."""
        if a is None and b is None:
            return None
        elif a is None:
            return b.copy() if b is not None else None
        elif b is None:
            return a.copy() if a is not None else None
        else:
            return alpha * a + (1 - alpha) * b

    def _bound_change(
        self, new_params: np.ndarray, old_params: np.ndarray, max_change: float
    ) -> np.ndarray:
        """Limit parameter changes to avoid extreme jumps."""
        if new_params is None or old_params is None:
            return new_params

        change = new_params - old_params
        norm = np.linalg.norm(change)

        if norm > max_change and norm > 1e-8:
            return old_params + change * (max_change / norm)

        return new_params


class RecoveryManager:
    """
    Manages recovery attempts with cooldown and basic statistics.
    """

    def __init__(
        self,
        cooldown: int = 5,
        max_recoveries: int = 10,
        min_spacing: int = 3,
        adaptive_cooldown: bool = True,
        strategy_config: Optional[dict] = None,
    ):
        self.cooldown = cooldown
        self.max_recoveries = max_recoveries
        self.min_spacing = min_spacing
        self.adaptive_cooldown = adaptive_cooldown

        # Initialize recovery strategy with custom config if provided
        strategy_config = strategy_config or {}
        self.strategy = RecoveryStrategy(**strategy_config)

        # Recovery tracking
        self.count = 0
        self.last_iteration = -cooldown
        self.recovery_types = []
        self.effectiveness = (
            []
        )  # List of tuples (energy_before, energy_after, improved)

    def can_recover(self, current_iteration: int) -> bool:
        """Check if recovery is allowed at this iteration."""
        if current_iteration is None:
            return False

        since_last = current_iteration - self.last_iteration

        # Check cooldown
        if since_last < self._current_cooldown():
            return False

        # Check max recoveries
        if self.max_recoveries > 0 and self.count >= self.max_recoveries:
            return False

        # Check minimum spacing
        if since_last < self.min_spacing:
            return False

        return True

    def attempt_recovery(
        self,
        current_iteration: int,
        energies: List[float],
        params_history: List[np.ndarray],
        current_lr: float,
        current_energy: float,
        current_distortion: float,
        threshold: float,
    ) -> Tuple[Optional[np.ndarray], float, bool, str]:
        """Attempt recovery if conditions are met."""

        if not self.can_recover(current_iteration):
            return (
                params_history[-1] if params_history else None,
                current_lr,
                False,
                "cooldown",
            )

        # Use strategy to compute recovery
        recovered, new_lr, success, rtype = self.strategy.compute_recovery(
            energies=energies,
            params_history=params_history,
            current_lr=current_lr,
            current_energy=current_energy,
            current_distortion=current_distortion,
            threshold=threshold,
        )

        if success:
            # Track recovery
            self.count += 1
            self.last_iteration = current_iteration
            self.recovery_types.append(rtype)
            # Store placeholder for effectiveness (will be updated later)
            self.effectiveness.append((current_energy, None, None))

        return recovered, new_lr, success, rtype

    def update_effectiveness(self, energy_before: float, energy_after: float) -> None:
        """Update effectiveness of the most recent recovery."""
        if not self.effectiveness:
            return

        # Update the last effectiveness entry
        old_energy, _, _ = self.effectiveness[-1]
        improved = (
            energy_after < old_energy
            if (old_energy is not None and energy_after is not None)
            else None
        )
        self.effectiveness[-1] = (old_energy, energy_after, improved)

    def _current_cooldown(self) -> int:
        """Compute current cooldown period (adaptive if enabled)."""
        base_cooldown = self.cooldown

        if not self.adaptive_cooldown or not self.effectiveness:
            return base_cooldown

        # Check last few recoveries for effectiveness
        recent_effectiveness = self.effectiveness[-3:]
        ineffective_count = sum(
            1 for _, _, improved in recent_effectiveness if improved is False
        )

        # Increase cooldown if recent recoveries were ineffective
        if ineffective_count > 0:
            return int(base_cooldown * (1 + 0.5 * ineffective_count))

        return base_cooldown

    def get_stats(self) -> dict:
        """Get recovery statistics."""
        effective_count = sum(
            1 for _, _, improved in self.effectiveness if improved is True
        )
        ineffective_count = sum(
            1 for _, _, improved in self.effectiveness if improved is False
        )
        unknown_count = sum(
            1 for _, _, improved in self.effectiveness if improved is None
        )

        return {
            "total_recoveries": self.count,
            "effective_recoveries": effective_count,
            "ineffective_recoveries": ineffective_count,
            "unknown_effectiveness": unknown_count,
            "recovery_types": self.recovery_types.copy(),
            "last_iteration": self.last_iteration,
            "in_cooldown": self.count > 0 and (self.last_iteration + self.cooldown > 0),
            "cooldown_remaining": max(
                0, self._current_cooldown() - (self.last_iteration + self.cooldown)
            ),
        }
