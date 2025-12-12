import numpy as np


# -------------------------
# Dynamic threshold and recovery trigger
# -------------------------
class DynamicThresholdModel:
    """
    Dynamic threshold:
    tau_t = tau_min + (tau_max - tau_min) * exp(-decay * t) + failure_tolerance * failure_rate
    """

    def __init__(
        self,
        tau_min: float = 0.08,
        tau_max: float = 0.4,
        decay_rate: float = 0.05,
        failure_tolerance: float = 0.15,
        noisy_multiplier: float = 1.0,
        kappa: float = 1.0,
        derivative_threshold: float = 0.001,
        degradation_threshold: float = 0.1,
    ):
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.decay_rate = decay_rate
        self.failure_tolerance = failure_tolerance
        self.noisy_multiplier = noisy_multiplier
        self.kappa = kappa
        self.derivative_threshold = derivative_threshold
        self.degradation_threshold = degradation_threshold

    def compute_threshold(self, iteration: int, failure_rate: float = 0.0) -> float:
        """Compute τ_t for a given iteration and failure rate."""
        base = self.tau_min + (self.tau_max - self.tau_min) * np.exp(
            -self.decay_rate * iteration
        )
        adjusted = base + self.failure_tolerance * failure_rate
        return float(adjusted * self.noisy_multiplier)

    def should_trigger_recovery(
        self,
        current_distortion: float,
        threshold: float,
        distortion_history: list,
        threshold_history: list,
        energy_history: list,
        current_iteration: int,
    ) -> bool:
        """
        Decide whether to trigger recovery using:
          1) current_distortion > threshold
          2) derivative condition OR significant absolute increase
          3) energy degradation OR persistent high distortion as fallbacks
        """

        if current_distortion is None or threshold is None:
            return False

        if not distortion_history or not threshold_history or not energy_history:
            return False

        # Condition 1
        if current_distortion <= threshold:
            return False

        # Derivative estimates (use previous step)
        dD = (
            current_distortion - distortion_history[-2]
            if len(distortion_history) >= 2
            else 0.0
        )
        dTau = threshold - threshold_history[-2] if len(threshold_history) >= 2 else 0.0

        # Condition 2: derivative-based or absolute jump
        cond_derivative = (dD > self.kappa * dTau) or (dD > self.derivative_threshold)

        # Condition 3: energy degradation
        cond_energy = self._energy_degrading(energy_history)

        # Condition 4: persistent high distortion
        cond_persistent = self._persistent_high_distortion(
            distortion_history, threshold_history
        )

        return cond_derivative or cond_energy or cond_persistent

    def _energy_degrading(self, energy_history: list) -> bool:
        """Return True if recent energies show consistent degradation or exceed recent minimum by threshold."""
        if not energy_history or len(energy_history) < 5:
            return False

        recent = energy_history[-5:]
        if all(recent[i] >= recent[i - 1] for i in range(1, len(recent))):
            return True

        lookback = min(8, len(energy_history))
        recent_min = min(energy_history[-lookback:])
        return energy_history[-1] > recent_min + self.degradation_threshold

    def _persistent_high_distortion(
        self, distortion_history: list, threshold_history: list
    ) -> bool:
        """Return True if distortion exceeded threshold in most recent steps."""
        if len(distortion_history) < 4 or len(threshold_history) < 4:
            return False

        recent_d = distortion_history[-4:]
        recent_t = threshold_history[-4:]
        high_count = sum(1 for d, t in zip(recent_d, recent_t) if d > t)
        return high_count >= 3

    def threshold_derivative(self, iteration: int) -> float:
        """Analytic derivative dτ/dt of the exponential term
        (ignoring failure_rate term)."""
        exp_term = (self.tau_max - self.tau_min) * np.exp(-self.decay_rate * iteration)
        return float(-self.decay_rate * exp_term)
