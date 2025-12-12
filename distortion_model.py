import numpy as np
from scipy import stats
# -------------------------
# Energy progression distortion
# -------------------------
class EnergyProgressionDistortion:
    """Energy progression distortion that aligns with energy improvement.
       
    d_E(t) = (Ê_t - E_t) / (σ_E + ε)
    """

    def __init__(self, window_size: int = 5, min_sigma = 0.01, epsilon = 1e-8):
        self.window_size = window_size
        self.min_sigma = min_sigma
        self.epsilon = epsilon

    def compute(self, energy_history):
        if not energy_history or len(energy_history) < self.window_size + 1:
            return 0.0

        recent = energy_history[-(self.window_size + 1):-1]

        t = np.arange(len(recent))
        slope, intercept, _, _, _ = stats.linregress(t, recent)
        predicted = intercept + slope * len(recent)
        current = energy_history[-1]
        # previous = energy_history[-2]
        # relative_diff = np.abs(current-previous)/np.abs()
        # # Compute standard deviation with safeguards
        # sigma = np.std(recent)
    
        # sigma = max(sigma, self.min_sigma)
        

        deviation = predicted - current  

        normalized_distortion = np.abs(deviation / (current + self.epsilon))
        
        return min(normalized_distortion, 1.0)
        


# -------------------------
# State fidelity distortion
# -------------------------
class StateFidelityDistortion:
    """State fidelity distortion - aligned with energy improvement."""
    
    def __init__(self, epsilon = 1e-8):
        self.epsilon = epsilon

    def compute(self, fidelity_history):
        """Compute fidelity-based distortion aligned with energy trends.
        
        Negative: Fidelity decrease when energy improves (expected - good)
        Positive: Fidelity decrease when energy degrades (unexpected - bad)
        """
        if not fidelity_history or len(fidelity_history) < 2:
            return 0.0

        
        F_t = fidelity_history[-1]  # F(ψ_t, ψ_{t-1})
        F_t_prev = fidelity_history[-2]  # F(ψ_{t-1}, ψ_{t-2})

        
        # Fidelity change (positive if fidelity decreased, negative if increased)
        fidelity_change = np.abs(F_t_prev - F_t) + self.epsilon
        
        # Normalize the change
        relative_change = np.abs(fidelity_change / F_t_prev)
    
        return min(abs(relative_change), 1.0)
      

# -------------------------
# Parameter stability distortion
# -------------------------
class ParameterStabilityDistortion:
    """Parameter stability aligned with energy improvement."""
    
    def __init__(self, epsilon = 1e-8):
        self.epsilon = epsilon
  

    def compute(self, params_current, params_previous):
        """Compute parameter distortion aligned with energy changes.
        
        Negative: Parameter changes when energy improves (good - exploration)
        Positive: Large parameter changes when energy degrades (bad - instability)
        """
        if params_current is None or params_previous is None:
            return 0.0

        
        # Ensure parameters are numpy arrays
        if not isinstance(params_current, np.ndarray):
            params_current = np.array(params_current)
        if not isinstance(params_previous, np.ndarray):
            params_previous = np.array(params_previous)
        
        # Compute relative parameter change
        norm_prev = np.linalg.norm(params_previous) + self.epsilon

        
        relative_change = np.linalg.norm(params_current - params_previous) / norm_prev

        return min(abs(relative_change),1.0)


# -------------------------
# Convergence rate distortion
# -------------------------
class ConvergenceRateDistortion:
    """Convergence rate distortion aligned with energy improvement.
    
    Negative: Convergence faster than expected (good)
    Positive: Convergence slower than expected (bad)
    """
    
    def __init__(self, 
                 delta_min = 1e-8, 
                 epsilon = 1e-8):
  
        self.delta_min = delta_min  # Minimum energy change to consider
        self.epsilon = epsilon

    def compute(self, energy_history):
        if not energy_history or len(energy_history) < 3:
            return 0.0

        current_energy = energy_history[-1]
        prev_energy = energy_history[-2]
        prev_prev_energy = energy_history[-3]
        
        # Calculate actual improvements
        improvement_t = prev_energy - current_energy  # Positive if improved
        improvement_prev = prev_prev_energy - prev_energy  # Positive if improved

        conv_rate = np.abs(improvement_t / (improvement_prev + self.epsilon))

        return min(abs(conv_rate),1.0)


# -------------------------
# Composite distortion model
# -------------------------
class CompositeDistortionModel:
    """
    Composite distortion model where negative values indicate good behavior.
    
    Total distortion = weighted sum where:
    - Negative values: Good/beneficial behavior (energy improving)
    - Positive values: Bad/problematic behavior (energy degrading)
    - Near zero: Normal/expected behavior
    """
    
    def __init__(
        self,
        energy_window: int = 5,
        weight_energy  = 0.4,
        weight_fidelity  = 0.3,
        weight_parameter  = 0.2,
        weight_convergence = 0.1,

    ):
    

        # Initialize component distortion models
        self.energy_distortion = EnergyProgressionDistortion(window_size=energy_window)
        self.fidelity_distortion = StateFidelityDistortion()
        self.parameter_distortion = ParameterStabilityDistortion()
        self.convergence_distortion = ConvergenceRateDistortion()

        # Weights (all positive, sum should be 1)
        self.weight_energy = weight_energy
        self.weight_fidelity = weight_fidelity
        self.weight_parameter = weight_parameter
        self.weight_convergence = weight_convergence
  
  
        
        # Normalize weights
        total_weight = weight_energy + weight_fidelity + weight_parameter + weight_convergence
        if total_weight > 0:
            self.weight_energy /= total_weight
            self.weight_fidelity /= total_weight
            self.weight_parameter /= total_weight
            self.weight_convergence /= total_weight

    def compute_component_distortions(self, energy_history, fidelity_history,
                                      params_current, params_previous):
        """Return a tuple of signed component distortions.
        
        Negative values: Good/beneficial behavior
        Positive values: Bad/problematic behavior
        """
        # Get energy context
        energy_current = energy_history[-1] if energy_history and len(energy_history) > 0 else None
        energy_previous = energy_history[-2] if energy_history and len(energy_history) > 1 else None
        
        # Compute each distortion with error handling
        d_energy = self.energy_distortion.compute(energy_history) 
        d_fidelity = self.fidelity_distortion.compute(fidelity_history) 
        d_param = self.parameter_distortion.compute( params_current, params_previous) 
        d_conv = self.convergence_distortion.compute(energy_history) 
        if energy_previous is None or (energy_current-energy_previous) < 0:
            return -d_energy, -d_fidelity, -d_param, -d_conv
        else:
            return d_energy, d_fidelity, d_param, d_conv

    def compute_total_distortion(self, component_distortions):
        """Compute weighted signed distortion.
        
        Returns:
        - Positive value: Problematic behavior that may need recovery
        - Negative value: Good behavior (continue as is)
        - Near zero: Normal behavior
        """
        d_energy, d_fidelity, d_param, d_conv = component_distortions

        # Compute weighted sum (all positie)
        total = (
            self.weight_energy * d_energy +
            self.weight_fidelity * d_fidelity +
            self.weight_parameter * d_param +
            self.weight_convergence * d_conv
        )
        # print("..............................................................")
        # print( d_energy,d_fidelity ,d_param,d_conv
            
        # )
        # Bound to [-1, 1]
        return max(min(total, 1.0), -1.0)
    
    def is_problematic(self, component_distortions):
        """Check if distortion indicates problematic behavior."""
        total = self.compute_total_distortion(component_distortions)
        return total > self.positivity_threshold
    
    def get_distortion_breakdown(self, component_distortions):
        """Return detailed breakdown of distortion components."""
        d_energy, d_fidelity, d_param, d_conv = component_distortions
        
        return {
            'energy': d_energy,
            'fidelity': d_fidelity,
            'parameter': d_param,
            'convergence': d_conv,
            'total': self.compute_total_distortion(component_distortions),
            'is_problematic': self.is_problematic(component_distortions),
            'interpretation': self._interpret_distortion(component_distortions)
        }
    
    def _interpret_distortion(self, component_distortions):
        """Provide human-readable interpretation of distortion."""
        total = self.compute_total_distortion(component_distortions)
        
        if total > 1.0:
            return "SEVERE: Significant problematic behavior"
        elif total > 0.5:
            return "MODERATE: Problematic behavior"
        elif total > 0.1:
            return "MILD: Slightly problematic"
        elif total < -1.0:
            return "EXCELLENT: Very good progress"
        elif total < -0.5:
            return "GOOD: Good progress"
        elif total < -0.1:
            return "OK: Normal improvement"
        elif abs(total) < 0.1:
            return "NORMAL: Expected behavior"
        else:
            return "NEUTRAL: No significant deviation"