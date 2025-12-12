# Quantum Distortion-Aware VQE Framework

[![zenodo](https://img.shields.io/badge/doi-10.5281/zenodo.17262788-blue)](https://doi.org/10.5281/zenodo.17262788)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/mit)

A Python framework implementing quantum distortion modeling for variational quantum algorithms, enabling error-resilient optimization without traditional error correction. Based on the paper **"Quantum Distortion Model for Running Variational Quantum Algorithms without Error Corrections"** by Ammar Daskin.

## 🔬 Overview

This framework enables variational quantum algorithms (like VQE) to operate effectively in noisy quantum environments by implementing distortion-aware monitoring and adaptive recovery strategies. Rather than eliminating errors entirely, the system:

1. **Monitors multiple distortion metrics** during optimization
2. **Detects anomalous behavior** using dynamic thresholds
3. **Triggers intelligent recovery** when distortion exceeds acceptable limits
4. **Adapts to different noise regimes** without requiring error correction overhead

## 📁 Project Structure

```
quantum-distortion-vqe/
├── vqe_simulator.py           # Main simulator and orchestration
├── distortion_model.py        # Distortion metrics implementation
├── failure_model.py           # Dynamic threshold and recovery triggers
├── recovery_strategy.py       # Adaptive recovery strategies
├── vqe_plotting.py           # Visualization utilities
├── vqe_circuit_hamiltonian.py # Quantum circuit and Hamiltonian definitions
├── config_generator.py       # Configuration management (imported)
└── README.md                 # This file
```

### Key Components

- **`AdaptiveVQESimulator`**: Main simulation orchestrator with four-case comparison
- **`CompositeDistortionModel`**: Implements multiple distortion metrics from the paper
- **`DynamicThresholdModel`**: Adaptive thresholds based on iteration and noise levels
- **`RecoveryManager`**: Executes recovery strategies when distortion is detected
- **`VQECircuit`**: Manages quantum circuit construction with configurable noise
- **`VQEPlotter`**: Comprehensive visualization of results and comparisons

### Core Features

1. **Multiple Distortion Metrics**:
   - Energy progression distortion
   - State fidelity distortion  
   - Parameter stability distortion
   - Convergence rate distortion

2. **Adaptive Recovery**:
   - Dynamic threshold modeling
   - Phase-aware distortion monitoring
   - Parameter restoration with learning rate adjustment
   - Cooldown mechanisms to prevent over-recovery

3. **Comprehensive Analysis**:
   - Four-case comparison (ideal, ideal+recovery, noisy, noisy+recovery)
   - Hamiltonian spectral analysis
   - Performance gap quantification
   - Recovery effectiveness tracking

## 🚀 Installation

### Prerequisites

- Python
- PennyLane 
- NumPy, SciPy
- Matplotlib

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/quantum-distortion-vqe.git
cd quantum-distortion-vqe

# Install dependencies
pip install pennylane numpy scipy matplotlib
```

## 💻 Quick Start

### Basic Usage

```python
from vqe_simulator import AdaptiveVQESimulator, VQEConfig

# Create configuration
config = VQEConfig(
    n_qubits=4,
    layers=2,
    max_iterations=200,
    # ... other configuration parameters
)

# Initialize simulator
simulator = AdaptiveVQESimulator(config)

# Analyze Hamiltonian
analysis = simulator.analyze_hamiltonian()
ground_energy = analysis['ground_energy']

# Run four-case comparison
histories = simulator.run_four_case_comparison(learning_rate=0.1)

# Analyze results
simulator.print_four_case_results(histories, ground_energy)
```

### Complete Example with Custom Configuration

```python
from vqe_simulator import main

# Run the complete demonstration
simulator, histories, ground_energy = main()
```

### Custom Distortion Weights

```python
config = {
    'n_qubits': 8,
    'layers': 3,
    'max_iterations': 200,
    
    # Custom distortion parameters
    'distortion_params': {
        'energy_window': 10,
        'weight_energy': 0.6,      # Energy progression importance
        'weight_fidelity': 0.2,    # State fidelity importance
        'weight_parameter': 0.1,   # Parameter stability importance
        'weight_convergence': 0.1, # Convergence rate importance
    },
    
    # Custom noise schedule
    'noise_schedule': {
        'noise_intervals': [
            (10, 30, 0.1),   # Early noise burst
            (50, 80, 0.05),  # Mid-optimization noise
            (120, 150, 0.2), # Late-stage noise spike
        ],
        'base_noise_level': 0.001,
    },
    
    # Recovery configuration
    'recovery_params': {
        'cooldown': 3,
        'max_recoveries': 10,
        'min_spacing': 2,
        'adaptive_cooldown': True,
        'strategy_config': {
            'recent_lookback': 20,
            'energy_threshold': 0.1,
            'lr_reduction': 0.8,
        }
    }
}
```

## 📊 Output and Visualization

The framework provides comprehensive visualization through the `VQEPlotter` class:

1. **Energy Convergence**: Comparison of all four cases
2. **Best Energy Achieved**: Bar chart comparison
3. **Performance Gaps**: Relative to ideal VQE
4. **Distortion Monitoring**: With dynamic thresholds
5. **Recovery Events**: Timing and frequency analysis
6. **Noise Schedule**: Visual representation of noise intervals

### Example Output

```
🧪 QUANTUM DISTORTION-AWARE VQE: FOUR-CASE COMPARISON
======================================================================
HAMILTONIAN ANALYSIS
======================================================================
Number of qubits: 8
Hilbert space dimension: 256
Ground state energy: -12.34567890
Spectral gap: 0.123456
Difficulty assessment: ✅ Reasonable spectral gap

FOUR-CASE VQE COMPARISON SIMULATION
======================================================================
Configuration: 8 qubits, 3 layers, 200 iterations
Learning rate: 0.01

Iter | Energy    | Distortion | Threshold | Phase  | Noise  | Recovery
-----------------------------------------------------------------------
  10 | -10.123456 |    0.2345  |    0.1890 | NOISY  | 0.1000 | 
  25 | -11.456789 |    0.1876  |    0.1623 | CLEAN  | 0.0000 | 🔄 RECOVERED
```

## 🔧 Advanced Configuration

### Noise Types
The framework works with multiple noise models used in Pennylane:
- `depolarizing`: Depolarizing noise channels
- `amplitude_damping`: Amplitude damping noise
- `phase_damping`: Phase damping noise  
- `bit_flip`: Bit flip errors
- `combined`: All noise types applied

### Threshold Parameters
```python
'threshold_params': {
    'tau_min': 0.08,           # Minimum threshold
    'tau_max': 0.4,            # Maximum threshold
    'decay_rate': 0.05,        # Exponential decay rate
    'failure_tolerance': 0.15, # Tolerance for failure rates
    'kappa': 1.0,              # Derivative sensitivity
    'derivative_threshold': 0.001,  # Minimum derivative for trigger
    'degradation_threshold': 0.1,   # Energy degradation threshold
    'noisy_multiplier': 1.0,   # Multiplier for noisy iterations
}
```

### Recovery Strategies
Three levels of recovery based on distortion severity:
- **Mild (distortion > 0.1)**: Small adjustment toward best parameters (α=0.9)
- **Moderate (distortion > 0.3)**: Blend of smoothed and best parameters (α=0.5)
- **Severe (distortion > 0.5)**: Strong reset toward best parameters (α=0.5)

## 🧪 Research Applications

This framework is designed for researchers investigating:
- Error resilience in variational quantum algorithms
- Quantum-classical hybrid optimization
- Noise-aware quantum computing
- Distortion modeling for quantum systems
- Adaptive recovery strategies for NISQ devices

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 Citation

If you use this code in your research, please cite the original paper:

```bibtex
@article{daskin2025quantum,
  author       = {Ammar, Daskin},
  title        = {Quantum Distortion Model for Running Variational Quantum Algorithms without Error Corrections},
  month        = oct,
  year         = 2025,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17262788},
  url          = {https://doi.org/10.5281/zenodo.17262788},
}
```

## 🛠️ Development

### Running Tests
```bash
# Run the main simulation
python vqe_simulator.py

# For custom experiments
python -c "
from vqe_simulator import AdaptiveVQESimulator, VQEConfig
# ... your experiment code
"
```

### Adding New Distortion Metrics
```python
from distortion_model import CompositeDistortionModel

class CustomDistortionMetric:
    def compute(self, data):
        # Implement your metric
        return distortion_value

# Add to CompositeDistortionModel
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## 📞 Support

For questions and issues:
- Open an issue on GitHub
- Refer to the original paper for theoretical background
- Check PennyLane documentation for quantum computing concepts

---

**Note**: This is research code intended for demonstration and experimentation. Performance may vary based on specific problem instances, noise characteristics, and hardware configurations.