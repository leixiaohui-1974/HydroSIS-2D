# HydroSIS-2D Examples

This directory contains practical examples demonstrating the capabilities of the HydroSIS-2D preprocessing and solver modules.

## Available Examples

### 1. MUSCL Order Comparison (`compare_muscl_orders.py`)

**Purpose**: Demonstrates the difference between first-order and second-order (MUSCL) spatial accuracy.

**What it does**:
- Runs a dam break simulation with both schemes
- Compares numerical diffusion and shock resolution
- Generates comparison plots
- Provides quantitative analysis

**Usage**:
```bash
python examples/compare_muscl_orders.py
```

**Output**:
- Terminal output with detailed analysis
- Comparison plots saved as PNG
- Quantitative metrics (Total Variation, shock sharpness)

**Key Results**:
- Second-order MUSCL reduces numerical diffusion (~9%)
- Sharper shock resolution (~1.6x)
- Computational overhead (~84%)

**Recommended for**:
- Understanding MUSCL benefits
- Choosing between first and second-order schemes
- Visual comparison for presentations

---

### 2. Performance Benchmark (`benchmark_performance.py`)

**Purpose**: Benchmarks different solver configurations to help users choose optimal settings.

**Tested Configurations**:
1. First-order + NumPy (baseline)
2. Second-order MUSCL + NumPy
3. First-order + Numba
4. Second-order + Numba (currently falls back to NumPy)

**Usage**:
```bash
python examples/benchmark_performance.py
```

**Customization**:
```python
from examples.benchmark_performance import run_benchmark

# Custom benchmark
run_benchmark(
    grid_sizes=[50, 100, 200],  # Grid sizes to test
    t_end=1.0,                   # Simulation time
    num_runs=3                   # Runs to average
)
```

**Output**:
- Performance comparison table
- Speedup analysis
- Recommendations based on grid size

**Key Findings**:
- Small grids (<100): NumPy is sufficient
- Medium grids (100-200): Numba provides modest speedup
- Large grids (>200): Numba speedup increases
- MUSCL overhead: ~40-80%

**Recommended for**:
- Choosing optimal configuration for your problem
- Understanding performance trade-offs
- Hardware-specific optimization

---

## Requirements

All examples require:
- Python 3.8+
- NumPy
- Matplotlib
- HydroSIS-2D preprocessing and solver modules

Optional (for Numba benchmarks):
- Numba 0.55+

Installation:
```bash
pip install numpy matplotlib numba
```

---

## Running Examples

### From the prepost directory:
```bash
cd /path/to/HydroSIS-2D/prepost
python examples/compare_muscl_orders.py
python examples/benchmark_performance.py
```

### From the examples directory:
```bash
cd /path/to/HydroSIS-2D/prepost/examples
python compare_muscl_orders.py
python benchmark_performance.py
```

---

## Understanding the Results

### MUSCL Comparison

**Total Variation (TV)**:
- Measures numerical diffusion
- Higher TV = less diffusive = better
- Second-order typically has 5-10% higher TV

**Shock Sharpness**:
- Measured by max gradient |dh/dx|
- Higher gradient = sharper shock = better
- Second-order typically 1.5-2x sharper

**Computational Cost**:
- Second-order is 40-100% slower
- Worth it for accuracy-critical simulations

### Performance Benchmark

**Speedup Factor**:
- > 1.0: Faster than baseline (good)
- = 1.0: Same as baseline (neutral)
- < 1.0: Slower than baseline (bad)

**Recommendations**:
- Development: Use first-order + NumPy (fast iteration)
- Production (small grids): Use first-order + NumPy
- Production (medium grids): Use first-order + Numba
- Production (large grids): Use first-order + Numba
- High accuracy: Use second-order + NumPy (accept overhead)

---

## Tips for Best Results

### For Comparison Example:
1. Use larger grids (nx=200-500) for more dramatic differences
2. Increase t_end for evolved flow features
3. Save plots for documentation/presentations

### For Benchmark:
1. Close other applications for accurate timing
2. Run with `num_runs=5` for more stable averages
3. Test on your actual problem grid size
4. Consider warm-up overhead for Numba (one-time cost)

---

## Creating Your Own Examples

Template structure:
```python
#!/usr/bin/env python3
"""
Example: Your Title

Description of what this example demonstrates.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.mesh_generation import MeshGenerator, DomainParams
from solver import ShallowWaterSolver, SolverConfig

def main():
    # Your example code here
    pass

if __name__ == '__main__':
    main()
```

---

## Future Examples (Planned)

- **Real-world test case**: Realistic flood simulation
- **Mesh refinement study**: Convergence analysis
- **Boundary condition showcase**: Different BC types
- **Numba + MUSCL**: When Numba-optimized MUSCL is available
- **GPU acceleration**: When GPU support is added

---

## Troubleshooting

**Issue**: Import errors
- **Solution**: Make sure you're in the `prepost` directory or examples directory

**Issue**: Numba examples fail
- **Solution**: Install Numba: `pip install numba`
- **Alternative**: Examples will skip Numba tests automatically

**Issue**: Plots don't show
- **Solution**: Install matplotlib: `pip install matplotlib`
- **Alternative**: Set `save_plots=True` to save without displaying

**Issue**: Out of memory
- **Solution**: Reduce grid sizes or t_end in examples

---

## Contributing

To add your own example:

1. Create a new Python script in `examples/`
2. Follow the template structure above
3. Add documentation header
4. Test it thoroughly
5. Update this README
6. Submit a pull request

---

## License

These examples are part of the HydroSIS-2D project.

---

## Support

For questions or issues:
- Check the main HydroSIS-2D documentation
- Review the docstrings in each example
- Open an issue on GitHub

---

**Last Updated**: October 29, 2025
**Examples Version**: 1.0
**Tested with**: HydroSIS-2D prepost module v1.0
