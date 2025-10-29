# HydroSIS-2D

A GPU-accelerated 2D shallow water equations solver for flood simulation and hydraulic modeling.

## Overview

HydroSIS-2D is a high-performance computational fluid dynamics tool designed for simulating 2D surface water flow using shallow water equations. The solver leverages GPU acceleration for efficient large-scale flood modeling, urban drainage analysis, and hydraulic studies.

## Key Features

### Core Simulation Engine
- **GPU-accelerated computation** using CUDA/OpenCL
- **2D shallow water equations** solver
- **High-order numerical schemes** for accuracy
- **Adaptive time stepping** for stability
- **Wet/dry front handling** for flood propagation

### Preprocessing and Postprocessing Toolkit ✨ NEW

The project now includes a comprehensive preprocessing and postprocessing suite in the `prepost/` directory:

#### Preprocessing Capabilities
- **Mesh Generation**: Uniform and adaptive structured mesh generation
- **Geometry Processing**: Terrain data handling (ASCII Grid format), synthetic terrain generation
- **Boundary Conditions**: Wall, inflow, outflow, periodic, time-series BCs
- **Initial Conditions**: Dam break, uniform, dry bed, custom field initialization
- **Simulation Configuration**: Integrated workflow for complete simulation setup

#### Postprocessing Capabilities
- **3D Visualization**: Water surface, terrain, and velocity field rendering using PyVista
- **Animation Generation**: Time series animations in MP4, GIF formats
- **Result Analysis**: VTK file loading, statistics computation, profile extraction
- **Scientific Colormaps**: Specialized color schemes for hydrodynamic data

**Toolkit Statistics**:
- 44 Python files
- 11,496 lines of code
- 145 tests (100% passing)
- 7 complete modules
- 10 working examples

See [`prepost/README.md`](prepost/README.md) for complete documentation.

## Project Structure

```
HydroSIS-2D/
├── src/                        # Core simulation engine (CUDA/C++)
├── prepost/                    # Preprocessing and postprocessing toolkit
│   ├── preprocessing/          # Mesh, geometry, BC, IC modules
│   ├── postprocessing/         # Visualization and analysis
│   ├── simulation/             # Integrated simulation configuration
│   ├── tests/                  # 145 comprehensive tests
│   ├── examples/               # 10 working examples
│   └── README.md               # Toolkit documentation
├── docs/                       # Documentation
└── README.md                   # This file
```

## Quick Start

### 1. Using the Preprocessing Toolkit

Create a complete dam break simulation setup:

```python
from simulation import create_dam_break_simulation

# Create configuration
config = create_dam_break_simulation(
    length=200.0,
    width=100.0,
    dam_position=0.5,
    upstream_depth=10.0,
    nx=100, ny=50,
    simulation_time=10.0
)

# Validate and export
is_valid, errors = config.validate()
if is_valid:
    config.export_configuration('output/dam_break')
```

This generates all necessary files:
- `simulation_config.json` - Simulation metadata
- `mesh.vtk` - Computational mesh
- `boundary_conditions.json` - BC specifications
- `initial_conditions.json` - IC specifications
- `initial_fields.npz` - Initial depth and velocity fields

### 2. Running the Simulation

Use the exported configuration with the HydroSIS-2D solver:

```bash
./hydrosis2d --config output/dam_break/simulation_config.json
```

### 3. Visualizing Results

Analyze and visualize simulation results:

```python
from postprocessing.result_analysis import ResultAnalyzer
from postprocessing.visualization_engine import VisualizationEngine

# Load results
analyzer = ResultAnalyzer()
analyzer.load_vtk_series('output/results/*.vtk')

# Create visualization
vis = VisualizationEngine()
result = analyzer.get_result_at_time(5.0)
vis.visualize_water_surface(result.mesh, result.depth, result.terrain)
vis.show()
```

## Installation

### Core Solver Dependencies

```bash
# CUDA toolkit (for GPU acceleration)
# OpenMP (for CPU parallelization)
# HDF5 (for data I/O)
```

### Python Toolkit Dependencies

```bash
cd prepost
pip install numpy scipy matplotlib pyvista pytest
```

## Documentation

- **Preprocessing/Postprocessing Toolkit**: [`prepost/README.md`](prepost/README.md)
- **Development Roadmap**: [`docs/PREPROCESSING_POSTPROCESSING_ROADMAP.md`](docs/PREPROCESSING_POSTPROCESSING_ROADMAP.md)
- **Mesh Generation Summary**: [`docs/MESH_GENERATION_SUMMARY.md`](docs/MESH_GENERATION_SUMMARY.md)
- **Visualization Summary**: [`docs/VISUALIZATION_MODULE_SUMMARY.md`](docs/VISUALIZATION_MODULE_SUMMARY.md)
- **Geometry Module Summary**: [`docs/GEOMETRY_MODULE_SUMMARY.md`](docs/GEOMETRY_MODULE_SUMMARY.md)

## Examples

Complete working examples are available in `prepost/examples/`:

1. **Dam Break**: Classical dam break scenario
2. **Channel Flow**: Steady channel flow with inflow/outflow
3. **Custom Terrain**: Flood over complex terrain
4. **Visualization**: 3D rendering and animation
5. **Result Analysis**: Statistics and profile extraction

```bash
cd prepost/examples
python example_complete_workflow.py
```

## Testing

The preprocessing/postprocessing toolkit includes comprehensive tests:

```bash
cd prepost
pytest tests/ -v
```

**Test Results**: 145 tests, 100% pass rate

## Applications

HydroSIS-2D is suitable for:
- **Flood modeling**: Dam break, levee breach, flood inundation mapping
- **Urban drainage**: Storm water runoff, urban flooding
- **River hydraulics**: Channel flow, flood routing
- **Coastal engineering**: Tsunami propagation, storm surge
- **Laboratory validation**: Benchmark test cases

## Performance

- GPU acceleration enables real-time to near-real-time simulation
- Handles domains with millions of cells efficiently
- Adaptive mesh refinement reduces computational cost
- Optimized for NVIDIA GPUs (CUDA) and AMD GPUs (OpenCL)

## Contributing

Contributions are welcome! Please ensure:
- Code follows project style guidelines
- All tests pass
- New features include documentation and tests

## License

[License information to be added]

## Citation

If you use HydroSIS-2D in your research, please cite:

```
[Citation information to be added]
```

## Contact

For questions, issues, or contributions:
- GitHub Issues: [https://github.com/leixiaohui-1974/HydroSIS-2D/issues](https://github.com/leixiaohui-1974/HydroSIS-2D/issues)

---

**Version**: 2.0 (with integrated preprocessing/postprocessing toolkit)
**Last Updated**: 2025-10-29