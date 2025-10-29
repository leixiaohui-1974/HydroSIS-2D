# HydroSIS-2D Preprocessing and Postprocessing Toolkit - User Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Getting Started](#getting-started)
4. [Basic Workflows](#basic-workflows)
5. [Advanced Topics](#advanced-topics)
6. [Troubleshooting](#troubleshooting)
7. [FAQ](#faq)

## Introduction

This user guide provides step-by-step instructions for using the HydroSIS-2D preprocessing and postprocessing toolkit. Whether you're setting up your first simulation or analyzing complex results, this guide will help you get the most out of the toolkit.

### Who Should Use This Guide

- New users setting up their first simulation
- Experienced users looking for advanced features
- Researchers analyzing simulation results
- Developers integrating the toolkit into custom workflows

### What You'll Learn

- How to create and configure simulations
- How to set up meshes, terrain, and boundary conditions
- How to visualize and analyze results
- Best practices and optimization tips

## Installation

### Prerequisites

Before installing the toolkit, ensure you have:

- Python 3.8 or higher
- pip package manager
- At least 4GB of RAM (8GB recommended for large simulations)

### Step 1: Install Core Dependencies

```bash
# Navigate to the toolkit directory
cd /path/to/HydroSIS-2D/prepost

# Install core numerical libraries
pip install numpy scipy matplotlib

# Install visualization libraries
pip install pyvista vtk

# Install testing framework (optional)
pip install pytest
```

### Step 2: Verify Installation

Test that the installation was successful:

```python
# Create a test script: test_install.py
import sys
import os
sys.path.insert(0, '/path/to/HydroSIS-2D/prepost')

from simulation import SimulationConfig
from preprocessing.mesh_generation import MeshGenerator
from preprocessing.geometry import GeometryGenerator

print("✓ All modules imported successfully!")
```

Run the test:
```bash
python test_install.py
```

### Troubleshooting Installation

**ImportError for pyvista**:
```bash
# Try installing specific version
pip install pyvista==0.38.0
```

**CUDA/GPU warnings**:
- These are normal if you don't have a GPU. The toolkit works fine on CPU.

## Getting Started

### Your First Simulation

Let's create a simple dam break simulation from scratch.

#### Step 1: Create Configuration

```python
# my_first_simulation.py
import sys
sys.path.insert(0, '/path/to/HydroSIS-2D/prepost')

from simulation import create_dam_break_simulation

# Create a dam break scenario
config = create_dam_break_simulation(
    length=100.0,       # 100 meter long domain
    width=50.0,         # 50 meter wide domain
    dam_position=0.5,   # Dam at middle
    upstream_depth=5.0, # 5 meter water depth upstream
    downstream_depth=0.0, # Dry downstream
    nx=50,              # 50 cells in x direction
    ny=25,              # 25 cells in y direction
    simulation_time=10.0 # Simulate 10 seconds
)

print("Simulation created successfully!")
```

#### Step 2: Validate Configuration

```python
# Validate the configuration
is_valid, errors = config.validate()

if is_valid:
    print("✓ Configuration is valid!")
else:
    print("✗ Configuration has errors:")
    for error in errors:
        print(f"  - {error}")
```

#### Step 3: Export Configuration

```python
# Export all configuration files
config.export_configuration('output/my_first_simulation')

print("Configuration exported to: output/my_first_simulation/")
print("Files created:")
print("  - simulation_config.json")
print("  - mesh.vtk")
print("  - boundary_conditions.json")
print("  - initial_conditions.json")
print("  - initial_fields.npz")
```

#### Step 4: Review Configuration

```python
# Print configuration summary
print(config.summary())

# Get detailed statistics
stats = config.get_statistics()
print(f"\nTotal water volume: {stats['initial_conditions']['total_volume']:.2f} m³")
print(f"Number of cells: {stats['mesh']['ncells']}")
print(f"Simulation time: {stats['simulation']['time']:.1f} seconds")
```

### Understanding the Output Files

After exporting, you'll have these files:

1. **simulation_config.json**: Main configuration metadata
   - Domain dimensions
   - Simulation parameters
   - File references

2. **mesh.vtk**: Computational mesh
   - Can be viewed in ParaView
   - Contains cell coordinates

3. **boundary_conditions.json**: Boundary conditions
   - Specification for each boundary (NORTH, SOUTH, EAST, WEST)
   - Type and parameters

4. **initial_conditions.json**: Initial conditions
   - Type and parameters
   - Statistics

5. **initial_fields.npz**: Numpy archive with initial data
   - `depth`: Initial water depth array
   - `velocity_x`: Initial x-velocity array
   - `velocity_y`: Initial y-velocity array

## Basic Workflows

### Workflow 1: Channel Flow Simulation

Create a steady channel flow with inflow and outflow:

```python
from simulation import create_channel_flow_simulation

# Create channel flow configuration
config = create_channel_flow_simulation(
    length=500.0,        # 500m long channel
    width=100.0,         # 100m wide channel
    inflow_depth=3.0,    # 3m inflow depth
    inflow_velocity=1.5, # 1.5 m/s inflow velocity
    slope=0.001,         # 0.1% bed slope
    nx=100,
    ny=50,
    simulation_time=200.0 # 200 seconds
)

# Validate and export
is_valid, errors = config.validate()
if is_valid:
    config.export_configuration('output/channel_flow')
    print("✓ Channel flow configuration ready!")
```

### Workflow 2: Custom Terrain Simulation

Create a simulation with complex terrain:

```python
from simulation import SimulationConfig
from preprocessing.geometry import GeometryGenerator
from preprocessing.boundary_conditions import InflowBC, OutflowBC, WallBC, BCLocation
from preprocessing.initial_conditions import DryBedIC

# Step 1: Create configuration
config = SimulationConfig(name="Flood Over Hills")

# Step 2: Setup domain
config.set_domain_and_mesh(
    xmin=0.0, xmax=1000.0,
    ymin=0.0, ymax=500.0,
    nx=200, ny=100
)

# Step 3: Create terrain with hills
terrain_gen = GeometryGenerator()
terrain = terrain_gen.composite_terrain(
    features=[
        ('inclined_plane', {
            'slope_x': 0.002,
            'base_elevation': 0.0
        }),
        ('gaussian_hill', {
            'center_x': 400.0,
            'center_y': 250.0,
            'height': 30.0,
            'width': 120.0
        }),
        ('gaussian_hill', {
            'center_x': 700.0,
            'center_y': 200.0,
            'height': 20.0,
            'width': 100.0
        })
    ],
    ncols=200,
    nrows=100,
    cellsize=5.0
)
config.set_terrain(terrain.data)

# Step 4: Setup boundary conditions
bc_manager = config.setup_boundary_conditions()
bc_manager.set_boundary(InflowBC(BCLocation.WEST, depth=4.0, velocity_x=2.0))
bc_manager.set_boundary(OutflowBC(BCLocation.EAST, outflow_type='zero_gradient'))
bc_manager.set_boundary(WallBC(BCLocation.NORTH))
bc_manager.set_boundary(WallBC(BCLocation.SOUTH))

# Step 5: Setup initial conditions (start with dry bed)
ic_manager = config.setup_initial_conditions()
ic_manager.set_initial_condition(DryBedIC())

# Step 6: Set simulation parameters
config.set_simulation_parameters(
    simulation_time=300.0,
    output_interval=10.0,
    cfl_number=0.5
)

# Step 7: Validate and export
is_valid, errors = config.validate()
if is_valid:
    config.export_configuration('output/flood_hills')
    print("✓ Complex terrain simulation ready!")
```

### Workflow 3: Time-Varying Inflow

Create a simulation with time-varying inflow (e.g., storm hydrograph):

```python
import numpy as np
from simulation import SimulationConfig
from preprocessing.boundary_conditions import TimeSeriesBC, OutflowBC, WallBC, BCLocation
from preprocessing.initial_conditions import DryBedIC

# Create configuration
config = SimulationConfig(name="Storm Hydrograph")
config.set_domain_and_mesh(0, 1000, 0, 200, 200, 40)
config.set_flat_terrain(0.0)

# Define storm hydrograph
times = np.array([0, 30, 60, 90, 120, 180, 240, 300])  # seconds
depths = np.array([1.0, 2.5, 4.0, 5.0, 4.5, 3.0, 2.0, 1.5])  # meters
velocities = np.array([0.5, 1.5, 2.5, 3.0, 2.8, 2.0, 1.5, 1.0])  # m/s

# Setup boundary conditions
bc_manager = config.setup_boundary_conditions()
bc_manager.set_boundary(TimeSeriesBC(
    BCLocation.WEST,
    times=times,
    depths=depths,
    velocity_x=velocities
))
bc_manager.set_boundary(OutflowBC(BCLocation.EAST, outflow_type='zero_gradient'))
bc_manager.set_boundary(WallBC(BCLocation.NORTH))
bc_manager.set_boundary(WallBC(BCLocation.SOUTH))

# Setup initial conditions
ic_manager = config.setup_initial_conditions()
ic_manager.set_initial_condition(DryBedIC())

# Set simulation parameters
config.set_simulation_parameters(
    simulation_time=300.0,
    output_interval=15.0,
    cfl_number=0.5
)

# Export
config.export_configuration('output/storm_hydrograph')
```

### Workflow 4: Visualizing Results

After running a simulation, visualize the results:

```python
from postprocessing.result_analysis import ResultAnalyzer
from postprocessing.visualization_engine import VisualizationEngine

# Load simulation results
analyzer = ResultAnalyzer()
analyzer.load_vtk_series('output/results/result_*.vtk')

print(f"Loaded {len(analyzer.results)} time steps")

# Get result at specific time
result = analyzer.get_result_at_time(50.0)

# Visualize water surface
vis = VisualizationEngine(offscreen=False)
vis.visualize_water_surface(
    result.mesh,
    result.depth,
    result.terrain,
    cmap='Blues',
    show_edges=False
)
vis.add_colorbar('Water Depth [m]')
vis.show()

# Save screenshot
vis.save_screenshot('water_surface_t50.png')
```

### Workflow 5: Creating Animations

Create an animation from time series results:

```python
from postprocessing.animation import AnimationGenerator

# Create animation generator
anim_gen = AnimationGenerator()

# Load time series
anim_gen.load_vtk_series('output/results/result_*.vtk')

# Create water surface animation
anim_gen.create_water_surface_animation(
    output_file='flood_animation.mp4',
    fps=10,
    cmap='Blues',
    show_colorbar=True,
    colorbar_label='Water Depth [m]'
)

print("Animation created: flood_animation.mp4")
```

### Workflow 6: Analyzing Results

Compute statistics and extract profiles:

```python
from postprocessing.result_analysis import ResultAnalyzer
from postprocessing.profile_extractor import ProfileExtractor
import matplotlib.pyplot as plt

# Load results
analyzer = ResultAnalyzer()
analyzer.load_vtk_series('output/results/result_*.vtk')

# Compute statistics over all time steps
stats = analyzer.compute_statistics('depth')

print(f"Maximum depth: {stats['max']:.2f} m")
print(f"Mean depth: {stats['mean']:.2f} m")
print(f"Total volume: {stats['total_volume']:.2f} m³")

# Check mass balance
mass_balance = analyzer.check_mass_balance()
print(f"Mass balance error: {mass_balance['relative_error']:.4%}")

# Extract profile along centerline
extractor = ProfileExtractor()
result = analyzer.get_result_at_time(100.0)

# Define profile path (from x=0 to x=1000 along y=250)
start_point = (0, 250)
end_point = (1000, 250)

profile_data = extractor.extract_profile(
    result.mesh,
    result.depth,
    start_point,
    end_point,
    num_points=100
)

# Plot profile
plt.figure(figsize=(10, 6))
plt.plot(profile_data['distance'], profile_data['values'], 'b-', linewidth=2)
plt.xlabel('Distance [m]')
plt.ylabel('Water Depth [m]')
plt.title('Centerline Profile at t=100s')
plt.grid(True, alpha=0.3)
plt.savefig('profile_t100.png', dpi=150)
print("Profile plot saved: profile_t100.png")
```

## Advanced Topics

### Custom Initial Conditions

Create custom initial conditions using a function:

```python
from preprocessing.initial_conditions import CustomFieldIC
import numpy as np

# Create configuration
config = SimulationConfig(name="Custom IC")
config.set_domain_and_mesh(0, 100, 0, 50, 100, 50)
config.set_flat_terrain(0.0)

# Define custom depth function
def custom_depth_func(x, y):
    """Create a wave pattern"""
    depth = 2.0 + 1.0 * np.sin(2 * np.pi * x / 100) * np.cos(2 * np.pi * y / 50)
    return np.maximum(depth, 0.0)  # Ensure non-negative

# Define custom velocity function
def custom_velocity_func(x, y):
    """Circular flow pattern"""
    center_x, center_y = 50.0, 25.0
    dx = x - center_x
    dy = y - center_y
    r = np.sqrt(dx**2 + dy**2)

    # Velocity proportional to radius
    velocity_magnitude = 0.1 * r

    # Tangential velocity
    u = -dy / (r + 1e-6) * velocity_magnitude
    v = dx / (r + 1e-6) * velocity_magnitude

    return u, v

# Create mesh coordinates
mesh = config.mesh
X, Y = np.meshgrid(
    np.linspace(0, 100, 100),
    np.linspace(0, 50, 50),
    indexing='ij'
)

# Generate fields
depth = custom_depth_func(X, Y)
u, v = custom_velocity_func(X, Y)

# Setup initial conditions
ic_manager = config.setup_initial_conditions()
ic = CustomFieldIC(depth=depth, velocity_x=u, velocity_y=v)
ic_manager.set_initial_condition(ic)

# Continue with boundary conditions and export...
```

### Loading External Terrain Data

Load terrain from an ASCII Grid file:

```python
from preprocessing.terrain_reader import TerrainReader
from preprocessing.terrain_processor import TerrainProcessor

# Read terrain file
terrain = TerrainReader.read('data/terrain_data.asc')

print(f"Terrain size: {terrain.ncols} × {terrain.nrows}")
print(f"Cell size: {terrain.cellsize} m")
print(f"Elevation range: {terrain.data.min():.2f} - {terrain.data.max():.2f} m")

# Process terrain
processor = TerrainProcessor(terrain)

# Smooth terrain to remove noise
smoothed = processor.smooth_terrain(method='gaussian', sigma=2.0)

# Interpolate to simulation mesh
config = SimulationConfig()
config.set_domain_and_mesh(
    xmin=terrain.xllcorner,
    xmax=terrain.xllcorner + terrain.ncols * terrain.cellsize,
    ymin=terrain.yllcorner,
    ymax=terrain.yllcorner + terrain.nrows * terrain.cellsize,
    nx=200, ny=100
)

# Interpolate terrain to mesh
terrain_on_mesh = processor.interpolate_to_mesh(config.mesh, method='linear')
config.set_terrain(terrain_on_mesh)

# Analyze terrain statistics
stats = processor.analyze_terrain()
print(f"Mean slope: {stats['mean_slope']:.4f}")
print(f"Max slope: {stats['max_slope']:.4f}")
```

### Adaptive Mesh Refinement

Use adaptive mesh refinement for focused resolution:

```python
from preprocessing.adaptive_mesh import AdaptiveMeshGenerator, RefinementZone
from preprocessing.mesh_generation import DomainParams

# Define domain
domain = DomainParams(xmin=0, xmax=1000, ymin=0, ymax=500)

# Create adaptive mesh generator
adaptive_gen = AdaptiveMeshGenerator(domain, base_resolution=(50, 25))

# Add circular refinement zone around structure
zone1 = RefinementZone(
    center=(300, 250),
    radius=100,
    refinement_level=2,  # 4x refinement
    description="Around dam"
)
adaptive_gen.add_refinement_zone(zone1)

# Add refinement in downstream area
zone2 = RefinementZone(
    bounds=(400, 800, 100, 400),
    refinement_level=1,  # 2x refinement
    description="Downstream flood area"
)
adaptive_gen.add_refinement_zone(zone2)

# Generate mesh
adaptive_mesh = adaptive_gen.generate()

print(f"Generated adaptive mesh:")
print(f"  Base cells: {50 * 25}")
print(f"  Total cells: {adaptive_mesh.ncells}")
print(f"  Refinement ratio: {adaptive_mesh.ncells / (50 * 25):.2f}x")

# Visualize refinement
adaptive_gen.visualize_refinement_levels('refinement_map.png')
```

### Periodic Boundary Conditions

Setup periodic boundaries for domains with repeating patterns:

```python
from preprocessing.boundary_conditions import PeriodicBC, BCLocation

config = SimulationConfig()
config.set_domain_and_mesh(0, 100, 0, 50, 100, 50)
config.set_flat_terrain(0.0)

# Setup boundary conditions
bc_manager = config.setup_boundary_conditions()

# Create periodic pair for x-direction
bc_manager.set_boundary(PeriodicBC(BCLocation.WEST, paired_with=BCLocation.EAST))
bc_manager.set_boundary(PeriodicBC(BCLocation.EAST, paired_with=BCLocation.WEST))

# Walls for y-direction
bc_manager.set_boundary(WallBC(BCLocation.NORTH))
bc_manager.set_boundary(WallBC(BCLocation.SOUTH))

# Validate
is_valid, errors = bc_manager.validate()
if is_valid:
    print("✓ Periodic boundary conditions configured correctly")
```

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: Import Errors

**Problem**: `ModuleNotFoundError: No module named 'preprocessing'`

**Solution**:
```python
import sys
sys.path.insert(0, '/path/to/HydroSIS-2D/prepost')
# Then import modules
```

Or set PYTHONPATH:
```bash
export PYTHONPATH=/path/to/HydroSIS-2D/prepost:$PYTHONPATH
```

#### Issue 2: Validation Errors

**Problem**: Configuration validation fails

**Solution**: Check error messages:
```python
is_valid, errors = config.validate()
if not is_valid:
    for error in errors:
        print(f"Error: {error}")
```

Common validation errors:
- Missing mesh or domain
- Missing boundary conditions
- Missing initial conditions
- Invalid simulation parameters (negative time, etc.)
- Inconsistent periodic boundary pairs

#### Issue 3: Memory Errors

**Problem**: `MemoryError` for large meshes

**Solution**:
- Reduce mesh resolution
- Use adaptive mesh refinement instead of uniform mesh
- Export in binary format instead of ASCII

```python
# Check estimated memory before generating
from preprocessing.mesh_quality import MeshQualityChecker

checker = MeshQualityChecker(mesh)
cost = checker.estimate_computational_cost(simulation_time=100.0)
print(f"Estimated memory: {cost['memory_mb']:.2f} MB")
```

#### Issue 4: CFL Condition Violations

**Problem**: Simulation crashes or produces unphysical results

**Solution**: Check CFL condition:
```python
from preprocessing.mesh_quality import MeshQualityChecker

checker = MeshQualityChecker(mesh)
cfl_result = checker.check_cfl_condition(
    max_velocity=5.0,  # Expected maximum velocity
    dt=0.1             # Time step
)

if not cfl_result['is_stable']:
    print(f"CFL violation! Max CFL: {cfl_result['cfl_max']:.3f}")
    print(f"Recommended time step: {cfl_result['recommended_dt']:.4f} s")
```

#### Issue 5: Visualization Not Showing

**Problem**: `vis.show()` doesn't display anything

**Solution**:
- Make sure `offscreen=False`:
```python
vis = VisualizationEngine(offscreen=False)
```
- If running on a server without display, use offscreen mode and save to file:
```python
vis = VisualizationEngine(offscreen=True)
vis.visualize_water_surface(...)
vis.save_screenshot('output.png')
```

## FAQ

### General Questions

**Q: What types of simulations can I create?**

A: The toolkit supports:
- Dam break scenarios
- Channel flow (steady and unsteady)
- Urban flooding
- Flood propagation over terrain
- Custom scenarios with any combination of BCs and ICs

**Q: Do I need a GPU to use the toolkit?**

A: No, the preprocessing and postprocessing toolkit works entirely on CPU. The GPU is only needed for running the actual HydroSIS-2D simulation solver.

**Q: Can I use my own terrain data?**

A: Yes! Import terrain from ASCII Grid format (.asc files) using `TerrainReader`, or create synthetic terrain using `GeometryGenerator`.

**Q: What output formats are supported?**

A: The toolkit exports:
- VTK files (ParaView compatible)
- JSON configuration files
- NumPy binary files (.npz)
- ASCII Grid files (.asc)

### Technical Questions

**Q: How do I choose the right mesh resolution?**

A: Follow these guidelines:
- Start with coarser mesh for testing
- Cell size should resolve important features (structures, channels)
- Check CFL condition for stability
- Use adaptive refinement for efficiency

See the mesh resolution table in the README for application-specific recommendations.

**Q: What's the difference between zero-gradient and critical depth outflow?**

A:
- **Zero-gradient**: Assumes no change in flow variables at boundary (∂h/∂x = 0)
- **Critical depth**: Enforces critical flow condition (Froude number = 1)

Use zero-gradient for subcritical outflows, critical depth for supercritical flows.

**Q: Can I modify initial conditions after creation?**

A: Yes:
```python
ic_manager = config.setup_initial_conditions()
ic_manager.set_initial_condition(UniformIC(depth=3.0))

# Later, change it
ic_manager.set_initial_condition(DamBreakIC(dam_position=0.5, ...))
```

**Q: How do I validate my mesh quality?**

A:
```python
from preprocessing.mesh_quality import MeshQualityChecker

checker = MeshQualityChecker(mesh)
metrics = checker.compute_metrics()
checker.print_report()

# Check for warnings
if metrics['warnings']:
    for warning in metrics['warnings']:
        print(f"Warning: {warning}")
```

### Best Practices

**Q: What's the recommended workflow?**

A:
1. Start with factory functions for common scenarios
2. Validate configuration before exporting
3. Test with coarse mesh first
4. Refine mesh resolution based on results
5. Document your configuration parameters

**Q: How should I organize my simulation files?**

A:
```
project/
├── input/
│   ├── terrain.asc
│   └── config.py
├── output/
│   ├── sim1/
│   ├── sim2/
│   └── ...
├── results/
│   ├── sim1/
│   └── sim2/
└── analysis/
    ├── plots/
    └── animations/
```

**Q: Should I use adaptive mesh refinement?**

A: Use adaptive refinement when:
- You have localized features (structures, sources)
- Terrain has sharp gradients
- You need efficiency (reduce cells while maintaining accuracy)

Use uniform mesh when:
- Domain is relatively simple
- You need predictable computational cost
- Resolution requirements are uniform

## Additional Resources

### Example Scripts

All examples are in `prepost/examples/`:
- `example_mesh_generation.py`: Mesh creation examples
- `example_geometry.py`: Terrain processing examples
- `example_boundary_conditions.py`: BC setup examples
- `example_initial_conditions.py`: IC setup examples
- `example_complete_workflow.py`: End-to-end workflows
- `example_visualization.py`: Visualization examples
- `example_result_analysis.py`: Analysis examples

### Documentation

- Main README: `prepost/README.md`
- Development Roadmap: `docs/PREPROCESSING_POSTPROCESSING_ROADMAP.md`
- Module Summaries: `docs/MESH_GENERATION_SUMMARY.md`, etc.
- API Reference: See README API section

### Getting Help

1. Check this user guide first
2. Review example scripts for similar use cases
3. Check the FAQ section
4. Review error messages carefully
5. Open an issue on GitHub with:
   - Minimal code to reproduce the problem
   - Full error message
   - System information (Python version, OS)

### Contributing

Found a bug or want to add a feature? Contributions welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

Make sure all tests pass:
```bash
cd prepost
pytest tests/ -v
```

---

**Document Version**: 1.0
**Last Updated**: 2025-10-29
**Feedback**: Please report issues or suggestions on GitHub
