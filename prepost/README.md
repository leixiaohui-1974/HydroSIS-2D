# HydroSIS-2D Pre/Post-Processing Toolkit

A comprehensive preprocessing and postprocessing suite for the HydroSIS-2D hydrodynamic simulation software.

## Features

### ✅ Preprocessing (Implemented)

- **Structured Mesh Generation**
  - Uniform Cartesian mesh generation
  - Configurable cell count or cell spacing
  - Target resolution-based generation

- **Adaptive Mesh Refinement**
  - User-defined rectangular refinement zones
  - Circular refinement zones (e.g., around structures)
  - Terrain gradient-based automatic refinement
  - Multi-level refinement (2x, 4x, 8x, ...)
  - Combined refinement strategies

- **Mesh Quality Assessment**
  - Cell size and aspect ratio analysis
  - CFL stability condition checking
  - Computational cost estimation
  - Automatic quality warnings
  - Comprehensive quality reports

- **Mesh I/O**
  - Export to HydroSIS-2D INI format
  - Export to VTK format (ParaView compatible)
  - Export to JSON metadata format
  - NumPy binary format for coordinates
  - Batch export to multiple formats

### ✅ Postprocessing (Implemented)

- **3D Visualization Engine**
  - Mesh structure visualization
  - Terrain elevation rendering
  - Water surface visualization (terrain + depth)
  - Velocity field visualization (magnitude and vectors)
  - Combined multi-field visualizations
  - High-quality screenshot export

- **Animation Generation**
  - Time series animations (water evolution, velocity)
  - Multiple output formats (MP4, GIF, PNG sequence)
  - Customizable frame rates and colormaps
  - Time labels and annotations

- **Colormap Utilities**
  - Scientific colormap recommendations
  - 40+ available colormaps
  - Field-type to colormap mapping

### 🚧 In Development

- Unstructured mesh generation (Gmsh integration)
- Geometry processing and CAD import
- Boundary condition interactive setup
- Result analysis tools
- GUI interface

## Installation

### Requirements

```bash
# Core dependencies
pip install numpy scipy

# Visualization (required for postprocessing)
pip install pyvista matplotlib

# Optional for testing
pip install pytest
```

### Setup

```bash
cd /path/to/HydroSIS-2D/prepost
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

## Quick Start

### Example 1: Basic Uniform Mesh

```python
from preprocessing.mesh_generation import MeshGenerator, DomainParams, MeshIO

# Define domain
domain = DomainParams(xmin=0, xmax=100, ymin=0, ymax=50)

# Create mesh generator
generator = MeshGenerator(domain)

# Generate uniform mesh
mesh = generator.generate_uniform_mesh(nx=100, ny=50)

# Export to HydroSIS-2D format
MeshIO.export_to_ini(mesh, "mesh_config.ini")
```

### Example 2: Adaptive Mesh with Refinement Zones

```python
from preprocessing.mesh_generation import AdaptiveMeshGenerator, DomainParams

# Create adaptive generator
domain = DomainParams(xmin=0, xmax=1000, ymin=0, ymax=500)
generator = AdaptiveMeshGenerator(domain)

# Add refinement zone around dam
generator.add_circular_refinement_zone(
    center_x=200, center_y=250,
    radius=100,
    refinement_level=2  # 4x finer
)

# Add rectangular zone downstream
generator.add_refinement_zone(
    xmin=300, xmax=600,
    ymin=150, ymax=350,
    refinement_level=1  # 2x finer
)

# Generate mesh
mesh = generator.generate_multiresolution_mesh(base_nx=50, base_ny=25)

print(f"Generated mesh: {mesh.nx} × {mesh.ny} = {mesh.ncells:,} cells")
```

### Example 3: Terrain-based Refinement

```python
import numpy as np
from preprocessing.mesh_generation import AdaptiveMeshGenerator, DomainParams

# Load or create terrain data
terrain = np.load("terrain_data.npy")  # Shape: (ny, nx)

# Create generator
domain = DomainParams(xmin=0, xmax=1000, ymin=0, ymax=500)
generator = AdaptiveMeshGenerator(domain)

# Generate refinement map from terrain gradients
generator.generate_refinement_map_from_terrain(
    terrain_data=terrain,
    gradient_threshold=0.1,
    max_refinement_level=2
)

# Generate mesh
mesh = generator.generate_multiresolution_mesh(50, 25)
```

### Example 4: Mesh Quality Assessment

```python
from preprocessing.mesh_generation import MeshQualityChecker

# Check mesh quality
checker = MeshQualityChecker(mesh)
metrics = checker.compute_metrics()

# Print comprehensive report
checker.print_report()

# Check CFL stability
cfl_result = checker.check_cfl_condition(max_velocity=5.0, dt=0.1)
print(f"CFL number: {cfl_result['cfl_max']:.3f}")
print(f"Stable: {cfl_result['is_stable']}")

# Estimate computational cost
cost = checker.estimate_computational_cost(simulation_time=100.0)
print(f"Estimated time steps: {cost['num_timesteps']:,}")
print(f"Memory required: {cost['memory_mb']:.2f} MB")
```

### Example 5: 3D Visualization

```python
from preprocessing.mesh_generation import MeshGenerator, DomainParams
from postprocessing.visualization import VisualizationEngine
import numpy as np

# Create mesh and data
domain = DomainParams(xmin=0, xmax=400, ymin=0, ymax=200)
generator = MeshGenerator(domain)
mesh = generator.generate_uniform_mesh(80, 40)

# Create terrain and water depth
terrain = 5.0 + 0.01 * mesh.x  # Sloped terrain
water_depth = np.maximum(0, 10.0 - 0.02 * mesh.x)  # Dam break initial condition

# Visualize water surface
engine = VisualizationEngine(offscreen=True)
engine.visualize_water_surface(mesh, water_depth, terrain, cmap='Blues')
engine.screenshot('water_surface.png')
engine.close()
```

### Example 6: Animation Generation

```python
from postprocessing.visualization import AnimationGenerator
import numpy as np

# Create animation generator
anim_gen = AnimationGenerator(mesh)

# Add time steps (simulate wave propagation)
for t in np.linspace(0, 10, 30):
    # Update water depth for each time step
    wave_front = 100 + 50 * t
    water_depth_t = np.where(mesh.x < wave_front, 10.0, 0.0)
    anim_gen.add_timestep(t, {'h': water_depth_t})

# Create animation
anim_gen.create_water_surface_animation(
    terrain,
    'flood_animation.gif',
    fps=10
)
```

## Running Examples

Complete working examples are provided in the `examples/` directory:

```bash
# Example 1: Basic mesh generation
cd prepost/examples
python example_basic_mesh.py

# Example 2: Adaptive mesh refinement
python example_adaptive_mesh.py

# Example 3: 3D Visualization and rendering
python example_visualization.py
```

Output files will be created in `examples/output/` directory.

## Running Tests

```bash
cd prepost/tests

# Test mesh generation
pytest test_mesh_generation.py -v

# Test visualization
pytest test_visualization.py -v
```

## Module Structure

```
prepost/
├── preprocessing/
│   ├── mesh_generation/
│   │   ├── mesh_generator.py       # Basic structured mesh generation
│   │   ├── adaptive_mesh.py        # Adaptive refinement
│   │   ├── mesh_quality.py         # Quality assessment
│   │   └── mesh_io.py              # Import/export utilities
│   ├── geometry/                   # [TODO] Geometry processing
│   └── boundary_conditions/        # [TODO] BC setup
├── postprocessing/
│   ├── visualization/              # [TODO] 3D visualization
│   └── analysis/                   # [TODO] Result analysis
├── gui/                            # [TODO] Graphical interface
├── tests/
│   └── test_mesh_generation.py     # Unit tests
└── examples/
    ├── example_basic_mesh.py       # Basic mesh example
    └── example_adaptive_mesh.py    # Adaptive mesh example
```

## API Reference

### Core Classes

#### `DomainParams`
Define computational domain extent.

**Parameters:**
- `xmin`, `xmax`: Domain extent in x-direction [m]
- `ymin`, `ymax`: Domain extent in y-direction [m]

**Properties:**
- `length_x`, `length_y`: Domain dimensions
- `area`: Total domain area

#### `MeshGenerator`
Basic structured mesh generator.

**Methods:**
- `generate_uniform_mesh(nx, ny)`: Generate mesh with specified cell count
- `generate_mesh_with_spacing(dx, dy)`: Generate mesh with cell spacing
- `generate_mesh_from_resolution(target_cell_area)`: Generate mesh targeting cell area
- `export_to_hydrosis_format(output_file)`: Export to INI format
- `visualize_mesh(show_every)`: Visualize mesh grid

#### `AdaptiveMeshGenerator`
Advanced mesh generator with adaptive refinement.

**Methods:**
- `add_refinement_zone(xmin, xmax, ymin, ymax, refinement_level)`: Add rectangular zone
- `add_circular_refinement_zone(center_x, center_y, radius, refinement_level)`: Add circular zone
- `generate_refinement_map_from_zones(base_nx, base_ny)`: Create refinement map from zones
- `generate_refinement_map_from_terrain(terrain_data, gradient_threshold, max_refinement_level)`: Create map from terrain
- `generate_multiresolution_mesh(base_nx, base_ny)`: Generate adaptive mesh
- `visualize_refinement_map()`: Visualize refinement levels
- `get_refinement_statistics()`: Get refinement statistics

#### `MeshQualityChecker`
Mesh quality assessment tool.

**Methods:**
- `compute_metrics()`: Compute quality metrics
- `check_cfl_condition(max_velocity, dt)`: Check CFL stability
- `estimate_computational_cost(simulation_time)`: Estimate cost
- `generate_quality_report()`: Generate text report
- `print_report()`: Print report to console

#### `MeshIO`
Mesh import/export utilities.

**Static Methods:**
- `export_to_ini(mesh, filepath)`: Export to HydroSIS-2D INI
- `export_to_vtk(mesh, filepath, data_arrays)`: Export to VTK
- `export_to_json(mesh, filepath)`: Export metadata to JSON
- `export_coordinates(mesh, filepath)`: Export coordinates to NPZ
- `import_from_json(filepath)`: Import mesh from JSON
- `export_batch(mesh, output_dir, base_name)`: Export to all formats

## Mesh Quality Metrics

The quality checker evaluates:

1. **Cell Size Distribution**: Min, max, mean cell sizes
2. **Aspect Ratio**: Ratio of dy/dx (recommended: 0.2 - 5.0)
3. **Uniformity Score**: 0-1 scale, 1 = perfectly uniform
4. **CFL Stability**: Checks if time step satisfies stability condition
5. **Computational Cost**: Estimates memory and FLOPS requirements

### Quality Warnings

The system automatically warns about:
- Extreme aspect ratios (< 0.2 or > 5.0)
- Very small cells (< 0.1 m, high computational cost)
- Very large cells (> 100 m, accuracy concerns)
- CFL condition violations
- Large mesh sizes (> 10M cells)

## File Formats

### HydroSIS-2D INI Format

Standard configuration format for the solver:

```ini
[Grid]
nx = 100
ny = 50
dx = 1.000000
dy = 1.000000
xmin = 0.000000
ymin = 0.000000

[Physics]
gravity = 9.81
cfl = 0.5
...
```

### VTK Format

Structured grid format compatible with ParaView:
- ASCII or binary format
- Includes cell-centered data fields
- Supports multiple scalar and vector fields

### JSON Format

Human-readable metadata format:
```json
{
  "type": "StructuredMesh",
  "nx": 100,
  "ny": 50,
  "domain": {
    "xmin": 0.0,
    "xmax": 100.0,
    ...
  }
}
```

## Best Practices

### Mesh Resolution Guidelines

| Application | Typical Cell Size | Grid Size |
|-------------|------------------|-----------|
| Large-scale flood | 10-100 m | 10³-10⁵ cells |
| Dam break | 1-10 m | 10⁴-10⁶ cells |
| Urban flooding | 0.5-5 m | 10⁵-10⁷ cells |
| Laboratory scale | 0.01-0.1 m | 10⁴-10⁵ cells |

### Adaptive Refinement Tips

1. **Refinement Zones**: Place around:
   - Dam/breach locations
   - Structures and obstacles
   - Observation/gauge points
   - Areas of interest

2. **Terrain-based**: Useful for:
   - Complex topography
   - Steep slopes
   - River channels
   - Sharp elevation changes

3. **Performance**: Each refinement level (2x) increases:
   - Cell count by 4x
   - Memory by 4x
   - Computation time by ~8x (4x cells × 2x smaller dt)

### CFL Stability

For shallow water equations:
- CFL number should be < 0.5 for stability
- Time step: `dt < CFL × min(dx, dy) / (|u| + √(gh))`
- Consider maximum expected velocity and depth

## Troubleshooting

### Import Errors

If you get import errors:
```bash
export PYTHONPATH=/path/to/HydroSIS-2D/prepost:$PYTHONPATH
```

### Matplotlib Not Available

Visualization is optional. Install with:
```bash
pip install matplotlib
```

### Large Mesh Memory Issues

For very large meshes (> 10M cells):
- Use VTK binary format instead of ASCII
- Consider coarser base resolution
- Limit refinement levels

## Development Roadmap

See `docs/PREPROCESSING_POSTPROCESSING_ROADMAP.md` for the complete development plan.

### Phase 1 (Current)
- ✅ Structured mesh generation
- ✅ Adaptive refinement
- ✅ Quality assessment
- ✅ Mesh I/O

### Phase 2 (Next)
- 🚧 Unstructured mesh (Gmsh integration)
- 🚧 Geometry processing
- 🚧 Boundary condition setup
- 🚧 3D visualization engine

### Phase 3 (Future)
- ⏳ GUI interface
- ⏳ Result analysis
- ⏳ Report generation

## Contributing

This is an active development project. Contributions welcome!

## License

Part of the HydroSIS-2D project.

## Support

For issues and questions, please refer to the main HydroSIS-2D documentation.

---

**Version**: 0.1.0
**Last Updated**: 2025-10-29
**Author**: HydroSIS-2D Development Team
