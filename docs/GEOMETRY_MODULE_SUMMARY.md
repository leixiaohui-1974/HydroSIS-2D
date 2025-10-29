# Geometry Processing Module Summary

## Overview

The Geometry Processing Module provides comprehensive tools for handling terrain elevation data and generating synthetic geometric features for HydroSIS-2D simulations. This module completes the essential preprocessing capabilities needed to prepare realistic simulation domains.

**Module Status**: ✅ Complete and tested
**Lines of Code**: ~1,200 lines across 3 core files
**Test Coverage**: 29 unit tests, 100% passing
**Documentation**: Complete with working examples

---

## Architecture

### Module Structure

```
preprocessing/geometry/
├── __init__.py              # Module exports
├── terrain_reader.py        # Terrain data I/O (250 lines)
├── terrain_processor.py     # Processing and analysis (440 lines)
└── geometry_generator.py    # Synthetic geometry (510 lines)

tests/
└── test_geometry.py         # Comprehensive tests (490 lines)

examples/
└── example_geometry_processing.py  # Demonstration (440 lines)
```

---

## Component Details

### 1. terrain_reader.py

**Purpose**: Read, write, and manage terrain elevation data from standard GIS formats.

**Key Classes**:

```python
class TerrainData:
    """Container for terrain elevation data with metadata"""
    - data: 2D numpy array of elevations
    - ncols, nrows: Grid dimensions
    - xllcorner, yllcorner: Lower-left corner coordinates
    - cellsize: Grid resolution
    - nodata_value: Value representing missing data

    # Properties
    - xmax, ymax: Upper bounds
    - extent: (xmin, xmax, ymin, ymax)

    # Methods
    - get_statistics(): Compute min, max, mean, std, median

class ASCIIGridReader:
    """Read/write ArcGIS ASCII Grid format (.asc files)"""
    @staticmethod
    def read(filepath) -> TerrainData
    @staticmethod
    def write(terrain, filepath)

class TerrainReader:
    """Main terrain reader interface"""
    @staticmethod
    def read_ascii_grid(filepath) -> TerrainData
    @staticmethod
    def from_array(data, **kwargs) -> TerrainData
```

**Supported Formats**:
- ✅ ASCII Grid (.asc, .txt) - ArcGIS standard format
- 🔜 GeoTIFF (.tif) - Planned for future release
- 🔜 NetCDF (.nc) - Planned for future release

**Example Usage**:
```python
from preprocessing.geometry import load_terrain, save_terrain

# Load terrain from file
terrain = load_terrain('elevation.asc')
print(f"Loaded {terrain.ncols}×{terrain.nrows} terrain")
print(f"Elevation range: {terrain.get_statistics()}")

# Create from numpy array
import numpy as np
data = np.random.rand(100, 50)
terrain = TerrainReader.from_array(data, cellsize=2.0)

# Save terrain
save_terrain(terrain, 'output.asc')
```

---

### 2. terrain_processor.py

**Purpose**: Process, analyze, and interpolate terrain data for simulation preparation.

**Key Class**:

```python
class TerrainProcessor:
    """Comprehensive terrain processing and analysis"""

    def __init__(self, terrain: TerrainData)

    # Interpolation
    def interpolate_to_mesh(mesh, method='linear') -> np.ndarray
        # Methods: 'linear', 'cubic'

    # Smoothing
    def smooth_gaussian(sigma=1.0) -> TerrainData
    def smooth_uniform(size=3) -> TerrainData

    # Terrain Analysis
    def compute_gradient() -> Tuple[np.ndarray, np.ndarray]
    def compute_slope() -> np.ndarray
    def compute_aspect() -> np.ndarray
    def compute_curvature() -> Tuple[np.ndarray, np.ndarray]

    # Mesh Refinement Support
    def identify_refinement_zones(slope_threshold=0.1) -> np.ndarray

    # Data Operations
    def resample(new_cellsize) -> TerrainData
    def clip_to_extent(xmin, xmax, ymin, ymax) -> TerrainData
    def fill_nodata(method='nearest') -> TerrainData
```

**Key Features**:

1. **Interpolation to Mesh**
   - Linear interpolation (fast, suitable for most cases)
   - Cubic interpolation (smooth, better for visualization)
   - Automatic bounds handling with configurable fill values
   - Seamless integration with StructuredMesh

2. **Smoothing Filters**
   - Gaussian smoothing with adjustable sigma
   - Uniform (box) smoothing with configurable window size
   - Preserves nodata values during filtering

3. **Terrain Analysis**
   - Gradient computation using central differences
   - Slope magnitude calculation
   - Aspect direction (0-360°)
   - Profile and planform curvature

4. **Adaptive Mesh Support**
   - Identify steep terrain requiring refinement
   - Configurable slope thresholds
   - Returns boolean mask for refinement zones

5. **Data Management**
   - Resample to different resolutions
   - Clip to specific extents
   - Fill missing data with interpolation

**Example Usage**:
```python
from preprocessing.geometry import TerrainProcessor, load_terrain
from preprocessing.mesh_generation import MeshGenerator, DomainParams

# Load terrain
terrain = load_terrain('elevation.asc')
processor = TerrainProcessor(terrain)

# Apply smoothing
terrain_smooth = processor.smooth_gaussian(sigma=2.0)

# Analyze terrain
slope = processor.compute_slope()
refinement_zones = processor.identify_refinement_zones(
    slope_threshold=0.15
)
print(f"Found {np.sum(refinement_zones)} cells requiring refinement")

# Interpolate to simulation mesh
domain = DomainParams(0, 1000, 0, 500)
generator = MeshGenerator(domain)
mesh = generator.generate_uniform_mesh(100, 50)

elevation = processor.interpolate_to_mesh(mesh, method='cubic')
```

**Convenience Function**:
```python
from preprocessing.geometry import process_terrain_for_simulation

# One-line terrain processing
elevation = process_terrain_for_simulation(
    terrain,
    mesh,
    smooth=True,
    smooth_sigma=1.0
)
```

---

### 3. geometry_generator.py

**Purpose**: Generate synthetic terrain and geometric features for testing and idealized scenarios.

**Key Class**:

```python
class GeometryGenerator:
    """Generate synthetic terrains and geometric features"""

    # Basic Shapes
    @staticmethod
    def flat_surface(ncols, nrows, elevation, ...) -> TerrainData
    @staticmethod
    def inclined_plane(ncols, nrows, slope_x, slope_y, ...) -> TerrainData

    # Topographic Features
    @staticmethod
    def gaussian_hill(ncols, nrows, center_x, center_y,
                     height, width, ...) -> TerrainData
    @staticmethod
    def valley(ncols, nrows, orientation, depth, width, ...) -> TerrainData

    # Complex Terrains
    @staticmethod
    def random_terrain(ncols, nrows, amplitude, wavelength,
                      seed=None, ...) -> TerrainData
    @staticmethod
    def composite_terrain(features, ncols, nrows, ...) -> TerrainData

    # Standard Test Cases
    @staticmethod
    def dam_break_channel(length, width, dam_position, ...) -> TerrainData
```

**Available Terrains**:

1. **Flat Surface**
   - Uniform elevation
   - Used for basic verification tests

2. **Inclined Plane**
   - Linear slope in x and/or y direction
   - Configurable gradient
   - Tests steady flow conditions

3. **Gaussian Hill**
   - Smooth, bell-shaped elevation
   - Configurable height and width
   - Tests flow around obstacles

4. **Valley**
   - Parabolic cross-section
   - Orientation along x or y axis
   - Tests channel flow

5. **Random Terrain**
   - Perlin-like noise generation
   - Configurable amplitude and wavelength
   - Reproducible with seed parameter
   - Natural-looking terrain for testing

6. **Dam Break Channel**
   - Classic benchmark problem setup
   - Step function at dam position
   - Standard validation case

7. **Composite Terrain**
   - Combine multiple features
   - Additive composition
   - Create complex scenarios

**Example Usage**:

```python
from preprocessing.geometry import GeometryGenerator, create_test_terrain

gen = GeometryGenerator()

# Simple hill
terrain = gen.gaussian_hill(
    ncols=100, nrows=100,
    height=15.0,
    width=30.0,
    cellsize=1.0
)

# Inclined plane with hill
terrain = gen.composite_terrain(
    features=[
        ('inclined_plane', {'slope_x': 0.01}),
        ('gaussian_hill', {
            'center_x': 50.0,
            'center_y': 50.0,
            'height': 10.0
        })
    ],
    ncols=100, nrows=100, cellsize=1.0
)

# Quick test terrain
terrain = create_test_terrain('hill', ncols=50, nrows=50)
```

---

## Integration with Other Modules

### Mesh Generation Integration

```python
from preprocessing.geometry import GeometryGenerator, TerrainProcessor
from preprocessing.mesh_generation import (
    MeshGenerator, AdaptiveMeshGenerator, DomainParams
)

# Generate terrain
terrain = GeometryGenerator.gaussian_hill(100, 100, height=20.0)

# Identify refinement zones
processor = TerrainProcessor(terrain)
refinement_zones = processor.identify_refinement_zones(0.15)

# Create adaptive mesh
domain = DomainParams(0, 99, 0, 99)
mesh_gen = AdaptiveMeshGenerator(domain)

# Add refinement based on terrain
for i in range(100):
    for j in range(100):
        if refinement_zones[i, j]:
            x = i * 1.0
            y = j * 1.0
            mesh_gen.add_circular_refinement_zone(x, y, 5.0, 2)

mesh = mesh_gen.generate_multiresolution_mesh(50, 50)
```

### Visualization Integration

```python
from preprocessing.geometry import load_terrain, TerrainProcessor
from preprocessing.mesh_generation import MeshGenerator, DomainParams
from postprocessing.visualization import VisualizationEngine

# Load and process terrain
terrain = load_terrain('elevation.asc')
processor = TerrainProcessor(terrain)

# Create mesh and interpolate
domain = DomainParams(0, 1000, 0, 500)
mesh = MeshGenerator(domain).generate_uniform_mesh(100, 50)
elevation = processor.interpolate_to_mesh(mesh)

# Visualize
engine = VisualizationEngine()
engine.visualize_terrain(mesh, elevation, cmap='terrain')
engine.screenshot('terrain.png')
```

---

## Testing

### Test Coverage

**test_geometry.py**: 29 comprehensive unit tests

```
TestTerrainData (3 tests)
  ✓ test_creation
  ✓ test_extent_properties
  ✓ test_statistics

TestASCIIGridReader (2 tests)
  ✓ test_write_read_roundtrip
  ✓ test_read_nonexistent_file

TestTerrainReader (1 test)
  ✓ test_from_array

TestGeometryGenerator (8 tests)
  ✓ test_flat_surface
  ✓ test_inclined_plane
  ✓ test_gaussian_hill
  ✓ test_valley
  ✓ test_random_terrain
  ✓ test_dam_break_channel
  ✓ test_composite_terrain
  ✓ test_create_test_terrain

TestTerrainProcessor (11 tests)
  ✓ test_interpolate_to_mesh
  ✓ test_smooth_gaussian
  ✓ test_smooth_uniform
  ✓ test_compute_gradient
  ✓ test_compute_slope
  ✓ test_compute_aspect
  ✓ test_curvature
  ✓ test_identify_refinement_zones
  ✓ test_resample
  ✓ test_clip_to_extent
  ✓ test_fill_nodata

TestConvenienceFunctions (4 tests)
  ✓ test_process_terrain_for_simulation
  ✓ test_load_save_terrain
  ✓ test_load_unsupported_format
  ✓ test_save_unsupported_format

All 29 tests passing (100% success rate)
```

### Running Tests

```bash
# Run all geometry tests
pytest tests/test_geometry.py -v

# Run specific test class
pytest tests/test_geometry.py::TestTerrainProcessor -v

# Run with coverage
pytest tests/test_geometry.py --cov=preprocessing.geometry
```

---

## Examples

### example_geometry_processing.py

Comprehensive demonstration with 6 examples:

**Example 1: Basic Terrain Generation**
- Creates 4 different terrain types
- Demonstrates flat, inclined, hill, and valley terrains
- Output: `terrain_types.png`

**Example 2: Terrain Processing**
- Applies smoothing filters
- Computes slope, aspect, curvature
- Shows before/after comparisons
- Output: `terrain_processing.png`

**Example 3: Terrain Interpolation**
- High-res terrain (200×200) to coarse mesh (50×25)
- Compares linear vs cubic interpolation
- Output: `terrain_interpolation.png`

**Example 4: Refinement Zones**
- Identifies steep areas requiring refinement
- Configurable slope threshold
- Output: `refinement_zones.png`

**Example 5: Save/Load Operations**
- Demonstrates file I/O
- Verifies data integrity
- Output: `terrain_save_load.png`, `test_terrain.asc`

**Example 6: Complete Workflow**
- End-to-end simulation setup
- Dam break channel example
- Exports mesh and terrain files
- Output: `simulation_setup.png`, `terrain.asc`, `simulation_mesh.vtk`

### Running Examples

```bash
cd prepost/examples
python example_geometry_processing.py

# Generates 6 PNG images and 2 data files
```

---

## Performance Characteristics

### Computational Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Load ASCII Grid | O(n) | n = number of cells |
| Save ASCII Grid | O(n) | n = number of cells |
| Interpolate to mesh | O(m) | m = target mesh cells |
| Gaussian smoothing | O(n·k²) | k = kernel size |
| Gradient computation | O(n) | Central differences |
| Slope/aspect | O(n) | From gradients |
| Curvature | O(n) | Second derivatives |
| Resampling | O(m) | m = new grid size |

### Memory Usage

- TerrainData: ~8 bytes per cell (float64)
- 1000×1000 grid: ~8 MB
- Typical simulation domain: < 50 MB

### Optimization Tips

1. **Use appropriate interpolation**:
   - Linear: Faster, suitable for most cases
   - Cubic: Slower, better for visualization

2. **Pre-smooth noisy terrain**:
   - Apply Gaussian filter before interpolation
   - Reduces artifacts in simulation

3. **Clip to simulation domain**:
   - Load only necessary extent
   - Reduces memory and processing time

4. **Cache processed terrain**:
   - Save interpolated elevation to file
   - Reuse across multiple simulations

---

## Future Enhancements

### Planned Features

1. **Additional File Formats**
   - GeoTIFF support via rasterio
   - NetCDF support via xarray
   - HDF5 support

2. **Advanced Processing**
   - Multi-scale decomposition
   - Feature extraction (peaks, ridges, channels)
   - Watershed analysis

3. **Coordinate Systems**
   - Projection transformations
   - Coordinate reference system (CRS) support
   - Integration with pyproj

4. **Performance**
   - Numba acceleration for tight loops
   - Parallel processing for large terrains
   - Lazy evaluation for large files

5. **Terrain Modification**
   - Cut/fill operations
   - Smoothing along specific paths
   - Embankment and levee generation

---

## API Reference

### Quick Reference

```python
# Import module
from preprocessing.geometry import (
    # Data structures
    TerrainData,

    # I/O
    TerrainReader,
    ASCIIGridReader,
    load_terrain,
    save_terrain,

    # Processing
    TerrainProcessor,
    process_terrain_for_simulation,

    # Generation
    GeometryGenerator,
    create_test_terrain
)

# Load terrain
terrain = load_terrain('file.asc')

# Process terrain
processor = TerrainProcessor(terrain)
slope = processor.compute_slope()
terrain_smooth = processor.smooth_gaussian(2.0)

# Generate synthetic terrain
gen = GeometryGenerator()
terrain = gen.gaussian_hill(100, 100, height=15.0)

# Quick test terrain
terrain = create_test_terrain('hill')
```

---

## Common Use Cases

### 1. Load Real Terrain and Prepare for Simulation

```python
from preprocessing.geometry import load_terrain, process_terrain_for_simulation
from preprocessing.mesh_generation import MeshGenerator, DomainParams

# Load terrain from GIS data
terrain = load_terrain('real_terrain.asc')

# Create simulation mesh matching terrain extent
extent = terrain.extent
domain = DomainParams(extent[0], extent[1], extent[2], extent[3])
mesh = MeshGenerator(domain).generate_uniform_mesh(200, 100)

# Process terrain with smoothing
elevation = process_terrain_for_simulation(
    terrain, mesh, smooth=True, smooth_sigma=2.0
)

# Use elevation in simulation...
```

### 2. Create Idealized Test Case

```python
from preprocessing.geometry import GeometryGenerator

# Dam break channel
terrain = GeometryGenerator.dam_break_channel(
    length=500.0,
    width=100.0,
    dam_position=250.0,
    cellsize=0.5
)

# Initial water depth (only upstream)
h0 = np.where(terrain.data > 5.0, 10.0, 0.0)
```

### 3. Adaptive Mesh from Terrain

```python
from preprocessing.geometry import load_terrain, TerrainProcessor
from preprocessing.mesh_generation import AdaptiveMeshGenerator

# Load terrain
terrain = load_terrain('terrain.asc')

# Identify steep zones
processor = TerrainProcessor(terrain)
steep_zones = processor.identify_refinement_zones(slope_threshold=0.2)

# Create adaptive mesh (pseudo-code for concept)
# In practice, integrate with AdaptiveMeshGenerator
mesh_gen = AdaptiveMeshGenerator(domain)
for i, j in np.argwhere(steep_zones):
    x, y = terrain.xllcorner + i*terrain.cellsize, ...
    mesh_gen.add_circular_refinement_zone(x, y, radius=10.0, level=2)
```

---

## Troubleshooting

### Common Issues

**Issue**: Terrain data and mesh don't align
**Solution**: Ensure terrain extent covers mesh domain. Use `clip_to_extent()` if needed.

**Issue**: Interpolation produces artifacts
**Solution**: Pre-smooth terrain with Gaussian filter (sigma=1-2).

**Issue**: Nodata values cause problems
**Solution**: Use `fill_nodata()` before processing.

**Issue**: File not loading
**Solution**: Check file format matches extension. Verify ASCII Grid header.

**Issue**: Memory error on large terrains
**Solution**: Clip to simulation extent before loading full terrain.

---

## Dependencies

**Required**:
- numpy >= 1.20
- scipy >= 1.7 (for interpolation)
- logging (standard library)

**Optional**:
- matplotlib (for visualization in examples)
- pytest (for running tests)

---

## License and Citation

Part of the HydroSIS-2D Pre/Post-Processing Toolkit.

If you use this module in research, please cite:
```
HydroSIS-2D Pre/Post-Processing Toolkit
Geometry Processing Module
https://github.com/your-repo/HydroSIS-2D
```

---

## Summary Statistics

- **Total Lines**: ~1,200 lines of production code
- **Test Lines**: 490 lines
- **Example Lines**: 440 lines
- **Test Coverage**: 29 tests, 100% passing
- **Classes**: 5 main classes
- **Functions**: 30+ public methods
- **File Formats**: 1 supported (ASCII Grid)
- **Terrain Types**: 6 synthetic generators
- **Processing Operations**: 10+ terrain analysis tools

---

**Module Status**: ✅ Production Ready
**Last Updated**: 2025-10-29
**Version**: 1.0.0
