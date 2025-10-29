# HydroSIS-2D Terrain Loading Guide

## Overview

HydroSIS-2D supports loading bed elevation data from external terrain files, enabling realistic simulations of floods, dam breaks, and river flows over complex topography. This guide explains how to create and use terrain files in your simulations.

## Supported File Formats

### 1. ASCII Grid Format (.asc)

The ASCII Grid format is a standard GIS raster format supported by ArcGIS, QGIS, and GDAL. It consists of a header section followed by elevation data.

**Header Format:**
```
ncols         <number of columns>
nrows         <number of rows>
xllcorner     <x-coordinate of lower-left corner>
yllcorner     <y-coordinate of lower-left corner>
cellsize      <cell size in map units>
NODATA_value  <value representing missing data, typically -9999>
```

**Data Section:**
- Space-separated elevation values
- Ordered from top-left to bottom-right (row by row)
- Each row contains `ncols` values
- Total of `nrows` lines of data

**Example (5x5 grid):**
```
ncols         5
nrows         5
xllcorner     0.0
yllcorner     0.0
cellsize      10.0
NODATA_value  -9999
10.0 9.5 9.0 8.5 8.0
10.0 9.5 9.0 8.5 8.0
10.0 9.5 9.0 8.5 8.0
10.0 9.5 9.0 8.5 8.0
10.0 9.5 9.0 8.5 8.0
```

### 2. Binary Format (.bin)

Raw binary format containing elevation values as floating-point numbers.

**Structure:**
- Sequential `float` (32-bit) or `double` (64-bit) values
- Ordered row by row (same as ASCII Grid data section)
- No header - grid dimensions must be specified in configuration
- Requires matching simulation grid size (nx × ny values)

**Note:** Binary format is more compact but requires exact grid size matching.

## Creating Terrain Files

### Using GIS Software

#### QGIS (Free & Open Source)

1. **Import DEM data:**
   - Raster → Conversion → Translate (Convert Format)
   - Select input DEM file
   - Choose "AAIGrid" as output format
   - Specify output path with `.asc` extension

2. **Resample if needed:**
   - Raster → Projections → Warp (Reproject)
   - Set target resolution with `-tr` options

3. **Crop to area of interest:**
   - Raster → Extraction → Clip Raster by Extent

#### ArcGIS

1. Use "Raster to ASCII" tool
2. Select input DEM raster
3. Specify output .asc file path

### Using GDAL Command Line

```bash
# Convert any raster format to ASCII Grid
gdal_translate -of AAIGrid input.tif output.asc

# Resample to specific resolution (e.g., 2m)
gdalwarp -tr 2 2 -r bilinear input.tif temp.tif
gdal_translate -of AAIGrid temp.tif output.asc

# Crop to bounding box
gdalwarp -te xmin ymin xmax ymax input.tif cropped.tif
gdal_translate -of AAIGrid cropped.tif output.asc
```

### Using Python (Custom Terrain Generation)

```python
import numpy as np

def create_terrain_asc(filename, nx, ny, cellsize, elevation_func):
    """
    Create ASCII Grid terrain file.

    Parameters:
    - filename: Output .asc file path
    - nx, ny: Grid dimensions (columns, rows)
    - cellsize: Cell size in meters
    - elevation_func: Function(x, y) -> elevation
    """
    with open(filename, 'w') as f:
        # Write header
        f.write(f"ncols         {nx}\n")
        f.write(f"nrows         {ny}\n")
        f.write(f"xllcorner     0.0\n")
        f.write(f"yllcorner     0.0\n")
        f.write(f"cellsize      {cellsize}\n")
        f.write(f"NODATA_value  -9999\n")

        # Write elevation data (top to bottom)
        for j in range(ny-1, -1, -1):  # Top to bottom
            row = []
            for i in range(nx):
                x = i * cellsize
                y = j * cellsize
                z = elevation_func(x, y)
                row.append(f"{z:.2f}")
            f.write(" ".join(row) + "\n")

# Example 1: Linear slope
def linear_slope(x, y):
    return 10.0 - x * 0.05  # 5% slope downward in x direction

create_terrain_asc("slope.asc", nx=100, ny=50, cellsize=2.0,
                   elevation_func=linear_slope)

# Example 2: Parabolic channel
def parabolic_channel(x, y):
    xc = 50.0 * 2.0  # Center x coordinate
    width = 30.0      # Channel width
    dx = (x - xc) / width
    return 0.2 * dx * dx  # Parabolic profile

create_terrain_asc("channel.asc", nx=100, ny=50, cellsize=2.0,
                   elevation_func=parabolic_channel)

# Example 3: Gaussian hill
def gaussian_hill(x, y):
    xc, yc = 100.0, 50.0  # Hill center
    sigma_x, sigma_y = 50.0, 30.0  # Spread
    height = 10.0
    dx = (x - xc) / sigma_x
    dy = (y - yc) / sigma_y
    return height * np.exp(-(dx*dx + dy*dy) / 2.0)

create_terrain_asc("hill.asc", nx=100, ny=50, cellsize=2.0,
                   elevation_func=gaussian_hill)
```

## Using Terrain Files in Simulations

### Command Line Usage

```bash
# Specify terrain file directly
./hydrosis --bed-elevation terrain.asc [other options]

# Use with configuration file
./hydrosis --config my_simulation.ini
```

### Configuration File (.ini)

```ini
[Domain]
nx = 200
ny = 100
xmin = 0.0
ymin = 0.0
dx = 2.0
dy = 2.0

[Terrain]
bed_elevation_file = examples/terrain_parabolic_channel.asc

[Simulation]
t_end = 60.0
dt = 0.01
output_interval = 1.0

[Initial Conditions]
# ... other settings ...
```

### Terrain Interpolation

When terrain file grid doesn't match simulation grid:
- **Bilinear interpolation** is automatically applied
- Terrain is resampled to simulation grid resolution
- Geographic coordinates are used for alignment

**Requirements:**
- Terrain must cover simulation domain
- Points outside terrain use nearest boundary value
- NODATA values are replaced with 0.0

## Example Terrain Files

HydroSIS-2D includes several example terrain files in the `examples/` directory:

### 1. Parabolic Channel (`terrain_parabolic_channel.asc`)

**Description:** U-shaped channel with parabolic cross-section

**Specifications:**
- Grid: 100 × 50 cells
- Cell size: 2.0 m
- Domain: 200m × 100m
- Elevation range: 0.0 - 8.0 m
- Formula: z = 0.2 × [(x - center) / width]²

**Use cases:**
- Channel flow simulations
- Dam break in parabolic valley
- River hydraulics testing

**Example configuration:**
```ini
[Terrain]
bed_elevation_file = examples/terrain_parabolic_channel.asc

[Initial Conditions]
h_left = 10.0    # High water on left
h_right = 0.1    # Low water on right
```

### 2. Gaussian Hill (`terrain_gaussian_hill.asc`)

**Description:** Smooth Gaussian-shaped hill

**Specifications:**
- Grid: 100 × 50 cells
- Cell size: 2.0 m
- Peak height: ~70 m
- Center: (100m, 50m)
- Sigma: σₓ=50m, σᵧ=30m

**Use cases:**
- Flow around obstacles
- Hillslope runoff
- Terrain-following flow testing

**Example configuration:**
```ini
[Terrain]
bed_elevation_file = examples/terrain_gaussian_hill.asc

[Initial Conditions]
initial_water_depth = 2.0  # Uniform initial depth
```

### 3. Linear Slope (`terrain_linear_slope.asc`)

**Description:** Constant downward slope in x-direction

**Specifications:**
- Grid: 100 × 50 cells
- Cell size: 2.0 m
- Elevation range: 0.0 - 10.0 m
- Slope: 5% (10m drop over 200m)
- Formula: z = 10.0 - x × 0.05

**Use cases:**
- Gravity-driven flow
- Flood wave propagation downslope
- Rainfall-runoff modeling

**Example configuration:**
```ini
[Terrain]
bed_elevation_file = examples/terrain_linear_slope.asc

[Boundary Conditions]
bc_left = 2     # Inflow at upslope end
bc_right = 3    # Outflow at downslope end
```

### 4. Dam Break Step (`terrain_dam_break.asc`)

**Description:** Step terrain representing a dam

**Specifications:**
- Grid: 100 × 50 cells
- Cell size: 2.0 m
- Left section: elevation = 5.0 m (dam/reservoir)
- Right section: elevation = 0.0 m (downstream)
- Step location: x = 100 m (middle of domain)

**Use cases:**
- Classic dam break problem
- Hydraulic jump formation
- Shock wave verification

**Example configuration:**
```ini
[Terrain]
bed_elevation_file = examples/terrain_dam_break.asc

[Initial Conditions]
# Water fills reservoir above dam
h_left = 15.0   # 10m water depth above 5m dam
h_right = 5.1   # Shallow water downstream
```

## Terrain Coordinate Systems

### Geographic Coordinates

The terrain file defines its own coordinate system via:
- `xllcorner`, `yllcorner`: Lower-left corner coordinates
- `cellsize`: Cell size in map units

### Simulation Domain Coordinates

HydroSIS-2D simulation domain is defined by:
- `xmin`, `ymin`: Domain origin
- `dx`, `dy`: Cell spacing
- `nx`, `ny`: Grid dimensions

### Coordinate Alignment

The terrain reader:
1. Reads terrain file with its native coordinates
2. Maps simulation grid cells to terrain coordinates
3. Interpolates elevation at each simulation cell center

**Important:** Ensure simulation domain overlaps terrain coverage!

## Advanced Topics

### Handling Large Terrain Files

For very large DEMs:

1. **Crop to area of interest** before simulation:
   ```bash
   gdalwarp -te xmin ymin xmax ymax large_dem.tif cropped.tif
   gdal_translate -of AAIGrid cropped.tif cropped.asc
   ```

2. **Resample to coarser resolution**:
   ```bash
   gdalwarp -tr 10 10 -r average high_res.tif low_res.tif
   ```

3. **Use binary format** for faster loading:
   ```python
   # Convert ASCII to binary
   terrain = np.loadtxt('terrain.asc', skiprows=6)
   terrain.astype(np.float32).tofile('terrain.bin')
   ```

### Synthetic Terrain Types

The `TerrainReader` class also supports generating synthetic terrains programmatically:

```cpp
// In C++ code (advanced users)
TerrainReader::create_synthetic_terrain(
    z_output, nx, ny, xmin, ymin, dx, dy, terrain_type
);

// terrain_type options:
// 0 = Flat (z = 0)
// 1 = Parabolic channel
// 2 = Gaussian hill
// 3 = Linear slope (1% in x-direction)
// 4 = V-shaped valley
// 5 = Step (for dam break)
```

### Combining Terrain with Source Terms

Terrain affects flow via bed slope source terms in shallow water equations:

**Momentum equations include:**
```
∂(hu)/∂t + ... = -g h ∂z/∂x + ...
∂(hv)/∂t + ... = -g h ∂z/∂y + ...
```

where `∂z/∂x` and `∂z/∂y` are bed slopes computed from terrain.

## Troubleshooting

### Common Issues

**1. "Failed to load terrain file"**
- Check file exists at specified path
- Verify file permissions (readable)
- Confirm ASCII Grid header format

**2. Simulation domain outside terrain coverage**
- Terrain's geographic extent must cover simulation domain
- Check `xllcorner`, `yllcorner`, and grid dimensions
- Solution: Expand terrain or reduce simulation domain

**3. Unexpected elevation values**
- Check NODATA_value setting (usually -9999)
- Verify elevation units (meters expected)
- Inspect terrain file for corruption

**4. Terrain interpolation artifacts**
- Occurs when terrain resolution >> simulation resolution
- Solution: Use finer terrain grid
- Or resample terrain to match simulation grid

**5. Performance issues with large files**
- ASCII Grid loading can be slow for large files
- Solution: Convert to binary format
- Or reduce terrain resolution

### Validation

Visualize loaded terrain:

```python
import numpy as np
import matplotlib.pyplot as plt

# Load ASCII Grid
data = np.loadtxt('terrain.asc', skiprows=6)

plt.figure(figsize=(10, 6))
plt.imshow(data, cmap='terrain', origin='lower')
plt.colorbar(label='Elevation (m)')
plt.title('Terrain Elevation')
plt.xlabel('x (cells)')
plt.ylabel('y (cells)')
plt.show()
```

Check for issues:
- Data range (min/max elevations reasonable?)
- Continuity (no sudden jumps or gaps?)
- NODATA values properly handled?

## Real-World Applications

### 1. Urban Flooding

**Data sources:**
- LiDAR-derived DEMs (0.5m - 2m resolution)
- Building outlines integrated into DEM
- Street-level elevation data

**Workflow:**
1. Obtain high-resolution DEM from city or national database
2. Clip to urban area of interest
3. Burn in building footprints (raise elevation)
4. Resample to simulation grid
5. Set rainfall or river inflow boundary conditions

### 2. Dam Break Analysis

**Data sources:**
- Dam and reservoir survey data
- Downstream valley DEM
- River bathymetry

**Workflow:**
1. Combine reservoir DEM with downstream DEM
2. Create step terrain at dam location
3. Initialize high water in reservoir
4. Simulate sudden dam failure

### 3. Coastal Flooding

**Data sources:**
- Coastal DEM with bathymetry
- Tidal elevation data
- Storm surge predictions

**Workflow:**
1. Merge land elevation with nearshore bathymetry
2. Set time-varying sea level at ocean boundary
3. Model storm surge propagation inland

### 4. River Flooding

**Data sources:**
- River channel bathymetry
- Floodplain DEM
- Hydrograph data (upstream discharge)

**Workflow:**
1. Integrate channel cross-sections into DEM
2. Set inflow boundary at upstream end
3. Model flood wave propagation and inundation

## References

### File Format Standards

- **ASCII Grid**: ESRI Arc/Info ASCII Grid format
  - https://en.wikipedia.org/wiki/Esri_grid
- **GeoTIFF**: For raster DEMs
  - https://www.ogc.org/standards/geotiff

### Data Sources

- **USGS National Map**: https://apps.nationalmap.gov/
  - 1m, 3m, 10m, 30m resolution DEMs (USA)
- **EU-DEM**: https://land.copernicus.eu/
  - 25m resolution (Europe)
- **ASTER GDEM**: https://asterweb.jpl.nasa.gov/
  - 30m resolution (global)
- **SRTM**: https://www2.jpl.nasa.gov/srtm/
  - 30m and 90m resolution (global)

### Tools

- **QGIS**: https://qgis.org/ (Free & Open Source GIS)
- **GDAL**: https://gdal.org/ (Geospatial Data Abstraction Library)
- **ArcGIS**: https://www.esri.com/en-us/arcgis/products/arcgis-desktop

## Summary

Key points:
- HydroSIS-2D supports ASCII Grid (.asc) and binary (.bin) formats
- ASCII Grid is recommended for ease of use and interoperability
- Terrain is automatically interpolated to simulation grid
- Example files demonstrate common test cases
- Real-world applications require careful DEM preprocessing

For questions or issues, see the main HydroSIS-2D documentation or contact the development team.
