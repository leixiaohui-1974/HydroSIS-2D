# HydroSIS-2D Real-World Application Cases

## Overview

This document describes real-world application cases for HydroSIS-2D. Each case demonstrates the solver's capabilities for practical flood modeling scenarios using realistic terrain, boundary conditions, and physical parameters.

## Available Application Cases

### 1. Dam Break Valley Flooding
### 2. Urban Flash Flooding
### 3. River Floodplain Inundation

---

## Case 1: Dam Break Valley Flooding

### Description

Simulation of catastrophic dam failure with downstream valley inundation. This case models the sudden release of reservoir water into a V-shaped valley, capturing the rapid flood wave propagation and valley flooding dynamics.

### Application Areas

- **Flood hazard assessment**: Determine inundation extent and arrival times
- **Emergency planning**: Evacuation route planning and timing
- **Dam safety analysis**: Consequence modeling for risk assessment
- **Infrastructure design**: Design of flood protection measures

### Physical Setup

**Domain:**
- Size: 2000m × 1000m
- Resolution: 5m (400 × 200 cells)
- Orientation: Dam at upstream end (x=0), valley extends downstream

**Terrain:**
- V-shaped valley with 15% side slopes
- Valley bottom width: 600m
- Longitudinal slope: 1% downward
- Elevation range: 30-80m
- Generated using `scripts/generate_valley_terrain.py`

**Initial Conditions:**
- Reservoir: 60m water depth behind dam
- Downstream: Nearly dry valley (0.01m depth)
- Dam location: x = 200m
- Dam failure: Instantaneous at t=0

**Boundary Conditions:**
- Upstream (left): Wall (closed reservoir)
- Downstream (right): Open outflow
- Sides (top/bottom): Wall (valley sides)

### Key Parameters

```ini
# Time
t_end = 300s (5 minutes)
dt = 0.01s
output_interval = 5s

# Friction
manning_n = 0.035 (natural valley)

# Numerical
theta = 1.3 (stable for complex terrain)
cfl = 0.4
```

### Expected Results

**Flood Wave Characteristics:**
- Wave front arrival at x=1000m: ~30-40 seconds
- Peak depth in valley: 30-40m near dam, 5-10m far downstream
- Flow velocity: 10-15 m/s initially, decreasing downstream
- Froude number: Supercritical (Fr > 1) near dam breach

**Physical Phenomena:**
- Rapid initial wave propagation (bore formation)
- Wave spreading and attenuation
- Reflection from valley sides
- Gradual deceleration and depth increase

**Output Files:**
- VTK files every 5 seconds
- Variables: h, u, v, z, eta, Fr
- Time series at 3 monitoring points along valley

### Running the Case

```bash
# Generate terrain (if not already done)
python3 scripts/generate_valley_terrain.py

# Run simulation
./hydrosis --config examples/applications/dam_break_valley.ini

# Results in: output/dam_break_valley/
```

### Visualization Tips

**ParaView visualization:**
1. Load VTK sequence
2. Create "Surface" plot colored by water depth (h)
3. Add "Contour" filter for water surface elevation (eta)
4. Use "Warp by Scalar" on bed elevation (z) for 3D view
5. Animate time sequence to see flood wave propagation

**Key visualizations:**
- Water depth evolution (shows inundation extent)
- Velocity magnitude (identifies high-hazard zones)
- Froude number (locates supercritical flow regions)
- Water surface elevation (eta) for free surface tracking

---

## Case 2: Urban Flash Flooding

### Description

Extreme rainfall event causing street flooding in an urban district. This case simulates surface runoff accumulation in a street grid network with buildings represented as elevated obstacles that block flow.

### Application Areas

- **Urban flood risk assessment**: Identify flood-prone streets
- **Drainage system design**: Evaluate capacity requirements
- **Emergency response**: Predict flooded areas and access routes
- **Climate adaptation**: Assess impacts of extreme rainfall events

### Physical Setup

**Domain:**
- Size: 1000m × 1000m urban district
- Resolution: 5m (200 × 200 cells)
- Street grid: 200m × 200m blocks

**Terrain:**
- Base elevation: 100m
- Building height: 10m above street level
- Street width: 100m
- Drainage slope: 0.1% (very gentle)
- Street area: ~72% of domain
- Generated using `scripts/generate_urban_terrain.py`

**Initial Conditions:**
- Initially dry streets (1mm water depth)
- No initial flow

**Rainfall:**
- Intensity: 100 mm/hr (extreme cloudbursts event)
- Duration: 30 minutes
- Total rainfall: 50mm
- Start: t = 0s

**Boundary Conditions:**
- All sides: Open boundaries (water can drain out)

### Key Parameters

```ini
# Time
t_end = 1800s (30 minutes)
output_interval = 60s (1 minute)

# Rainfall
rainfall_rate = 100 mm/hr = 0.0000278 m/s
rainfall_duration = 1800s

# Friction
manning_n = 0.015 (paved streets)

# Numerical
theta = 1.4
cfl = 0.45
epsilon_h = 0.005m (dry threshold)
```

### Expected Results

**Flooding Progression:**
- Initial ponding: 5-10 minutes (streets begin to fill)
- Maximum depth: 0.5-1.5m at street intersections (low points)
- Flow patterns: Water flows along streets toward drainage points
- Critical areas: Street intersections, low-lying zones

**Hydraulic Behavior:**
- Subcritical flow in streets (Fr < 1)
- Ponding at intersections and dead ends
- Flow acceleration along sloped streets
- Building blockage effects on flow routing

**Output Files:**
- VTK files every 1 minute
- Time series at 4 street intersections
- Variables: h, u, v, z, eta, Fr

### Running the Case

```bash
# Generate terrain
python3 scripts/generate_urban_terrain.py

# Run simulation
./hydrosis --config examples/applications/urban_flooding.ini

# Results in: output/urban_flooding/
```

### Visualization Tips

**Flood depth mapping:**
1. Color streets by water depth
2. Buildings appear as elevated areas (white/gray)
3. Use "Threshold" filter to show only flooded areas (h > 0.05m)

**Hazard assessment:**
- Depth < 0.3m: Low hazard (passable)
- Depth 0.3-0.6m: Moderate hazard (difficult)
- Depth > 0.6m: High hazard (dangerous)

**Velocity-depth product (flood intensity):**
- Create calculator: `h * sqrt(u*u + v*v)`
- Values > 0.5 m²/s indicate high hazard

---

## Case 3: River Floodplain Inundation

### Description

River flood event with overbank flow and floodplain inundation. This case simulates a flood wave propagating down a meandering river channel, overtopping the banks, and spreading across the floodplain.

### Application Areas

- **Flood forecasting**: Predict inundation extent for given upstream flows
- **Floodplain mapping**: Delineate flood zones for land-use planning
- **Levee design**: Determine required levee heights
- **Ecological modeling**: Assess floodplain connectivity
- **Bridge hydraulics**: Evaluate bridge performance during floods

### Physical Setup

**Domain:**
- Size: 3000m × 1000m river reach
- Resolution: 5m (600 × 200 cells)
- River orientation: Flows in +x direction

**Terrain:**
- Meandering channel centerline
- Channel width: 100m at bankfull
- Channel depth: 5m below floodplain
- Parabolic cross-section in channel
- Floodplain slope: 1% transverse
- Streamwise slope: 0.05%
- Meander wavelength: 1000m, amplitude: 50m
- Generated using `scripts/generate_river_terrain.py`

**Initial Conditions:**
- Uniform flow at base level
- Depth: 2m (within channel)
- Velocity: 1 m/s downstream
- Discharge: ~200 m³/s

**Boundary Conditions:**
- Upstream (left): Time-varying inflow (flood hydrograph)
- Downstream (right): Open outflow
- Sides (top/bottom): Wall (floodplain boundaries)

**Flood Hydrograph:**
- Base flow: 2m depth, 1 m/s (Q ≈ 200 m³/s)
- Peak flow: 8m depth, 3 m/s (Q ≈ 2400 m³/s)
- Time to peak: 30 minutes
- Peak duration: 30 minutes
- Recession: 60 minutes
- Total duration: 2 hours

### Key Parameters

```ini
# Time
t_end = 7200s (2 hours)
output_interval = 300s (5 minutes)

# Friction
manning_n = 0.030 (natural channel)

# Numerical
theta = 1.4
cfl = 0.4

# Inflow hydrograph
base_depth = 2m
peak_depth = 8m
time_to_peak = 1800s
```

### Expected Results

**Flood Progression:**
- t = 0-30 min: Flood wave arrival, water level rises
- t = 30-60 min: Peak flow, extensive overbank flooding
- t = 60-120 min: Recession, floodplain drainage

**Hydraulic Characteristics:**
- In-channel flow: Subcritical (Fr ≈ 0.3-0.5)
- Overbank flow: Very slow (0.1-0.5 m/s)
- Floodplain depth: 0.5-3m depending on location
- Inundation extent: 200-400m from channel centerline

**Physical Phenomena:**
- Channel-floodplain interaction
- Backwater effects near meander bends
- Floodplain storage and attenuation
- Flow momentum exchange at bank edges
- Asymmetric flooding due to meanders

**Output Files:**
- VTK files every 5 minutes (25 time steps)
- Time series at 5 locations (channel and floodplain)
- Variables: h, u, v, z, eta, Fr, bed_shear

### Running the Case

```bash
# Generate terrain
python3 scripts/generate_river_terrain.py

# Run simulation
./hydrosis --config examples/applications/river_flooding.ini

# Results in: output/river_flooding/
```

### Visualization Tips

**Flood extent mapping:**
1. Color by water depth (h)
2. Use "Contour" filter on eta for flood boundary
3. Extract floodplain areas with threshold h > 0.1m

**Channel vs. floodplain:**
- In-channel: High velocity, moderate depth
- Floodplain: Low velocity, shallow depth
- Transition: Identified by velocity gradients

**Hydrograph analysis:**
- Extract time series at monitoring points
- Plot depth vs. time to see flood wave passage
- Calculate peak attenuation between upstream/downstream

**3D visualization:**
- Warp terrain by z (bed elevation)
- Overlay water surface (eta)
- Shows channel meandering and floodplain topography

---

## Comparison of Application Cases

| Case | Domain Size | Resolution | Duration | Key Phenomenon | Primary Application |
|------|-------------|------------|----------|----------------|---------------------|
| Dam Break Valley | 2×1 km | 5m | 5 min | Rapid flood wave | Dam safety |
| Urban Flooding | 1×1 km | 5m | 30 min | Rainfall runoff | Urban planning |
| River Flooding | 3×1 km | 5m | 2 hours | Overbank flow | Flood forecasting |

## Common Analysis Tasks

### 1. Maximum Inundation Extent

Extract maximum water depth over entire simulation:

**ParaView:**
- Filters → Temporal → Temporal Statistics
- Select "Maximum" for water depth (h)
- Color by h_maximum

**Purpose:** Flood hazard mapping, insurance, land-use planning

### 2. Flood Arrival Time

Time when water depth exceeds threshold (e.g., 0.1m):

**Method:**
- Export time series at multiple points
- Find first time when h > threshold
- Create contour map of arrival times

**Purpose:** Evacuation planning, warning system design

### 3. Flow Velocity Hazard

Identify dangerous flow areas using velocity-depth product:

**Criterion:**
- Low hazard: h×V < 0.3 m²/s
- Moderate: 0.3 < h×V < 0.6 m²/s
- High: h×V > 0.6 m²/s

**Purpose:** Pedestrian and vehicle safety assessment

### 4. Peak Discharge

Calculate discharge through cross-section:

**Formula:** Q = ∫ h·u·dy

**Method:**
- Extract slice across flow direction
- Integrate h×u over width
- Track over time for hydrograph

**Purpose:** Hydraulic structure design, channel capacity

### 5. Volume Balance

Check mass conservation:

**Components:**
- Initial volume
- Inflow volume (rainfall or boundary)
- Outflow volume
- Final volume
- Storage change

**Purpose:** Model validation, understanding flood storage

## Advanced Customization

### Modifying Terrain

Edit terrain generation scripts:
- Adjust channel geometry (width, depth, slope)
- Add features (levees, bridges, buildings)
- Incorporate real DEM data

### Custom Hydrographs

Create time-varying boundary conditions:
- Measured flood hydrographs
- Design storm events
- Dam release scenarios

### Parameter Sensitivity

Test sensitivity to:
- Manning's roughness (friction effects)
- Grid resolution (numerical accuracy)
- Boundary conditions (model extent)
- Initial conditions (spin-up effects)

### Multi-GPU Scaling

For large domains, enable multi-GPU:

```ini
[Multi-GPU]
use_multi_gpu = true
num_gpus = 2  # or 4, 8, etc.
domain_decomposition = x  # streamwise for river, x or y for others
```

**Recommended for:**
- Domain > 1000 × 1000 cells
- High-resolution urban areas
- Long river reaches
- Real-time forecasting applications

## Validation and Verification

### Analytical Solutions

Compare with analytical solutions where available:
- Dam break: Ritter solution (flat bed)
- Channel flow: Manning's equation (uniform flow)
- Wave propagation: Shallow water wave theory

### Benchmark Cases

Standard test problems:
- Thacker's planar beach (analytical wetting/drying)
- Malpasset dam break (field data)
- Toce River experiment (physical model data)

### Sensitivity Analysis

Verify numerical robustness:
- Grid convergence (refine dx, dy)
- Time step independence (reduce dt)
- Limiter effects (vary theta)

## Troubleshooting

### Common Issues

**1. Simulation crashes or becomes unstable**

Causes:
- Time step too large
- Steep terrain gradients
- Dry bed handling issues

Solutions:
- Reduce CFL number (try 0.3-0.4)
- Increase epsilon_h (dry threshold)
- Use more conservative limiter (theta = 1.0-1.3)
- Check terrain for unrealistic features

**2. Unrealistic flooding patterns**

Causes:
- Incorrect terrain
- Wrong boundary conditions
- Inappropriate initial conditions

Solutions:
- Visualize terrain independently (check elevation range)
- Verify boundary condition types match physics
- Ensure initial conditions are in equilibrium

**3. Slow performance**

Causes:
- Large domain with fine resolution
- Excessive output frequency
- Small time step

Solutions:
- Enable multi-GPU if available
- Reduce output frequency
- Optimize grid resolution (balance accuracy vs. speed)
- Use adaptive time stepping

**4. Mass conservation errors**

Causes:
- Boundary condition implementation
- Wetting/drying issues
- Numerical scheme accuracy

Solutions:
- Check boundary fluxes
- Adjust dry threshold (epsilon_h)
- Use finer grid near critical areas

### Getting Help

- Check documentation: docs/README.md
- Review test cases: tests/validation/
- Contact development team
- Report issues on GitHub

## References

### Dam Break Modeling

- **Ritter, A. (1892)**: Analytical dam break solution
- **Fraccarollo & Toro (1995)**: Experimental data for validation
- **IMPACT Project**: European dam failure research

### Urban Flooding

- **Galloway et al. (2018)**: Urban flood modeling best practices
- **HR Wallingford**: Urban drainage and flooding guidelines
- **CRED EM-DAT**: Disaster database (flood statistics)

### River Hydraulics

- **Chow (1959)**: Open Channel Hydraulics
- **USGS**: Stream flow measurements and data
- **HEC-RAS**: USACE river modeling standards

### Shallow Water Modeling

- **Toro (2001)**: Shock-Capturing Methods for Free-Surface Shallow Flows
- **LeVeque (2002)**: Finite Volume Methods for Hyperbolic Problems
- **Begnudelli & Sanders (2006)**: Conservative wetting/drying

## Summary

These application cases demonstrate HydroSIS-2D's capability to simulate realistic flood scenarios with:

- **Complex terrain**: Valleys, urban areas, river channels
- **Realistic physics**: Friction, wetting/drying, supercritical flow
- **Practical applications**: Hazard assessment, infrastructure design, forecasting
- **Scalability**: From local (1 km²) to regional (10+ km²) domains

Users can adapt these cases as templates for their own applications by:
1. Generating custom terrain (real DEM or synthetic)
2. Adjusting domain size and resolution
3. Setting appropriate initial and boundary conditions
4. Configuring physical parameters (roughness, time scales)
5. Defining output requirements (frequency, variables, locations)

The cases serve as starting points for flood risk assessment, engineering design, emergency planning, and scientific research using HydroSIS-2D.
