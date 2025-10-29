# Configuration File Usage Guide

## Overview

HydroSIS-2D now supports configuration files for easy parameter management. You can specify all simulation parameters in a single `.ini` file and override them with command line arguments if needed.

## Usage

### Basic Usage

```bash
# Run with configuration file
./hydrosis --config examples/config_dam_break.ini

# Run with config and enable VTK output
./hydrosis --config examples/config_lake_at_rest.ini --vtk

# Run with config and override grid size
./hydrosis --config examples/config_dam_break.ini --nx 1024 --ny 512
```

### Command Line Arguments

All command line arguments override config file values:

```bash
./hydrosis --config my_case.ini \
           --nx 2048 \           # Override grid size
           --cfl 0.4 \           # Override CFL number
           --tend 50.0 \         # Override end time
           --vtk \               # Enable VTK output
           --validate            # Enable validation
```

## Configuration File Format

Configuration files use simple INI format with `key = value` pairs.

### Example Configuration

```ini
# Grid Parameters
nx = 512
ny = 256
dx = 1.0
dy = 1.0
xmin = 0.0
ymin = 0.0

# Physical Parameters
gravity = 9.81
cfl = 0.5
h_dry = 1e-4
friction_type = 0

# Time Parameters
t_start = 0.0
t_end = 10.0
dt_max = 0.1
output_interval = 1.0

# Boundary Conditions
bc_left = 0
bc_right = 0
bc_bottom = 0
bc_top = 0

# Solver Options
use_lts = false
riemann_solver = 1
slope_limiter = 0
order = 2
```

## Parameter Reference

### Grid Parameters

- **nx**: Number of cells in x-direction
- **ny**: Number of cells in y-direction
- **dx**: Cell size in x-direction (meters)
- **dy**: Cell size in y-direction (meters)
- **xmin**: Minimum x-coordinate of domain
- **ymin**: Minimum y-coordinate of domain

### Physical Parameters

- **gravity**: Gravitational acceleration (m/s²), default: 9.81
- **cfl**: CFL number for time step, default: 0.5
- **h_dry**: Dry tolerance (meters), default: 1e-4
- **friction_type**: Friction model (0: Manning, 1: Chezy)

### Time Parameters

- **t_start**: Start time (seconds), default: 0.0
- **t_end**: End time (seconds)
- **dt_max**: Maximum time step (seconds), default: 0.1
- **output_interval**: Output interval (seconds), default: 1.0

### Boundary Conditions

- **bc_left**: Left boundary type
- **bc_right**: Right boundary type
- **bc_bottom**: Bottom boundary type
- **bc_top**: Top boundary type

Boundary types:
- 0: Wall (reflective)
- 1: Open (zero gradient) - *Coming in Task 1.1*
- 2: Inflow (fixed state) - *Coming in Task 1.1*
- 3: Outflow (radiation) - *Coming in Task 1.1*

### Solver Options

- **use_lts**: Use local time stepping (true/false)
- **riemann_solver**: Riemann solver type
  - 0: HLL
  - 1: HLLC (recommended)
  - 2: Roe
- **slope_limiter**: Slope limiter for MUSCL reconstruction
  - 0: minmod
  - 1: superbee
  - 2: MC (Monotonized Central)
- **order**: Spatial order (1 or 2)

## Example Cases

### 1. Dam Break

```bash
./hydrosis --config examples/config_dam_break.ini --test 0 --vtk
```

### 2. Lake at Rest

```bash
./hydrosis --config examples/config_lake_at_rest.ini --test 5 --validate
```

### 3. High-Resolution Simulation

```bash
./hydrosis --config examples/config_dam_break.ini \
           --nx 2048 --ny 1024 \
           --tend 20.0 \
           --vtk
```

### 4. Multi-GPU

```bash
mpirun -np 4 ./hydrosis --config examples/config_dam_break.ini \
                         --multi-gpu \
                         --nx 4096 --ny 2048
```

## Tips

1. **Start with examples**: Modify existing config files in `examples/`
2. **Test small first**: Use small grid sizes for testing, then scale up
3. **Comments**: Use `#` for comments in config files
4. **Override sparingly**: Use command line overrides for quick parameter sweeps
5. **Check output**: Config values are printed at simulation start

## Validation

To verify your configuration is loaded correctly:

```bash
./hydrosis --config your_config.ini --help
# Then run normally
./hydrosis --config your_config.ini
```

The configuration summary is printed before simulation starts.

## Troubleshooting

**Config file not found:**
```
Error: Cannot open configuration file: myconfig.ini
```
→ Check file path and permissions

**Invalid parameter:**
```
Warning: Invalid integer for key 'nx'
```
→ Check parameter format in config file

**Missing required parameter:**
→ All parameters have defaults; simulation will use default values

## Next Steps

After Task 1.1 completion, you'll be able to use:
- Open boundaries for river flows
- Inflow boundaries for upstream conditions
- Outflow boundaries for downstream conditions

Stay tuned for updates!
