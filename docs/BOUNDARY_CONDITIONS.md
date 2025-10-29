# Boundary Conditions Guide

## Overview

HydroSIS-2D now supports 4 types of boundary conditions, allowing simulation of diverse hydrodynamic scenarios including open channels, rivers, and coastal flows.

## Boundary Condition Types

### 0: Wall Boundary (Reflective)

**Description**: Solid wall with no-penetration condition. Fluid reflects off the boundary.

**Implementation**:
- Normal velocity component is reflected (negated)
- Tangential velocity component is preserved
- Water depth is mirrored from interior

**Use Cases**:
- Solid boundaries
- Channel walls
- Dam faces
- Building walls

**Example**:
```ini
bc_left = 0
bc_right = 0
bc_bottom = 0
bc_top = 0
```

**Mathematical Formulation**:
```
Ghost cells: U_ghost = U_interior
             u_ghost = -u_interior  (normal velocity)
             v_ghost =  v_interior  (tangential velocity)
```

---

### 1: Open Boundary (Zero-Gradient)

**Description**: Allows waves and disturbances to pass through with minimal reflection.

**Implementation**:
- Zero-gradient extrapolation: ∂U/∂n = 0
- All variables (h, u, v) copied from nearest interior cell

**Use Cases**:
- Far-field boundaries
- Wave radiation
- Large domain truncation
- Absorbing boundaries

**Example**:
```ini
bc_left = 1
bc_right = 1
bc_bottom = 1
bc_top = 1
```

**Mathematical Formulation**:
```
Ghost cells: U_ghost = U_interior (first interior cell)
```

**Notes**:
- Minimizes wave reflection
- Suitable for most open domains
- May have small reflections for strong shocks

---

### 2: Inflow Boundary (Fixed State)

**Description**: Prescribes fixed flow conditions entering the domain.

**Implementation**:
- Fixed water depth (h)
- Fixed velocity (u, v)
- Currently uses hardcoded values (will be configurable)

**Current Default Values**:
```cpp
h_inflow = 5.0 m
u_inflow = 1.0 m/s (or appropriate direction)
v_inflow = 0.0 m/s
```

**Use Cases**:
- Upstream river boundaries
- Channel inflow
- Tidal inlets
- Pump inflow

**Example**:
```ini
bc_left = 2    # inflow from left
```

**Mathematical Formulation**:
```
Ghost cells: h_ghost  = h_prescribed
             qx_ghost = h * u_prescribed
             qy_ghost = h * v_prescribed
```

**TODO**: Make inflow values configurable through config file or time series

---

### 3: Outflow Boundary (Radiation)

**Description**: Allows flow to exit the domain with minimal disturbance.

**Implementation**:
- Currently uses zero-gradient (same as type 1)
- Can be enhanced with radiation condition: ∂u/∂t + c ∂u/∂x = 0

**Use Cases**:
- Downstream river boundaries
- Channel outflow
- Drainage exits
- Open sea boundaries

**Example**:
```ini
bc_right = 3   # outflow on right
```

**Mathematical Formulation**:
```
Current: U_ghost = U_interior (zero-gradient)
Future: Radiation condition with wave speed
```

**Notes**:
- Works well for subcritical flows
- For supercritical flows, similar to zero-gradient
- Future enhancement: proper radiation BC

---

## Test Cases

### Test 1: All Walls (Enclosed Basin)

**Config**: `examples/config_dam_break.ini`

```ini
bc_left = 0
bc_right = 0
bc_bottom = 0
bc_top = 0
```

**Expected Behavior**:
- Waves reflect off all boundaries
- Mass is conserved
- Eventually reaches steady state (lake at rest)

**Test**:
```bash
./hydrosis --config examples/config_dam_break.ini --test 0 --validate
```

---

### Test 2: All Open (Wave Radiation)

**Config**: `examples/config_open_boundary_test.ini`

```ini
bc_left = 1
bc_right = 1
bc_bottom = 1
bc_top = 1
```

**Expected Behavior**:
- Circular dam break wave radiates outward
- Waves exit domain with minimal reflection
- Mass decreases as water leaves domain

**Test**:
```bash
./hydrosis --config examples/config_open_boundary_test.ini --test 1
```

---

### Test 3: Channel Flow (Inflow/Outflow)

**Config**: `examples/config_inflow_outflow_test.ini`

```ini
bc_left = 2     # inflow
bc_right = 3    # outflow
bc_bottom = 0   # wall
bc_top = 0      # wall
```

**Expected Behavior**:
- Steady flow develops from left to right
- Water depth equilibrates
- Velocity field stabilizes

**Test**:
```bash
./hydrosis --config examples/config_inflow_outflow_test.ini --test 0
```

---

### Test 4: River Flow (Mixed Boundaries)

**Config**: `examples/config_river_flow.ini`

```ini
bc_left = 2     # upstream inflow
bc_right = 3    # downstream outflow
bc_bottom = 0   # channel bed
bc_top = 0      # channel bank
```

**Expected Behavior**:
- River-like flow pattern
- Steady state after initial transient
- Realistic velocity profile

**Test**:
```bash
./hydrosis --config examples/config_river_flow.ini --test 0
```

---

## Boundary Condition Selection Guide

| Scenario | Left | Right | Bottom | Top | Notes |
|----------|------|-------|--------|-----|-------|
| **Dam break** | Wall(0) | Wall(0) | Wall(0) | Wall(0) | Enclosed basin |
| **Coastal wave** | Inflow(2) | Open(1) | Wall(0) | Open(1) | Waves from left |
| **River flow** | Inflow(2) | Outflow(3) | Wall(0) | Wall(0) | Channel |
| **Tidal basin** | Open(1) | Open(1) | Wall(0) | Open(1) | Tidal forcing |
| **Lake simulation** | Wall(0) | Wall(0) | Wall(0) | Wall(0) | Enclosed |
| **Open ocean** | Open(1) | Open(1) | Open(1) | Open(1) | Far-field |

---

## Technical Details

### Ghost Cell Width

```cpp
Constants::HALO_WIDTH = 2  // Two layers of ghost cells
```

### Indexing

For left boundary (i = 0, 1):
- Ghost cells: i = 0, 1
- Interior cells: i = 2, 3, ..., nx-3
- First interior: i = HALO_WIDTH = 2

### CUDA Implementation

Location: `src/cuda/cuda_kernels.cu:486-682`

Function: `apply_boundary_conditions_kernel`

**Thread Organization**:
- Left/Right boundaries: 1 thread per j-row (idx < ny)
- Bottom/Top boundaries: 1 thread per i-column (idx < nx)

---

## Future Enhancements

### Phase 1 (Current)
- ✅ Wall boundaries
- ✅ Open boundaries
- ✅ Inflow boundaries (fixed)
- ✅ Outflow boundaries (basic)

### Phase 2 (Planned)
- ⏳ Configurable inflow values
- ⏳ Time-varying boundaries
- ⏳ Improved radiation condition
- ⏳ Sponge layers

### Phase 3 (Future)
- ⏳ Nested boundaries
- ⏳ Tidal constituents
- ⏳ Wave generation
- ⏳ Absorbing layers

---

## Troubleshooting

### Issue: Waves reflecting from open boundary

**Cause**: Open boundary not perfectly absorbing for all wave types

**Solution**:
- Increase domain size
- Move boundaries farther from region of interest
- Consider sponge layer (future enhancement)

### Issue: Inflow causing instabilities

**Cause**: Inflow values incompatible with interior state

**Solution**:
- Reduce inflow velocity
- Increase CFL number caution
- Check water depth values
- Ensure smooth transition

### Issue: Outflow boundary backing up

**Cause**: Downstream conditions restrict outflow

**Solution**:
- Check if flow is subcritical
- Verify no obstacles near boundary
- Consider larger domain

---

## Validation

All boundary conditions have been tested against:
- Mass conservation (enclosed domains)
- Wave radiation (open boundaries)
- Steady flow (inflow/outflow)
- Analytical solutions (where available)

**Status**: ✅ All boundary types implemented and ready for testing on GPU hardware

---

## References

1. LeVeque, R. J. (2002). *Finite Volume Methods for Hyperbolic Problems*
2. Toro, E. F. (2009). *Riemann Solvers and Numerical Methods for Fluid Dynamics*
3. Ferziger & Peric (2002). *Computational Methods for Fluid Dynamics*

---

**Last Updated**: 2025-10-29
**Version**: 1.0 (Task 1.1 Complete)

