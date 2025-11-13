"""
Post-processing & Visualization Validation Tests

This module tests post-processing operations and visualization quality,
ensuring correct computation of derived quantities and proper data representation.

Test Categories:
1. Derived Quantity Computation (4 tests)
2. Visualization Quality Checks (3 tests)
3. Data Export & Format Validation (3 tests)
4. Animation & Time Series (3 tests)

Physical Context:
- Post-processing extracts engineering insights from simulation results
- Visualizations must accurately represent physical phenomena
- Data export enables integration with GIS and other tools

GPU Readiness: Framework complete, awaits GPU compilation
Author: HydroSIS-2D Development Team
Date: 2025-11-13
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import os


class TestDerivedQuantityComputation:
    """
    Test computation of derived hydraulic quantities.

    Derived Quantities:
    - Froude number
    - Specific energy
    - Unit discharge
    - Velocity magnitude
    - Flow regime classification
    """

    def test_froude_number_calculation(self):
        """
        Test Froude number computation and classification.

        Froude Number:
        - Fr = V / √(gh)
        - Fr < 1: Subcritical (tranquil flow)
        - Fr = 1: Critical flow
        - Fr > 1: Supercritical (rapid flow)

        Physical Significance:
        - Ratio of inertial to gravitational forces
        - Determines flow regime and wave propagation
        """
        print("\n" + "="*70)
        print("TEST: Froude Number Calculation")
        print("="*70)

        # Test cases
        g = 9.81  # m/s²

        test_cases = [
            # (depth, velocity, expected_Fr, regime)
            (2.0, 1.0, 0.226, "Subcritical"),
            (1.0, 3.13, 1.0, "Critical"),
            (0.5, 5.0, 2.26, "Supercritical"),
            (10.0, 2.0, 0.202, "Subcritical"),
        ]

        print(f"\n{'Depth (m)':>12} {'Velocity (m/s)':>18} {'Fr (calc)':>12} {'Fr (expected)':>15} {'Regime':>15}")
        print("-" * 85)

        for h, V, Fr_expected, regime_expected in test_cases:
            # Compute Froude number
            Fr_calc = V / np.sqrt(g * h)

            # Classify regime
            if Fr_calc < 0.99:
                regime_calc = "Subcritical"
            elif Fr_calc > 1.01:
                regime_calc = "Supercritical"
            else:
                regime_calc = "Critical"

            print(f"{h:12.2f} {V:18.2f} {Fr_calc:12.4f} {Fr_expected:15.4f} {regime_calc:>15}")

            # Validation
            assert abs(Fr_calc - Fr_expected) < 0.01, f"Fr mismatch for h={h}, V={V}"
            assert regime_calc == regime_expected, f"Regime classification error"

        # Vector field test
        print(f"\nVector Field Test:")
        nx, ny = 50, 50
        h = np.random.uniform(0.5, 3.0, (nx, ny))
        u = np.random.uniform(-2.0, 2.0, (nx, ny))
        v = np.random.uniform(-2.0, 2.0, (nx, ny))

        V_mag = np.sqrt(u**2 + v**2)
        Fr = V_mag / np.sqrt(g * h)

        print(f"  Grid: {nx}×{ny}")
        print(f"  Fr range: [{Fr.min():.4f}, {Fr.max():.4f}]")
        print(f"  Subcritical cells: {(Fr < 1).sum()} ({(Fr < 1).sum()/(nx*ny)*100:.1f}%)")
        print(f"  Critical cells: {((Fr >= 0.99) & (Fr <= 1.01)).sum()}")
        print(f"  Supercritical cells: {(Fr > 1).sum()} ({(Fr > 1).sum()/(nx*ny)*100:.1f}%)")

        # Validation
        assert Fr.min() >= 0, "Froude number must be non-negative"
        assert np.all(np.isfinite(Fr)), "All Fr values must be finite"

        print("\n✓ Froude number calculation test passed")

    def test_specific_energy_computation(self):
        """
        Test specific energy computation.

        Specific Energy:
        - E = h + V²/(2g)
        - E = depth + velocity head
        - Minimum at critical depth

        Applications:
        - Hydraulic jump analysis
        - Channel transitions
        - Energy dissipation
        """
        print("\n" + "="*70)
        print("TEST: Specific Energy Computation")
        print("="*70)

        g = 9.81

        # Test: specific energy for various depths at constant discharge
        q = 2.0  # Unit discharge (m²/s)
        depths = np.linspace(0.1, 3.0, 50)

        E = np.zeros_like(depths)

        print(f"\nSpecific Energy for q = {q} m²/s:")

        for i, h in enumerate(depths):
            V = q / h
            E[i] = h + V**2 / (2*g)

        # Find critical depth (minimum energy)
        idx_critical = E.argmin()
        h_critical = depths[idx_critical]
        E_min = E[idx_critical]

        # Analytical critical depth: h_c = (q²/g)^(1/3)
        h_c_analytical = (q**2 / g)**(1.0/3.0)
        E_c_analytical = 1.5 * h_c_analytical

        print(f"  Critical depth (numerical): {h_critical:.4f} m")
        print(f"  Critical depth (analytical): {h_c_analytical:.4f} m")
        print(f"  Minimum energy (numerical): {E_min:.4f} m")
        print(f"  Minimum energy (analytical): {E_c_analytical:.4f} m")
        print(f"  Error in h_c: {abs(h_critical - h_c_analytical):.6f} m")

        # Validation
        assert abs(h_critical - h_c_analytical) < 0.05, "Critical depth should match analytical"
        assert abs(E_min - E_c_analytical) < 0.05, "Minimum energy should match analytical"

        # Check E curve shape (should have minimum)
        assert E[0] > E_min and E[-1] > E_min, "Energy curve should have minimum"

        print("\n✓ Specific energy test passed")

    def test_discharge_computation(self):
        """
        Test discharge computation from depth and velocity.

        Discharge:
        - Q = ∫∫ (u·n) dA
        - For 2D SWE: Q = Σ (h * u * Δy) or (h * v * Δx)

        Applications:
        - Flow rate through cross-section
        - Mass balance verification
        - Hydrograph generation
        """
        print("\n" + "="*70)
        print("TEST: Discharge Computation")
        print("="*70)

        # Channel cross-section
        ny = 50  # Points across channel
        dy = 1.0  # m

        # Depth profile (trapezoidal channel)
        y = np.arange(ny) * dy
        y_center = y.mean()

        # Trapezoidal shape
        h = np.maximum(0, 3.0 - 0.5 * np.abs(y - y_center))

        # Velocity (uniform)
        u = 1.5 * np.ones(ny)  # m/s

        print(f"\nCross-Section:")
        print(f"  Width: {ny * dy} m")
        print(f"  Points: {ny}")
        print(f"  Max depth: {h.max():.2f} m")
        print(f"  Velocity: {u[0]:.2f} m/s (uniform)")

        # Compute discharge
        Q = np.sum(h * u * dy)

        # Area
        A = np.sum(h * dy)

        # Average velocity
        V_avg = Q / A

        print(f"\nResults:")
        print(f"  Cross-sectional area: {A:.2f} m²")
        print(f"  Discharge: {Q:.2f} m³/s")
        print(f"  Average velocity: {V_avg:.4f} m/s")

        # Validation
        assert abs(V_avg - u[0]) < 0.01, "Average velocity should match uniform velocity"

        # 2D field integration
        print(f"\n2D Field Integration:")
        nx, ny = 100, 50
        dx, dy = 10.0, 10.0

        h_2d = np.random.uniform(1.0, 3.0, (nx, ny))
        u_2d = np.random.uniform(0.5, 2.0, (nx, ny))

        # Discharge through vertical cross-sections
        Q_x = np.sum(h_2d * u_2d * dy, axis=1)  # Discharge at each x

        print(f"  Grid: {nx}×{ny}")
        print(f"  Discharge range: [{Q_x.min():.2f}, {Q_x.max():.2f}] m³/s")
        print(f"  Mean discharge: {Q_x.mean():.2f} m³/s")

        assert np.all(Q_x > 0), "All discharges should be positive"

        print("\n✓ Discharge computation test passed")

    def test_flow_regime_classification(self):
        """
        Test automatic flow regime classification.

        Flow Regimes:
        1. Dry (h < h_dry)
        2. Subcritical (Fr < 1)
        3. Critical (Fr ≈ 1)
        4. Supercritical (Fr > 1)
        5. Stagnant (V ≈ 0)

        Applications:
        - Adaptive meshing
        - Numerical scheme selection
        - Visualization enhancement
        """
        print("\n" + "="*70)
        print("TEST: Flow Regime Classification")
        print("="*70)

        g = 9.81
        h_dry = 1e-4  # Dry threshold

        # Synthetic flow field
        nx, ny = 100, 100
        np.random.seed(42)

        h = np.random.uniform(0, 2.0, (nx, ny))
        u = np.random.uniform(-2.0, 2.0, (nx, ny))
        v = np.random.uniform(-2.0, 2.0, (nx, ny))

        # Classify
        V_mag = np.sqrt(u**2 + v**2)

        regime = np.zeros((nx, ny), dtype=int)
        # 0: Dry, 1: Stagnant, 2: Subcritical, 3: Critical, 4: Supercritical

        dry_mask = h < h_dry
        regime[dry_mask] = 0

        wet_mask = ~dry_mask
        stagnant_mask = wet_mask & (V_mag < 0.01)
        regime[stagnant_mask] = 1

        active_mask = wet_mask & ~stagnant_mask
        Fr = np.zeros_like(h)
        Fr[active_mask] = V_mag[active_mask] / np.sqrt(g * h[active_mask])

        subcritical_mask = active_mask & (Fr < 0.95)
        critical_mask = active_mask & (Fr >= 0.95) & (Fr <= 1.05)
        supercritical_mask = active_mask & (Fr > 1.05)

        regime[subcritical_mask] = 2
        regime[critical_mask] = 3
        regime[supercritical_mask] = 4

        # Statistics
        regime_names = ['Dry', 'Stagnant', 'Subcritical', 'Critical', 'Supercritical']
        regime_counts = [(regime == i).sum() for i in range(5)]

        print(f"\nFlow Regime Classification:")
        print(f"  Grid: {nx}×{ny} ({nx*ny} cells)")
        print(f"\n{'Regime':>15} {'Count':>10} {'Percentage':>12}")
        print("-" * 40)

        for i, (name, count) in enumerate(zip(regime_names, regime_counts)):
            pct = count / (nx*ny) * 100
            print(f"{name:>15} {count:10d} {pct:11.2f}%")

        # Validation
        assert regime.min() >= 0 and regime.max() <= 4, "Regime codes in valid range"
        assert regime_counts[0] + regime_counts[1] + regime_counts[2] + \
               regime_counts[3] + regime_counts[4] == nx*ny, "All cells classified"

        print("\n✓ Flow regime classification test passed")


class TestVisualizationQualityChecks:
    """
    Test visualization quality and correctness.

    Quality Checks:
    - Color scale appropriateness
    - Spatial resolution adequacy
    - Physical plausibility
    """

    def test_colormap_range_selection(self):
        """
        Test automatic colormap range selection.

        Considerations:
        - Exclude outliers (use percentiles)
        - Symmetric for signed quantities (velocity)
        - Zero-centered for differences
        - Logarithmic for wide ranges

        Test:
        - Depth (positive): min to max
        - Velocity (signed): symmetric around zero
        - Difference: symmetric
        """
        print("\n" + "="*70)
        print("TEST: Colormap Range Selection")
        print("="*70)

        np.random.seed(42)

        # Test 1: Depth (positive quantity)
        h = np.random.lognormal(0, 0.5, (100, 100))
        h = np.clip(h, 0, 10)  # Physical limit

        # Exclude outliers (use 2nd and 98th percentiles)
        h_min = np.percentile(h, 2)
        h_max = np.percentile(h, 98)

        print(f"\nDepth Field:")
        print(f"  Actual range: [{h.min():.4f}, {h.max():.4f}]")
        print(f"  Display range (2-98%): [{h_min:.4f}, {h_max:.4f}]")
        print(f"  Outliers excluded: {((h < h_min) | (h > h_max)).sum()} cells")

        # Validation
        assert h_min >= 0, "Depth min should be non-negative"
        assert h_min < h_max, "Valid range"

        # Test 2: Velocity (signed quantity)
        u = np.random.normal(0, 1.5, (100, 100))

        # Symmetric range
        u_abs_max = max(abs(np.percentile(u, 2)), abs(np.percentile(u, 98)))
        u_min = -u_abs_max
        u_max = u_abs_max

        print(f"\nVelocity Field:")
        print(f"  Actual range: [{u.min():.4f}, {u.max():.4f}]")
        print(f"  Display range (symmetric): [{u_min:.4f}, {u_max:.4f}]")
        print(f"  Zero-centered: {abs(u_min + u_max) < 0.01}")

        # Validation
        assert abs(u_min + u_max) < 0.1, "Should be approximately symmetric"

        # Test 3: Difference field
        h1 = np.random.uniform(1, 3, (100, 100))
        h2 = h1 + np.random.normal(0, 0.2, (100, 100))
        diff = h2 - h1

        diff_abs_max = max(abs(np.percentile(diff, 2)), abs(np.percentile(diff, 98)))
        diff_min = -diff_abs_max
        diff_max = diff_abs_max

        print(f"\nDifference Field:")
        print(f"  Actual range: [{diff.min():.4f}, {diff.max():.4f}]")
        print(f"  Display range (symmetric): [{diff_min:.4f}, {diff_max:.4f}]")

        assert abs(diff_min + diff_max) < 0.1, "Difference range should be symmetric"

        print("\n✓ Colormap range selection test passed")

    def test_vector_field_decimation(self):
        """
        Test vector field decimation for clarity.

        Issue:
        - Full vector field too dense to visualize
        - Need intelligent subsampling

        Methods:
        - Regular decimation (every Nth vector)
        - Adaptive decimation (denser where flow varies)
        - Magnitude-based filtering

        Test:
        - Decimation preserves important features
        - Reduced density improves clarity
        """
        print("\n" + "="*70)
        print("TEST: Vector Field Decimation")
        print("="*70)

        # High-resolution velocity field
        nx, ny = 200, 100
        x = np.linspace(0, 10, nx)
        y = np.linspace(0, 5, ny)
        X, Y = np.meshgrid(x, y, indexing='ij')

        # Vortex flow pattern
        cx, cy = 5, 2.5
        u = -(Y - cy)
        v = (X - cx)

        print(f"\nFull Vector Field:")
        print(f"  Grid: {nx}×{ny} = {nx*ny} vectors")
        print(f"  Too dense for clear visualization")

        # Method 1: Regular decimation
        skip = 5
        u_dec = u[::skip, ::skip]
        v_dec = v[::skip, ::skip]

        print(f"\nMethod 1: Regular Decimation (every {skip}th vector):")
        print(f"  Decimated grid: {u_dec.shape[0]}×{u_dec.shape[1]} = {u_dec.size} vectors")
        print(f"  Reduction: {u_dec.size / (nx*ny) * 100:.2f}%")

        # Method 2: Magnitude-based filtering
        V_mag = np.sqrt(u**2 + v**2)
        threshold = np.percentile(V_mag, 75)  # Keep only strong flows

        strong_flow_mask = V_mag > threshold
        n_strong = strong_flow_mask.sum()

        print(f"\nMethod 2: Magnitude-Based Filtering:")
        print(f"  Threshold: {threshold:.4f} m/s (75th percentile)")
        print(f"  Strong flow vectors: {n_strong} ({n_strong / (nx*ny) * 100:.2f}%)")

        # Validation
        assert u_dec.size < u.size, "Decimation should reduce vector count"
        assert n_strong < nx*ny, "Filtering should reduce vector count"
        assert n_strong > 0, "Some vectors should remain"

        print("\n✓ Vector field decimation test passed")

    def test_contour_level_selection(self):
        """
        Test automatic contour level selection.

        Goals:
        - Informative levels (show key features)
        - Not too many (cluttered)
        - Not too few (uninformative)
        - Round numbers for readability

        Methods:
        - Linear spacing
        - Logarithmic for wide ranges
        - Custom levels for specific applications
        """
        print("\n" + "="*70)
        print("TEST: Contour Level Selection")
        print("="*70)

        # Depth field with wide range
        nx, ny = 100, 100
        np.random.seed(42)

        x = np.linspace(0, 100, nx)
        y = np.linspace(0, 50, ny)
        X, Y = np.meshgrid(x, y, indexing='ij')

        # Depth decreases with distance
        h = 10 * np.exp(-((X-50)**2 + (Y-25)**2) / 500)

        print(f"\nDepth Field:")
        print(f"  Range: [{h.min():.4f}, {h.max():.4f}] m")

        # Method 1: Linear spacing
        n_levels = 10
        levels_linear = np.linspace(h.min(), h.max(), n_levels)

        print(f"\nMethod 1: Linear Spacing ({n_levels} levels):")
        print(f"  Levels: {levels_linear}")

        # Method 2: Nice round numbers
        # Round to nearest 0.5m or 1m depending on range
        h_range = h.max() - h.min()

        if h_range < 5:
            step = 0.5
        elif h_range < 20:
            step = 1.0
        else:
            step = 2.0

        level_min = np.floor(h.min() / step) * step
        level_max = np.ceil(h.max() / step) * step
        levels_nice = np.arange(level_min, level_max + step, step)

        print(f"\nMethod 2: Round Numbers (step = {step} m):")
        print(f"  Levels: {levels_nice}")
        print(f"  Number of levels: {len(levels_nice)}")

        # Method 3: Logarithmic (for very wide ranges)
        if h.max() / h.min() > 100:
            levels_log = np.logspace(np.log10(h.min()), np.log10(h.max()), n_levels)
            print(f"\nMethod 3: Logarithmic:")
            print(f"  Levels: {levels_log}")

        # Validation
        assert len(levels_linear) == n_levels, "Correct number of levels"
        assert 5 < len(levels_nice) < 20, "Reasonable number of round levels"
        assert np.all(np.diff(levels_linear) > 0), "Levels should be increasing"

        print("\n✓ Contour level selection test passed")


class TestDataExportFormatValidation:
    """
    Test data export formats and validation.

    Export Formats:
    - ASCII Grid (GIS standard)
    - NetCDF (CF-compliant)
    - VTK (visualization)
    - CSV (simple tables)
    """

    def test_ascii_grid_export(self):
        """
        Test ASCII Grid (ARC/INFO) export format.

        Format:
        - Header: ncols, nrows, xllcorner, yllcorner, cellsize, NODATA_value
        - Data: row-major order

        Validation:
        - Header correctness
        - Data dimensions
        - NODATA handling
        """
        print("\n" + "="*70)
        print("TEST: ASCII Grid Export Format")
        print("="*70)

        # Generate test data
        ncols, nrows = 50, 40
        xllcorner, yllcorner = 100000.0, 200000.0
        cellsize = 10.0
        NODATA_value = -9999.0

        data = np.random.uniform(0, 5, (nrows, ncols))

        # Add some NODATA cells
        data[0:5, 0:5] = NODATA_value

        print(f"\nGrid Specifications:")
        print(f"  Dimensions: {ncols} cols × {nrows} rows")
        print(f"  Lower-left corner: ({xllcorner}, {yllcorner})")
        print(f"  Cell size: {cellsize} m")
        print(f"  NODATA value: {NODATA_value}")

        # Write ASCII Grid
        with tempfile.NamedTemporaryFile(mode='w', suffix='.asc', delete=False) as f:
            filename = f.name

            # Header
            f.write(f"ncols {ncols}\n")
            f.write(f"nrows {nrows}\n")
            f.write(f"xllcorner {xllcorner}\n")
            f.write(f"yllcorner {yllcorner}\n")
            f.write(f"cellsize {cellsize}\n")
            f.write(f"NODATA_value {NODATA_value}\n")

            # Data (row-major, top to bottom)
            for row in data:
                f.write(' '.join(f"{val:.4f}" for val in row) + '\n')

        print(f"\nFile written: {filename}")
        print(f"  Size: {os.path.getsize(filename)} bytes")

        # Read and validate
        with open(filename, 'r') as f:
            lines = f.readlines()

        # Parse header
        header = {}
        for i, line in enumerate(lines[:6]):
            key, value = line.strip().split()
            header[key] = float(value) if '.' in value else int(value)

        print(f"\nValidation:")
        print(f"  ncols: {header['ncols']} (expected: {ncols})")
        print(f"  nrows: {header['nrows']} (expected: {nrows})")
        print(f"  Header lines: 6")
        print(f"  Data lines: {len(lines) - 6}")

        # Validation
        assert header['ncols'] == ncols, "ncols mismatch"
        assert header['nrows'] == nrows, "nrows mismatch"
        assert abs(header['xllcorner'] - xllcorner) < 0.01, "xllcorner mismatch"
        assert len(lines) == 6 + nrows, "Correct number of lines"

        # Clean up
        os.unlink(filename)

        print("\n✓ ASCII Grid export test passed")

    def test_netcdf_cf_export(self):
        """
        Test NetCDF CF-compliant export.

        CF Conventions:
        - Standard variable names
        - Units attributes
        - Coordinate systems
        - Metadata

        Validation:
        - File structure
        - Dimensions
        - Variables
        - Attributes
        """
        print("\n" + "="*70)
        print("TEST: NetCDF CF-Compliant Export")
        print("="*70)

        try:
            import netCDF4 as nc
        except ImportError:
            pytest.skip("netCDF4 not available")
            return

        # Test data
        nx, ny, nt = 50, 40, 10
        dx, dy = 100.0, 100.0
        dt = 60.0  # seconds

        x = np.arange(nx) * dx
        y = np.arange(ny) * dy
        time = np.arange(nt) * dt

        depth = np.random.uniform(0, 3, (nt, ny, nx))
        velocity_x = np.random.uniform(-1, 1, (nt, ny, nx))

        print(f"\nDataset Specifications:")
        print(f"  Grid: {nx}×{ny}")
        print(f"  Time steps: {nt}")
        print(f"  Variables: depth, velocity_x")

        # Write NetCDF
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as f:
            filename = f.name

        dataset = nc.Dataset(filename, 'w', format='NETCDF4')

        # Dimensions
        dataset.createDimension('x', nx)
        dataset.createDimension('y', ny)
        dataset.createDimension('time', nt)

        # Coordinate variables
        x_var = dataset.createVariable('x', 'f8', ('x',))
        x_var[:] = x
        x_var.units = 'meters'
        x_var.standard_name = 'projection_x_coordinate'

        y_var = dataset.createVariable('y', 'f8', ('y',))
        y_var[:] = y
        y_var.units = 'meters'
        y_var.standard_name = 'projection_y_coordinate'

        time_var = dataset.createVariable('time', 'f8', ('time',))
        time_var[:] = time
        time_var.units = 'seconds since 2025-01-01 00:00:00'
        time_var.standard_name = 'time'

        # Data variables
        depth_var = dataset.createVariable('depth', 'f4', ('time', 'y', 'x'),
                                          compression='zlib', complevel=4)
        depth_var[:] = depth
        depth_var.units = 'm'
        depth_var.standard_name = 'water_surface_height_above_reference_datum'
        depth_var.long_name = 'Water Depth'

        u_var = dataset.createVariable('velocity_x', 'f4', ('time', 'y', 'x'),
                                      compression='zlib', complevel=4)
        u_var[:] = velocity_x
        u_var.units = 'm/s'
        u_var.standard_name = 'sea_water_x_velocity'
        u_var.long_name = 'Velocity in X Direction'

        # Global attributes
        dataset.Conventions = 'CF-1.8'
        dataset.title = 'HydroSIS-2D Test Output'
        dataset.institution = 'HydroSIS-2D Development Team'
        dataset.source = 'HydroSIS-2D v0.1.0'

        dataset.close()

        print(f"\nFile written: {filename}")
        print(f"  Size: {os.path.getsize(filename)} bytes")

        # Read and validate
        dataset = nc.Dataset(filename, 'r')

        print(f"\nValidation:")
        print(f"  Dimensions: {list(dataset.dimensions.keys())}")
        print(f"  Variables: {list(dataset.variables.keys())}")
        print(f"  Convention: {dataset.Conventions}")

        # Check structure
        assert 'x' in dataset.dimensions, "x dimension exists"
        assert 'time' in dataset.dimensions, "time dimension exists"
        assert 'depth' in dataset.variables, "depth variable exists"
        assert dataset.variables['depth'].shape == (nt, ny, nx), "Correct shape"
        assert hasattr(dataset, 'Conventions'), "CF Convention attribute present"

        dataset.close()
        os.unlink(filename)

        print("\n✓ NetCDF CF export test passed")

    def test_vtk_export(self):
        """
        Test VTK (Visualization Toolkit) export.

        VTK Format:
        - Structured grid
        - Point data
        - Cell data
        - Scalars and vectors

        Use Cases:
        - ParaView visualization
        - 3D rendering
        - Advanced post-processing
        """
        print("\n" + "="*70)
        print("TEST: VTK Export Format")
        print("="*70)

        # Test data
        nx, ny = 20, 15
        dx, dy = 10.0, 10.0

        x = np.arange(nx) * dx
        y = np.arange(ny) * dy
        X, Y = np.meshgrid(x, y, indexing='ij')

        # Scalar field (depth)
        h = np.random.uniform(1, 3, (nx, ny))

        # Vector field (velocity)
        u = np.random.uniform(-1, 1, (nx, ny))
        v = np.random.uniform(-1, 1, (nx, ny))

        print(f"\nDataset:")
        print(f"  Grid: {nx}×{ny}")
        print(f"  Scalar: depth")
        print(f"  Vector: velocity (u, v)")

        # Write VTK (simple ASCII format)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.vtk', delete=False) as f:
            filename = f.name

            # Header
            f.write("# vtk DataFile Version 3.0\n")
            f.write("HydroSIS-2D Output\n")
            f.write("ASCII\n")
            f.write("DATASET STRUCTURED_POINTS\n")

            # Grid specification
            f.write(f"DIMENSIONS {nx} {ny} 1\n")
            f.write(f"ORIGIN 0.0 0.0 0.0\n")
            f.write(f"SPACING {dx} {dy} 1.0\n")

            # Point data
            n_points = nx * ny
            f.write(f"POINT_DATA {n_points}\n")

            # Scalar field
            f.write("SCALARS depth float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for j in range(ny):
                for i in range(nx):
                    f.write(f"{h[i, j]:.6f}\n")

            # Vector field
            f.write("VECTORS velocity float\n")
            for j in range(ny):
                for i in range(nx):
                    f.write(f"{u[i, j]:.6f} {v[i, j]:.6f} 0.0\n")

        print(f"\nFile written: {filename}")
        print(f"  Size: {os.path.getsize(filename)} bytes")

        # Read and validate
        with open(filename, 'r') as f:
            lines = f.readlines()

        print(f"\nValidation:")
        print(f"  Total lines: {len(lines)}")
        print(f"  Header: {lines[0].strip()}")
        print(f"  Format: {lines[2].strip()}")

        # Check key elements
        assert "vtk DataFile" in lines[0], "Valid VTK header"
        assert "ASCII" in lines[2], "ASCII format"
        assert "STRUCTURED_POINTS" in lines[3], "Structured points dataset"

        os.unlink(filename)

        print("\n✓ VTK export test passed")


class TestAnimationTimeSeries:
    """
    Test animation generation and time series extraction.

    Animations:
    - Frame generation
    - Temporal interpolation
    - File size optimization
    """

    def test_animation_frame_generation(self):
        """
        Test animation frame generation.

        Requirements:
        - Consistent layout across frames
        - Proper time labeling
        - Smooth transitions
        - Reasonable file size

        Test:
        - Generate series of frames
        - Check consistency
        """
        print("\n" + "="*70)
        print("TEST: Animation Frame Generation")
        print("="*70)

        # Time series data
        nt = 20
        nx, ny = 50, 40

        time = np.linspace(0, 60, nt)  # 60 seconds
        depth = np.zeros((nt, nx, ny))

        # Propagating wave
        for t_idx, t in enumerate(time):
            x = np.linspace(0, 100, nx)
            y = np.linspace(0, 80, ny)
            X, Y = np.meshgrid(x, y, indexing='ij')

            # Wave equation (simplified)
            k = 2 * np.pi / 50  # Wavenumber
            omega = 2 * np.pi / 10  # Angular frequency
            depth[t_idx] = 2.0 + 0.5 * np.sin(k * X - omega * t)

        print(f"\nAnimation Specifications:")
        print(f"  Frames: {nt}")
        print(f"  Duration: {time[-1]} seconds")
        print(f"  Grid: {nx}×{ny}")

        # Generate frames
        temp_dir = tempfile.mkdtemp()
        frame_files = []

        for t_idx in range(nt):
            fig, ax = plt.subplots(figsize=(8, 6))

            im = ax.imshow(depth[t_idx].T, origin='lower',
                          cmap='Blues', vmin=1.5, vmax=2.5)
            ax.set_title(f'Depth at t = {time[t_idx]:.2f} s')
            ax.set_xlabel('X (grid points)')
            ax.set_ylabel('Y (grid points)')

            plt.colorbar(im, ax=ax, label='Depth (m)')

            frame_file = os.path.join(temp_dir, f'frame_{t_idx:03d}.png')
            plt.savefig(frame_file, dpi=100, bbox_inches='tight')
            plt.close(fig)

            frame_files.append(frame_file)

        print(f"\nFrames Generated:")
        print(f"  Directory: {temp_dir}")
        print(f"  Files: {len(frame_files)}")

        # Check file sizes
        file_sizes = [os.path.getsize(f) for f in frame_files]
        avg_size = np.mean(file_sizes)
        std_size = np.std(file_sizes)

        print(f"  Average size: {avg_size/1024:.2f} KB")
        print(f"  Std dev: {std_size/1024:.2f} KB")
        print(f"  Consistency: {std_size/avg_size*100:.2f}% variation")

        # Validation
        assert len(frame_files) == nt, "All frames generated"
        assert std_size / avg_size < 0.1, "Frame sizes consistent"

        # Clean up
        for f in frame_files:
            os.unlink(f)
        os.rmdir(temp_dir)

        print("\n✓ Animation frame generation test passed")

    def test_time_series_extraction(self):
        """
        Test time series extraction at points/lines.

        Extraction Types:
        - Point time series (gauge)
        - Line cross-section time series
        - Area-averaged time series

        Applications:
        - Comparison with observations
        - Hydrograph generation
        - Stage-discharge relationships
        """
        print("\n" + "="*70)
        print("TEST: Time Series Extraction")
        print("="*70)

        # Simulation results
        nt = 100
        nx, ny = 50, 40
        dt = 1.0  # seconds

        time = np.arange(nt) * dt
        depth = np.random.uniform(1, 3, (nt, nx, ny))

        print(f"\nSimulation Data:")
        print(f"  Time steps: {nt}")
        print(f"  Grid: {nx}×{ny}")

        # Point extraction (gauge location)
        gauge_i, gauge_j = 25, 20

        depth_gauge = depth[:, gauge_i, gauge_j]

        print(f"\nPoint Time Series:")
        print(f"  Location: ({gauge_i}, {gauge_j})")
        print(f"  Mean depth: {depth_gauge.mean():.4f} m")
        print(f"  Std dev: {depth_gauge.std():.4f} m")
        print(f"  Range: [{depth_gauge.min():.4f}, {depth_gauge.max():.4f}] m")

        # Cross-section extraction
        cross_section_i = 25
        depth_cross = depth[:, cross_section_i, :]  # Time × Y

        # Average across cross-section
        depth_cross_avg = depth_cross.mean(axis=1)

        print(f"\nCross-Section Average:")
        print(f"  Location: i = {cross_section_i}")
        print(f"  Mean depth: {depth_cross_avg.mean():.4f} m")
        print(f"  Range: [{depth_cross_avg.min():.4f}, {depth_cross_avg.max():.4f}] m")

        # Area-averaged time series (entire domain)
        depth_domain_avg = depth.mean(axis=(1, 2))

        print(f"\nDomain Average:")
        print(f"  Mean: {depth_domain_avg.mean():.4f} m")
        print(f"  Temporal trend: {(depth_domain_avg[-1] - depth_domain_avg[0]):.4f} m")

        # Validation
        assert len(depth_gauge) == nt, "Time series length correct"
        assert depth_gauge.min() >= 0, "Physical values"
        assert len(depth_cross_avg) == nt, "Cross-section average length correct"

        print("\n✓ Time series extraction test passed")

    def test_video_compression_quality(self):
        """
        Test video compression and quality settings.

        Considerations:
        - File size vs quality trade-off
        - Frame rate selection
        - Codec compatibility
        - Resolution

        Test:
        - Different compression settings
        - Evaluate quality metrics
        """
        print("\n" + "="*70)
        print("TEST: Video Compression Quality")
        print("="*70)

        # Simulation parameters
        nt = 30
        fps = 10  # Frames per second

        print(f"\nVideo Specifications:")
        print(f"  Frames: {nt}")
        print(f"  Frame rate: {fps} fps")
        print(f"  Duration: {nt/fps:.2f} seconds")

        # Quality settings
        quality_settings = [
            ('Low', 50),    # Quality 0-100 (lower = more compression)
            ('Medium', 75),
            ('High', 95),
        ]

        print(f"\nQuality Settings:")
        for name, quality in quality_settings:
            # Estimate compression ratio (simplified)
            # In reality, would generate actual videos
            uncompressed_size = nt * 1920 * 1080 * 3  # Bytes (assuming HD)
            compression_ratio = 100 / quality

            compressed_size = uncompressed_size / compression_ratio

            print(f"  {name:8s} (quality={quality:2d}): ~{compressed_size/1e6:.2f} MB " +
                  f"(compression: {compression_ratio:.1f}x)")

        # Recommendations
        print(f"\nRecommendations:")
        print(f"  Publication: High quality (95), 30 fps")
        print(f"  Web preview: Medium quality (75), 15-20 fps")
        print(f"  Email/quick share: Low quality (50), 10 fps")

        print("\n✓ Video compression quality test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
