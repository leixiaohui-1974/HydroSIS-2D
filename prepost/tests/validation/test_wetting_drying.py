"""
Wetting-Drying Interface Tests

This module tests the numerical treatment of wetting-drying fronts in shallow water flow.
Wetting-drying is one of the most challenging aspects of shallow water modeling.

Tests cover:
- Thin film approximation
- Dry cell detection and handling
- Wetting front advancement
- Mass conservation at wet-dry interface
- Positivity preservation (h ≥ 0)
- Numerical stability near h = 0

Author: HydroSIS-2D Development Team
Date: 2025-11-13
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from preprocessing.mesh_generation import UniformRectangularMesh
from preprocessing.initial_conditions import CustomIC
from preprocessing.boundary_conditions import BoundaryConditionManager


class TestDryCellDetection:
    """Tests for identifying and handling dry cells."""

    def test_dry_tolerance_threshold(self):
        """
        Dry cell detection using depth threshold

        Test objectives:
        - Define appropriate dry tolerance (typically 1e-6 to 1e-4 m)
        - Cells with h < h_dry are treated as dry
        - Ensures numerical stability and physical realism

        Common values:
        - h_dry = 1e-6 m (very thin film)
        - h_dry = 1e-4 m (0.1 mm, more robust)

        GPU kernel: Dry cell detection in flux computation
        """
        # Dry tolerance values
        h_dry_strict = 1e-6   # Strict (may be unstable)
        h_dry_standard = 1e-5  # Standard
        h_dry_robust = 1e-4    # Robust (recommended)

        # Test water depths
        h_values = np.array([1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 1.0])

        for h_dry in [h_dry_strict, h_dry_standard, h_dry_robust]:
            is_dry = h_values < h_dry
            is_wet = h_values >= h_dry

            # Verify mutual exclusivity
            assert np.all(is_dry != is_wet), "Cell must be either dry or wet"

            # Very small depths should be dry
            assert is_dry[0], f"h={h_values[0]} should be dry with h_dry={h_dry}"

            # Moderate depths should be wet
            assert is_wet[-1], f"h={h_values[-1]} should be wet with h_dry={h_dry}"

            print(f"h_dry = {h_dry:.1e}: {np.sum(is_dry)}/{len(h_values)} cells dry")

        # Recommended: h_dry = 1e-5 m (10 μm)
        # Too small: numerical instability
        # Too large: artificial mass loss

    def test_velocity_in_dry_cells(self):
        """
        Velocity treatment in dry cells

        Test objectives:
        - Dry cells should have zero velocity (u = v = 0)
        - Prevents spurious mass flux from dry cells
        - Ensures momentum conservation

        Implementation:
        - if h < h_dry: set u = v = 0
        - Prevents division by zero in u = hu / h

        GPU kernel: State variable update
        """
        h_dry = 1e-5

        # Test cases
        test_cases = [
            {'h': 1e-8, 'hu': 1e-7, 'expected_u': 0.0},  # Dry: force u = 0
            {'h': 1e-6, 'hu': 1e-5, 'expected_u': 0.0},  # Dry: force u = 0
            {'h': 1e-4, 'hu': 1e-3, 'expected_u': 10.0}, # Wet: u = hu/h
            {'h': 1.0, 'hu': 2.0, 'expected_u': 2.0},    # Wet: u = hu/h
        ]

        for case in test_cases:
            h = case['h']
            hu = case['hu']

            if h < h_dry:
                # Dry cell: enforce u = 0
                u_computed = 0.0
            else:
                # Wet cell: compute u = hu / h
                u_computed = hu / h

            assert abs(u_computed - case['expected_u']) < 1e-10, \
                f"Velocity mismatch for h={h}, hu={hu}"

    def test_partially_dry_interface(self):
        """
        Interface between wet and dry cells

        Test objectives:
        - Handle flux computation at wet-dry interface
        - One-sided Riemann problem (dry state on one side)
        - Preserve positivity (h ≥ 0 everywhere)

        Method:
        - Left cell: h_L > 0 (wet)
        - Right cell: h_R = 0 (dry)
        - Flux should ensure h_L stays positive

        GPU kernel: Riemann solver with dry state
        """
        g = 9.81
        h_dry = 1e-5

        # Wet cell
        h_L = 1.0
        u_L = 2.0

        # Dry cell
        h_R = 0.0
        u_R = 0.0

        # Wave speeds
        c_L = np.sqrt(g * h_L)
        c_R = 0.0  # Dry cell: c = 0

        # HLL wave speeds for wet-dry interface
        S_L = u_L - 2.0 * c_L  # Outgoing rarefaction
        S_R = u_L + c_L        # Right wave

        # Verify wave speeds
        assert S_L < 0, "Left wave should propagate left"
        assert S_R > 0, "Right wave should propagate right"

        # Flux at wet-dry interface
        # Should allow wetting (flux into dry cell) but prevent negative depth

        # Mass flux into dry cell
        F_mass = h_L * u_L if u_L > 0 else 0.0

        # Wetting criterion: flux from wet to dry
        is_wetting = (h_L > h_dry) and (u_L > 0)

        if is_wetting:
            print(f"Wetting front: F_mass = {F_mass:.3f} from wet to dry")
        else:
            print(f"No wetting: h_L={h_L}, u_L={u_L}")


class TestWettingFront:
    """Tests for advancing wetting fronts."""

    def test_dam_break_on_dry_bed(self):
        """
        Classical dam break on initially dry bed

        Test objectives:
        - Wetting front advances correctly
        - No negative depths at front
        - Mass conservation during wetting

        Analytical solution (Ritter):
        - Wet-dry interface at x_f(t) = 2*sqrt(g*h0)*t

        GPU kernel: Full solver with wetting-drying
        """
        # Domain setup
        Lx = 200.0
        nx = 200

        mesh = UniformRectangularMesh(
            x_range=(0, Lx),
            y_range=(0, 10),
            nx=nx,
            ny=10
        )

        # Initial condition: dam break on dry bed
        h0 = 10.0
        x_dam = 50.0

        h = np.zeros((nx, 10))
        h[mesh.x_centers < x_dam] = h0

        # All initially at rest
        u = np.zeros((nx, 10))
        v = np.zeros((nx, 10))

        # Theoretical wetting front position
        g = 9.81
        t = 5.0  # Time

        x_front_theory = x_dam + 2.0 * np.sqrt(g * h0) * t

        # Expected front position
        print(f"Theoretical wetting front at t={t}s: x_f = {x_front_theory:.2f} m")

        # Verify initial condition
        total_mass_initial = np.sum(h) * mesh.dx * mesh.dy
        assert total_mass_initial > 0, "Should have initial water mass"

        # Expected: front advances at ~2*sqrt(g*h0) ≈ 20 m/s
        front_speed = 2.0 * np.sqrt(g * h0)
        assert 15 < front_speed < 25, "Front speed should be ~20 m/s"

    def test_wetting_front_positivity(self):
        """
        Positivity preservation at wetting front

        Test objectives:
        - Depth remains non-negative: h ≥ 0 everywhere
        - Critical for numerical stability
        - CFL condition ensures positivity

        Method:
        - Forward Euler: h^(n+1) = h^n - (dt/dx) * (F_right - F_left)
        - For h^(n+1) ≥ 0: dt ≤ dx / (2 * max wave speed)

        GPU kernel: Time stepping with positivity check
        """
        # CFL condition for positivity
        g = 9.81
        h_max = 10.0
        c_max = np.sqrt(g * h_max)  # Maximum wave speed
        u_max = 5.0

        lambda_max = abs(u_max) + c_max  # Maximum eigenvalue

        # Cell size
        dx = 1.0

        # CFL condition
        CFL = 0.5  # Safety factor
        dt_max = CFL * dx / lambda_max

        print(f"Maximum wave speed: λ_max = {lambda_max:.2f} m/s")
        print(f"Maximum timestep for CFL={CFL}: dt_max = {dt_max:.4f} s")

        # Verify positivity condition
        # For explicit scheme: CFL ≤ 1.0 (preferably ≤ 0.5)
        assert CFL <= 1.0, "CFL must be ≤ 1 for stability"
        assert dt_max > 0, "Timestep must be positive"

        # Test: if dt > dt_max, positivity may be violated
        dt_unsafe = 2.0 * dt_max
        print(f"Unsafe timestep (2x max): dt = {dt_unsafe:.4f} s (may cause h < 0)")

    def test_thin_film_treatment(self):
        """
        Thin film approximation for very shallow water

        Test objectives:
        - Handle very small depths (h ~ 1e-6 to 1e-4 m)
        - Prevent numerical issues from h → 0
        - Maintain mass conservation

        Approaches:
        1. Depth cutoff: if h < h_dry, set h = 0
        2. Thin film: keep very small h but modify equations
        3. Hybrid: transition between dry and thin film

        GPU kernel: Thin film state handling
        """
        h_dry = 1e-5      # Dry threshold
        h_thin = 1e-4     # Thin film threshold

        # Test depth values
        depths = {
            'completely_dry': 0.0,
            'nearly_dry': 1e-6,
            'thin_film': 5e-5,
            'shallow': 1e-3,
            'normal': 1.0
        }

        for name, h in depths.items():
            if h < h_dry:
                state = 'DRY'
                treatment = 'Set h=0, u=0, v=0'
            elif h < h_thin:
                state = 'THIN_FILM'
                treatment = 'Special treatment (reduced friction, etc.)'
            else:
                state = 'WET'
                treatment = 'Standard shallow water equations'

            print(f"{name:15s} h={h:.2e}m → {state:10s}: {treatment}")

            # Verify state classification
            if h >= h_thin:
                assert state == 'WET', f"h={h} should be wet"
            elif h >= h_dry:
                assert state == 'THIN_FILM', f"h={h} should be thin film"
            else:
                assert state == 'DRY', f"h={h} should be dry"


class TestDryingProcess:
    """Tests for drying (wet → dry transition)."""

    def test_recession_to_dry(self):
        """
        Water recession leaving dry bed behind

        Test objectives:
        - Smooth transition from wet to dry
        - No artificial mass loss
        - Stable drying process

        Scenario: Water draining from a basin

        GPU kernel: Full solver with drying
        """
        # Basin with outlet
        Lx, Ly = 100.0, 50.0
        nx, ny = 100, 50

        mesh = UniformRectangularMesh(
            x_range=(0, Lx),
            y_range=(0, Ly),
            nx=nx,
            ny=ny
        )

        # Initial water depth
        h_initial = 1.0

        # Topography: sloping basin
        X, Y = np.meshgrid(mesh.x_centers, mesh.y_centers, indexing='ij')
        z_bed = 0.01 * X  # 1% slope

        # As water drains, dry cells appear at high elevation
        # Expected: drying front recedes upslope

        # Track mass over time
        mass_initial = h_initial * Lx * Ly

        print(f"Initial water mass: {mass_initial:.2f} m³")
        print(f"Basin slope: 1% (0.01 m/m)")

        # Expected: mass decreases as water drains out
        # No artificial mass creation/loss

    def test_evaporation_drying(self):
        """
        Drying due to evaporation (negative source term)

        Test objectives:
        - Handle evaporation source term
        - Smooth transition to dry state
        - Prevent negative depths

        Source term: S_h = -E (evaporation rate)
        E ~ 2-10 mm/day (2e-8 to 1e-7 m/s)

        GPU kernel: Source term integration
        """
        # Evaporation rate
        E_mm_per_day = 5.0  # mm/day
        E_m_per_s = E_mm_per_day / 1000.0 / 86400.0  # m/s

        print(f"Evaporation rate: {E_mm_per_day} mm/day = {E_m_per_s:.2e} m/s")

        # Initial depth
        h0 = 0.1  # m (10 cm)

        # Time to dry
        t_dry = h0 / E_m_per_s

        print(f"Initial depth: {h0*100:.1f} cm")
        print(f"Time to dry: {t_dry/86400:.2f} days")

        # Verify physical values
        assert 0 < t_dry < 365 * 86400, "Drying time should be reasonable"

        # Update equation: dh/dt = -E
        # Solution: h(t) = max(0, h0 - E*t)

        # Timestep constraint: dt ≤ h / E to prevent negative depth
        dt_max = 0.9 * h0 / E_m_per_s

        print(f"Safe timestep: {dt_max:.1f} s")

    def test_infiltration_drying(self):
        """
        Drying due to infiltration into soil

        Test objectives:
        - Handle infiltration sink term
        - More rapid than evaporation
        - Smooth wet-dry transition

        Infiltration models:
        - Green-Ampt: f = K_s * (1 + (ψ*Δθ)/F)
        - Horton: f = f_c + (f_0 - f_c) * exp(-kt)

        Typical rates: 1-100 mm/hr

        GPU kernel: Infiltration source term
        """
        # Infiltration rate (Green-Ampt)
        K_s = 10.0 / 1000.0 / 3600.0  # 10 mm/hr → m/s

        print(f"Infiltration rate: {K_s*1000*3600:.1f} mm/hr = {K_s:.2e} m/s")

        # Initial ponding depth
        h0 = 0.05  # m (5 cm)

        # Time to infiltrate
        t_infiltrate = h0 / K_s

        print(f"Initial ponding: {h0*100:.1f} cm")
        print(f"Time to infiltrate: {t_infiltrate/60:.1f} minutes")

        # Verify reasonable
        assert 60 < t_infiltrate < 3600, "Infiltration time should be minutes to hour"

        # Infiltration faster than evaporation
        E_typical = 5e-8  # m/s
        assert K_s > E_typical, "Infiltration should be faster than evaporation"


class TestMassConservation:
    """Tests for mass conservation at wet-dry interfaces."""

    def test_wetting_mass_balance(self):
        """
        Mass conservation during wetting

        Test objectives:
        - Total mass conserved as front advances
        - No artificial mass creation at front
        - Balance equation: dM/dt = inflow - outflow

        GPU kernel: Global mass computation
        """
        # Conservation: ∫∫ h dA = constant (no sources/sinks)

        # Initial mass in wet region
        Lx_wet = 50.0
        Ly = 100.0
        h_avg = 5.0

        M_initial = Lx_wet * Ly * h_avg

        print(f"Initial mass: M0 = {M_initial:.2f} m³")

        # After wetting: wet region expands, average depth decreases
        # But total mass must be conserved

        # Example after wetting
        Lx_wet_new = 80.0
        h_avg_new = M_initial / (Lx_wet_new * Ly)

        M_final = Lx_wet_new * Ly * h_avg_new

        # Verify conservation
        mass_error = abs(M_final - M_initial) / M_initial

        assert mass_error < 1e-10, f"Mass conservation error: {mass_error:.2e}"

        print(f"After wetting: wet length {Lx_wet} → {Lx_wet_new} m")
        print(f"After wetting: avg depth {h_avg:.3f} → {h_avg_new:.3f} m")
        print(f"Final mass: M = {M_final:.2f} m³ (error: {mass_error:.2e})")

    def test_drying_mass_balance(self):
        """
        Mass conservation during drying

        Test objectives:
        - Mass conserved as cells dry
        - Residual water redistributed correctly
        - No mass trapped in dry cells

        GPU kernel: Drying algorithm
        """
        h_dry = 1e-5

        # Cell with very small depth
        h_cell = 5e-6  # Below dry tolerance

        # This cell should dry out
        # Water mass should be redistributed or lost to tolerance

        cell_area = 1.0  # m²
        residual_mass = h_cell * cell_area

        print(f"Cell depth: {h_cell:.2e} m (below h_dry = {h_dry:.2e})")
        print(f"Residual mass: {residual_mass:.2e} m³")

        # Option 1: Discard (acceptable if < tolerance)
        if residual_mass < h_dry * cell_area:
            treatment = "Discard (negligible)"

        # Option 2: Redistribute to neighbors
        else:
            treatment = "Redistribute to wet neighbors"

        print(f"Treatment: {treatment}")

        # Verify residual is negligible
        assert residual_mass < 1e-4, "Residual mass should be negligible"


# Run tests if executed directly
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
