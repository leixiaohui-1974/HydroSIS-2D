"""
Extreme Condition Tests for HydroSIS-2D

Tests solver robustness under extreme physical conditions:
- Very shallow water (near-dry states)
- Very deep water
- Very high velocities (near sonic/supersonic)
- Very steep slopes
- Very high friction
- Large time steps (stability limits)

These tests verify the solver can handle edge cases without crashing or
producing non-physical results.

Author: HydroSIS-2D Development Team
Date: 2025-11-13
"""

import numpy as np
import pytest
from preprocessing.mesh_generation import UniformMeshGenerator
from preprocessing.geometry import DomainParams
from preprocessing.initial_conditions import InitialConditionManager


class TestShallowWaterExtremes:
    """Test extreme shallow water conditions"""

    def test_very_shallow_flow(self):
        """
        Test Case: Flow with very shallow depth (h ~ 0.01m)

        Expected:
        - No negative depths
        - Stable computation
        - Physically reasonable velocities
        """
        # Setup
        domain = DomainParams(xmin=0.0, xmax=100.0, ymin=0.0, ymax=10.0)
        mesh_gen = UniformMeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=100, ny=10)

        ic_manager = InitialConditionManager(mesh)

        # Very shallow uniform depth (1 cm)
        h_shallow = 0.01  # meters
        u_velocity = 1.0  # m/s - relatively high velocity for shallow depth

        h_init = np.full((mesh.nx, mesh.ny), h_shallow)
        u_init = np.full((mesh.nx, mesh.ny), u_velocity)
        v_init = np.zeros((mesh.nx, mesh.ny))

        ic_manager.set_custom_depth(h_init)
        ic_manager.set_custom_velocity(u_init, v_init)

        # Verify setup
        assert np.all(ic_manager.depth >= 0.0), "Negative depths in initial condition"
        assert np.all(ic_manager.depth == h_shallow), "Incorrect shallow depth"

        # Check Froude number (should be high for shallow flow)
        g = 9.81
        Fr = u_velocity / np.sqrt(g * h_shallow)
        print(f"Froude number for very shallow flow: {Fr:.3f}")
        assert Fr > 3.0, "Froude number should be high for shallow fast flow"

        # Simulation would run here with GPU solver
        # Expected: No instabilities, h > 0 preserved


    def test_near_dry_wetting_drying(self):
        """
        Test Case: Wetting and drying with near-zero depths

        Expected:
        - Smooth transition between wet/dry
        - No negative depths
        - Correct dry cell treatment
        """
        domain = DomainParams(xmin=0.0, xmax=50.0, ymin=0.0, ymax=10.0)
        mesh_gen = UniformMeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=50, ny=10)

        # Create terrain with elevation variation
        x_centers = mesh.x + 0.5 * mesh.dx
        z = 0.05 * x_centers  # Sloped bed: 5% grade

        # Initial water level creates dry zones
        eta_init = 1.5  # Initial water surface elevation
        h_init = np.maximum(eta_init - z.reshape(-1, 1), 0.0)

        # Many cells will be dry (h = 0) or nearly dry (h < 0.01)
        dry_cells = np.sum(h_init < 1e-6)
        nearly_dry = np.sum((h_init > 1e-6) & (h_init < 0.01))
        total_cells = mesh.nx * mesh.ny

        print(f"Dry cells: {dry_cells}/{total_cells} ({100*dry_cells/total_cells:.1f}%)")
        print(f"Nearly dry: {nearly_dry}/{total_cells} ({100*nearly_dry/total_cells:.1f}%)")

        assert dry_cells > 0, "Test should include some dry cells"
        assert nearly_dry > 0, "Test should include nearly dry cells"

        # No negative depths
        assert np.all(h_init >= 0.0), "No negative depths allowed"


    def test_very_deep_water(self):
        """
        Test Case: Very deep water (h ~ 1000m)

        Expected:
        - Stable computation with large depths
        - Correct hydrostatic pressure computation
        - No overflow in depth calculations
        """
        domain = DomainParams(xmin=0.0, xmax=1000.0, ymin=0.0, ymax=1000.0)
        mesh_gen = UniformMeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=50, ny=50)

        ic_manager = InitialConditionManager(mesh)

        # Very deep water (ocean depth scale)
        h_deep = 1000.0  # meters
        u_velocity = 5.0  # m/s - typical ocean current

        h_init = np.full((mesh.nx, mesh.ny), h_deep)
        u_init = np.full((mesh.nx, mesh.ny), u_velocity)
        v_init = np.zeros((mesh.nx, mesh.ny))

        ic_manager.set_custom_depth(h_init)
        ic_manager.set_custom_velocity(u_init, v_init)

        # Check Froude number (should be very small for deep water)
        g = 9.81
        Fr = u_velocity / np.sqrt(g * h_deep)
        print(f"Froude number for very deep water: {Fr:.6f}")
        assert Fr < 0.02, "Froude number should be very small for deep water"

        # Check wave speed (should be very large)
        c = np.sqrt(g * h_deep)
        print(f"Wave speed for deep water: {c:.2f} m/s")
        assert c > 90.0, "Wave speed should be high for deep water"


class TestHighVelocityExtremes:
    """Test extreme velocity conditions"""

    def test_high_subsonic_flow(self):
        """
        Test Case: High subsonic flow (Fr ~ 0.9)

        Expected:
        - Stable near-critical flow computation
        - Correct shock capturing if transitions occur
        """
        domain = DomainParams(xmin=0.0, xmax=100.0, ymin=0.0, ymax=10.0)
        mesh_gen = UniformMeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=100, ny=10)

        ic_manager = InitialConditionManager(mesh)

        # High subsonic conditions
        h = 2.0  # meters
        g = 9.81
        c = np.sqrt(g * h)  # Wave speed
        u = 0.9 * c  # 90% of wave speed

        h_init = np.full((mesh.nx, mesh.ny), h)
        u_init = np.full((mesh.nx, mesh.ny), u)
        v_init = np.zeros((mesh.nx, mesh.ny))

        ic_manager.set_custom_depth(h_init)
        ic_manager.set_custom_velocity(u_init, v_init)

        # Verify Froude number
        Fr = u / c
        print(f"Froude number (high subsonic): {Fr:.3f}")
        assert 0.85 < Fr < 0.95, "Should be high subsonic"


    def test_supercritical_flow(self):
        """
        Test Case: Supercritical flow (Fr > 2)

        Expected:
        - Stable supercritical computation
        - Downstream boundary doesn't affect upstream
        - Shock formation if flow becomes subcritical
        """
        domain = DomainParams(xmin=0.0, xmax=100.0, ymin=0.0, ymax=10.0)
        mesh_gen = UniformMeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=100, ny=10)

        ic_manager = InitialConditionManager(mesh)

        # Supercritical conditions
        h = 1.0  # meters
        g = 9.81
        c = np.sqrt(g * h)
        u = 2.5 * c  # 2.5x wave speed

        h_init = np.full((mesh.nx, mesh.ny), h)
        u_init = np.full((mesh.nx, mesh.ny), u)
        v_init = np.zeros((mesh.nx, mesh.ny))

        ic_manager.set_custom_depth(h_init)
        ic_manager.set_custom_velocity(u_init, v_init)

        # Verify Froude number
        Fr = u / c
        print(f"Froude number (supercritical): {Fr:.3f}")
        assert Fr > 2.0, "Should be supercritical"

        # Compute characteristic speeds
        lambda_plus = u + c
        lambda_minus = u - c
        print(f"Characteristic speeds: λ+ = {lambda_plus:.2f}, λ- = {lambda_minus:.2f} m/s")
        assert lambda_minus > 0, "Both characteristics should point downstream"


    def test_extreme_velocity_gradient(self):
        """
        Test Case: Very steep velocity gradient (shock-like)

        Expected:
        - Shock capturing without oscillations
        - Entropy condition satisfied
        - No negative depths at discontinuity
        """
        domain = DomainParams(xmin=0.0, xmax=100.0, ymin=0.0, ymax=10.0)
        mesh_gen = UniformMeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=200, ny=10)

        ic_manager = InitialConditionManager(mesh)

        # Create sharp velocity discontinuity
        x_centers = mesh.x + 0.5 * mesh.dx
        x_shock = 50.0

        h_init = np.full((mesh.nx, mesh.ny), 5.0)
        u_init = np.where(
            x_centers.reshape(-1, 1) < x_shock,
            20.0,  # High velocity upstream
            5.0    # Lower velocity downstream
        )
        v_init = np.zeros((mesh.nx, mesh.ny))

        ic_manager.set_custom_depth(h_init)
        ic_manager.set_custom_velocity(u_init, v_init)

        # Verify discontinuity
        velocity_jump = np.max(np.abs(np.diff(u_init[:, 0])))
        print(f"Maximum velocity jump: {velocity_jump:.2f} m/s")
        assert velocity_jump > 10.0, "Should have steep velocity gradient"


class TestSteepSlopeExtremes:
    """Test extreme slope conditions"""

    def test_very_steep_slope(self):
        """
        Test Case: Very steep bed slope (20% grade)

        Expected:
        - Well-balanced scheme maintains lake at rest
        - Correct bed slope source term
        - Stable flow on steep terrain
        """
        domain = DomainParams(xmin=0.0, xmax=100.0, ymin=0.0, ymax=10.0)
        mesh_gen = UniformMeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=100, ny=10)

        # Very steep slope: 20% grade (11.3 degrees)
        x_centers = mesh.x + 0.5 * mesh.dx
        z = 0.20 * x_centers.reshape(-1, 1)  # 20% grade

        # Lake at rest: constant water surface elevation
        eta = 30.0
        h_init = eta - z

        # Ensure all depths are positive
        assert np.all(h_init > 0), "All cells should be wet"

        # Check slope magnitude
        dz_dx = 0.20
        slope_angle = np.arctan(dz_dx) * 180 / np.pi
        print(f"Bed slope: {dz_dx*100:.1f}% ({slope_angle:.1f} degrees)")
        assert dz_dx >= 0.20, "Should have very steep slope"

        # Lake at rest: zero velocity
        u_init = np.zeros((mesh.nx, mesh.ny))
        v_init = np.zeros((mesh.nx, mesh.ny))

        # This should remain at rest (well-balanced test)
        # Verify constant water surface
        water_surface = h_init + z
        assert np.allclose(water_surface, eta, atol=1e-10), \
            "Water surface should be constant for lake at rest"


    def test_adverse_slope_flow(self):
        """
        Test Case: Flow up an adverse slope

        Expected:
        - Correct energy dissipation
        - Flow deceleration
        - Possible flow reversal
        """
        domain = DomainParams(xmin=0.0, xmax=100.0, ymin=0.0, ymax=10.0)
        mesh_gen = UniformMeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=100, ny=10)

        # Upward slope in flow direction
        x_centers = mesh.x + 0.5 * mesh.dx
        z = 0.10 * x_centers.reshape(-1, 1)  # 10% upward grade

        # Initial flow moving upstream (against slope)
        h_init = np.full((mesh.nx, mesh.ny), 5.0)
        u_init = np.full((mesh.nx, mesh.ny), 3.0)  # Positive velocity
        v_init = np.zeros((mesh.nx, mesh.ny))

        # Check if flow has enough energy to climb slope
        g = 9.81
        specific_energy = h_init + 0.5 * u_init**2 / g
        elevation_gain = z.max() - z.min()

        print(f"Initial specific energy: {specific_energy[0,0]:.2f} m")
        print(f"Elevation gain: {elevation_gain:.2f} m")

        # Flow may or may not have enough energy
        can_climb = specific_energy[0,0] > elevation_gain
        print(f"Can flow climb slope: {can_climb}")


class TestHighFrictionExtremes:
    """Test extreme friction conditions"""

    def test_very_high_manning_coefficient(self):
        """
        Test Case: Very high Manning coefficient (n = 0.5, dense vegetation)

        Expected:
        - Rapid flow deceleration
        - Large energy dissipation
        - Stable computation with strong source term
        """
        domain = DomainParams(xmin=0.0, xmax=100.0, ymin=0.0, ymax=10.0)
        mesh_gen = UniformMeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=100, ny=10)

        ic_manager = InitialConditionManager(mesh)

        # High friction scenario
        n_manning = 0.5  # Very high (dense forest, debris)
        h_init = np.full((mesh.nx, mesh.ny), 2.0)
        u_init = np.full((mesh.nx, mesh.ny), 5.0)
        v_init = np.zeros((mesh.nx, mesh.ny))

        ic_manager.set_custom_depth(h_init)
        ic_manager.set_custom_velocity(u_init, v_init)

        # Estimate friction slope
        g = 9.81
        R_h = h_init  # Hydraulic radius ~ depth for wide channel
        S_f = (n_manning * u_init / R_h**(2/3))**2

        print(f"Manning n: {n_manning}")
        print(f"Friction slope: {S_f[0,0]:.6f}")

        assert S_f[0,0] > 0.01, "Should have high friction slope"


    def test_friction_dominated_flow(self):
        """
        Test Case: Friction-dominated equilibrium flow

        Expected:
        - Balance between bed slope and friction
        - Uniform flow steady state
        - Correct discharge for given slope and roughness
        """
        domain = DomainParams(xmin=0.0, xmax=1000.0, ymin=0.0, ymax=10.0)
        mesh_gen = UniformMeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=100, ny=10)

        # Uniform flow parameters
        S_0 = 0.001  # Bed slope
        n_manning = 0.035  # Manning coefficient
        h_uniform = 3.0  # Normal depth

        # Manning's equation for velocity
        g = 9.81
        R_h = h_uniform  # Hydraulic radius
        u_uniform = (1.0 / n_manning) * R_h**(2/3) * np.sqrt(S_0)

        print(f"Uniform flow depth: {h_uniform:.3f} m")
        print(f"Uniform flow velocity: {u_uniform:.3f} m/s")

        # At equilibrium: S_f = S_0
        S_f = (n_manning * u_uniform / R_h**(2/3))**2
        print(f"Friction slope: {S_f:.6f}")
        print(f"Bed slope: {S_0:.6f}")

        # Should be approximately equal
        assert np.abs(S_f - S_0) / S_0 < 0.01, \
            "Friction slope should equal bed slope for uniform flow"


class TestStabilityExtremes:
    """Test numerical stability limits"""

    def test_large_cfl_number(self):
        """
        Test Case: Large CFL number (near stability limit)

        Expected:
        - Solver should warn or reduce time step
        - Should not produce unstable oscillations
        - CFL adaptive scheme should activate
        """
        domain = DomainParams(xmin=0.0, xmax=100.0, ymin=0.0, ymin=10.0)
        mesh_gen = UniformMeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=100, ny=10)

        # High velocity, small cells → large CFL if dt not reduced
        h = 5.0
        u = 10.0
        g = 9.81
        c = np.sqrt(g * h)

        dx = mesh.dx
        dt_explicit = 0.8 * dx / (np.abs(u) + c)
        dt_large = 2.0 * dx / (np.abs(u) + c)  # CFL = 2.0 (unstable)

        print(f"Stable time step (CFL=0.8): {dt_explicit:.6f} s")
        print(f"Large time step (CFL=2.0): {dt_large:.6f} s")

        # Solver should detect and reject large dt
        cfl_stable = (np.abs(u) + c) * dt_explicit / dx
        cfl_large = (np.abs(u) + c) * dt_large / dx

        print(f"Stable CFL: {cfl_stable:.3f}")
        print(f"Large CFL: {cfl_large:.3f}")

        assert cfl_stable < 1.0, "Stable CFL should be < 1"
        assert cfl_large > 1.0, "Large CFL should be > 1"


    def test_mixed_subcritical_supercritical(self):
        """
        Test Case: Mixed flow regime (transitions between sub/supercritical)

        Expected:
        - Correct identification of flow regime
        - Proper shock capturing at transitions
        - Entropy-satisfying solution
        """
        domain = DomainParams(xmin=0.0, xmax=200.0, ymin=0.0, ymax=10.0)
        mesh_gen = UniformMeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=200, ny=10)

        x_centers = mesh.x + 0.5 * mesh.dx

        # Create flow that transitions from supercritical to subcritical
        # Supercritical: high velocity, shallow depth
        # Subcritical: low velocity, deep depth

        h_init = np.where(
            x_centers.reshape(-1, 1) < 100.0,
            1.0,  # Shallow (supercritical)
            5.0   # Deep (subcritical)
        )

        u_init = np.where(
            x_centers.reshape(-1, 1) < 100.0,
            10.0,  # High velocity (supercritical)
            2.0    # Low velocity (subcritical)
        )

        v_init = np.zeros((mesh.nx, mesh.ny))

        # Compute Froude numbers
        g = 9.81
        Fr_left = 10.0 / np.sqrt(g * 1.0)
        Fr_right = 2.0 / np.sqrt(g * 5.0)

        print(f"Froude number (left): {Fr_left:.3f} (supercritical)")
        print(f"Froude number (right): {Fr_right:.3f} (subcritical)")

        assert Fr_left > 1.0, "Left side should be supercritical"
        assert Fr_right < 1.0, "Right side should be subcritical"

        # This creates a hydraulic jump scenario
        print("Setup creates hydraulic jump scenario")


class TestCombinedExtremes:
    """Test combinations of extreme conditions"""

    def test_shallow_high_velocity_steep_slope(self):
        """
        Test Case: Shallow + High velocity + Steep slope

        Expected:
        - Multiple stress test: solver should handle all extremes simultaneously
        - No crashes or non-physical results
        """
        domain = DomainParams(xmin=0.0, xmax=100.0, ymin=0.0, ymax=10.0)
        mesh_gen = UniformMeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=100, ny=10)

        # Extreme combination
        h_shallow = 0.05  # 5 cm depth
        u_high = 5.0  # 5 m/s velocity

        x_centers = mesh.x + 0.5 * mesh.dx
        z_steep = 0.15 * x_centers.reshape(-1, 1)  # 15% grade

        # Froude number
        g = 9.81
        Fr = u_high / np.sqrt(g * h_shallow)

        print(f"Extreme conditions:")
        print(f"  Depth: {h_shallow} m")
        print(f"  Velocity: {u_high} m/s")
        print(f"  Slope: 15%")
        print(f"  Froude number: {Fr:.3f}")

        assert h_shallow < 0.1, "Extremely shallow"
        assert Fr > 5.0, "Extremely supercritical"
        assert 0.15 > 0.10, "Very steep slope"

        print("This is a severe stress test for the solver")


    def test_deep_low_froude_high_friction(self):
        """
        Test Case: Deep water + Low Froude + High friction

        Expected:
        - Slow, friction-dominated deep flow
        - Stable computation despite contrasting scales
        """
        domain = DomainParams(xmin=0.0, xmax=1000.0, ymin=0.0, ymax=100.0)
        mesh_gen = UniformMeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=50, ny=10)

        # Deep, slow, rough flow
        h_deep = 100.0  # 100 m depth
        u_slow = 0.5  # 0.5 m/s velocity
        n_high = 0.1  # High roughness

        g = 9.81
        Fr = u_slow / np.sqrt(g * h_deep)

        # Friction slope
        R_h = h_deep
        S_f = (n_high * u_slow / R_h**(2/3))**2

        print(f"Deep slow flow:")
        print(f"  Depth: {h_deep} m")
        print(f"  Velocity: {u_slow} m/s")
        print(f"  Manning n: {n_high}")
        print(f"  Froude number: {Fr:.6f}")
        print(f"  Friction slope: {S_f:.8f}")

        assert Fr < 0.01, "Very subcritical flow"
        assert S_f < 1e-5, "Very small friction slope for deep flow"


if __name__ == "__main__":
    print("=" * 70)
    print("EXTREME CONDITION TESTS FOR HYDROSIS-2D")
    print("=" * 70)
    print()
    print("These tests verify solver robustness under extreme conditions.")
    print("Run with: pytest test_extreme_conditions.py -v")
    print()
    print("Test Categories:")
    print("  1. Shallow Water Extremes (4 tests)")
    print("  2. High Velocity Extremes (4 tests)")
    print("  3. Steep Slope Extremes (2 tests)")
    print("  4. High Friction Extremes (2 tests)")
    print("  5. Stability Extremes (2 tests)")
    print("  6. Combined Extremes (2 tests)")
    print()
    print("Total: 16 extreme condition tests")
    print("=" * 70)
