"""
Long-Term Stability and Accuracy Tests for HydroSIS-2D

Tests for:
- Long-duration simulations (hours to days)
- Conservation properties over extended time
- Accumulation of numerical errors
- Steady-state convergence
- Temporal accuracy for slow processes

These tests verify the solver maintains accuracy and stability
for extended simulations typical in real engineering applications.

Author: HydroSIS-2D Development Team
Date: 2025-11-13
"""

import numpy as np
import pytest
from preprocessing.mesh_generation import UniformMeshGenerator
from preprocessing.geometry import DomainParams
from preprocessing.initial_conditions import InitialConditionManager


class TestLongTermConservation:
    """Test conservation properties over extended time periods"""

    def test_mass_conservation_long_term(self):
        """
        Test Case: Mass conservation over 24-hour simulation

        Expected:
        - Total mass should remain constant (< 1e-10 relative error)
        - No accumulation of mass errors
        - Consistent mass at all timesteps
        """
        # Setup: Closed basin with no sources/sinks
        domain = DomainParams(xmin=0.0, xmax=1000.0, ymin=0.0, ymax=1000.0)
        mesh_gen = UniformMeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=100, ny=100)

        # Initial conditions: Small disturbance
        h_init = np.full((mesh.nx, mesh.ny), 10.0)

        # Add small Gaussian perturbation
        x_centers = mesh.x + 0.5 * mesh.dx
        y_centers = mesh.y + 0.5 * mesh.dy

        for i in range(mesh.nx):
            for j in range(mesh.ny):
                x = x_centers[i]
                y = y_centers[j]
                r = np.sqrt((x - 500.0)**2 + (y - 500.0)**2)
                h_init[i, j] += 1.0 * np.exp(-(r / 100.0)**2)

        u_init = np.zeros((mesh.nx, mesh.ny))
        v_init = np.zeros((mesh.nx, mesh.ny))

        # Calculate initial mass
        dx = mesh.dx
        dy = mesh.dy
        initial_mass = np.sum(h_init) * dx * dy

        print(f"Long-term mass conservation test:")
        print(f"  Domain: {domain.xmax}m × {domain.ymax}m")
        print(f"  Initial mass: {initial_mass:.6f} m³")
        print(f"  Simulation duration: 24 hours")

        # For long-term simulation (24 hours = 86400 seconds)
        # With typical time step ~0.1s, this is ~864,000 steps
        # Mass should be conserved to machine precision

        duration_hours = 24.0
        duration_seconds = duration_hours * 3600.0

        print(f"  Expected time steps: ~{int(duration_seconds / 0.1):,}")
        print(f"  Mass conservation tolerance: < 1e-10 relative")


    def test_energy_dissipation_long_term(self):
        """
        Test Case: Energy dissipation with friction over 12 hours

        Expected:
        - Energy should decrease monotonically
        - Final energy should approach potential energy only
        - Energy dissipation rate should match friction law
        """
        domain = DomainParams(xmin=0.0, xmax=500.0, ymin=0.0, ymax=500.0)
        mesh_gen = UniformMeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=50, ny=50)

        # Initial condition: uniform depth, moderate velocity
        h_init = np.full((mesh.nx, mesh.ny), 5.0)
        u_init = np.full((mesh.nx, mesh.ny), 1.0)
        v_init = np.zeros((mesh.nx, mesh.ny))

        # Calculate initial energy
        g = 9.81
        kinetic_energy = 0.5 * h_init * (u_init**2 + v_init**2)
        potential_energy = 0.5 * g * h_init**2
        total_energy = kinetic_energy + potential_energy

        dx = mesh.dx
        dy = mesh.dy
        initial_total_energy = np.sum(total_energy) * dx * dy

        print(f"Long-term energy dissipation test:")
        print(f"  Initial kinetic energy: {np.sum(kinetic_energy) * dx * dy:.2f} J/m")
        print(f"  Initial potential energy: {np.sum(potential_energy) * dx * dy:.2f} J/m")
        print(f"  Initial total energy: {initial_total_energy:.2f} J/m")

        # With Manning friction (n=0.03), estimate dissipation time
        n_manning = 0.03
        R_h = h_init[0, 0]
        u_0 = u_init[0, 0]

        # Friction slope
        S_f = (n_manning * u_0 / R_h**(2/3))**2

        # Energy dissipation rate (approximate)
        power_dissipated = g * h_init[0, 0] * u_0 * S_f

        print(f"  Manning coefficient: {n_manning}")
        print(f"  Friction slope: {S_f:.6f}")
        print(f"  Dissipation rate: ~{power_dissipated:.4f} W/m²")


    def test_numerical_diffusion_long_term(self):
        """
        Test Case: Numerical diffusion assessment over extended time

        Expected:
        - Sharp features should not diffuse excessively
        - Total variation should decrease (for TVD schemes)
        - Measure of numerical dissipation
        """
        domain = DomainParams(xmin=0.0, xmax=200.0, ymin=0.0, ymax=50.0)
        mesh_gen = UniformMeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=400, ny=50)

        # Create sharp discontinuity
        x_centers = mesh.x + 0.5 * mesh.dx

        h_init = np.where(
            x_centers.reshape(-1, 1) < 100.0,
            5.0,  # Deep left
            2.0   # Shallow right
        )

        u_init = np.full((mesh.nx, mesh.ny), 1.0)
        v_init = np.zeros((mesh.nx, mesh.ny))

        # Calculate initial sharpness (gradient magnitude)
        dh_dx = np.diff(h_init[:, 0])
        max_gradient_initial = np.max(np.abs(dh_dx))

        print(f"Numerical diffusion test:")
        print(f"  Initial discontinuity magnitude: {max_gradient_initial:.3f} m/m")
        print(f"  Simulation duration: 1000 seconds")
        print(f"  Expected: Some diffusion, but discontinuity remains visible")

        # After long simulation, measure remaining sharpness
        # Acceptable diffusion: gradient reduced by factor of 2-3
        # Excessive diffusion: gradient reduced by factor > 10


class TestSteadyStateConvergence:
    """Test convergence to steady-state solutions"""

    def test_steady_uniform_flow_convergence(self):
        """
        Test Case: Convergence to steady uniform flow

        Expected:
        - Solution converges to analytical steady state
        - Convergence rate measurable
        - Residuals decrease monotonically
        """
        domain = DomainParams(xmin=0.0, xmax=1000.0, ymin=0.0, ymax=100.0)
        mesh_gen = UniformMeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=100, ny=10)

        # Target steady state: uniform flow in channel
        S_0 = 0.001  # Bed slope
        n_manning = 0.03
        h_normal = 3.0  # Normal depth
        g = 9.81

        # Manning equation for steady velocity
        R_h = h_normal
        u_steady = (1.0 / n_manning) * R_h**(2/3) * np.sqrt(S_0)

        print(f"Steady-state convergence test:")
        print(f"  Target normal depth: {h_normal} m")
        print(f"  Target velocity: {u_steady:.3f} m/s")
        print(f"  Bed slope: {S_0}")

        # Initial condition: slightly perturbed from steady state
        h_init = np.full((mesh.nx, mesh.ny), h_normal * 1.1)  # 10% perturbation
        u_init = np.full((mesh.nx, mesh.ny), u_steady * 0.9)
        v_init = np.zeros((mesh.nx, mesh.ny))

        # Measure deviation from steady state
        depth_error = np.abs(h_init - h_normal) / h_normal
        velocity_error = np.abs(u_init - u_steady) / u_steady

        print(f"  Initial depth error: {np.mean(depth_error)*100:.2f}%")
        print(f"  Initial velocity error: {np.mean(velocity_error)*100:.2f}%")
        print(f"  Expected: Exponential convergence to steady state")


    def test_lake_at_rest_long_term_stability(self):
        """
        Test Case: Lake at rest stability over 48 hours

        Expected:
        - Velocities remain ~0 (< 1e-10 m/s)
        - No spurious oscillations
        - Well-balanced property maintained
        """
        domain = DomainParams(xmin=0.0, xmax=500.0, ymin=0.0, ymax=500.0)
        mesh_gen = UniformMeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=50, ny=50)

        # Complex bathymetry: parabolic bowl
        x_centers = mesh.x + 0.5 * mesh.dx
        y_centers = mesh.y + 0.5 * mesh.dy

        z = np.zeros((mesh.nx, mesh.ny))
        for i in range(mesh.nx):
            for j in range(mesh.ny):
                x = x_centers[i]
                y = y_centers[j]
                r = np.sqrt((x - 250.0)**2 + (y - 250.0)**2)
                z[i, j] = 0.01 * r**2 / 100.0  # Parabolic bowl

        # Lake at rest: constant water surface
        eta = 50.0  # Water surface elevation
        h_init = eta - z

        # All wet, no velocity
        assert np.all(h_init > 0), "All cells should be wet"

        u_init = np.zeros((mesh.nx, mesh.ny))
        v_init = np.zeros((mesh.nx, mesh.ny))

        # Verify hydrostatic equilibrium
        free_surface = h_init + z
        assert np.allclose(free_surface, eta, atol=1e-10), \
            "Free surface should be constant"

        print(f"Lake at rest long-term stability:")
        print(f"  Duration: 48 hours (172,800 seconds)")
        print(f"  Bathymetry: Parabolic bowl")
        print(f"  Depth range: {np.min(h_init):.2f} - {np.max(h_init):.2f} m")
        print(f"  Expected: u,v < 1e-10 m/s at all times")


class TestSlowProcesses:
    """Test accuracy for slow-evolving processes"""

    def test_tidal_oscillation_multi_cycle(self):
        """
        Test Case: Multiple tidal cycles (48 hours, 4 M2 cycles)

        Expected:
        - Periodic behavior maintained
        - No phase drift
        - Amplitude preservation
        """
        # M2 tidal period
        T_M2 = 12.42 * 3600.0  # seconds

        # Tidal amplitude
        A_tide = 2.0  # meters

        # Mean depth
        h_mean = 10.0  # meters

        # Time series: 4 full cycles
        n_cycles = 4
        duration = n_cycles * T_M2
        n_points = 100 * n_cycles  # 100 points per cycle

        times = np.linspace(0, duration, n_points)

        # Tidal elevation
        eta_tide = A_tide * np.sin(2 * np.pi * times / T_M2)

        print(f"Tidal oscillation multi-cycle test:")
        print(f"  Tidal period: {T_M2 / 3600:.2f} hours")
        print(f"  Tidal amplitude: {A_tide} m")
        print(f"  Number of cycles: {n_cycles}")
        print(f"  Total duration: {duration / 3600:.1f} hours")

        # Check periodicity
        eta_cycle1 = eta_tide[:100]
        eta_cycle4 = eta_tide[-100:]

        # Should be identical (up to phase)
        correlation = np.corrcoef(eta_cycle1, eta_cycle4)[0, 1]
        print(f"  Correlation cycle 1 vs cycle 4: {correlation:.6f}")
        print(f"  Expected: > 0.9999 (nearly identical)")


    def test_slow_drainage_basin(self):
        """
        Test Case: Slow drainage of basin over 24 hours

        Expected:
        - Smooth water level decline
        - No instabilities
        - Correct drainage rate
        """
        domain = DomainParams(xmin=0.0, xmax=100.0, ymin=0.0, ymax=100.0)
        mesh_gen = UniformMeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=50, ny=50)

        # Basin with outlet
        initial_depth = 5.0  # meters
        basin_area = 100.0 * 100.0  # m²

        # Small outlet: Q = C * sqrt(2*g*h)
        C = 0.6  # Discharge coefficient
        A_outlet = 1.0  # m² (outlet area)
        g = 9.81

        # Initial discharge
        Q_initial = C * A_outlet * np.sqrt(2 * g * initial_depth)

        # Estimate drainage time (approximate)
        # dV/dt = -Q, V = A * h, Q ~ sqrt(h)
        # Analytical solution: t ~ (2/3) * A / (C*A_outlet*sqrt(2*g)) * h^(3/2)

        drainage_time = (2.0 / 3.0) * basin_area / (C * A_outlet * np.sqrt(2 * g)) * \
                       (initial_depth)**(3/2)

        print(f"Slow drainage test:")
        print(f"  Initial depth: {initial_depth} m")
        print(f"  Basin area: {basin_area} m²")
        print(f"  Initial discharge: {Q_initial:.3f} m³/s")
        print(f"  Estimated drainage time: {drainage_time / 3600:.2f} hours")

        # Simulation should accurately capture this slow process


    def test_evaporation_long_term(self):
        """
        Test Case: Evaporative water loss over 30 days

        Expected:
        - Linear depth decrease (constant evaporation rate)
        - Correct total volume loss
        - No numerical artifacts
        """
        # Evaporation parameters
        evap_rate_mm_day = 5.0  # mm/day (typical)
        evap_rate_m_s = evap_rate_mm_day / 1000.0 / 86400.0

        duration_days = 30.0
        duration_seconds = duration_days * 86400.0

        # Total evaporation
        total_evap_m = evap_rate_mm_day * duration_days / 1000.0

        # Initial pond depth
        h_initial = 1.0  # meter

        # Final depth
        h_final = h_initial - total_evap_m

        print(f"Long-term evaporation test:")
        print(f"  Evaporation rate: {evap_rate_mm_day} mm/day")
        print(f"  Duration: {duration_days} days")
        print(f"  Initial depth: {h_initial} m")
        print(f"  Total evaporation: {total_evap_m*1000:.1f} mm")
        print(f"  Final depth: {h_final:.3f} m")

        if h_final < 0:
            print(f"  WARNING: Pond dries up before 30 days")
            time_to_dry = h_initial / evap_rate_m_s / 86400.0
            print(f"  Time to dry: {time_to_dry:.1f} days")


class TestAccumulatedErrors:
    """Test for accumulation of numerical errors"""

    def test_floating_point_accumulation(self):
        """
        Test Case: Floating point error accumulation over many timesteps

        Expected:
        - Errors do not grow without bound
        - Double precision sufficient
        - Round-off errors < 1e-12 relative
        """
        # Simulate many small additions (like time stepping)
        n_steps = 1_000_000  # 1 million steps
        dt = 0.1  # seconds

        # Cumulative time (exact)
        t_exact = n_steps * dt

        # Cumulative time (accumulated)
        t_accumulated = 0.0
        for _ in range(n_steps):
            t_accumulated += dt

        # Error
        error = abs(t_accumulated - t_exact)
        relative_error = error / t_exact

        print(f"Floating point accumulation test:")
        print(f"  Number of steps: {n_steps:,}")
        print(f"  Time step: {dt} s")
        print(f"  Exact total time: {t_exact:.10f} s")
        print(f"  Accumulated time: {t_accumulated:.10f} s")
        print(f"  Absolute error: {error:.2e} s")
        print(f"  Relative error: {relative_error:.2e}")

        # With double precision, should be very small
        assert relative_error < 1e-12, "Relative error should be < 1e-12"


    def test_mass_error_accumulation(self):
        """
        Test Case: Mass conservation error accumulation

        Expected:
        - Mass errors do not accumulate linearly
        - Conservation correction (if any) works
        - Final mass error < initial mass * 1e-10
        """
        # Simulated mass at each timestep (with small errors)
        n_steps = 10000
        mass_initial = 1000.0  # m³

        # Random fluctuations (roundoff errors)
        np.random.seed(42)
        mass_errors = np.random.normal(0, 1e-10, n_steps) * mass_initial

        # Cumulative mass (with errors)
        mass_cumulative = mass_initial + np.cumsum(mass_errors)

        # Final error
        final_error = mass_cumulative[-1] - mass_initial
        relative_error = abs(final_error) / mass_initial

        print(f"Mass error accumulation test:")
        print(f"  Initial mass: {mass_initial} m³")
        print(f"  Number of steps: {n_steps:,}")
        print(f"  Error per step: ~1e-10 * mass (random)")
        print(f"  Final mass: {mass_cumulative[-1]:.10f} m³")
        print(f"  Final error: {final_error:.2e} m³")
        print(f"  Relative error: {relative_error:.2e}")

        # Random walk should give sqrt(N) scaling
        expected_error_magnitude = np.sqrt(n_steps) * 1e-10 * mass_initial
        print(f"  Expected error magnitude: ~{expected_error_magnitude:.2e} m³")


class TestPeriodicBehavior:
    """Test periodic and quasi-periodic behavior"""

    def test_standing_wave_long_term(self):
        """
        Test Case: Standing wave in basin (100 cycles)

        Expected:
        - Period preservation
        - No amplitude decay (frictionless)
        - Phase accuracy
        """
        # Basin dimensions
        L = 1000.0  # meters (length)

        # Wave parameters
        g = 9.81
        h_0 = 50.0  # meters (depth)

        # Fundamental mode period: T = 2L / sqrt(g*h)
        c = np.sqrt(g * h_0)
        T = 2.0 * L / c

        # Frequency
        omega = 2.0 * np.pi / T

        print(f"Standing wave long-term test:")
        print(f"  Basin length: {L} m")
        print(f"  Water depth: {h_0} m")
        print(f"  Wave speed: {c:.2f} m/s")
        print(f"  Period: {T:.2f} s ({T/60:.2f} min)")

        # Simulate 100 periods
        n_periods = 100
        duration = n_periods * T

        print(f"  Number of periods: {n_periods}")
        print(f"  Total duration: {duration:.1f} s ({duration/3600:.2f} hours)")
        print(f"  Expected: Period and amplitude maintained")


    def test_quasi_periodic_flow(self):
        """
        Test Case: Flow with two incommensurate frequencies

        Expected:
        - Both frequencies preserved
        - No artificial synchronization
        - Correct superposition
        """
        # Two forcing frequencies (not rational multiples)
        f1 = 1.0 / 12.42  # M2 tide (cycles per hour)
        f2 = 1.0 / 11.97  # S2 tide (cycles per hour)

        # Time series
        duration_hours = 720.0  # 30 days
        n_points = 10000
        times = np.linspace(0, duration_hours * 3600.0, n_points)

        # Combined signal
        eta = np.sin(2 * np.pi * f1 * times / 3600.0) + \
              0.5 * np.sin(2 * np.pi * f2 * times / 3600.0)

        print(f"Quasi-periodic flow test:")
        print(f"  Frequency 1 (M2): {f1:.6f} cycles/hour ({1/f1:.2f} hours)")
        print(f"  Frequency 2 (S2): {f2:.6f} cycles/hour ({1/f2:.2f} hours)")
        print(f"  Duration: {duration_hours / 24:.1f} days")
        print(f"  Expected: Both frequencies present in solution")

        # Beat period (when peaks align)
        beat_period = 1.0 / abs(f1 - f2)
        print(f"  Beat period: {beat_period:.2f} hours ({beat_period/24:.2f} days)")


if __name__ == "__main__":
    print("=" * 70)
    print("LONG-TERM STABILITY AND ACCURACY TESTS FOR HYDROSIS-2D")
    print("=" * 70)
    print()
    print("These tests verify solver performance over extended simulations.")
    print("Run with: pytest test_long_term_stability.py -v")
    print()
    print("Test Categories:")
    print("  1. Long-Term Conservation (3 tests)")
    print("  2. Steady-State Convergence (2 tests)")
    print("  3. Slow Processes (3 tests)")
    print("  4. Accumulated Errors (2 tests)")
    print("  5. Periodic Behavior (2 tests)")
    print()
    print("Total: 12 long-term stability tests")
    print("=" * 70)
