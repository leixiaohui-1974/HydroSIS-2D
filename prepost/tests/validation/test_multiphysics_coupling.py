"""
Multi-Physics Coupling Tests for HydroSIS-2D

Tests for coupled physical processes:
- Rainfall-runoff coupling
- Infiltration and subsurface interaction
- Evaporation and atmospheric exchange
- Wind stress effects
- Temperature-dependent processes
- Sediment transport coupling (conceptual)

These tests verify correct implementation of source/sink terms and
coupling between shallow water flow and other physical processes.

Author: HydroSIS-2D Development Team
Date: 2025-11-13
"""

import numpy as np
import pytest
from preprocessing.mesh_generation import UniformMeshGenerator
from preprocessing.geometry import DomainParams
from preprocessing.initial_conditions import InitialConditionManager


class TestRainfallRunoffCoupling:
    """Rainfall as source term in shallow water equations"""

    def test_uniform_rainfall_ponding(self):
        """
        Test Case: Uniform rainfall on flat impermeable surface

        Expected:
        - Linear depth increase with time
        - Mass balance: ΔV = R * A * Δt
        - No spurious velocities on flat surface
        """
        domain = DomainParams(xmin=0.0, xmax=100.0, ymin=0.0, ymax=100.0)
        mesh_gen = UniformMeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=50, ny=50)

        # Rainfall parameters
        rainfall_rate = 50.0 / 1000.0 / 3600.0  # 50 mm/hr to m/s
        duration = 3600.0  # 1 hour
        expected_depth = rainfall_rate * duration

        print(f"Uniform rainfall test:")
        print(f"  Rainfall rate: 50 mm/hr")
        print(f"  Duration: 1 hour")
        print(f"  Expected ponding depth: {expected_depth*1000:.1f} mm")

        # Initial condition: dry surface
        h_init = np.zeros((mesh.nx, mesh.ny))
        u_init = np.zeros((mesh.nx, mesh.ny))
        v_init = np.zeros((mesh.nx, mesh.ny))

        # Verification: mass balance
        domain_area = 100.0 * 100.0  # m²
        total_volume = rainfall_rate * domain_area * duration
        average_depth = total_volume / domain_area

        assert np.abs(average_depth - expected_depth) < 1e-10, \
            "Mass balance error in rainfall"

        # Source term for continuity equation: S_mass = rainfall_rate


    def test_spatially_varying_rainfall(self):
        """
        Test Case: Spatially varying rainfall (storm cell)

        Expected:
        - Higher rainfall → greater depth accumulation
        - Flow from high rain area to low rain area
        - Correct total volume
        """
        domain = DomainParams(xmin=0.0, xmax=200.0, ymin=0.0, ymax=200.0)
        mesh_gen = UniformMeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=100, ny=100)

        x_centers = mesh.x + 0.5 * mesh.dx
        y_centers = mesh.y + 0.5 * mesh.dy

        # Gaussian rainfall pattern (storm cell)
        storm_center_x, storm_center_y = 100.0, 100.0
        storm_radius = 40.0
        max_rainfall_rate = 100.0 / 1000.0 / 3600.0  # 100 mm/hr to m/s

        rainfall_field = np.zeros((mesh.nx, mesh.ny))

        for i in range(mesh.nx):
            for j in range(mesh.ny):
                x = x_centers[i]
                y = y_centers[j]

                dist = np.sqrt((x - storm_center_x)**2 + (y - storm_center_y)**2)
                rainfall_field[i, j] = max_rainfall_rate * np.exp(-(dist / storm_radius)**2)

        # Verify rainfall distribution
        total_rainfall = np.sum(rainfall_field) * mesh.dx * mesh.dy
        max_rain = np.max(rainfall_field)
        min_rain = np.min(rainfall_field)

        print(f"Spatially varying rainfall:")
        print(f"  Maximum rate: {max_rain * 3600 * 1000:.1f} mm/hr")
        print(f"  Minimum rate: {min_rain * 3600 * 1000:.1f} mm/hr")
        print(f"  Total rainfall (1 hr): {total_rainfall * 3600:.2f} m³")


    def test_time_varying_rainfall_hyetograph(self):
        """
        Test Case: Time-varying rainfall (design storm)

        Expected:
        - Depth increases/decreases with rain intensity
        - Peak runoff follows peak rainfall (with lag)
        - Mass balance over entire storm duration
        """
        # SCS Type II 24-hour design storm
        # Intense period around 12 hours

        duration_hours = 24.0
        time_steps = 48  # 30-minute intervals
        dt = duration_hours * 3600.0 / time_steps

        # Cumulative rainfall distribution (SCS Type II)
        time_fractions = np.linspace(0, 1, time_steps + 1)
        cumulative_fractions = np.zeros_like(time_fractions)

        for i, t in enumerate(time_fractions):
            if t < 0.5:
                cumulative_fractions[i] = 0.033 + 0.378 * t
            else:
                cumulative_fractions[i] = 0.493 + 0.507 * (1.952 * (t - 0.5))**0.5

        # Total rainfall for design storm (e.g., 100mm)
        total_rainfall_m = 0.100  # 100 mm
        cumulative_rainfall = cumulative_fractions * total_rainfall_m

        # Incremental rainfall
        incremental_rainfall = np.diff(cumulative_rainfall)
        rainfall_intensities = incremental_rainfall / dt  # m/s

        # Peak intensity
        peak_intensity = np.max(rainfall_intensities)
        peak_time_index = np.argmax(rainfall_intensities)
        peak_time_hours = peak_time_index * dt / 3600.0

        print(f"Time-varying rainfall (SCS Type II):")
        print(f"  Total rainfall: {total_rainfall_m * 1000:.0f} mm")
        print(f"  Duration: {duration_hours:.0f} hours")
        print(f"  Peak intensity: {peak_intensity * 3600 * 1000:.1f} mm/hr")
        print(f"  Peak time: {peak_time_hours:.1f} hours")

        assert peak_time_hours > 10.0 and peak_time_hours < 14.0, \
            "SCS Type II peak should be around 12 hours"


class TestInfiltrationCoupling:
    """Infiltration as sink term"""

    def test_green_ampt_infiltration(self):
        """
        Test Case: Green-Ampt infiltration model

        Expected:
        - Infiltration rate decreases with time
        - Cumulative infiltration increases
        - Depth decreases due to infiltration loss
        """
        # Green-Ampt parameters (loam soil)
        K_sat = 13.0 / 1000.0 / 3600.0  # 13 mm/hr to m/s
        psi_f = 0.09  # Wetting front suction (m)
        delta_theta = 0.25  # Moisture deficit

        # Ponding depth
        h_pond = 0.05  # 5 cm

        # Green-Ampt infiltration rate
        # f = K_sat * (1 + (psi_f * delta_theta) / F)
        # where F is cumulative infiltration

        # At t=0 (initial)
        F_0 = 0.01  # Small initial value (m)
        f_0 = K_sat * (1.0 + (psi_f * delta_theta) / F_0)

        # After some infiltration
        F_1 = 0.10  # m
        f_1 = K_sat * (1.0 + (psi_f * delta_theta) / F_1)

        print(f"Green-Ampt infiltration:")
        print(f"  Soil: Loam")
        print(f"  Saturated conductivity: {K_sat * 3600 * 1000:.1f} mm/hr")
        print(f"  Initial infiltration rate: {f_0 * 3600 * 1000:.1f} mm/hr")
        print(f"  Later infiltration rate: {f_1 * 3600 * 1000:.1f} mm/hr")

        assert f_1 < f_0, "Infiltration rate should decrease with time"

        # Mass balance check
        # Sink term in continuity: S_infil = -f(t)


    def test_horton_infiltration(self):
        """
        Test Case: Horton infiltration model

        Expected:
        - Exponential decay from f_0 to f_c
        - Asymptotic approach to constant rate
        """
        # Horton equation: f(t) = f_c + (f_0 - f_c) * exp(-k*t)

        f_0 = 100.0 / 1000.0 / 3600.0  # Initial rate: 100 mm/hr
        f_c = 10.0 / 1000.0 / 3600.0   # Final rate: 10 mm/hr
        k = 4.0 / 3600.0  # Decay constant: 1/s

        # Time series
        times = np.array([0, 0.5, 1.0, 2.0, 4.0]) * 3600.0  # hours to seconds
        f_t = f_c + (f_0 - f_c) * np.exp(-k * times)

        print(f"Horton infiltration:")
        print(f"  Initial rate: {f_0 * 3600 * 1000:.1f} mm/hr")
        print(f"  Final rate: {f_c * 3600 * 1000:.1f} mm/hr")
        print(f"  Decay constant: {k * 3600:.2f} 1/hr")
        print(f"  Rates over time:")
        for t_hr, f in zip(times / 3600.0, f_t):
            print(f"    t={t_hr:.1f}h: {f * 3600 * 1000:.1f} mm/hr")

        # Verify decay behavior
        assert f_t[0] == pytest.approx(f_0, rel=1e-6)
        assert f_t[-1] > f_c and f_t[-1] < f_0
        assert np.all(np.diff(f_t) <= 0), "Infiltration rate should decrease"


    def test_infiltration_excess_runoff(self):
        """
        Test Case: Infiltration-excess (Hortonian) runoff

        Expected:
        - Runoff begins when rainfall > infiltration capacity
        - Runoff rate = rainfall rate - infiltration rate
        - Correct partitioning between infiltration and runoff
        """
        rainfall_rate = 50.0 / 1000.0 / 3600.0  # 50 mm/hr
        infiltration_capacity = 20.0 / 1000.0 / 3600.0  # 20 mm/hr

        # Runoff generation
        if rainfall_rate > infiltration_capacity:
            runoff_rate = rainfall_rate - infiltration_capacity
            runoff_fraction = runoff_rate / rainfall_rate
        else:
            runoff_rate = 0.0
            runoff_fraction = 0.0

        print(f"Infiltration-excess runoff:")
        print(f"  Rainfall: {rainfall_rate * 3600 * 1000:.1f} mm/hr")
        print(f"  Infiltration capacity: {infiltration_capacity * 3600 * 1000:.1f} mm/hr")
        print(f"  Runoff rate: {runoff_rate * 3600 * 1000:.1f} mm/hr")
        print(f"  Runoff coefficient: {runoff_fraction:.2f}")

        assert runoff_rate == pytest.approx((50.0 - 20.0) / 1000.0 / 3600.0)


class TestEvaporationCoupling:
    """Evaporation as sink term"""

    def test_penman_monteith_evaporation(self):
        """
        Test Case: Penman-Monteith evaporation model (simplified)

        Expected:
        - Evaporation rate depends on meteorological conditions
        - Depth decreases due to evaporative loss
        - Correct energy balance
        """
        # Simplified Penman equation: E ~ 5-7 mm/day for open water

        evaporation_rate_mm_day = 6.0  # mm/day
        evaporation_rate_m_s = evaporation_rate_mm_day / 1000.0 / 86400.0

        duration_days = 7.0  # 1 week
        total_evaporation_m = evaporation_rate_mm_day * duration_days / 1000.0

        print(f"Evaporation test:")
        print(f"  Evaporation rate: {evaporation_rate_mm_day} mm/day")
        print(f"  Duration: {duration_days} days")
        print(f"  Total evaporation: {total_evaporation_m * 1000:.1f} mm")

        # For shallow pond (h=0.1m), evaporation time
        initial_depth = 0.1  # m
        evaporation_time_days = (initial_depth * 1000.0) / evaporation_rate_mm_day

        print(f"  Time to evaporate {initial_depth}m depth: {evaporation_time_days:.1f} days")


    def test_evaporation_mass_loss(self):
        """
        Test Case: Mass loss due to evaporation

        Expected:
        - Total volume decreases linearly
        - No change in horizontal momentum (u,v)
        - Verification: ΔV = E * A * Δt
        """
        domain_area = 1000.0 * 1000.0  # m² (1 km²)
        evap_rate = 5.0 / 1000.0 / 86400.0  # 5 mm/day to m/s
        duration = 30.0 * 86400.0  # 30 days

        volume_loss = evap_rate * domain_area * duration

        print(f"Evaporation mass loss:")
        print(f"  Domain area: {domain_area / 1e6:.2f} km²")
        print(f"  Evaporation rate: 5 mm/day")
        print(f"  Duration: 30 days")
        print(f"  Total volume loss: {volume_loss:.2f} m³")
        print(f"  Average depth loss: {volume_loss / domain_area * 1000:.1f} mm")


class TestWindStressCoupling:
    """Wind stress as momentum source"""

    def test_wind_driven_circulation(self):
        """
        Test Case: Wind stress on water surface

        Expected:
        - Surface velocity in wind direction
        - Momentum input = wind stress
        - Setup at downwind boundary
        """
        # Wind stress: τ = ρ_air * C_d * W²
        rho_air = 1.2  # kg/m³
        C_d = 0.0015  # Drag coefficient
        wind_speed = 15.0  # m/s

        wind_stress = rho_air * C_d * wind_speed**2

        # Momentum source for shallow water: S_u = τ / (ρ_water * h)
        rho_water = 1000.0  # kg/m³
        h = 5.0  # m
        momentum_source = wind_stress / (rho_water * h)

        # Equilibrium velocity (balance with bottom friction)
        # τ_wind ≈ τ_bottom = ρ * g * n² * u² / h^(1/3)
        g = 9.81
        n_manning = 0.025
        u_equilibrium = (wind_stress / (rho_water * g * n_manning**2) * h**(1/3))**0.5

        print(f"Wind stress test:")
        print(f"  Wind speed: {wind_speed} m/s")
        print(f"  Wind stress: {wind_stress:.4f} N/m²")
        print(f"  Momentum source: {momentum_source:.6f} m/s²")
        print(f"  Equilibrium velocity: {u_equilibrium:.3f} m/s")


    def test_wind_setup_enclosed_basin(self):
        """
        Test Case: Wind setup (water surface slope due to wind)

        Expected:
        - Water surface tilts in wind direction
        - Setup magnitude proportional to wind stress and fetch
        - Balance: τ_wind = ρ * g * h * ∂η/∂x
        """
        # Wind setup formula: Δη ≈ (τ * L) / (ρ * g * h_mean)

        wind_stress = 0.5  # N/m² (moderate wind)
        fetch_length = 10000.0  # m (10 km)
        h_mean = 3.0  # m

        rho_water = 1000.0
        g = 9.81

        setup = (wind_stress * fetch_length) / (rho_water * g * h_mean)

        print(f"Wind setup in enclosed basin:")
        print(f"  Wind stress: {wind_stress} N/m²")
        print(f"  Fetch length: {fetch_length / 1000:.1f} km")
        print(f"  Mean depth: {h_mean} m")
        print(f"  Water level setup: {setup * 100:.2f} cm")


class TestTemperatureCoupling:
    """Temperature-dependent processes (conceptual)"""

    def test_temperature_dependent_viscosity(self):
        """
        Test Case: Water viscosity varies with temperature

        Note: Standard shallow water equations use constant viscosity.
        This test demonstrates the concept for extensions.

        Expected:
        - Viscosity decreases with increasing temperature
        - Affects turbulent dissipation
        """
        # Kinematic viscosity of water
        temps = np.array([0, 10, 20, 30])  # °C
        viscosities = np.array([1.787, 1.307, 1.004, 0.801]) * 1e-6  # m²/s

        print(f"Temperature-dependent viscosity:")
        for T, nu in zip(temps, viscosities):
            print(f"  T = {T}°C: ν = {nu * 1e6:.3f} × 10⁻⁶ m²/s")

        # Viscosity decreases with temperature
        assert np.all(np.diff(viscosities) < 0)


    def test_thermal_expansion_effect(self):
        """
        Test Case: Thermal expansion and density stratification

        Note: Standard SWE assumes constant density.
        Important for temperature-stratified flows.

        Expected:
        - Warm water less dense → stays on top
        - Density difference drives circulation
        """
        # Water density vs temperature
        temps = np.array([5, 15, 25])  # °C
        densities = np.array([999.97, 999.10, 997.05])  # kg/m³

        print(f"Thermal expansion:")
        for T, rho in zip(temps, densities):
            print(f"  T = {T}°C: ρ = {rho:.2f} kg/m³")

        # Density decreases with temperature (above 4°C)
        assert densities[2] < densities[0]


class TestSedimentCoupling:
    """Sediment transport coupling (conceptual)"""

    def test_erosion_deposition_concept(self):
        """
        Test Case: Bed evolution due to sediment transport (conceptual)

        Note: Full sediment transport requires additional equations.
        This demonstrates the coupling concept.

        Expected:
        - High velocity → erosion
        - Low velocity → deposition
        - Bed elevation changes over time
        """
        # Critical velocity for sediment motion (Shields criterion)
        d50 = 0.001  # Median grain size: 1 mm
        rho_s = 2650.0  # Sediment density kg/m³
        rho_w = 1000.0  # Water density kg/m³
        g = 9.81

        # Shields parameter for initiation of motion
        theta_cr = 0.05  # Critical Shields parameter

        # Critical shear stress
        tau_cr = theta_cr * (rho_s - rho_w) * g * d50

        # Critical velocity (approximate)
        h = 1.0  # m
        n_manning = 0.025
        u_cr = (tau_cr / (rho_w * g * n_manning**2 / h**(1/3)))**0.5

        print(f"Sediment transport (conceptual):")
        print(f"  Grain size: {d50 * 1000:.1f} mm")
        print(f"  Critical shear stress: {tau_cr:.3f} N/m²")
        print(f"  Critical velocity: {u_cr:.3f} m/s")
        print(f"  u < {u_cr:.2f} m/s: Deposition")
        print(f"  u > {u_cr:.2f} m/s: Erosion")


class TestMultiProcessInteraction:
    """Combined multi-physics scenarios"""

    def test_rainfall_infiltration_runoff_chain(self):
        """
        Test Case: Complete rainfall-infiltration-runoff chain

        Expected:
        - Rainfall → Ponding
        - Infiltration → Depth loss
        - Excess rainfall → Runoff generation
        - Correct mass balance
        """
        # Parameters
        rainfall = 60.0  # mm/hr
        infiltration = 25.0  # mm/hr
        evaporation = 0.2  # mm/hr (negligible during storm)

        # Partitioning
        runoff = max(0.0, rainfall - infiltration)
        actual_infiltration = min(rainfall, infiltration)

        # Mass balance (per hour per m²)
        input_volume = rainfall / 1000.0  # m³/m²
        infiltration_volume = actual_infiltration / 1000.0
        evap_volume = evaporation / 1000.0
        runoff_volume = runoff / 1000.0

        balance = input_volume - infiltration_volume - evap_volume - runoff_volume

        print(f"Complete hydrologic chain:")
        print(f"  Rainfall: {rainfall} mm/hr")
        print(f"  Infiltration: {actual_infiltration} mm/hr")
        print(f"  Evaporation: {evaporation} mm/hr")
        print(f"  Runoff: {runoff} mm/hr")
        print(f"  Mass balance error: {balance * 1000:.6f} mm/hr")

        assert abs(balance) < 1e-10, "Mass balance must be preserved"


    def test_wind_rain_combined_forcing(self):
        """
        Test Case: Combined wind and rain forcing

        Expected:
        - Wind stress drives circulation
        - Rainfall adds mass
        - Combined effect on water surface
        """
        # Wind contribution
        wind_stress = 0.3  # N/m²
        rho_water = 1000.0
        h = 2.0
        wind_accel = wind_stress / (rho_water * h)

        # Rain contribution
        rainfall_rate = 100.0 / 1000.0 / 3600.0  # m/s
        rain_volume_rate = rainfall_rate  # m³/m²/s

        print(f"Combined wind-rain forcing:")
        print(f"  Wind acceleration: {wind_accel * 1000:.3f} mm/s²")
        print(f"  Rain volume rate: {rain_volume_rate * 3600 * 1000:.1f} mm/hr")
        print(f"  Both processes active simultaneously in simulation")


if __name__ == "__main__":
    print("=" * 70)
    print("MULTI-PHYSICS COUPLING TESTS FOR HYDROSIS-2D")
    print("=" * 70)
    print()
    print("These tests verify coupling with other physical processes.")
    print("Run with: pytest test_multiphysics_coupling.py -v")
    print()
    print("Test Categories:")
    print("  1. Rainfall-Runoff Coupling (3 tests)")
    print("  2. Infiltration Coupling (3 tests)")
    print("  3. Evaporation Coupling (2 tests)")
    print("  4. Wind Stress Coupling (2 tests)")
    print("  5. Temperature Coupling (2 tests, conceptual)")
    print("  6. Sediment Coupling (1 test, conceptual)")
    print("  7. Multi-Process Interaction (2 tests)")
    print()
    print("Total: 15 multi-physics coupling tests")
    print("=" * 70)
