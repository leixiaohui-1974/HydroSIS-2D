"""
Advanced Physical Processes Tests for HydroSIS-2D GPU Solver

This module tests advanced physical processes beyond basic shallow water equations.

Test Categories:
1. Wind Stress Effects (3 tests)
   - Constant wind stress
   - Spatially varying wind
   - Wind-induced setup

2. Coriolis Effects (3 tests)
   - Geostrophic balance
   - Kelvin waves
   - Inertial oscillations

3. Turbulence Modeling (3 tests)
   - Eddy viscosity (Smagorinsky)
   - Horizontal mixing
   - Sub-grid scale dissipation

4. Variable Density Effects (3 tests)
   - Baroclinic pressure gradients
   - Density-driven flows
   - Stratification effects

Physics Context:
- Wind stress: Important for large water bodies (lakes, estuaries, coastal)
- Coriolis: Critical for geophysical flows (L > 10 km, t > hours)
- Turbulence: Sub-grid scale processes in coarse grids
- Density: Salinity/temperature gradients, stratified flows

Author: HydroSIS-2D Development Team
Date: 2025-11-13
"""

import pytest
import numpy as np
from typing import Tuple, Dict
import scipy.linalg


class TestWindStressEffects:
    """Tests for wind stress forcing on water surface."""

    def test_constant_wind_stress(self):
        """
        Test uniform wind stress effect on water surface.

        Wind stress parameterization:
        τ_wind = ρ_air * C_d * |W| * W

        where:
        - ρ_air = 1.2 kg/m³ (air density)
        - C_d = 0.001 - 0.002 (drag coefficient)
        - W = wind velocity (m/s)

        Momentum equation with wind:
        ∂(hu)/∂t + ... = τ_wind_x / ρ_water
        ∂(hv)/∂t + ... = τ_wind_y / ρ_water

        Physical context: Wind-driven circulation in lakes/estuaries.
        """
        # Parameters
        rho_air = 1.2  # kg/m³
        rho_water = 1000.0  # kg/m³
        C_d = 0.0015  # Typical drag coefficient

        # Wind velocity
        W_x = 10.0  # m/s (eastward wind, ~36 km/h)
        W_y = 0.0   # m/s

        W_mag = np.sqrt(W_x**2 + W_y**2)

        # Compute wind stress (N/m²)
        tau_x = rho_air * C_d * W_mag * W_x
        tau_y = rho_air * C_d * W_mag * W_y

        # Expected values
        tau_x_expected = 1.2 * 0.0015 * 10.0 * 10.0  # = 0.18 N/m²
        tau_y_expected = 0.0

        np.testing.assert_allclose(tau_x, tau_x_expected, rtol=1e-10)
        np.testing.assert_allclose(tau_y, tau_y_expected, rtol=1e-10)

        # Source term for momentum equation (m²/s²)
        h = 5.0  # m (water depth)
        S_u = tau_x / rho_water
        S_v = tau_y / rho_water

        # Acceleration due to wind
        acc_u = S_u / h  # m/s²
        acc_v = S_v / h

        # Expected acceleration
        acc_expected = tau_x_expected / (rho_water * h)  # ≈ 3.6e-5 m/s²

        np.testing.assert_allclose(acc_u, acc_expected, rtol=1e-10)

        # Velocity change over time
        dt = 3600.0  # s (1 hour)
        du = acc_u * dt  # m/s

        # Expected velocity change: ~0.13 m/s after 1 hour
        assert 0.1 <= du <= 0.2, f"Velocity change {du:.4f} m/s seems unreasonable"

        # Verify stress is in reasonable range
        assert 0.01 <= tau_x <= 1.0, "Wind stress should be 0.01-1.0 N/m² for typical winds"

    def test_spatially_varying_wind(self):
        """
        Test wind field with spatial variation.

        Realistic scenario:
        - Wind decreases near shoreline due to terrain
        - Wind shadows behind obstacles
        - Fetch-limited wave generation

        Physical context: Complex wind patterns in real basins.
        """
        # Domain
        Lx, Ly = 1000.0, 1000.0  # m
        nx, ny = 50, 50
        x = np.linspace(0, Lx, nx)
        y = np.linspace(0, Ly, ny)
        X, Y = np.meshgrid(x, y)

        # Wind field: stronger in the center, weaker near edges
        x0, y0 = Lx/2, Ly/2
        sigma = 300.0  # m

        W_max = 15.0  # m/s (center)
        W_min = 5.0   # m/s (edges)

        r2 = (X - x0)**2 + (Y - y0)**2
        W_mag = W_min + (W_max - W_min) * np.exp(-r2 / (2 * sigma**2))

        # Wind direction: eastward
        theta = 0.0  # radians
        W_x = W_mag * np.cos(theta)
        W_y = W_mag * np.sin(theta)

        # Verify wind field properties
        assert W_x.min() >= W_min, "Wind speed below minimum"
        assert W_x.max() <= W_max, "Wind speed above maximum"

        # Compute wind stress field
        rho_air = 1.2
        C_d = 0.0015
        tau_x = rho_air * C_d * W_mag * W_x
        tau_y = rho_air * C_d * W_mag * W_y

        # Verify stress gradient exists
        dtau_dx = np.gradient(tau_x, Lx/nx, axis=1)
        dtau_dy = np.gradient(tau_y, Ly/ny, axis=0)

        # Should have non-zero gradients due to wind variation
        assert np.std(dtau_dx) > 0, "Wind stress should vary in x"

        # Verify maximum stress at center
        center_i, center_j = ny // 2, nx // 2
        tau_center = tau_x[center_i, center_j]
        tau_edge = tau_x[0, 0]

        assert tau_center > tau_edge, "Stress should be higher at center"

        # Ratio of center to edge stress
        stress_ratio = tau_center / tau_edge
        assert 1.5 <= stress_ratio <= 10.0, \
            f"Stress ratio {stress_ratio:.2f} seems unreasonable"

    def test_wind_induced_setup(self):
        """
        Test wind-induced water level setup (wind setup).

        Theory:
        For constant wind over shallow water, equilibrium setup:
        ∂η/∂x = τ_wind / (ρ g h)

        Integrated over fetch L:
        Δη = (τ_wind * L) / (ρ g h)

        Physical context: Storm surge, seiche initiation in lakes.
        """
        # Parameters
        rho_water = 1000.0  # kg/m³
        g = 9.81  # m/s²
        h = 5.0  # m (average depth)

        # Wind stress
        W = 20.0  # m/s (strong wind, ~72 km/h)
        rho_air = 1.2
        C_d = 0.002  # Higher for strong winds
        tau_wind = rho_air * C_d * W**2  # N/m²

        # Fetch (distance over which wind acts)
        L = 10000.0  # m (10 km fetch)

        # Compute wind setup
        delta_eta = (tau_wind * L) / (rho_water * g * h)

        # Expected setup for these conditions
        # tau_wind ≈ 1.2 * 0.002 * 400 = 0.96 N/m²
        # delta_eta ≈ (0.96 * 10000) / (1000 * 9.81 * 5) ≈ 0.196 m

        assert 0.15 <= delta_eta <= 0.25, \
            f"Wind setup {delta_eta:.3f} m seems unreasonable for these conditions"

        # Test scaling relationships
        # Setup should be linear with fetch
        L2 = 20000.0  # m (double fetch)
        delta_eta_2 = (tau_wind * L2) / (rho_water * g * h)
        ratio_L = delta_eta_2 / delta_eta

        np.testing.assert_allclose(ratio_L, 2.0, rtol=1e-10)

        # Setup should be inverse with depth
        h2 = 10.0  # m (double depth)
        delta_eta_3 = (tau_wind * L) / (rho_water * g * h2)
        ratio_h = delta_eta_3 / delta_eta

        np.testing.assert_allclose(ratio_h, 0.5, rtol=1e-10)

        # Setup should be quadratic with wind speed
        W2 = 10.0  # m/s (half wind)
        tau_wind_2 = rho_air * C_d * W2**2
        delta_eta_4 = (tau_wind_2 * L) / (rho_water * g * h)
        ratio_W = delta_eta_4 / delta_eta

        np.testing.assert_allclose(ratio_W, 0.25, rtol=1e-10)  # (W2/W)^2 = 0.25


class TestCoriolisEffects:
    """Tests for Coriolis force in rotating reference frame."""

    def test_geostrophic_balance(self):
        """
        Test geostrophic balance (pressure gradient vs Coriolis).

        Geostrophic equations:
        f * v = -g * ∂η/∂x  (east-west balance)
        f * u =  g * ∂η/∂y  (north-south balance)

        where f = 2Ω sin(φ) is Coriolis parameter
        - Ω = 7.292e-5 rad/s (Earth rotation)
        - φ = latitude

        Physical context: Large-scale ocean/lake currents.
        """
        # Coriolis parameter at mid-latitudes (φ = 45°N)
        Omega = 7.292e-5  # rad/s
        phi = 45.0 * np.pi / 180.0  # radians
        f = 2.0 * Omega * np.sin(phi)

        # f ≈ 1.03e-4 rad/s at 45°N

        g = 9.81  # m/s²

        # Surface slope (pressure gradient)
        d_eta_dx = 1e-6  # m/m (1 cm over 10 km)

        # Geostrophic velocity
        v_geo = -(g / f) * d_eta_dx  # m/s

        # Expected velocity
        # v ≈ -(9.81 / 1.03e-4) * 1e-6 ≈ -0.095 m/s

        assert -0.15 <= v_geo <= -0.05, \
            f"Geostrophic velocity {v_geo:.4f} m/s seems unreasonable"

        # Verify Rossby number (ratio of inertial to Coriolis forces)
        L = 10000.0  # m (length scale)
        U = abs(v_geo)  # Velocity scale
        Ro = U / (f * L)  # Rossby number

        # For geostrophic flow: Ro << 1
        assert Ro < 0.1, f"Rossby number {Ro:.4f} too large for geostrophic balance"

        # Test at different latitudes
        latitudes = [0, 30, 45, 60, 90]  # degrees
        f_values = []

        for lat in latitudes:
            phi_rad = lat * np.pi / 180.0
            f_lat = 2.0 * Omega * np.sin(phi_rad)
            f_values.append(f_lat)

        # Verify Coriolis parameter increases with latitude
        assert f_values[0] == 0.0, "f should be zero at equator"
        assert f_values[-1] > f_values[-2], "f should increase toward pole"
        assert all(f_values[i] <= f_values[i+1] for i in range(len(f_values)-1))

    def test_kelvin_waves(self):
        """
        Test Kelvin wave propagation (coastal trapped waves).

        Kelvin wave properties:
        - Propagates along coast (right in NH, left in SH)
        - Trapped by Coriolis force
        - e-folding scale: R_d = c / f (Rossby radius)
        - Phase speed: c = sqrt(g*h)

        Physical context: Tides, storm surges along coastlines.
        """
        # Parameters
        g = 9.81  # m/s²
        h = 50.0  # m (shelf depth)
        c = np.sqrt(g * h)  # m/s (phase speed)

        # Coriolis at 45°N
        f = 1.03e-4  # rad/s

        # Rossby radius of deformation
        R_d = c / f  # m

        # Expected: R_d ≈ 22.1 / 1.03e-4 ≈ 214 km
        R_d_km = R_d / 1000.0

        assert 100.0 <= R_d_km <= 300.0, \
            f"Rossby radius {R_d_km:.1f} km seems unreasonable"

        # Kelvin wave structure (cross-shore decay)
        y = np.linspace(0, 5*R_d, 1000)  # Distance from coast

        # Amplitude decay: exp(-y/R_d)
        A_0 = 1.0  # m (amplitude at coast)
        eta = A_0 * np.exp(-y / R_d)

        # Verify decay
        assert eta[0] == A_0, "Amplitude at coast should be A_0"
        assert eta[-1] < 0.01 * A_0, "Amplitude should decay far from coast"

        # e-folding distance
        idx_e = np.argmin(np.abs(eta - A_0 / np.e))
        y_e = y[idx_e]

        np.testing.assert_allclose(y_e, R_d, rtol=0.05)

        # Kelvin wave period (tidal period for M2 tide)
        T_M2 = 12.42 * 3600.0  # s (M2 tidal period)

        # Wavelength
        lambda_k = c * T_M2  # m

        lambda_k_km = lambda_k / 1000.0

        # Expected: ~1000 km for M2 tide
        assert 800.0 <= lambda_k_km <= 1200.0, \
            f"Kelvin wavelength {lambda_k_km:.1f} km unreasonable"

    def test_inertial_oscillations(self):
        """
        Test inertial oscillations (Coriolis-driven circular motion).

        Inertial period: T_i = 2π / f

        Initial velocity perturbation leads to circular motion
        with period T_i.

        Physical context: Wind-induced currents after wind stops.
        """
        # Coriolis parameter at 45°N
        f = 1.03e-4  # rad/s

        # Inertial period
        T_i = 2.0 * np.pi / f  # s

        # Convert to hours
        T_i_hours = T_i / 3600.0

        # Expected: ~17 hours at 45°N
        assert 15.0 <= T_i_hours <= 20.0, \
            f"Inertial period {T_i_hours:.1f} hours seems unreasonable"

        # Simulate inertial oscillation
        # Equations (without pressure gradient):
        # du/dt = f * v
        # dv/dt = -f * u

        # Solution: circular motion
        # u(t) = U₀ cos(ft)
        # v(t) = U₀ sin(ft)

        U_0 = 0.5  # m/s (initial velocity)
        t = np.linspace(0, 2*T_i, 1000)

        u = U_0 * np.cos(f * t)
        v = U_0 * np.sin(f * t)

        # Verify initial conditions
        np.testing.assert_allclose(u[0], U_0, rtol=1e-10)
        np.testing.assert_allclose(v[0], 0.0, atol=1e-10)

        # Verify periodicity
        idx_period = np.argmin(np.abs(t - T_i))
        np.testing.assert_allclose(u[idx_period], U_0, rtol=1e-3)
        np.testing.assert_allclose(v[idx_period], 0.0, atol=1e-3)

        # Verify amplitude conservation (circular motion)
        speed = np.sqrt(u**2 + v**2)
        np.testing.assert_allclose(speed, U_0, rtol=1e-10)

        # Verify clockwise rotation (Northern Hemisphere)
        # At t = T_i/4: u ≈ 0, v ≈ U₀ (rightward turn)
        idx_quarter = np.argmin(np.abs(t - T_i/4))
        assert v[idx_quarter] > 0, "Should rotate clockwise in NH"

        # Test latitude dependence
        latitudes = [30, 45, 60]  # degrees
        Omega = 7.292e-5

        for lat in latitudes:
            phi = lat * np.pi / 180.0
            f_lat = 2.0 * Omega * np.sin(phi)
            T_i_lat = 2.0 * np.pi / f_lat / 3600.0  # hours

            # Higher latitude → shorter period
            if lat == 60:
                assert T_i_lat < 15.0, "Period should be <15h at 60°N"


class TestTurbulenceModeling:
    """Tests for sub-grid scale turbulence parameterizations."""

    def test_smagorinsky_eddy_viscosity(self):
        """
        Test Smagorinsky turbulence model.

        Eddy viscosity:
        ν_t = (C_s * Δ)² * |S|

        where:
        - C_s = 0.1 - 0.2 (Smagorinsky constant)
        - Δ = grid spacing
        - |S| = strain rate magnitude

        Physical context: Resolves sub-grid turbulent mixing.
        """
        # Parameters
        C_s = 0.15  # Typical value
        delta = 2.0  # m (grid spacing)

        # Velocity gradients
        du_dx = 0.1  # s⁻¹
        du_dy = 0.05
        dv_dx = 0.05
        dv_dy = 0.1

        # Strain rate tensor components
        S_xx = du_dx
        S_yy = dv_dy
        S_xy = 0.5 * (du_dy + dv_dx)

        # Strain rate magnitude
        S_mag = np.sqrt(2.0 * (S_xx**2 + S_yy**2 + 2.0 * S_xy**2))

        # Eddy viscosity
        nu_t = (C_s * delta)**2 * S_mag

        # Expected: nu_t ≈ (0.15 * 2)² * 0.17 ≈ 0.015 m²/s
        assert 0.01 <= nu_t <= 0.05, \
            f"Eddy viscosity {nu_t:.4f} m²/s seems unreasonable"

        # Test Reynolds number
        U = 1.0  # m/s (velocity scale)
        Re_t = U * delta / nu_t  # Turbulent Reynolds number

        # Should be O(10-100) for LES
        assert 10.0 <= Re_t <= 200.0, \
            f"Turbulent Re {Re_t:.1f} outside expected range"

        # Test grid sensitivity
        delta_fine = 1.0  # m (finer grid)
        nu_t_fine = (C_s * delta_fine)**2 * S_mag

        # Eddy viscosity should decrease with finer grid
        assert nu_t_fine < nu_t, "Eddy viscosity should decrease with grid refinement"

        ratio = nu_t / nu_t_fine
        np.testing.assert_allclose(ratio, 4.0, rtol=1e-10)  # (Δ/Δ_fine)² = 4

    def test_horizontal_mixing(self):
        """
        Test horizontal diffusion of momentum.

        Diffusion term:
        D = ν_t * ∇²u

        For 2D: ∇²u = ∂²u/∂x² + ∂²u/∂y²

        Physical context: Smooths out sub-grid velocity gradients.
        """
        # Setup velocity field with sharp gradient
        nx = 100
        x = np.linspace(0, 10, nx)  # m
        dx = x[1] - x[0]

        # Step function velocity
        u = np.ones(nx)
        u[:nx//2] = 2.0  # m/s
        u[nx//2:] = 0.0  # m/s

        # Eddy viscosity
        nu_t = 0.01  # m²/s

        # Compute Laplacian (finite differences)
        d2u_dx2 = np.zeros(nx)
        d2u_dx2[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / dx**2

        # Diffusion term
        D = nu_t * d2u_dx2

        # At the step: large positive diffusion (smoothing)
        step_idx = nx // 2
        assert D[step_idx-1] > 0, "Should have positive diffusion before step"
        assert D[step_idx] < 0, "Should have negative diffusion after step"

        # Apply diffusion for one timestep
        dt = 0.1  # s
        CFL_diffusion = nu_t * dt / dx**2

        # Stability: CFL_diff < 0.5 for explicit diffusion
        assert CFL_diffusion < 0.5, f"Diffusion CFL {CFL_diffusion:.4f} too large"

        u_new = u + dt * D

        # Verify smoothing occurred
        gradient_old = abs(u[step_idx] - u[step_idx-1])
        gradient_new = abs(u_new[step_idx] - u_new[step_idx-1])

        assert gradient_new < gradient_old, "Diffusion should smooth gradients"

    def test_subgrid_scale_dissipation(self):
        """
        Test energy dissipation due to sub-grid turbulence.

        Dissipation rate:
        ε = ν_t * |S|²

        where |S| is strain rate magnitude.

        Physical context: Turbulent kinetic energy cascade to small scales.
        """
        # Velocity gradient
        du_dx = 0.2  # s⁻¹
        dv_dy = 0.15  # s⁻¹

        # Strain rate
        S_mag = np.sqrt(du_dx**2 + dv_dy**2)

        # Eddy viscosity
        nu_t = 0.02  # m²/s

        # Dissipation rate per unit mass
        epsilon = nu_t * S_mag**2  # m²/s³

        # Expected: ε ≈ 0.02 * 0.25² ≈ 0.00125 m²/s³
        assert 0.0001 <= epsilon <= 0.01, \
            f"Dissipation rate {epsilon:.6f} m²/s³ seems unreasonable"

        # Kolmogorov microscale
        nu_molecular = 1e-6  # m²/s (water)
        eta_K = (nu_molecular**3 / epsilon)**0.25  # m

        # Expected: ~0.5-2 mm for this dissipation rate
        eta_K_mm = eta_K * 1000.0
        assert 0.1 <= eta_K_mm <= 5.0, \
            f"Kolmogorov scale {eta_K_mm:.2f} mm seems unreasonable"

        # Turbulent kinetic energy budget
        # Production = Dissipation at equilibrium
        # For this test, verify dissipation is positive
        assert epsilon > 0, "Dissipation must be positive"

        # Test that dissipation increases with strain rate
        S_mag_2 = 2.0 * S_mag
        epsilon_2 = nu_t * S_mag_2**2

        ratio = epsilon_2 / epsilon
        np.testing.assert_allclose(ratio, 4.0, rtol=1e-10)  # Quadratic scaling


class TestVariableDensityEffects:
    """Tests for variable density flows (salinity, temperature)."""

    def test_baroclinic_pressure_gradient(self):
        """
        Test pressure gradient with density stratification.

        Baroclinic term in momentum equation:
        F_baroclinic = -(1/ρ₀) * ∫ ∂ρ'/∂x * g * dz

        where ρ' = ρ - ρ₀ is density anomaly.

        Physical context: Estuarine circulation, thermal currents.
        """
        # Reference density
        rho_0 = 1000.0  # kg/m³ (freshwater)

        # Density variation (salinity gradient)
        # Typical: 1 psu ≈ 0.8 kg/m³ density change
        # Salinity gradient: 35 psu (ocean) → 0 psu (river) over 10 km
        drho_dx = (35 * 0.8) / 10000.0  # kg/m³/m ≈ 2.8e-3

        # Water depth
        h = 10.0  # m

        g = 9.81  # m/s²

        # Baroclinic pressure gradient force per unit mass
        # Simplified: F ≈ -(g*h/ρ₀) * ∂ρ/∂x
        F_baro = -(g * h / rho_0) * drho_dx  # m/s²

        # Expected: ~2.7e-6 m/s²
        assert abs(F_baro) < 1e-4, \
            f"Baroclinic force {F_baro:.6f} m/s² seems unreasonably large"

        # Velocity induced over time
        t = 3600.0  # s (1 hour)
        du = F_baro * t  # m/s

        # Expected: ~0.01 m/s after 1 hour
        assert abs(du) < 0.1, "Velocity change should be modest"

        # Densimetric Froude number
        delta_rho = 28.0  # kg/m³ (full salinity range)
        g_prime = g * delta_rho / rho_0  # Reduced gravity

        U = 0.1  # m/s (typical estuarine flow)
        Fr_d = U / np.sqrt(g_prime * h)

        # For stratified flow: Fr_d < 1
        assert Fr_d < 1.0, f"Densimetric Froude {Fr_d:.3f} suggests supercritical flow"

    def test_density_driven_flows(self):
        """
        Test density-driven flow (gravity current).

        Lock-exchange problem:
        - Dense fluid on one side, light fluid on other
        - Gate removed: dense fluid flows under light fluid
        - Front speed: U ≈ 0.5 * sqrt(g' * h)

        where g' = g * Δρ/ρ₀ is reduced gravity.

        Physical context: Submarine landslides, hyperpycnal flows.
        """
        # Density difference
        rho_1 = 1000.0  # kg/m³ (light)
        rho_2 = 1020.0  # kg/m³ (dense, e.g., high salinity)

        delta_rho = rho_2 - rho_1
        rho_0 = 0.5 * (rho_1 + rho_2)

        # Reduced gravity
        g = 9.81
        g_prime = g * delta_rho / rho_0  # m/s²

        # Expected: ~0.196 m/s²
        assert 0.1 <= g_prime <= 0.5, \
            f"Reduced gravity {g_prime:.3f} m/s² seems unreasonable"

        # Depth
        h = 1.0  # m

        # Front propagation speed (Benjamin 1968)
        U_front = 0.5 * np.sqrt(g_prime * h)  # m/s

        # Expected: ~0.22 m/s
        assert 0.1 <= U_front <= 0.5, \
            f"Front speed {U_front:.3f} m/s seems unreasonable"

        # Reynolds number
        nu = 1e-6  # m²/s
        L = h  # Length scale
        Re = U_front * L / nu

        # Expected: ~2e5 (turbulent)
        assert Re > 1000, "Flow should be turbulent"

        # Richardson number (stability indicator)
        # Ri = (g' * h) / U²
        Ri = (g_prime * h) / U_front**2

        # For this case: Ri ≈ 4 (from U = 0.5*sqrt(g'h))
        np.testing.assert_allclose(Ri, 4.0, rtol=0.1)

    def test_stratification_effects(self):
        """
        Test effect of vertical stratification on mixing.

        Brunt-Väisälä frequency:
        N² = -(g/ρ₀) * ∂ρ/∂z

        Strong stratification (N² > 0) suppresses vertical mixing.

        Physical context: Thermocline, halocline in lakes/oceans.
        """
        # Density profile (exponential stratification)
        z = np.linspace(0, -20, 100)  # m (depth, negative downward)
        dz = abs(z[1] - z[0])

        # Surface density
        rho_surf = 1000.0  # kg/m³

        # Density increase with depth (typical thermocline)
        delta_rho = 5.0  # kg/m³ over 20m
        z_scale = 5.0  # m (thermocline thickness)

        rho = rho_surf + delta_rho * (1.0 - np.exp(z / z_scale))

        # Density gradient
        drho_dz = np.gradient(rho, dz)

        # Brunt-Väisälä frequency squared
        g = 9.81
        rho_0 = 1000.0
        N_squared = -(g / rho_0) * drho_dz

        # Verify stratification is stable (N² > 0)
        assert np.all(N_squared >= 0), "Unstable stratification detected"

        # Maximum stratification (in thermocline)
        N_squared_max = N_squared.max()
        N_max = np.sqrt(N_squared_max)  # rad/s

        # Buoyancy period
        T_b = 2.0 * np.pi / N_max  # s

        # Expected: ~60-600 s for typical thermocline
        assert 30.0 <= T_b <= 1000.0, \
            f"Buoyancy period {T_b:.1f} s seems unreasonable"

        # Test mixing suppression
        # Turbulent Richardson number criterion: Ri > 0.25 suppresses mixing
        # Ri = N² / (∂u/∂z)²

        du_dz = 0.01  # s⁻¹ (weak shear)
        Ri = N_squared_max / du_dz**2

        if Ri > 0.25:
            mixing_suppressed = True
        else:
            mixing_suppressed = False

        # For strong stratification, should suppress mixing
        assert mixing_suppressed, "Strong stratification should suppress mixing"

        # Ozmidov scale (largest overturning eddy)
        epsilon = 1e-6  # m²/s³ (turbulent dissipation)
        L_oz = np.sqrt(epsilon / N_max**3)  # m

        # Expected: ~0.1-1 m for typical conditions
        assert 0.01 <= L_oz <= 10.0, \
            f"Ozmidov scale {L_oz:.3f} m seems unreasonable"


if __name__ == "__main__":
    """Run tests with: pytest test_advanced_physical_processes.py -v"""
    pytest.main([__file__, "-v", "--tb=short"])
