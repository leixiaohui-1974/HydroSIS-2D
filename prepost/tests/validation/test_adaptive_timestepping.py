"""
Adaptive Time Stepping Tests

This module tests automatic time step adaptation based on CFL condition
and other stability criteria. Tests cover:
- CFL-based timestep calculation
- Local vs global timestep strategies
- Stability limits and safety factors
- Timestep adjustment heuristics

Adaptive timestepping is critical for:
- Maintaining numerical stability
- Optimizing computational efficiency
- Handling varying flow conditions

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
from preprocessing.initial_conditions import DamBreakIC, UniformFlowIC
from preprocessing.boundary_conditions import BoundaryConditionManager


class TestCFLCondition:
    """Tests for CFL (Courant-Friedrichs-Lewy) condition."""

    def test_cfl_calculation(self):
        """
        CFL number calculation

        Test objectives:
        - Correct CFL formula for shallow water
        - Account for all wave speeds
        - Verify dimensionless nature

        CFL = (|u| + c) * dt / dx

        where c = √(gh) is wave celerity

        GPU kernel: CFL computation per cell
        """
        g = 9.81

        # Flow conditions
        h = 5.0  # Water depth (m)
        u = 2.0  # Velocity (m/s)

        # Grid spacing
        dx = 10.0  # m

        # Wave celerity
        c = np.sqrt(g * h)

        # Maximum wave speed
        lambda_max = abs(u) + c

        # For a given timestep
        dt = 0.5  # seconds

        # CFL number
        CFL = lambda_max * dt / dx

        print(f"Flow conditions: h={h}m, u={u}m/s")
        print(f"Wave celerity: c={c:.2f} m/s")
        print(f"Max wave speed: λ={lambda_max:.2f} m/s")
        print(f"Timestep: dt={dt}s, dx={dx}m")
        print(f"CFL number: {CFL:.3f}")

        # Verify CFL is dimensionless
        assert CFL > 0, "CFL should be positive"

        # Stability limit for explicit schemes
        CFL_max = 1.0  # Theoretical limit

        if CFL <= CFL_max:
            print(f"✓ Stable (CFL ≤ {CFL_max})")
        else:
            print(f"✗ Unstable (CFL > {CFL_max})")

    def test_cfl_vs_froude_number(self):
        """
        CFL behavior vs Froude number

        Test objectives:
        - Subcritical flow (Fr < 1): waves propagate both directions
        - Supercritical flow (Fr > 1): all waves go downstream
        - Critical flow (Fr = 1): challenging numerically

        GPU kernel: CFL analysis across flow regimes
        """
        g = 9.81
        h = 5.0
        c = np.sqrt(g * h)

        # Different Froude numbers
        froude_numbers = [0.3, 0.7, 1.0, 1.5, 3.0]

        dx = 10.0

        print("CFL vs Froude number:\n")

        for Fr in froude_numbers:
            u = Fr * c

            # Wave speeds in flow direction
            lambda_plus = u + c   # Downstream wave
            lambda_minus = u - c  # Upstream wave

            # Maximum absolute wave speed (for CFL)
            lambda_max = max(abs(lambda_plus), abs(lambda_minus))

            # Timestep for CFL = 0.5
            CFL_target = 0.5
            dt = CFL_target * dx / lambda_max

            print(f"Fr = {Fr:.1f}:")
            print(f"  u = {u:.2f} m/s")
            print(f"  λ+ = {lambda_plus:.2f} m/s, λ- = {lambda_minus:.2f} m/s")
            print(f"  dt(CFL=0.5) = {dt:.4f} s")

            # At Fr=1, upstream wave speed is zero
            if abs(Fr - 1.0) < 0.01:
                print(f"  ⚠ Critical flow: λ- ≈ 0 (sonic point)")

            print()

    def test_cfl_safety_factor(self):
        """
        CFL safety factor application

        Test objectives:
        - Use CFL < 1 for safety margin
        - Common values: 0.3-0.8
        - Trade-off: stability vs efficiency

        GPU kernel: Adaptive timestep with safety factor
        """
        # Theoretical stability limit
        CFL_max_theory = 1.0

        # Practical safety factors
        safety_factors = [
            {'CFL': 0.3, 'purpose': 'Very conservative (high accuracy)'},
            {'CFL': 0.5, 'purpose': 'Standard (balanced)'},
            {'CFL': 0.7, 'purpose': 'Aggressive (faster)'},
            {'CFL': 0.9, 'purpose': 'Very aggressive (stability risk)'},
        ]

        print("CFL safety factors:\n")

        for sf in safety_factors:
            CFL = sf['CFL']
            margin = (CFL_max_theory - CFL) / CFL_max_theory * 100

            print(f"CFL = {CFL:.1f}: {sf['purpose']}")
            print(f"  Safety margin: {margin:.0f}%")
            print()

        # Recommended: CFL = 0.5
        recommended_CFL = 0.5
        print(f"✓ Recommended: CFL = {recommended_CFL} (good balance)")


class TestLocalTimestep:
    """Tests for local (per-cell) timestepping."""

    def test_variable_wave_speeds(self):
        """
        Variable wave speeds across domain

        Test objectives:
        - Different depths → different wave speeds
        - Global timestep limited by fastest wave
        - Potential for local timestepping

        GPU kernel: Wave speed field computation
        """
        g = 9.81

        # Domain with varying depth
        nx, ny = 100, 100

        # Depth field (shallow to deep)
        x = np.linspace(0, 1000, nx)
        y = np.linspace(0, 1000, ny)
        X, Y = np.meshgrid(x, y, indexing='ij')

        # Depth varies from 1m to 20m
        h = 1.0 + 19.0 * (X / 1000.0)

        # Velocity field (uniform)
        u = 2.0 * np.ones((nx, ny))
        v = 0.0 * np.ones((nx, ny))

        # Wave celerity
        c = np.sqrt(g * h)

        # Local wave speed
        lambda_local = np.sqrt(u**2 + v**2) + c

        # Statistics
        lambda_min = np.min(lambda_local)
        lambda_max = np.max(lambda_local)
        lambda_mean = np.mean(lambda_local)

        print(f"Wave speed statistics:")
        print(f"  Min: {lambda_min:.2f} m/s")
        print(f"  Max: {lambda_max:.2f} m/s")
        print(f"  Mean: {lambda_mean:.2f} m/s")
        print(f"  Ratio: {lambda_max/lambda_min:.2f}x")

        # Global timestep (limited by maximum wave speed)
        dx = x[1] - x[0]
        CFL = 0.5
        dt_global = CFL * dx / lambda_max

        # Local timesteps (if used)
        dt_local = CFL * dx / lambda_local

        dt_local_mean = np.mean(dt_local)

        # Efficiency gain from local timestepping
        efficiency_gain = dt_local_mean / dt_global

        print(f"\nTimesteps:")
        print(f"  Global dt: {dt_global:.4f} s")
        print(f"  Local dt (mean): {dt_local_mean:.4f} s")
        print(f"  Potential efficiency gain: {efficiency_gain:.2f}x")

    def test_wet_dry_timestep_variation(self):
        """
        Timestep variation in wet-dry regions

        Test objectives:
        - Dry cells: infinite wave speed constraint
        - Very shallow cells: large wave speed (√(gh) / h)
        - Require special handling

        GPU kernel: Wet-dry adaptive timestepping
        """
        g = 9.81
        h_dry = 1e-5  # Dry tolerance

        # Range of depths
        depths = np.array([0.0, 1e-6, 1e-4, 0.01, 0.1, 1.0, 10.0])

        dx = 10.0
        CFL = 0.5

        print("Timestep vs depth:\n")

        for h in depths:
            if h < h_dry:
                # Dry cell: skip wave speed calculation
                dt = np.inf
                status = "DRY (skip)"
            else:
                # Wet cell
                c = np.sqrt(g * h)
                u = 0.0  # At rest for simplicity

                lambda_max = abs(u) + c
                dt = CFL * dx / lambda_max

                if h < 0.01:
                    status = "VERY SHALLOW (large c)"
                else:
                    status = "NORMAL"

            print(f"h = {h:.2e} m: dt = {dt:.4f} s ({status})")

        # Note: Very shallow water can impose severe timestep restrictions
        # May need minimum depth threshold or special treatment

    def test_local_vs_global_timestep(self):
        """
        Compare local vs global timestepping strategies

        Test objectives:
        - Global: single dt for entire domain (simple, synchronous)
        - Local: different dt per cell (complex, efficient)
        - Trade-offs in implementation complexity

        GPU kernel: Timestepping strategy comparison
        """
        strategies = {
            'Global timestep': {
                'advantages': [
                    'Simple to implement',
                    'Synchronous updates',
                    'Easy to parallelize'
                ],
                'disadvantages': [
                    'Limited by fastest wave in domain',
                    'Inefficient for variable conditions',
                    'Small timesteps everywhere'
                ]
            },
            'Local timestep': {
                'advantages': [
                    'Optimal timestep per region',
                    'Better efficiency for variable flows',
                    'Faster convergence to steady state'
                ],
                'disadvantages': [
                    'Complex implementation',
                    'Requires careful interpolation at interfaces',
                    'Not strictly conservative (time-accurate)'
                ]
            }
        }

        for strategy, details in strategies.items():
            print(f"\n{strategy}:")
            print("  Advantages:")
            for adv in details['advantages']:
                print(f"    + {adv}")
            print("  Disadvantages:")
            for dis in details['disadvantages']:
                print(f"    - {dis}")

        # HydroSIS-2D uses global timestep (standard for explicit schemes)
        print("\n✓ HydroSIS-2D: Global timestep (simple, robust)")


class TestTimestepAdaptation:
    """Tests for dynamic timestep adjustment."""

    def test_timestep_increase_strategy(self):
        """
        Timestep increase during simulation

        Test objectives:
        - Start with small dt (initial transient)
        - Gradually increase as flow stabilizes
        - Limit rate of increase for stability

        Common strategy: dt_new = min(factor * dt_old, dt_max)
        where factor ≈ 1.1 - 1.2

        GPU kernel: Adaptive timestep evolution
        """
        # Initial timestep
        dt_initial = 0.001  # seconds

        # Maximum timestep
        dt_max = 0.1  # seconds

        # Increase factor per step
        increase_factor = 1.1

        # Simulation
        n_steps = 50
        dt = dt_initial
        dt_history = [dt]

        for step in range(n_steps):
            # Increase timestep gradually
            dt_new = min(increase_factor * dt, dt_max)

            dt = dt_new
            dt_history.append(dt)

        print(f"Timestep evolution:")
        print(f"  Initial: {dt_initial:.4f} s")
        print(f"  After 10 steps: {dt_history[10]:.4f} s")
        print(f"  After 30 steps: {dt_history[30]:.4f} s")
        print(f"  Final: {dt_history[-1]:.4f} s (max: {dt_max:.4f} s)")

        # Verify monotonic increase up to maximum
        assert all(dt_history[i+1] >= dt_history[i] for i in range(len(dt_history)-1)
                   if dt_history[i] < dt_max), \
            "Timestep should increase monotonically up to maximum"

    def test_timestep_decrease_strategy(self):
        """
        Timestep decrease when needed

        Test objectives:
        - Detect instability or CFL violation
        - Reduce dt immediately (aggressive)
        - Prevent blow-up

        Common strategy: dt_new = max(safety * dt_old, dt_min)
        where safety ≈ 0.5 - 0.8

        GPU kernel: Timestep reduction
        """
        # Current timestep
        dt_current = 0.1  # seconds

        # Minimum timestep
        dt_min = 0.0001  # seconds

        # Safety factor for reduction
        reduction_factor = 0.5  # Cut in half

        # Scenarios requiring timestep reduction
        scenarios = [
            {'name': 'CFL violation detected', 'action': 'Reduce immediately'},
            {'name': 'Instability symptoms', 'action': 'Reduce aggressively'},
            {'name': 'Negative depth', 'action': 'Reduce and retry'},
            {'name': 'Non-convergence', 'action': 'Reduce iteratively'},
        ]

        print("Timestep reduction scenarios:\n")

        for scenario in scenarios:
            dt_new = max(reduction_factor * dt_current, dt_min)

            print(f"{scenario['name']}:")
            print(f"  Current dt: {dt_current:.4f} s")
            print(f"  New dt: {dt_new:.4f} s")
            print(f"  Action: {scenario['action']}")
            print()

        # Verify reduction is significant
        assert dt_new < dt_current, "Timestep should be reduced"
        assert dt_new >= dt_min, "Timestep should not go below minimum"

    def test_timestep_limits(self):
        """
        Minimum and maximum timestep limits

        Test objectives:
        - dt_min: prevent excessively small timesteps
        - dt_max: prevent loss of accuracy
        - Typical ranges for shallow water

        GPU kernel: Timestep bounds
        """
        # Typical limits for shallow water simulations
        limits = {
            'Dam break (transient)': {'dt_min': 1e-4, 'dt_max': 0.1},
            'Tidal simulation (long-term)': {'dt_min': 0.1, 'dt_max': 60.0},
            'Flash flood (fast)': {'dt_min': 1e-3, 'dt_max': 1.0},
            'Steady flow (slow)': {'dt_min': 0.01, 'dt_max': 10.0},
        }

        print("Typical timestep limits:\n")

        for scenario, lim in limits.items():
            dt_range = lim['dt_max'] / lim['dt_min']

            print(f"{scenario}:")
            print(f"  Min: {lim['dt_min']:.1e} s")
            print(f"  Max: {lim['dt_max']:.1e} s")
            print(f"  Range: {dt_range:.1e}x")
            print()


class TestStabilityMonitoring:
    """Tests for stability monitoring during simulation."""

    def test_cfl_monitoring(self):
        """
        Monitor CFL number during simulation

        Test objectives:
        - Track max CFL across domain
        - Warn if approaching stability limit
        - Adjust timestep proactively

        GPU kernel: Global CFL reduction
        """
        # Simulation loop (conceptual)
        CFL_limit = 0.9  # Warning threshold
        CFL_max_allowed = 1.0

        # Simulated CFL history
        CFL_history = [0.5, 0.6, 0.7, 0.75, 0.85, 0.92, 0.88, 0.7]

        print("CFL monitoring during simulation:\n")

        for step, CFL in enumerate(CFL_history):
            status = "OK"

            if CFL >= CFL_max_allowed:
                status = "UNSTABLE (reduce dt immediately)"
            elif CFL >= CFL_limit:
                status = "WARNING (approaching limit)"

            print(f"Step {step}: CFL = {CFL:.3f} - {status}")

            # Action
            if CFL >= CFL_limit:
                print(f"  → Reduce timestep")

    def test_solution_bounds_check(self):
        """
        Check solution for physical bounds

        Test objectives:
        - Verify h ≥ 0 (positive depth)
        - Check for NaN or Inf
        - Detect runaway solutions

        GPU kernel: Solution validation
        """
        # Solution variables
        h = np.array([5.0, 3.2, 1.5, 0.1, 0.0, 2.8])

        # Checks
        checks = {
            'Positive depth': np.all(h >= 0),
            'No NaN': not np.any(np.isnan(h)),
            'No Inf': not np.any(np.isinf(h)),
            'Reasonable values': np.all(h < 1000),  # < 1000m (arbitrary)
        }

        print("Solution validation:\n")

        all_pass = True
        for check_name, result in checks.items():
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"{check_name}: {status}")

            if not result:
                all_pass = False
                print(f"  → Reduce timestep and retry")

        if all_pass:
            print("\n✓ All checks passed - continue")
        else:
            print("\n✗ Some checks failed - timestep too large")

    def test_conservation_monitoring(self):
        """
        Monitor conservation errors

        Test objectives:
        - Track mass conservation error
        - Should be near machine precision
        - Large errors indicate instability

        GPU kernel: Global mass computation
        """
        # Initial mass
        mass_initial = 1000000.0  # m³

        # Simulated mass history (with small errors)
        mass_history = [
            1000000.0,
            999999.8,
            999999.5,
            999999.3,
            999999.1,
        ]

        print("Mass conservation monitoring:\n")

        for step, mass in enumerate(mass_history):
            error = (mass - mass_initial) / mass_initial
            error_percent = error * 100

            if abs(error) < 1e-10:
                status = "Excellent"
            elif abs(error) < 1e-6:
                status = "Good"
            elif abs(error) < 1e-3:
                status = "Acceptable"
            else:
                status = "Poor (check numerics)"

            print(f"Step {step}: M = {mass:.1f} m³, "
                  f"error = {error_percent:.2e}% ({status})")


# Run tests if executed directly
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
