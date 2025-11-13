"""
Boundary Condition Scenario Tests

This module tests various boundary condition scenarios that are common
in real-world applications:
- Channel flow with walls
- Open channel with outflow
- Periodic channel (river meanders)
- Mixed boundary conditions
- Time-varying boundary conditions
"""

import pytest
import numpy as np
from typing import Dict
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from preprocessing.mesh_generation import MeshGenerator, DomainParams
from preprocessing.geometry import GeometryGenerator
from preprocessing.boundary_conditions import BoundaryConditionManager
from preprocessing.initial_conditions import InitialConditionManager


class TestChannelFlowScenarios:
    """Test various channel flow scenarios"""

    def test_straight_channel_uniform_flow(self):
        """
        Test Case: Uniform flow in straight rectangular channel

        Expected: Constant depth and velocity after steady state
        """
        # Create channel domain
        domain = DomainParams(xmin=0.0, xmax=1000.0, ymin=0.0, ymax=50.0)
        mesh_gen = MeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=200, ny=10)

        # Flat bed
        z = np.zeros((mesh.nx, mesh.ny))

        # Initial conditions: uniform flow
        ic_manager = InitialConditionManager(mesh)
        h_init = np.full((mesh.nx, mesh.ny), 5.0)  # 5m depth
        u_init = np.full((mesh.nx, mesh.ny), 2.0)  # 2 m/s velocity
        v_init = np.zeros((mesh.nx, mesh.ny))

        ic_manager.set_depth_array(h_init)
        ic_manager.set_velocity_array(u_init, v_init)

        # Boundary conditions
        bc_manager = BoundaryConditionManager(domain)
        # Left: inflow, Right: outflow, Top/Bottom: walls

        # Verify setup
        assert ic_manager.depth.mean() == pytest.approx(5.0, rel=1e-6)
        assert ic_manager.velocity_x.mean() == pytest.approx(2.0, rel=1e-6)

        # After simulation, should maintain uniform flow
        # (Requires GPU solver to verify)

    def test_channel_with_contraction(self):
        """
        Test Case: Flow through channel contraction

        Expected: Velocity increases through contraction
        """
        # Create channel domain
        domain = DomainParams(xmin=0.0, xmax=100.0, ymin=0.0, ymax=20.0)
        mesh_gen = MeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=200, ny=40)

        # Create contraction geometry
        x_centers = mesh.x
        y_centers = mesh.y

        # Mask cells outside contraction
        in_channel = np.ones((mesh.nx, mesh.ny), dtype=bool)
        for i in range(mesh.nx):
            x = x_centers[i]
            if 40 < x < 60:
                # Contraction region
                width = 20.0 - 0.5 * (x - 50.0)**2 / 10.0  # Parabolic contraction
                for j in range(mesh.ny):
                    y = y_centers[j]
                    if y < (10.0 - width/2) or y > (10.0 + width/2):
                        in_channel[i, j] = False

        # Initial conditions
        ic_manager = InitialConditionManager(mesh)
        h_init = np.full((mesh.nx, mesh.ny), 3.0)
        u_init = np.full((mesh.nx, mesh.ny), 1.0)
        v_init = np.zeros((mesh.nx, mesh.ny))

        # Set depth to zero outside channel
        h_init[~in_channel] = 0.0

        ic_manager.set_depth_array(h_init)
        ic_manager.set_velocity_array(u_init, v_init)

        # Verify setup
        assert h_init[in_channel].mean() == pytest.approx(3.0, rel=1e-6)

        # After simulation:
        # - Velocity should increase in contraction (continuity)
        # - Depth may decrease in contraction (energy)

    def test_channel_with_expansion(self):
        """
        Test Case: Flow through channel expansion

        Expected: Velocity decreases, possible recirculation
        """
        # Create channel domain
        domain = DomainParams(xmin=0.0, xmax=100.0, ymin=0.0, ymax=30.0)
        mesh_gen = MeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=200, ny=60)

        # Create expansion geometry
        x_centers = mesh.x
        y_centers = mesh.y

        # Mask cells outside channel
        in_channel = np.ones((mesh.nx, mesh.ny), dtype=bool)
        for i in range(mesh.nx):
            x = x_centers[i]
            if x < 50:
                # Narrow section (10m wide)
                for j in range(mesh.ny):
                    y = y_centers[j]
                    if y < 10.0 or y > 20.0:
                        in_channel[i, j] = False
            # After x=50, full width (30m)

        # Initial conditions: narrow section filled
        ic_manager = InitialConditionManager(mesh)
        h_init = np.zeros((mesh.nx, mesh.ny))
        u_init = np.zeros((mesh.nx, mesh.ny))
        v_init = np.zeros((mesh.nx, mesh.ny))

        h_init[in_channel & (x_centers[:, np.newaxis] < 50)] = 2.0
        u_init[in_channel & (x_centers[:, np.newaxis] < 50)] = 3.0

        ic_manager.set_depth_array(h_init)
        ic_manager.set_velocity_array(u_init, v_init)

        # After simulation:
        # - Flow expands into wider section
        # - Velocity decreases
        # - Possible recirculation zones at expansion corners


class TestOpenBoundaryScenarios:
    """Test open boundary conditions"""

    def test_outflow_boundary_subcritical(self):
        """
        Test Case: Subcritical outflow boundary

        Expected: Information propagates upstream
        """
        # Create domain
        domain = DomainParams(xmin=0.0, xmax=100.0, ymin=0.0, ymax=20.0)
        mesh_gen = MeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=100, ny=20)

        # Flat bed
        z = np.zeros((mesh.nx, mesh.ny))

        # Initial conditions: subcritical flow (Fr < 1)
        ic_manager = InitialConditionManager(mesh)
        g = 9.81
        h = 5.0  # 5m depth
        u = 3.0  # 3 m/s
        Fr = u / np.sqrt(g * h)  # Should be < 1

        assert Fr < 1.0, f"Flow should be subcritical, Fr={Fr:.3f}"

        h_init = np.full((mesh.nx, mesh.ny), h)
        u_init = np.full((mesh.nx, mesh.ny), u)
        v_init = np.zeros((mesh.nx, mesh.ny))

        ic_manager.set_depth_array(h_init)
        ic_manager.set_velocity_array(u_init, v_init)

        # After simulation with outflow BC:
        # - Waves can propagate upstream
        # - Outflow BC allows information to exit

    def test_outflow_boundary_supercritical(self):
        """
        Test Case: Supercritical outflow boundary

        Expected: No upstream influence
        """
        # Create domain
        domain = DomainParams(xmin=0.0, xmax=100.0, ymin=0.0, ymax=20.0)
        mesh_gen = MeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=100, ny=20)

        # Steep slope for supercritical flow
        x_centers = mesh.x
        z = np.zeros((mesh.nx, mesh.ny))
        for i in range(mesh.nx):
            z[i, :] = 10.0 - 0.05 * x_centers[i]  # 5% slope

        # Initial conditions: supercritical flow (Fr > 1)
        ic_manager = InitialConditionManager(mesh)
        g = 9.81
        h = 2.0  # 2m depth
        u = 8.0  # 8 m/s
        Fr = u / np.sqrt(g * h)  # Should be > 1

        assert Fr > 1.0, f"Flow should be supercritical, Fr={Fr:.3f}"

        h_init = np.full((mesh.nx, mesh.ny), h)
        u_init = np.full((mesh.nx, mesh.ny), u)
        v_init = np.zeros((mesh.nx, mesh.ny))

        ic_manager.set_depth_array(h_init)
        ic_manager.set_velocity_array(u_init, v_init)

        # After simulation with outflow BC:
        # - No upstream propagation
        # - Disturbances at outflow don't affect interior

    def test_critical_depth_outflow(self):
        """
        Test Case: Critical depth outflow boundary

        Expected: Fr = 1 at boundary
        """
        # Create domain
        domain = DomainParams(xmin=0.0, xmax=100.0, ymin=0.0, ymax=20.0)
        mesh_gen = MeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=100, ny=20)

        # Initial conditions
        ic_manager = InitialConditionManager(mesh)
        h_init = np.full((mesh.nx, mesh.ny), 3.0)
        u_init = np.full((mesh.nx, mesh.ny), 4.0)
        v_init = np.zeros((mesh.nx, mesh.ny))

        ic_manager.set_depth_array(h_init)
        ic_manager.set_velocity_array(u_init, v_init)

        # After simulation with critical depth BC:
        # - Critical depth enforced at boundary
        # - h_crit = (Q²/g)^(1/3)


class TestPeriodicBoundaryScenarios:
    """Test periodic boundary conditions"""

    def test_periodic_wave_propagation(self):
        """
        Test Case: Wave propagation with periodic BC

        Expected: Waves re-enter domain from opposite side
        """
        # Create square domain
        domain = DomainParams(xmin=0.0, xmax=100.0, ymin=0.0, ymax=100.0)
        mesh_gen = MeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=100, ny=100)

        # Flat bed
        z = np.zeros((mesh.nx, mesh.ny))

        # Initial conditions: Gaussian disturbance
        ic_manager = InitialConditionManager(mesh)
        x_centers = mesh.x
        y_centers = mesh.y

        h_init = np.ones((mesh.nx, mesh.ny)) * 5.0
        for i in range(mesh.nx):
            for j in range(mesh.ny):
                x = x_centers[i]
                y = y_centers[j]
                r = np.sqrt((x - 50)**2 + (y - 50)**2)
                h_init[i, j] += 1.0 * np.exp(-r**2 / 100.0)

        u_init = np.zeros((mesh.nx, mesh.ny))
        v_init = np.zeros((mesh.nx, mesh.ny))

        ic_manager.set_depth_array(h_init)
        ic_manager.set_velocity_array(u_init, v_init)

        # After simulation with periodic BC:
        # - Waves exit one boundary and re-enter from opposite
        # - Total mass conserved
        # - No reflections at boundaries

    def test_periodic_vortex(self):
        """
        Test Case: Vortex with periodic BC

        Expected: Vortex persists indefinitely
        """
        # Create square domain
        domain = DomainParams(xmin=0.0, xmax=100.0, ymin=0.0, ymax=100.0)
        mesh_gen = MeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=100, ny=100)

        # Flat bed
        z = np.zeros((mesh.nx, mesh.ny))

        # Initial conditions: Vortex velocity field
        ic_manager = InitialConditionManager(mesh)
        x_centers = mesh.x
        y_centers = mesh.y

        h_init = np.full((mesh.nx, mesh.ny), 5.0)
        u_init = np.zeros((mesh.nx, mesh.ny))
        v_init = np.zeros((mesh.nx, mesh.ny))

        # Create vortex centered at (50, 50)
        for i in range(mesh.nx):
            for j in range(mesh.ny):
                x = x_centers[i] - 50.0
                y = y_centers[j] - 50.0
                r = np.sqrt(x**2 + y**2)
                if r > 0.1:
                    # Tangential velocity: v_theta = Gamma / (2*pi*r)
                    Gamma = 100.0  # Circulation
                    v_theta = Gamma / (2 * np.pi * r)
                    # Convert to Cartesian
                    u_init[i, j] = -v_theta * y / r
                    v_init[i, j] = v_theta * x / r

        ic_manager.set_depth_array(h_init)
        ic_manager.set_velocity_array(u_init, v_init)

        # After simulation with periodic BC:
        # - Vortex should persist (in frictionless case)
        # - Or decay gradually (with friction)


class TestMixedBoundaryScenarios:
    """Test scenarios with mixed boundary conditions"""

    def test_coastal_setup(self):
        """
        Test Case: Coastal setup with land boundaries

        Expected: Proper handling of complex geometry
        """
        # Create coastal domain
        domain = DomainParams(xmin=0.0, xmax=200.0, ymin=0.0, ymax=200.0)
        mesh_gen = MeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=100, ny=100)

        # Bathymetry: sloping beach
        x_centers = mesh.x
        y_centers = mesh.y
        z = np.zeros((mesh.nx, mesh.ny))

        for i in range(mesh.nx):
            x = x_centers[i]
            # Sloping beach: z = 0.05 * (x - 100) for x > 100
            if x > 100:
                z[i, :] = 0.05 * (x - 100)

        # Initial conditions: wave approaching shore
        ic_manager = InitialConditionManager(mesh)
        h_init = np.zeros((mesh.nx, mesh.ny))
        u_init = np.zeros((mesh.nx, mesh.ny))
        v_init = np.zeros((mesh.nx, mesh.ny))

        for i in range(mesh.nx):
            x = x_centers[i]
            if x < 100:
                # Deep water: 10m depth
                h_init[i, :] = 10.0 - z[i, :]
                # Wave: sinusoidal perturbation
                for j in range(mesh.ny):
                    y = y_centers[j]
                    h_init[i, j] += 0.5 * np.sin(2 * np.pi * y / 50.0)

        h_init = np.maximum(h_init, 0.0)  # Ensure non-negative

        ic_manager.set_depth_array(h_init)
        ic_manager.set_velocity_array(u_init, v_init)

        # Boundary conditions:
        # - Left: inflow (ocean)
        # - Right: wall (land)
        # - Top/Bottom: periodic or open

        # After simulation:
        # - Waves propagate toward shore
        # - Wave shoaling and breaking
        # - Runup on beach

    def test_river_junction(self):
        """
        Test Case: River junction (Y-shape)

        Expected: Flow splitting at junction
        """
        # Create junction domain
        domain = DomainParams(xmin=0.0, xmax=100.0, ymin=0.0, ymax=100.0)
        mesh_gen = MeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=100, ny=100)

        # Create Y-shaped channel
        x_centers = mesh.x
        y_centers = mesh.y
        in_channel = np.zeros((mesh.nx, mesh.ny), dtype=bool)

        for i in range(mesh.nx):
            x = x_centers[i]
            for j in range(mesh.ny):
                y = y_centers[j]

                # Main stem (x < 50, 40 < y < 60)
                if x < 50 and 40 < y < 60:
                    in_channel[i, j] = True

                # Upper branch (x >= 50, y > 50)
                elif x >= 50 and 50 < y < 70:
                    # Diagonal branch
                    if y < 50 + 0.4 * (x - 50) + 10:
                        in_channel[i, j] = True

                # Lower branch (x >= 50, y < 50)
                elif x >= 50 and 30 < y < 50:
                    # Diagonal branch
                    if y > 50 - 0.4 * (x - 50) - 10:
                        in_channel[i, j] = True

        # Initial conditions
        ic_manager = InitialConditionManager(mesh)
        h_init = np.zeros((mesh.nx, mesh.ny))
        u_init = np.zeros((mesh.nx, mesh.ny))
        v_init = np.zeros((mesh.nx, mesh.ny))

        # Fill channels
        h_init[in_channel] = 3.0
        u_init[in_channel & (x_centers[:, np.newaxis] < 50)] = 2.0

        ic_manager.set_depth_array(h_init)
        ic_manager.set_velocity_array(u_init, v_init)

        # After simulation:
        # - Flow splits at junction
        # - Part goes to upper branch, part to lower branch
        # - Split ratio depends on geometry and depths


class TestTimeVaryingBoundaries:
    """Test time-varying boundary conditions"""

    def test_tidal_boundary(self):
        """
        Test Case: Tidal boundary condition

        Expected: Sinusoidal water level oscillation
        """
        # Create coastal domain
        domain = DomainParams(xmin=0.0, xmax=1000.0, ymin=0.0, ymax=100.0)
        mesh_gen = MeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=200, ny=20)

        # Flat bed at mean sea level
        z = np.zeros((mesh.nx, mesh.ny))

        # Initial conditions: mean sea level
        ic_manager = InitialConditionManager(mesh)
        h_init = np.full((mesh.nx, mesh.ny), 5.0)
        u_init = np.zeros((mesh.nx, mesh.ny))
        v_init = np.zeros((mesh.nx, mesh.ny))

        ic_manager.set_depth_array(h_init)
        ic_manager.set_velocity_array(u_init, v_init)

        # Time-varying boundary condition at left:
        # h(t) = h_mean + A * sin(2*pi*t/T)
        # where T = 12.42 hours (M2 tidal period)
        # A = 1m (tidal amplitude)

        # After simulation:
        # - Water level oscillates with tidal period
        # - Tidal wave propagates inland
        # - Phase lag increases with distance

    def test_flood_hydrograph(self):
        """
        Test Case: Flood hydrograph boundary condition

        Expected: Flood wave propagation
        """
        # Create river domain
        domain = DomainParams(xmin=0.0, xmax=10000.0, ymin=0.0, ymax=500.0)
        mesh_gen = MeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=200, ny=10)

        # Sloping river bed
        x_centers = mesh.x
        z = np.zeros((mesh.nx, mesh.ny))
        for i in range(mesh.nx):
            z[i, :] = 100.0 - 0.001 * x_centers[i]  # 0.1% slope

        # Initial conditions: low flow
        ic_manager = InitialConditionManager(mesh)
        h_init = np.full((mesh.nx, mesh.ny), 2.0)
        u_init = np.full((mesh.nx, mesh.ny), 0.5)
        v_init = np.zeros((mesh.nx, mesh.ny))

        ic_manager.set_depth_array(h_init)
        ic_manager.set_velocity_array(u_init, v_init)

        # Time-varying boundary condition at upstream:
        # Q(t) = Q_base + Q_peak * exp(-(t-t_peak)²/σ²)
        # Gaussian flood hydrograph

        # After simulation:
        # - Flood wave propagates downstream
        # - Peak attenuates with distance
        # - Timing of peak arrival varies with location


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
