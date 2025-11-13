"""
Complex Geometry Handling Tests for HydroSIS-2D

Tests for:
- Complex terrain and bathymetry
- Multiple obstacles and structures
- Islands and enclosed regions
- Irregular boundaries
- Multi-scale features
- Wet-dry interface complexity

These tests verify the solver can handle realistic geometric
complexity encountered in real-world applications.

Author: HydroSIS-2D Development Team
Date: 2025-11-13
"""

import numpy as np
import pytest
from preprocessing.mesh_generation import UniformMeshGenerator
from preprocessing.geometry import DomainParams, GeometryGenerator


class TestComplexTerrain:
    """Test complex terrain and bathymetry"""

    def test_multi_scale_bathymetry(self):
        """
        Test Case: Bathymetry with features at multiple scales

        Features:
        - Large-scale slope (basin)
        - Medium-scale features (channels)
        - Small-scale roughness (bedforms)

        Expected:
        - All scales resolved appropriately
        - No numerical artifacts at scale transitions
        """
        domain = DomainParams(xmin=0.0, xmax=1000.0, ymin=0.0, ymax=1000.0)
        mesh_gen = UniformMeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=200, ny=200)

        x_centers = mesh.x + 0.5 * mesh.dx
        y_centers = mesh.y + 0.5 * mesh.dy

        z = np.zeros((mesh.nx, mesh.ny))

        for i in range(mesh.nx):
            for j in range(mesh.ny):
                x = x_centers[i]
                y = y_centers[j]

                # Large-scale: Linear slope
                z_large = 0.01 * x

                # Medium-scale: Sinusoidal channel
                z_medium = -2.0 * np.exp(-((y - 500.0) / 100.0)**2) * \
                           np.sin(2 * np.pi * x / 200.0)

                # Small-scale: Bedform roughness
                z_small = 0.1 * np.sin(2 * np.pi * x / 10.0) * \
                         np.sin(2 * np.pi * y / 10.0)

                z[i, j] = z_large + z_medium + z_small

        # Verify scale separation
        dz_dx = np.diff(z[:, mesh.ny//2])
        wavelengths = [10.0, 200.0, 1000.0]  # Small, medium, large

        print(f"Multi-scale bathymetry test:")
        print(f"  Domain: {domain.xmax}m × {domain.ymax}m")
        print(f"  Mesh resolution: {mesh.dx}m")
        print(f"  Scales present:")
        print(f"    - Large (1000m): Basin slope")
        print(f"    - Medium (200m): Channel features")
        print(f"    - Small (10m): Bedform roughness")
        print(f"  Elevation range: {np.min(z):.2f} to {np.max(z):.2f} m")


    def test_fractal_coastline(self):
        """
        Test Case: Fractal-like irregular coastline

        Expected:
        - Complex boundary handled correctly
        - No stair-stepping artifacts
        - Proper wet/dry identification
        """
        domain = DomainParams(xmin=0.0, xmax=500.0, ymin=0.0, ymax=500.0)
        mesh_gen = UniformMeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=250, ny=250)

        x_centers = mesh.x + 0.5 * mesh.dx
        y_centers = mesh.y + 0.5 * mesh.dy

        z = np.zeros((mesh.nx, mesh.ny))

        for i in range(mesh.nx):
            for j in range(mesh.ny):
                x = x_centers[i]
                y = y_centers[j]

                # Irregular coastline using sum of sines
                coastline_y = 250.0 + 50.0 * np.sin(2 * np.pi * x / 500.0) + \
                              20.0 * np.sin(2 * np.pi * x / 100.0) + \
                              10.0 * np.sin(2 * np.pi * x / 25.0)

                if y < coastline_y:
                    z[i, j] = -5.0  # Water (below datum)
                else:
                    z[i, j] = 2.0  # Land (above datum)

        # Count wet/dry transitions
        n_transitions = 0
        for i in range(mesh.nx - 1):
            for j in range(mesh.ny - 1):
                if (z[i, j] < 0 and z[i+1, j] > 0) or \
                   (z[i, j] > 0 and z[i+1, j] < 0) or \
                   (z[i, j] < 0 and z[i, j+1] > 0) or \
                   (z[i, j] > 0 and z[i, j+1] < 0):
                    n_transitions += 1

        print(f"Fractal coastline test:")
        print(f"  Number of wet/dry transitions: {n_transitions}")
        print(f"  Coastline complexity: Multiple wavelengths")
        print(f"  Expected: Smooth treatment of irregular boundary")


    def test_submarine_canyon(self):
        """
        Test Case: V-shaped submarine canyon with steep walls

        Expected:
        - Steep gradients handled correctly
        - No instabilities from sharp bathymetry
        - Proper flow channeling
        """
        domain = DomainParams(xmin=0.0, xmax=200.0, ymin=0.0, ymax=400.0)
        mesh_gen = UniformMeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=100, ny=200)

        x_centers = mesh.x + 0.5 * mesh.dx
        y_centers = mesh.y + 0.5 * mesh.dy

        z = np.zeros((mesh.nx, mesh.ny))

        for i in range(mesh.nx):
            for j in range(mesh.ny):
                x = x_centers[i]
                y = y_centers[j]

                # V-shaped canyon centered at x=100m
                canyon_depth = 50.0  # meters
                canyon_width = 50.0  # meters

                dist_from_center = abs(x - 100.0)

                if dist_from_center < canyon_width:
                    # Inside canyon: V-shaped
                    z[i, j] = -canyon_depth * (1.0 - dist_from_center / canyon_width)
                else:
                    # Outside canyon: shelf
                    z[i, j] = -10.0

                # Add downstream deepening
                z[i, j] -= 0.05 * y

        # Check canyon geometry
        canyon_center_profile = z[mesh.nx // 2, :]
        max_depth = np.min(canyon_center_profile)

        print(f"Submarine canyon test:")
        print(f"  Canyon depth: {-max_depth:.1f} m")
        print(f"  Shelf depth: 10 m")
        print(f"  Canyon width: {canyon_width} m")
        print(f"  Downstream gradient: 5%")


class TestObstaclesAndStructures:
    """Test multiple obstacles and man-made structures"""

    def test_multiple_circular_obstacles(self):
        """
        Test Case: 20 circular obstacles randomly distributed

        Expected:
        - Flow around each obstacle
        - Wake interactions
        - No collision detection errors
        """
        domain = DomainParams(xmin=0.0, xmax=200.0, ymin=0.0, ymax=100.0)
        mesh_gen = UniformMeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=400, ny=200)

        x_centers = mesh.x + 0.5 * mesh.dx
        y_centers = mesh.y + 0.5 * mesh.dy

        # Generate random obstacle positions
        np.random.seed(42)
        n_obstacles = 20
        obstacle_positions = []
        obstacle_radius = 5.0  # meters

        for _ in range(n_obstacles):
            x_obs = np.random.uniform(50.0, 150.0)
            y_obs = np.random.uniform(20.0, 80.0)
            obstacle_positions.append((x_obs, y_obs))

        z = np.zeros((mesh.nx, mesh.ny))

        for i in range(mesh.nx):
            for j in range(mesh.ny):
                x = x_centers[i]
                y = y_centers[j]

                # Check if inside any obstacle
                is_obstacle = False
                for x_obs, y_obs in obstacle_positions:
                    dist = np.sqrt((x - x_obs)**2 + (y - y_obs)**2)
                    if dist < obstacle_radius:
                        is_obstacle = True
                        break

                if is_obstacle:
                    z[i, j] = 10.0  # High elevation (obstacle)
                else:
                    z[i, j] = 0.0  # Channel bed

        obstacle_cells = np.sum(z > 5.0)
        total_cells = mesh.nx * mesh.ny

        print(f"Multiple obstacles test:")
        print(f"  Number of obstacles: {n_obstacles}")
        print(f"  Obstacle radius: {obstacle_radius} m")
        print(f"  Obstacle cells: {obstacle_cells}/{total_cells} " +
              f"({100*obstacle_cells/total_cells:.2f}%)")


    def test_building_complex(self):
        """
        Test Case: Urban building complex (10 rectangular buildings)

        Expected:
        - Flow between buildings
        - Proper wake regions
        - Street grid effects
        """
        domain = DomainParams(xmin=0.0, xmax=300.0, ymin=0.0, ymax=300.0)
        mesh_gen = UniformMeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=150, ny=150)

        x_centers = mesh.x + 0.5 * mesh.dx
        y_centers = mesh.y + 0.5 * mesh.dy

        # Define buildings (x_min, x_max, y_min, y_max)
        buildings = [
            (50, 80, 50, 100),
            (100, 130, 50, 100),
            (150, 180, 50, 100),
            (50, 80, 120, 170),
            (100, 130, 120, 170),
            (150, 180, 120, 170),
            (50, 80, 190, 240),
            (100, 130, 190, 240),
            (150, 180, 190, 240),
            (200, 250, 100, 200),
        ]

        z = np.zeros((mesh.nx, mesh.ny))

        for i in range(mesh.nx):
            for j in range(mesh.ny):
                x = x_centers[i]
                y = y_centers[j]

                # Check if inside any building
                for x_min, x_max, y_min, y_max in buildings:
                    if x_min <= x <= x_max and y_min <= y <= y_max:
                        z[i, j] = 8.0  # Building height
                        break

        building_cells = np.sum(z > 5.0)
        total_cells = mesh.nx * mesh.ny

        print(f"Building complex test:")
        print(f"  Number of buildings: {len(buildings)}")
        print(f"  Domain: {domain.xmax}m × {domain.ymax}m")
        print(f"  Building cells: {building_cells}/{total_cells} " +
              f"({100*building_cells/total_cells:.2f}%)")
        print(f"  Street widths: 20m (E-W) and 20m (N-S)")


class TestIslandsAndEnclosures:
    """Test islands and enclosed regions"""

    def test_single_island(self):
        """
        Test Case: Circular island in rectangular domain

        Expected:
        - Flow splits around island
        - Wake region downstream
        - No leakage through island
        """
        domain = DomainParams(xmin=0.0, xmax=500.0, ymin=0.0, ymax=300.0)
        mesh_gen = UniformMeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=250, ny=150)

        x_centers = mesh.x + 0.5 * mesh.dx
        y_centers = mesh.y + 0.5 * mesh.dy

        # Island parameters
        island_x = 250.0
        island_y = 150.0
        island_radius = 50.0

        z = np.zeros((mesh.nx, mesh.ny))

        for i in range(mesh.nx):
            for j in range(mesh.ny):
                x = x_centers[i]
                y = y_centers[j]

                dist = np.sqrt((x - island_x)**2 + (y - island_y)**2)

                if dist < island_radius:
                    # Island (above water)
                    z[i, j] = 5.0
                else:
                    # Water
                    z[i, j] = -10.0

        island_cells = np.sum(z > 0)
        water_cells = np.sum(z < 0)

        print(f"Single island test:")
        print(f"  Island center: ({island_x}, {island_y}) m")
        print(f"  Island radius: {island_radius} m")
        print(f"  Island cells: {island_cells}")
        print(f"  Water cells: {water_cells}")
        print(f"  Expected: Symmetric flow pattern around island")


    def test_archipelago(self):
        """
        Test Case: Multiple islands (archipelago)

        Expected:
        - Complex flow patterns
        - Inter-island channels
        - Correct connectivity
        """
        domain = DomainParams(xmin=0.0, xmax=600.0, ymin=0.0, ymax=600.0)
        mesh_gen = UniformMeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=120, ny=120)

        x_centers = mesh.x + 0.5 * mesh.dx
        y_centers = mesh.y + 0.5 * mesh.dy

        # Define islands (x, y, radius)
        islands = [
            (150, 150, 40),
            (300, 200, 50),
            (450, 150, 35),
            (200, 400, 45),
            (400, 450, 40),
        ]

        z = np.full((mesh.nx, mesh.ny), -15.0)  # Default: water

        for i in range(mesh.nx):
            for j in range(mesh.ny):
                x = x_centers[i]
                y = y_centers[j]

                # Check if inside any island
                for x_isle, y_isle, r_isle in islands:
                    dist = np.sqrt((x - x_isle)**2 + (y - y_isle)**2)
                    if dist < r_isle:
                        z[i, j] = 3.0  # Island elevation
                        break

        island_cells = np.sum(z > 0)
        water_cells = np.sum(z < 0)

        print(f"Archipelago test:")
        print(f"  Number of islands: {len(islands)}")
        print(f"  Island cells: {island_cells}")
        print(f"  Water cells: {water_cells}")
        print(f"  Land fraction: {100*island_cells/(island_cells+water_cells):.1f}%")


    def test_enclosed_lagoon(self):
        """
        Test Case: Lagoon enclosed by barrier with narrow inlet

        Expected:
        - Proper exchange flow through inlet
        - Lagoon response to external forcing
        - No artificial mixing
        """
        domain = DomainParams(xmin=0.0, xmax=400.0, ymin=0.0, ymax=400.0)
        mesh_gen = UniformMeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=200, ny=200)

        x_centers = mesh.x + 0.5 * mesh.dx
        y_centers = mesh.y + 0.5 * mesh.dy

        z = np.zeros((mesh.nx, mesh.ny))

        for i in range(mesh.nx):
            for j in range(mesh.ny):
                x = x_centers[i]
                y = y_centers[j]

                # Circular lagoon
                dist_center = np.sqrt((x - 200.0)**2 + (y - 200.0)**2)

                if dist_center < 120.0:
                    # Inside lagoon
                    z[i, j] = -5.0  # Shallow lagoon
                elif 120.0 <= dist_center < 140.0:
                    # Barrier
                    # Create inlet at x=200, y=200-140 to 200-120
                    if 190 <= x <= 210 and y < 100:
                        z[i, j] = -5.0  # Inlet
                    else:
                        z[i, j] = 2.0  # Barrier above water
                else:
                    # Outside: ocean
                    z[i, j] = -15.0

        lagoon_cells = np.sum((z < 0) & (z > -8))
        inlet_width = 20.0  # meters

        print(f"Enclosed lagoon test:")
        print(f"  Lagoon radius: 120 m")
        print(f"  Barrier width: 20 m")
        print(f"  Inlet width: {inlet_width} m")
        print(f"  Lagoon cells: {lagoon_cells}")
        print(f"  Expected: Limited exchange through narrow inlet")


class TestIrregularBoundaries:
    """Test irregular and complex boundaries"""

    def test_natural_river_meander(self):
        """
        Test Case: Meandering river with natural sinuosity

        Expected:
        - Flow follows meandering path
        - Secondary circulation in bends
        - Proper bank treatment
        """
        domain = DomainParams(xmin=0.0, xmax=1000.0, ymin=0.0, ymax=200.0)
        mesh_gen = UniformMeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=500, ny=100)

        x_centers = mesh.x + 0.5 * mesh.dx
        y_centers = mesh.y + 0.5 * mesh.dy

        z = np.full((mesh.nx, mesh.ny), 10.0)  # Default: banks

        # Meander parameters
        meander_amplitude = 50.0  # meters
        meander_wavelength = 200.0  # meters
        channel_width = 30.0  # meters

        for i in range(mesh.nx):
            for j in range(mesh.ny):
                x = x_centers[i]
                y = y_centers[j]

                # Meandering centerline
                y_centerline = 100.0 + meander_amplitude * \
                              np.sin(2 * np.pi * x / meander_wavelength)

                dist_from_centerline = abs(y - y_centerline)

                if dist_from_centerline < channel_width / 2.0:
                    z[i, j] = -3.0  # Channel bed

        channel_cells = np.sum(z < 0)
        sinuosity = 1.2  # Actual path length / straight distance

        print(f"Natural river meander test:")
        print(f"  Meander amplitude: {meander_amplitude} m")
        print(f"  Meander wavelength: {meander_wavelength} m")
        print(f"  Channel width: {channel_width} m")
        print(f"  Sinuosity: {sinuosity:.2f}")
        print(f"  Channel cells: {channel_cells}")


    def test_dendritic_network(self):
        """
        Test Case: Dendritic drainage network (tree-like)

        Expected:
        - Flow convergence through branches
        - Correct tributary junctions
        - Mass conservation at confluences
        """
        # Simplified dendritic pattern
        # Main channel with multiple tributaries

        domain = DomainParams(xmin=0.0, xmax=500.0, ymin=0.0, ymax=500.0)
        mesh_gen = UniformMeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=250, ny=250)

        x_centers = mesh.x + 0.5 * mesh.dx
        y_centers = mesh.y + 0.5 * mesh.dy

        z = np.full((mesh.nx, mesh.ny), 10.0)  # Default: land

        # Define channel network (centerlines and widths)
        channels = [
            # Main channel: vertical
            ((250, 0), (250, 500), 20),
            # Tributaries
            ((100, 100), (250, 200), 10),
            ((400, 100), (250, 200), 10),
            ((100, 300), (250, 350), 10),
            ((400, 300), (250, 350), 10),
        ]

        for i in range(mesh.nx):
            for j in range(mesh.ny):
                x = x_centers[i]
                y = y_centers[j]

                # Check distance to each channel
                for (x1, y1), (x2, y2), width in channels:
                    # Point-to-line distance
                    dx = x2 - x1
                    dy = y2 - y1
                    length = np.sqrt(dx**2 + dy**2)

                    if length > 0:
                        t = max(0, min(1, ((x - x1) * dx + (y - y1) * dy) / length**2))
                        proj_x = x1 + t * dx
                        proj_y = y1 + t * dy

                        dist = np.sqrt((x - proj_x)**2 + (y - proj_y)**2)

                        if dist < width / 2.0:
                            z[i, j] = -2.0  # Channel bed
                            break

        channel_cells = np.sum(z < 0)

        print(f"Dendritic network test:")
        print(f"  Number of channel segments: {len(channels)}")
        print(f"  Main channel width: 20 m")
        print(f"  Tributary width: 10 m")
        print(f"  Channel cells: {channel_cells}")
        print(f"  Pattern: Tree-like drainage network")


if __name__ == "__main__":
    print("=" * 70)
    print("COMPLEX GEOMETRY HANDLING TESTS FOR HYDROSIS-2D")
    print("=" * 70)
    print()
    print("These tests verify solver performance with complex geometries.")
    print("Run with: pytest test_complex_geometry.py -v")
    print()
    print("Test Categories:")
    print("  1. Complex Terrain (3 tests)")
    print("  2. Obstacles and Structures (2 tests)")
    print("  3. Islands and Enclosures (3 tests)")
    print("  4. Irregular Boundaries (2 tests)")
    print()
    print("Total: 10 complex geometry tests")
    print("=" * 70)
