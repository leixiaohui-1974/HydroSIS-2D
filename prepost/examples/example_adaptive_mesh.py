#!/usr/bin/env python3
"""
Example 2: Adaptive Mesh Generation

Demonstrates adaptive mesh refinement with user-defined zones
and terrain-based refinement.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from preprocessing.mesh_generation import (
    AdaptiveMeshGenerator,
    DomainParams,
    MeshQualityChecker,
    MeshIO
)


def create_synthetic_terrain(nx, ny, domain):
    """
    Create synthetic terrain with various features

    Returns terrain elevation array
    """
    # Create coordinate arrays
    x = np.linspace(domain.xmin, domain.xmax, nx)
    y = np.linspace(domain.ymin, domain.ymax, ny)
    X, Y = np.meshgrid(x, y)

    # Base flat terrain
    terrain = np.ones_like(X) * 5.0

    # Add a channel (depression)
    channel_y = domain.ymax / 2
    channel_width = domain.ymax / 10
    channel_depth = 3.0
    terrain -= channel_depth * np.exp(-((Y - channel_y) / channel_width) ** 2)

    # Add a hill
    hill_x = domain.xmax * 0.7
    hill_y = domain.ymax * 0.7
    hill_radius = domain.xmax * 0.15
    hill_height = 10.0
    dist_to_hill = np.sqrt((X - hill_x) ** 2 + (Y - hill_y) ** 2)
    terrain += hill_height * np.exp(-(dist_to_hill / hill_radius) ** 2)

    # Add a slope
    terrain += 0.01 * X

    return terrain


def main():
    print("=" * 60)
    print("Example 2: Adaptive Mesh Generation")
    print("=" * 60)
    print()

    # Define domain (1000 m × 500 m)
    domain = DomainParams(xmin=0.0, xmax=1000.0, ymin=0.0, ymax=500.0)
    print(f"Domain: {domain.length_x} m × {domain.length_y} m")
    print()

    # Create adaptive mesh generator
    generator = AdaptiveMeshGenerator(domain)

    # ==================================================================
    # PART 1: User-defined refinement zones
    # ==================================================================
    print("=" * 60)
    print("Part 1: User-defined Refinement Zones")
    print("=" * 60)
    print()

    # Add refinement zone 1: Around dam location (high priority)
    dam_x, dam_y = 200.0, 250.0
    dam_radius = 100.0
    generator.add_circular_refinement_zone(
        center_x=dam_x,
        center_y=dam_y,
        radius=dam_radius,
        refinement_level=2  # 4x finer
    )
    print(f"✓ Added refinement zone 1: Dam area (level 2)")
    print(f"  Center: ({dam_x}, {dam_y}), Radius: {dam_radius} m")

    # Add refinement zone 2: Downstream region
    generator.add_refinement_zone(
        xmin=300, xmax=600,
        ymin=150, ymax=350,
        refinement_level=1,  # 2x finer
        priority=1
    )
    print(f"✓ Added refinement zone 2: Downstream region (level 1)")

    # Add refinement zone 3: Observation point
    obs_x, obs_y = 800.0, 250.0
    generator.add_circular_refinement_zone(
        center_x=obs_x,
        center_y=obs_y,
        radius=50.0,
        refinement_level=1
    )
    print(f"✓ Added refinement zone 3: Observation point (level 1)")
    print()

    # Generate mesh with zones
    base_nx, base_ny = 50, 25
    print(f"Base mesh resolution: {base_nx} × {base_ny}")

    mesh_zones = generator.generate_multiresolution_mesh(base_nx, base_ny)
    print(f"Final mesh: {mesh_zones.nx} × {mesh_zones.ny} = {mesh_zones.ncells:,} cells")
    print(f"Cell size: dx={mesh_zones.dx:.3f} m, dy={mesh_zones.dy:.3f} m")
    print()

    # Get refinement statistics
    stats = generator.get_refinement_statistics()
    print("Refinement Statistics:")
    print(f"  Base cells: {stats['total_base_cells']}")
    print(f"  Max refinement level: {stats['max_refinement_level']}")
    print(f"  Cells to refine: {stats['cells_to_refine']}")
    for level, info in stats['cells_by_level'].items():
        print(f"  Level {level}: {info['count']} cells ({info['percentage']:.1f}%)")
    print()

    # ==================================================================
    # PART 2: Terrain-based refinement
    # ==================================================================
    print("=" * 60)
    print("Part 2: Terrain-based Refinement")
    print("=" * 60)
    print()

    # Create new generator for terrain-based refinement
    generator2 = AdaptiveMeshGenerator(domain)

    # Generate synthetic terrain
    terrain_nx, terrain_ny = 100, 50
    terrain = create_synthetic_terrain(terrain_nx, terrain_ny, domain)
    print(f"Generated synthetic terrain: {terrain_nx} × {terrain_ny}")
    print(f"Elevation range: [{terrain.min():.2f}, {terrain.max():.2f}] m")
    print()

    # Generate refinement map from terrain gradients
    refinement_map = generator2.generate_refinement_map_from_terrain(
        terrain_data=terrain,
        gradient_threshold=0.05,
        max_refinement_level=2
    )
    print("Generated terrain-based refinement map")

    # Generate mesh
    mesh_terrain = generator2.generate_multiresolution_mesh(
        base_nx=50,
        base_ny=25,
        use_terrain=True,
        terrain_data=terrain
    )
    print(f"Final mesh: {mesh_terrain.nx} × {mesh_terrain.ny} = {mesh_terrain.ncells:,} cells")
    print()

    # ==================================================================
    # PART 3: Combined approach
    # ==================================================================
    print("=" * 60)
    print("Part 3: Combined Approach (Zones + Terrain)")
    print("=" * 60)
    print()

    generator3 = AdaptiveMeshGenerator(domain)

    # Add user-defined zones
    generator3.add_circular_refinement_zone(200, 250, 100, refinement_level=2)

    # Generate zone-based map first
    generator3.generate_refinement_map_from_zones(50, 25)

    # Then overlay terrain-based refinement
    terrain_map = generator3.generate_refinement_map_from_terrain(
        terrain_data=terrain,
        gradient_threshold=0.1,
        max_refinement_level=1
    )

    # Combine maps (take maximum refinement level)
    combined_map = np.maximum(generator3.refinement_map, terrain_map)
    generator3.refinement_map = combined_map

    mesh_combined = generator3.generate_multiresolution_mesh(50, 25)
    print(f"Combined mesh: {mesh_combined.nx} × {mesh_combined.ny} = {mesh_combined.ncells:,} cells")
    print()

    # ==================================================================
    # Quality checks and export
    # ==================================================================
    print("=" * 60)
    print("Mesh Quality Assessment")
    print("=" * 60)

    checker = MeshQualityChecker(mesh_zones)
    checker.print_report()

    # Export meshes
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("Exporting Meshes")
    print("=" * 60)

    # Export mesh with zones
    MeshIO.export_to_ini(mesh_zones, os.path.join(output_dir, "mesh_adaptive_zones.ini"))
    MeshIO.export_to_vtk(mesh_zones, os.path.join(output_dir, "mesh_adaptive_zones.vtk"))
    print("✓ Exported mesh with refinement zones")

    # Export mesh with terrain refinement
    MeshIO.export_to_ini(mesh_terrain, os.path.join(output_dir, "mesh_adaptive_terrain.ini"))
    MeshIO.export_to_vtk(
        mesh_terrain,
        os.path.join(output_dir, "mesh_adaptive_terrain.vtk"),
        data_arrays={'elevation': terrain}
    )
    print("✓ Exported mesh with terrain-based refinement")

    # Export combined mesh
    MeshIO.export_to_ini(mesh_combined, os.path.join(output_dir, "mesh_adaptive_combined.ini"))
    print("✓ Exported combined mesh")

    # ==================================================================
    # Visualization
    # ==================================================================
    print("\n" + "=" * 60)
    print("Visualization")
    print("=" * 60)

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Plot 1: Refinement map for zones
        ax = axes[0, 0]
        generator.visualize_refinement_map()
        plt.sca(ax)
        plt.title("Refinement Map: User-defined Zones")

        # Plot 2: Refinement map for terrain
        ax = axes[0, 1]
        im = ax.imshow(
            refinement_map.T,
            origin='lower',
            extent=[domain.xmin, domain.xmax, domain.ymin, domain.ymax],
            cmap='YlOrRd',
            interpolation='nearest'
        )
        plt.colorbar(im, ax=ax, label='Refinement Level')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title('Refinement Map: Terrain-based')
        ax.set_aspect('equal')

        # Plot 3: Terrain elevation
        ax = axes[1, 0]
        im = ax.imshow(
            terrain.T,
            origin='lower',
            extent=[domain.xmin, domain.xmax, domain.ymin, domain.ymax],
            cmap='terrain',
            interpolation='bilinear'
        )
        plt.colorbar(im, ax=ax, label='Elevation [m]')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title('Synthetic Terrain')
        ax.set_aspect('equal')

        # Plot 4: Combined refinement map
        ax = axes[1, 1]
        im = ax.imshow(
            combined_map.T,
            origin='lower',
            extent=[domain.xmin, domain.xmax, domain.ymin, domain.ymax],
            cmap='YlOrRd',
            interpolation='nearest'
        )
        plt.colorbar(im, ax=ax, label='Refinement Level')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title('Refinement Map: Combined')
        ax.set_aspect('equal')

        plt.tight_layout()

        plot_file = os.path.join(output_dir, "adaptive_mesh_visualization.png")
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization: {plot_file}")

        # plt.show()  # Uncomment to display

    except ImportError:
        print("⚠ Matplotlib not available, skipping visualization")

    print()
    print("=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Review the generated mesh files in 'output/' directory")
    print("  2. Visualize VTK files in ParaView")
    print("  3. Use the INI files with HydroSIS-2D solver")
    print("  4. Adjust refinement zones based on your specific needs")


if __name__ == "__main__":
    main()
