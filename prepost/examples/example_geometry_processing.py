"""
Example: Geometry Processing for HydroSIS-2D

Demonstrates terrain generation, processing, and preparation for simulation.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from preprocessing.geometry import (
    GeometryGenerator,
    TerrainProcessor,
    create_test_terrain,
    save_terrain,
    load_terrain,
    process_terrain_for_simulation
)
from preprocessing.mesh_generation import DomainParams, MeshGenerator


def example_1_basic_terrain_generation():
    """Example 1: Generate various terrain types"""
    print("=" * 60)
    print("Example 1: Basic Terrain Generation")
    print("=" * 60)

    gen = GeometryGenerator()

    # Create different terrain types
    terrains = {
        'Flat Surface': gen.flat_surface(ncols=100, nrows=100, elevation=5.0, cellsize=1.0),
        'Inclined Plane': gen.inclined_plane(ncols=100, nrows=100, slope_x=0.02, cellsize=1.0),
        'Gaussian Hill': gen.gaussian_hill(ncols=100, nrows=100, height=15.0, width=30.0, cellsize=1.0),
        'Valley': gen.valley(ncols=100, nrows=100, orientation='x', depth=10.0, width=25.0, cellsize=1.0)
    }

    # Plot terrains
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (name, terrain) in enumerate(terrains.items()):
        ax = axes[idx]
        im = ax.imshow(terrain.data.T, origin='lower', cmap='terrain', aspect='auto')
        ax.set_title(f'{name}\n({terrain.ncols}×{terrain.nrows} cells)')
        ax.set_xlabel('X [cells]')
        ax.set_ylabel('Y [cells]')
        plt.colorbar(im, ax=ax, label='Elevation [m]')

        # Print statistics
        stats = terrain.get_statistics()
        print(f"\n{name}:")
        print(f"  Size: {terrain.ncols} × {terrain.nrows}")
        print(f"  Elevation range: [{stats['min']:.2f}, {stats['max']:.2f}] m")
        print(f"  Mean elevation: {stats['mean']:.2f} m")

    plt.tight_layout()
    plt.savefig('terrain_types.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved terrain visualization to 'terrain_types.png'")


def example_2_terrain_processing():
    """Example 2: Terrain processing operations"""
    print("\n" + "=" * 60)
    print("Example 2: Terrain Processing")
    print("=" * 60)

    # Create base terrain with hill
    terrain = GeometryGenerator.gaussian_hill(
        ncols=100,
        nrows=100,
        height=20.0,
        width=30.0,
        cellsize=1.0
    )

    processor = TerrainProcessor(terrain)

    # Compute terrain derivatives
    print("\nComputing terrain derivatives...")
    grad_x, grad_y = processor.compute_gradient()
    slope = processor.compute_slope()
    aspect = processor.compute_aspect()
    profile_curv, planform_curv = processor.compute_curvature()

    print(f"  Max slope: {np.max(slope):.4f}")
    print(f"  Max gradient magnitude: {np.max(np.sqrt(grad_x**2 + grad_y**2)):.4f}")

    # Apply smoothing
    print("\nApplying smoothing filters...")
    terrain_smooth_gauss = processor.smooth_gaussian(sigma=2.0)
    terrain_smooth_uniform = processor.smooth_uniform(size=5)

    # Plot results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Original terrain
    im0 = axes[0, 0].imshow(terrain.data.T, origin='lower', cmap='terrain')
    axes[0, 0].set_title('Original Terrain')
    plt.colorbar(im0, ax=axes[0, 0], label='Elevation [m]')

    # Slope
    im1 = axes[0, 1].imshow(slope.T, origin='lower', cmap='hot')
    axes[0, 1].set_title('Slope')
    plt.colorbar(im1, ax=axes[0, 1], label='Slope')

    # Aspect
    im2 = axes[0, 2].imshow(aspect.T, origin='lower', cmap='hsv', vmin=0, vmax=360)
    axes[0, 2].set_title('Aspect')
    plt.colorbar(im2, ax=axes[0, 2], label='Aspect [°]')

    # Gaussian smoothed
    im3 = axes[1, 0].imshow(terrain_smooth_gauss.data.T, origin='lower', cmap='terrain')
    axes[1, 0].set_title('Gaussian Smoothed (σ=2)')
    plt.colorbar(im3, ax=axes[1, 0], label='Elevation [m]')

    # Uniform smoothed
    im4 = axes[1, 1].imshow(terrain_smooth_uniform.data.T, origin='lower', cmap='terrain')
    axes[1, 1].set_title('Uniform Smoothed (size=5)')
    plt.colorbar(im4, ax=axes[1, 1], label='Elevation [m]')

    # Profile curvature
    im5 = axes[1, 2].imshow(profile_curv.T, origin='lower', cmap='RdBu_r')
    axes[1, 2].set_title('Profile Curvature')
    plt.colorbar(im5, ax=axes[1, 2], label='Curvature')

    for ax in axes.flatten():
        ax.set_xlabel('X [cells]')
        ax.set_ylabel('Y [cells]')

    plt.tight_layout()
    plt.savefig('terrain_processing.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved processing visualization to 'terrain_processing.png'")


def example_3_terrain_interpolation():
    """Example 3: Interpolate terrain to simulation mesh"""
    print("\n" + "=" * 60)
    print("Example 3: Terrain Interpolation to Mesh")
    print("=" * 60)

    # Create high-resolution terrain
    print("\nCreating high-resolution terrain (200×200)...")
    terrain = GeometryGenerator.composite_terrain(
        features=[
            ('inclined_plane', {'slope_x': 0.005, 'base_elevation': 0.0}),
            ('gaussian_hill', {'center_x': 60.0, 'center_y': 50.0, 'height': 12.0, 'width': 25.0}),
            ('gaussian_hill', {'center_x': 140.0, 'center_y': 50.0, 'height': 8.0, 'width': 20.0})
        ],
        ncols=200,
        nrows=100,
        cellsize=1.0
    )

    # Create coarser simulation mesh
    print("Creating simulation mesh (50×25)...")
    domain = DomainParams(xmin=0.0, xmax=199.0, ymin=0.0, ymax=99.0)
    generator = MeshGenerator(domain)
    mesh = generator.generate_uniform_mesh(nx=50, ny=25)

    # Interpolate terrain to mesh
    print("Interpolating terrain to mesh...")
    processor = TerrainProcessor(terrain)
    elevation_linear = processor.interpolate_to_mesh(mesh, method='linear')
    elevation_cubic = processor.interpolate_to_mesh(mesh, method='cubic')

    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Original terrain
    im0 = axes[0].imshow(terrain.data.T, origin='lower', cmap='terrain', aspect='auto')
    axes[0].set_title(f'Original Terrain\n({terrain.ncols}×{terrain.nrows} cells)')
    plt.colorbar(im0, ax=axes[0], label='Elevation [m]')

    # Linear interpolation
    im1 = axes[1].imshow(elevation_linear.T, origin='lower', cmap='terrain', aspect='auto')
    axes[1].set_title(f'Linear Interpolation\n({mesh.nx}×{mesh.ny} cells)')
    plt.colorbar(im1, ax=axes[1], label='Elevation [m]')

    # Cubic interpolation
    im2 = axes[2].imshow(elevation_cubic.T, origin='lower', cmap='terrain', aspect='auto')
    axes[2].set_title(f'Cubic Interpolation\n({mesh.nx}×{mesh.ny} cells)')
    plt.colorbar(im2, ax=axes[2], label='Elevation [m]')

    for ax in axes:
        ax.set_xlabel('X [cells]')
        ax.set_ylabel('Y [cells]')

    plt.tight_layout()
    plt.savefig('terrain_interpolation.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved interpolation comparison to 'terrain_interpolation.png'")

    print(f"\nInterpolation statistics:")
    print(f"  Original - min: {np.min(terrain.data):.2f}, max: {np.max(terrain.data):.2f}")
    print(f"  Linear   - min: {np.min(elevation_linear):.2f}, max: {np.max(elevation_linear):.2f}")
    print(f"  Cubic    - min: {np.min(elevation_cubic):.2f}, max: {np.max(elevation_cubic):.2f}")


def example_4_refinement_zones():
    """Example 4: Identify mesh refinement zones"""
    print("\n" + "=" * 60)
    print("Example 4: Identify Refinement Zones")
    print("=" * 60)

    # Create terrain with steep features
    terrain = GeometryGenerator.composite_terrain(
        features=[
            ('flat_surface', {'elevation': 0.0}),
            ('gaussian_hill', {'center_x': 50.0, 'center_y': 50.0, 'height': 30.0, 'width': 15.0}),
            ('valley', {'orientation': 'x', 'depth': 15.0, 'width': 10.0, 'base_elevation': 0.0})
        ],
        ncols=100,
        nrows=100,
        cellsize=1.0
    )

    processor = TerrainProcessor(terrain)

    # Compute slope and identify refinement zones
    slope = processor.compute_slope()
    refinement_zones = processor.identify_refinement_zones(slope_threshold=0.15)

    print(f"\nTerrain statistics:")
    print(f"  Max slope: {np.max(slope):.4f}")
    print(f"  Cells requiring refinement: {np.sum(refinement_zones)} ({100*np.sum(refinement_zones)/refinement_zones.size:.1f}%)")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Terrain
    im0 = axes[0].imshow(terrain.data.T, origin='lower', cmap='terrain')
    axes[0].set_title('Terrain Elevation')
    plt.colorbar(im0, ax=axes[0], label='Elevation [m]')

    # Slope
    im1 = axes[1].imshow(slope.T, origin='lower', cmap='hot')
    axes[1].set_title('Terrain Slope')
    plt.colorbar(im1, ax=axes[1], label='Slope')

    # Refinement zones
    im2 = axes[2].imshow(refinement_zones.T, origin='lower', cmap='RdYlGn_r')
    axes[2].set_title('Refinement Zones\n(slope > 0.15)')
    plt.colorbar(im2, ax=axes[2], label='Refine?')

    for ax in axes:
        ax.set_xlabel('X [cells]')
        ax.set_ylabel('Y [cells]')

    plt.tight_layout()
    plt.savefig('refinement_zones.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved refinement zones to 'refinement_zones.png'")


def example_5_save_load_terrain():
    """Example 5: Save and load terrain files"""
    print("\n" + "=" * 60)
    print("Example 5: Save and Load Terrain")
    print("=" * 60)

    # Generate test terrain
    print("\nGenerating test terrain...")
    terrain = GeometryGenerator.random_terrain(
        ncols=50,
        nrows=50,
        amplitude=8.0,
        wavelength=15.0,
        base_elevation=5.0,
        cellsize=2.0,
        seed=42
    )

    # Save to ASCII Grid format
    filename = 'test_terrain.asc'
    print(f"Saving terrain to '{filename}'...")
    save_terrain(terrain, filename)

    # Load back
    print(f"Loading terrain from '{filename}'...")
    terrain_loaded = load_terrain(filename)

    # Verify
    print("\nVerifying loaded terrain...")
    print(f"  Original size: {terrain.ncols} × {terrain.nrows}")
    print(f"  Loaded size:   {terrain_loaded.ncols} × {terrain_loaded.nrows}")
    print(f"  Cell size match: {terrain.cellsize == terrain_loaded.cellsize}")
    print(f"  Data match: {np.allclose(terrain.data, terrain_loaded.data)}")

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    im0 = axes[0].imshow(terrain.data.T, origin='lower', cmap='terrain')
    axes[0].set_title('Original Terrain')
    plt.colorbar(im0, ax=axes[0], label='Elevation [m]')

    im1 = axes[1].imshow(terrain_loaded.data.T, origin='lower', cmap='terrain')
    axes[1].set_title('Loaded Terrain')
    plt.colorbar(im1, ax=axes[1], label='Elevation [m]')

    for ax in axes:
        ax.set_xlabel('X [cells]')
        ax.set_ylabel('Y [cells]')

    plt.tight_layout()
    plt.savefig('terrain_save_load.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comparison to 'terrain_save_load.png'")
    print(f"✓ Terrain file saved to '{filename}'")


def example_6_full_workflow():
    """Example 6: Complete workflow for simulation setup"""
    print("\n" + "=" * 60)
    print("Example 6: Complete Workflow")
    print("=" * 60)

    print("\nStep 1: Generate terrain...")
    terrain = GeometryGenerator.dam_break_channel(
        length=200.0,
        width=50.0,
        dam_position=100.0,
        upstream_elevation=0.0,
        downstream_elevation=0.0,
        cellsize=0.5
    )

    print("\nStep 2: Create simulation mesh...")
    domain = DomainParams(xmin=0.0, xmax=199.5, ymin=0.0, ymax=49.5)
    generator = MeshGenerator(domain)
    mesh = generator.generate_uniform_mesh(nx=100, ny=25)

    print("\nStep 3: Process terrain for simulation...")
    elevation = process_terrain_for_simulation(
        terrain,
        mesh,
        smooth=True,
        smooth_sigma=1.0
    )

    print("\nStep 4: Export mesh and terrain...")
    from preprocessing.mesh_generation.mesh_io import MeshIO
    MeshIO.export_to_vtk(mesh, 'simulation_mesh.vtk')
    save_terrain(terrain, 'terrain.asc')

    print("\nStep 5: Visualize setup...")
    fig = plt.figure(figsize=(14, 8))

    # 3D view
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    X, Y = np.meshgrid(
        np.linspace(domain.xmin, domain.xmax, mesh.nx),
        np.linspace(domain.ymin, domain.ymax, mesh.ny),
        indexing='ij'
    )
    surf = ax1.plot_surface(X, Y, elevation, cmap='terrain', alpha=0.8)
    ax1.set_title('3D Terrain View')
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.set_zlabel('Elevation [m]')
    plt.colorbar(surf, ax=ax1, shrink=0.5)

    # Top view
    ax2 = fig.add_subplot(2, 2, 2)
    im2 = ax2.imshow(elevation.T, origin='lower', cmap='terrain', aspect='auto')
    ax2.set_title('Top View - Elevation')
    ax2.set_xlabel('X [cells]')
    ax2.set_ylabel('Y [cells]')
    plt.colorbar(im2, ax=ax2, label='Elevation [m]')

    # Profile at centerline
    ax3 = fig.add_subplot(2, 2, 3)
    centerline = elevation[:, mesh.ny // 2]
    x_coords = np.linspace(domain.xmin, domain.xmax, mesh.nx)
    ax3.plot(x_coords, centerline, 'b-', linewidth=2)
    ax3.set_title('Centerline Profile')
    ax3.set_xlabel('X [m]')
    ax3.set_ylabel('Elevation [m]')
    ax3.grid(True, alpha=0.3)
    ax3.axvline(x=100.0, color='r', linestyle='--', label='Dam position')
    ax3.legend()

    # Mesh visualization
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(mesh.x, mesh.y, 'k.', markersize=0.5, alpha=0.5)
    ax4.set_title(f'Mesh Layout ({mesh.nx}×{mesh.ny} cells)')
    ax4.set_xlabel('X [m]')
    ax4.set_ylabel('Y [m]')
    ax4.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('simulation_setup.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved simulation setup to 'simulation_setup.png'")

    print("\n" + "=" * 60)
    print("Simulation Ready!")
    print("=" * 60)
    print(f"  Mesh: {mesh.nx} × {mesh.ny} = {mesh.ncells} cells")
    print(f"  Domain: [{domain.xmin}, {domain.xmax}] × [{domain.ymin}, {domain.ymax}] m")
    print(f"  Cell size: Δx={mesh.dx:.2f} m, Δy={mesh.dy:.2f} m")
    print(f"  Elevation range: [{np.min(elevation):.2f}, {np.max(elevation):.2f}] m")
    print(f"\n  Files exported:")
    print(f"    - simulation_mesh.vtk")
    print(f"    - terrain.asc")


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("GEOMETRY PROCESSING EXAMPLES")
    print("HydroSIS-2D Pre/Post-Processing Toolkit")
    print("=" * 60)

    # Run examples
    example_1_basic_terrain_generation()
    example_2_terrain_processing()
    example_3_terrain_interpolation()
    example_4_refinement_zones()
    example_5_save_load_terrain()
    example_6_full_workflow()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - terrain_types.png")
    print("  - terrain_processing.png")
    print("  - terrain_interpolation.png")
    print("  - refinement_zones.png")
    print("  - terrain_save_load.png")
    print("  - simulation_setup.png")
    print("  - test_terrain.asc")
    print("  - terrain.asc")
    print("  - simulation_mesh.vtk")


if __name__ == '__main__':
    main()
