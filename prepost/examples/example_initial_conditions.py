# -*- coding: utf-8 -*-
"""
Example: Initial Conditions Setup for HydroSIS-2D

Demonstrates various initial condition configurations.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from preprocessing.initial_conditions import (
    UniformIC,
    DamBreakIC,
    DryBedIC,
    GaussianHumpIC,
    ParabolicBowlIC,
    CustomFieldIC,
    InitialConditionManager
)
from preprocessing.mesh_generation import DomainParams, MeshGenerator
from preprocessing.geometry import GeometryGenerator


def example_1_uniform_initial_condition():
    """Example 1: Uniform initial condition"""
    print("=" * 60)
    print("Example 1: Uniform Initial Condition")
    print("=" * 60)

    # Create mesh
    domain = DomainParams(xmin=0.0, xmax=100.0, ymin=0.0, ymax=50.0)
    mesh = MeshGenerator(domain).generate_uniform_mesh(50, 25)

    # Create initial condition
    ic = UniformIC(depth=3.0, velocity_x=0.5, velocity_y=0.0)

    # Create manager
    manager = InitialConditionManager(mesh)
    manager.set_initial_condition(ic)

    # Print summary
    print(manager.summary())

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    im0 = axes[0].imshow(manager.depth.T, origin='lower', cmap='Blues')
    axes[0].set_title('Water Depth [m]')
    axes[0].set_xlabel('X [cells]')
    axes[0].set_ylabel('Y [cells]')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(manager.velocity_x.T, origin='lower', cmap='RdBu_r')
    axes[1].set_title('X-Velocity [m/s]')
    axes[1].set_xlabel('X [cells]')
    axes[1].set_ylabel('Y [cells]')
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(manager.velocity_y.T, origin='lower', cmap='RdBu_r')
    axes[2].set_title('Y-Velocity [m/s]')
    axes[2].set_xlabel('X [cells]')
    axes[2].set_ylabel('Y [cells]')
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.savefig('ic_uniform.png', dpi=150, bbox_inches='tight')
    print("\n[OK] Saved visualization to 'ic_uniform.png'")


def example_2_dam_break():
    """Example 2: Classic dam break"""
    print("\n" + "=" * 60)
    print("Example 2: Dam Break Initial Condition")
    print("=" * 60)

    # Create mesh
    domain = DomainParams(xmin=0.0, xmax=200.0, ymin=0.0, ymax=100.0)
    mesh = MeshGenerator(domain).generate_uniform_mesh(100, 50)

    # Dam break IC
    ic = DamBreakIC(
        dam_position=0.5,
        upstream_depth=10.0,
        downstream_depth=0.0,
        orientation='x'
    )

    # Create manager
    manager = InitialConditionManager(mesh)
    manager.set_initial_condition(ic)

    print(manager.summary())

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 2D view of depth
    im0 = axes[0, 0].imshow(manager.depth.T, origin='lower', cmap='Blues', vmin=0, vmax=10)
    axes[0, 0].set_title('Water Depth [m]')
    axes[0, 0].set_xlabel('X [cells]')
    axes[0, 0].set_ylabel('Y [cells]')
    axes[0, 0].axvline(x=50, color='r', linestyle='--', label='Dam')
    axes[0, 0].legend()
    plt.colorbar(im0, ax=axes[0, 0])

    # 1D profile at centerline
    centerline = manager.depth[:, 25]
    x_coords = np.linspace(domain.xmin, domain.xmax, mesh.nx)
    axes[0, 1].plot(x_coords, centerline, 'b-', linewidth=2)
    axes[0, 1].axvline(x=100, color='r', linestyle='--', linewidth=2, label='Dam')
    axes[0, 1].set_xlabel('X [m]')
    axes[0, 1].set_ylabel('Water Depth [m]')
    axes[0, 1].set_title('Centerline Profile')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    axes[0, 1].set_ylim([0, 11])

    # 3D surface
    X, Y = np.meshgrid(np.linspace(0, 100, mesh.nx), np.linspace(0, 50, mesh.ny), indexing='ij')
    ax3d = fig.add_subplot(2, 2, 3, projection='3d')
    surf = ax3d.plot_surface(X, Y, manager.depth, cmap='Blues', alpha=0.8)
    ax3d.set_xlabel('X [m]')
    ax3d.set_ylabel('Y [m]')
    ax3d.set_zlabel('Depth [m]')
    ax3d.set_title('3D View')

    # Statistics
    stats_text = f"""Statistics:

Total Volume: {manager.get_total_volume():.0f} m^3
Wet Cells: {np.sum(manager.depth > 0)}
Max Depth: {np.max(manager.depth):.2f} m
Mean Depth: {np.mean(manager.depth):.2f} m"""

    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                   family='monospace')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig('ic_dam_break.png', dpi=150, bbox_inches='tight')
    print("\n[OK] Saved visualization to 'ic_dam_break.png'")


def example_3_gaussian_hump():
    """Example 3: Gaussian hump (wave propagation test)"""
    print("\n" + "=" * 60)
    print("Example 3: Gaussian Hump")
    print("=" * 60)

    # Create mesh
    domain = DomainParams(xmin=0.0, xmax=100.0, ymin=0.0, ymax=100.0)
    mesh = MeshGenerator(domain).generate_uniform_mesh(100, 100)

    # Gaussian hump IC
    ic = GaussianHumpIC(
        center_x=0.5,
        center_y=0.5,
        amplitude=2.0,
        width=0.1,
        base_depth=1.0
    )

    # Create manager
    manager = InitialConditionManager(mesh)
    manager.set_initial_condition(ic)

    print(manager.summary())

    # Visualize
    fig = plt.figure(figsize=(14, 10))

    # 2D contour
    ax1 = fig.add_subplot(2, 2, 1)
    levels = np.linspace(1.0, 3.0, 20)
    contourf = ax1.contourf(manager.depth.T, levels=levels, cmap='Blues', origin='lower')
    ax1.contour(manager.depth.T, levels=levels, colors='k', alpha=0.3, linewidths=0.5, origin='lower')
    ax1.set_title('Water Depth - Contours [m]')
    ax1.set_xlabel('X [cells]')
    ax1.set_ylabel('Y [cells]')
    ax1.set_aspect('equal')
    plt.colorbar(contourf, ax=ax1)

    # 3D surface
    X, Y = np.meshgrid(np.linspace(0, 100, mesh.nx), np.linspace(0, 100, mesh.ny), indexing='ij')
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    surf = ax2.plot_surface(X, Y, manager.depth, cmap='Blues', alpha=0.8)
    ax2.set_xlabel('X [m]')
    ax2.set_ylabel('Y [m]')
    ax2.set_zlabel('Depth [m]')
    ax2.set_title('3D Surface View')
    ax2.view_init(elev=30, azim=45)

    # Cross-section through center (X-direction)
    ax3 = fig.add_subplot(2, 2, 3)
    centerline_x = manager.depth[:, 50]
    x_coords = np.linspace(0, 100, mesh.nx)
    ax3.plot(x_coords, centerline_x, 'b-', linewidth=2, label='X-profile')
    ax3.set_xlabel('X [m]')
    ax3.set_ylabel('Water Depth [m]')
    ax3.set_title('Cross-Section Through Center')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Cross-section through center (Y-direction)
    ax4 = fig.add_subplot(2, 2, 4)
    centerline_y = manager.depth[50, :]
    y_coords = np.linspace(0, 100, mesh.ny)
    ax4.plot(y_coords, centerline_y, 'r-', linewidth=2, label='Y-profile')
    ax4.set_xlabel('Y [m]')
    ax4.set_ylabel('Water Depth [m]')
    ax4.set_title('Cross-Section Through Center')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    plt.savefig('ic_gaussian_hump.png', dpi=150, bbox_inches='tight')
    print("\n[OK] Saved visualization to 'ic_gaussian_hump.png'")


def example_4_parabolic_bowl():
    """Example 4: Parabolic bowl (well-balanced test)"""
    print("\n" + "=" * 60)
    print("Example 4: Parabolic Bowl")
    print("=" * 60)

    # Create mesh
    domain = DomainParams(xmin=-100.0, xmax=100.0, ymin=-100.0, ymax=100.0)
    mesh = MeshGenerator(domain).generate_uniform_mesh(100, 100)

    # Parabolic bowl IC
    ic = ParabolicBowlIC(bowl_depth=1.0, water_depth=0.5)

    # Create manager
    manager = InitialConditionManager(mesh)
    manager.set_initial_condition(ic)

    print(manager.summary())

    # Compute terrain from parabolic bowl
    x = np.linspace(-1, 1, mesh.nx)
    y = np.linspace(-1, 1, mesh.ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    r_squared = X**2 + Y**2
    terrain = 1.0 * r_squared * 100  # Scale for visualization

    # Visualize
    fig = plt.figure(figsize=(14, 6))

    # Water depth
    ax1 = fig.add_subplot(1, 3, 1)
    im1 = ax1.imshow(manager.depth.T, origin='lower', cmap='Blues')
    ax1.set_title('Water Depth [m]')
    ax1.set_xlabel('X [cells]')
    ax1.set_ylabel('Y [cells]')
    plt.colorbar(im1, ax=ax1)

    # Bed elevation
    ax2 = fig.add_subplot(1, 3, 2)
    im2 = ax2.imshow(terrain.T, origin='lower', cmap='terrain')
    ax2.set_title('Bed Elevation [m]')
    ax2.set_xlabel('X [cells]')
    ax2.set_ylabel('Y [cells]')
    plt.colorbar(im2, ax=ax2)

    # 3D view
    X_plot, Y_plot = np.meshgrid(np.linspace(-100, 100, mesh.nx),
                                  np.linspace(-100, 100, mesh.ny), indexing='ij')
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.plot_surface(X_plot, Y_plot, terrain, cmap='terrain', alpha=0.5, label='Bed')
    ax3.plot_surface(X_plot, Y_plot, terrain + manager.depth, cmap='Blues', alpha=0.7)
    ax3.set_xlabel('X [m]')
    ax3.set_ylabel('Y [m]')
    ax3.set_zlabel('Elevation [m]')
    ax3.set_title('3D View (Bed + Water)')
    ax3.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.savefig('ic_parabolic_bowl.png', dpi=150, bbox_inches='tight')
    print("\n[OK] Saved visualization to 'ic_parabolic_bowl.png'")


def example_5_with_terrain():
    """Example 5: Initial condition with terrain"""
    print("\n" + "=" * 60)
    print("Example 5: Initial Condition with Terrain")
    print("=" * 60)

    # Create mesh
    domain = DomainParams(xmin=0.0, xmax=200.0, ymin=0.0, ymax=100.0)
    mesh = MeshGenerator(domain).generate_uniform_mesh(100, 50)

    # Generate terrain with hill
    terrain_gen = GeometryGenerator()
    terrain_data = terrain_gen.composite_terrain(
        features=[
            ('inclined_plane', {'slope_x': 0.01, 'base_elevation': 0.0}),
            ('gaussian_hill', {'center_x': 100.0, 'center_y': 50.0, 'height': 10.0, 'width': 30.0})
        ],
        ncols=100,
        nrows=50,
        xllcorner=0.0,
        yllcorner=0.0,
        cellsize=2.0
    )

    # Uniform initial water depth
    ic = UniformIC(depth=5.0)

    # Create manager
    manager = InitialConditionManager(mesh, terrain=terrain_data.data)
    manager.set_initial_condition(ic)

    print(manager.summary())

    # Compute water surface elevation
    wse = manager.get_water_surface_elevation()

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Terrain
    im0 = axes[0, 0].imshow(manager.terrain.T, origin='lower', cmap='terrain')
    axes[0, 0].set_title('Terrain Elevation [m]')
    axes[0, 0].set_xlabel('X [cells]')
    axes[0, 0].set_ylabel('Y [cells]')
    plt.colorbar(im0, ax=axes[0, 0])

    # Water depth
    im1 = axes[0, 1].imshow(manager.depth.T, origin='lower', cmap='Blues')
    axes[0, 1].set_title('Water Depth [m]')
    axes[0, 1].set_xlabel('X [cells]')
    axes[0, 1].set_ylabel('Y [cells]')
    plt.colorbar(im1, ax=axes[0, 1])

    # Water surface elevation
    im2 = axes[1, 0].imshow(wse.T, origin='lower', cmap='viridis')
    axes[1, 0].set_title('Water Surface Elevation [m]')
    axes[1, 0].set_xlabel('X [cells]')
    axes[1, 0].set_ylabel('Y [cells]')
    plt.colorbar(im2, ax=axes[1, 0])

    # Centerline profile
    centerline_terrain = manager.terrain[:, 25]
    centerline_wse = wse[:, 25]
    x_coords = np.linspace(domain.xmin, domain.xmax, mesh.nx)

    axes[1, 1].fill_between(x_coords, 0, centerline_terrain, color='brown', alpha=0.5, label='Terrain')
    axes[1, 1].fill_between(x_coords, centerline_terrain, centerline_wse, color='blue', alpha=0.5, label='Water')
    axes[1, 1].plot(x_coords, centerline_wse, 'b-', linewidth=2, label='Water Surface')
    axes[1, 1].set_xlabel('X [m]')
    axes[1, 1].set_ylabel('Elevation [m]')
    axes[1, 1].set_title('Centerline Profile')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig('ic_with_terrain.png', dpi=150, bbox_inches='tight')
    print("\n[OK] Saved visualization to 'ic_with_terrain.png'")


def example_6_export_import():
    """Example 6: Export and import initial conditions"""
    print("\n" + "=" * 60)
    print("Example 6: Export/Import Initial Conditions")
    print("=" * 60)

    # Create mesh
    domain = DomainParams(xmin=0.0, xmax=100.0, ymin=0.0, ymax=50.0)
    mesh = MeshGenerator(domain).generate_uniform_mesh(50, 25)

    # Dam break IC
    ic = DamBreakIC(dam_position=0.6, upstream_depth=8.0, downstream_depth=1.0)

    # Create manager
    manager = InitialConditionManager(mesh)
    manager.set_initial_condition(ic)

    # Export to JSON
    manager.export_to_json('ic_config.json')
    print("[OK] Exported configuration to 'ic_config.json'")

    # Export to NumPy
    manager.export_arrays_to_numpy('ic_fields')
    print("[OK] Exported arrays to 'ic_fields.npz'")

    # Export to VTK
    manager.export_to_vtk('ic_initial.vtk')
    print("[OK] Exported VTK to 'ic_initial.vtk'")

    # Import from NumPy
    manager_loaded = InitialConditionManager.import_arrays_from_numpy('ic_fields.npz', mesh)
    print("[OK] Imported arrays from 'ic_fields.npz'")

    # Verify
    print("\nVerification:")
    print(f"  Depth arrays match: {np.allclose(manager.depth, manager_loaded.depth)}")
    print(f"  Velocity arrays match: {np.allclose(manager.velocity_x, manager_loaded.velocity_x)}")

    print("\n" + manager.summary())


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("INITIAL CONDITIONS EXAMPLES")
    print("HydroSIS-2D Pre/Post-Processing Toolkit")
    print("=" * 60)

    example_1_uniform_initial_condition()
    example_2_dam_break()
    example_3_gaussian_hump()
    example_4_parabolic_bowl()
    example_5_with_terrain()
    example_6_export_import()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - ic_uniform.png")
    print("  - ic_dam_break.png")
    print("  - ic_gaussian_hump.png")
    print("  - ic_parabolic_bowl.png")
    print("  - ic_with_terrain.png")
    print("  - ic_config.json")
    print("  - ic_fields.npz")
    print("  - ic_initial.vtk")


if __name__ == '__main__':
    main()
