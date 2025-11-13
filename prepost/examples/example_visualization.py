#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example 3: 3D Visualization and Rendering

Demonstrates advanced 3D visualization capabilities for HydroSIS-2D results.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from preprocessing.mesh_generation import MeshGenerator, DomainParams
from postprocessing.visualization import VisualizationEngine, AnimationGenerator
from postprocessing.visualization import get_colormap, print_colormap_guide


def create_synthetic_data(mesh, scenario='dam_break'):
    """
    Create synthetic simulation data for demonstration

    Args:
        mesh: StructuredMesh object
        scenario: 'dam_break', 'channel_flow', or 'wave'

    Returns:
        Tuple of (terrain, water_depth, u_velocity, v_velocity)
    """
    x = mesh.x
    y = mesh.y

    # Create terrain
    if scenario == 'dam_break':
        # Flat terrain with slight slope
        terrain = 5.0 + 0.001 * x

        # Initial water depth (dam break configuration)
        water_depth = np.zeros_like(x)
        dam_x = mesh.domain.xmin + mesh.domain.length_x * 0.3

        water_depth[x < dam_x] = 10.0  # Upstream water

        # Velocities (initially at rest)
        u = np.zeros_like(x)
        v = np.zeros_like(y)

    elif scenario == 'channel_flow':
        # Channel with varying bed elevation
        center_y = (mesh.domain.ymin + mesh.domain.ymax) / 2
        channel_width = mesh.domain.length_y / 4

        # Parabolic channel cross-section
        terrain = 2.0 + 0.5 * ((y - center_y) / channel_width) ** 2

        # Uniform flow
        water_depth = 3.0 * np.ones_like(x)

        # Velocity in x-direction, zero in y
        u = 2.0 * np.ones_like(x)
        v = np.zeros_like(y)

    elif scenario == 'wave':
        # Flat terrain
        terrain = np.zeros_like(x)

        # Standing wave pattern
        water_depth = 5.0 + 0.5 * np.sin(2 * np.pi * x / mesh.domain.length_x) * \
                           np.sin(2 * np.pi * y / mesh.domain.length_y)

        # Velocity field (circular pattern)
        center_x = (mesh.domain.xmin + mesh.domain.xmax) / 2
        center_y = (mesh.domain.ymin + mesh.domain.ymax) / 2

        u = 0.5 * (y - center_y) / mesh.domain.length_y
        v = -0.5 * (x - center_x) / mesh.domain.length_x

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    return terrain, water_depth, u, v


def example_1_mesh_visualization():
    """Example 1: Visualize mesh structure"""
    print("=" * 60)
    print("Example 1: Mesh Structure Visualization")
    print("=" * 60)
    print()

    # Create mesh
    domain = DomainParams(xmin=0, xmax=200, ymin=0, ymax=100)
    generator = MeshGenerator(domain)
    mesh = generator.generate_uniform_mesh(50, 25)

    print(f"Created mesh: {mesh.nx} x {mesh.ny} cells")

    # Create visualization engine
    engine = VisualizationEngine(offscreen=True)
    engine.visualize_mesh(mesh, show_edges=True, opacity=0.3)

    # Save screenshot
    output_dir = "output/visualization"
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "mesh_structure.png")
    engine.screenshot(output_file)
    print(f"[OK] Saved: {output_file}")

    engine.close()
    print()


def example_2_terrain_visualization():
    """Example 2: Visualize terrain"""
    print("=" * 60)
    print("Example 2: Terrain Visualization")
    print("=" * 60)
    print()

    # Create mesh
    domain = DomainParams(xmin=0, xmax=500, ymin=0, ymax=200)
    generator = MeshGenerator(domain)
    mesh = generator.generate_uniform_mesh(100, 40)

    # Create terrain with features
    x = mesh.x
    y = mesh.y

    # Base slope
    terrain = 10.0 + 0.01 * x

    # Add a hill
    hill_x, hill_y = 250, 100
    hill_radius = 80
    dist_to_hill = np.sqrt((x - hill_x)**2 + (y - hill_y)**2)
    terrain += 15.0 * np.exp(-(dist_to_hill / hill_radius)**2)

    # Add a valley
    valley_y = 100
    valley_width = 30
    terrain -= 5.0 * np.exp(-((y - valley_y) / valley_width)**2)

    print(f"Terrain elevation range: [{terrain.min():.2f}, {terrain.max():.2f}] m")

    # Visualize
    engine = VisualizationEngine(offscreen=True)
    engine.visualize_terrain(mesh, terrain, cmap='terrain')

    output_dir = "output/visualization"
    output_file = os.path.join(output_dir, "terrain.png")
    engine.screenshot(output_file)
    print(f"[OK] Saved: {output_file}")

    engine.close()
    print()

    return mesh, terrain


def example_3_water_surface():
    """Example 3: Water surface visualization"""
    print("=" * 60)
    print("Example 3: Water Surface Visualization")
    print("=" * 60)
    print()

    # Create mesh and data
    domain = DomainParams(xmin=0, xmax=400, ymin=0, ymax=200)
    generator = MeshGenerator(domain)
    mesh = generator.generate_uniform_mesh(80, 40)

    terrain, water_depth, u, v = create_synthetic_data(mesh, scenario='dam_break')

    print(f"Water depth range: [{water_depth.min():.2f}, {water_depth.max():.2f}] m")

    # Visualize water surface
    engine = VisualizationEngine(offscreen=True)
    engine.visualize_water_surface(
        mesh, water_depth, terrain,
        cmap='Blues', opacity=0.7
    )

    output_dir = "output/visualization"
    output_file = os.path.join(output_dir, "water_surface.png")
    engine.screenshot(output_file)
    print(f"[OK] Saved: {output_file}")

    engine.close()
    print()


def example_4_velocity_field():
    """Example 4: Velocity field visualization"""
    print("=" * 60)
    print("Example 4: Velocity Field Visualization")
    print("=" * 60)
    print()

    # Create mesh and data
    domain = DomainParams(xmin=0, xmax=300, ymin=0, ymax=150)
    generator = MeshGenerator(domain)
    mesh = generator.generate_uniform_mesh(60, 30)

    terrain, water_depth, u, v = create_synthetic_data(mesh, scenario='channel_flow')

    # Calculate velocity magnitude
    vel_mag = np.sqrt(u**2 + v**2)
    print(f"Velocity magnitude range: [{vel_mag.min():.2f}, {vel_mag.max():.2f}] m/s")

    # Visualize velocity magnitude
    engine = VisualizationEngine(offscreen=True)
    engine.visualize_velocity_magnitude(
        mesh, u, v, terrain=terrain,
        cmap='jet'
    )

    output_dir = "output/visualization"
    output_file = os.path.join(output_dir, "velocity_field.png")
    engine.screenshot(output_file)
    print(f"[OK] Saved: {output_file}")

    engine.close()
    print()


def example_5_combined_visualization():
    """Example 5: Combined visualization (water + vectors)"""
    print("=" * 60)
    print("Example 5: Combined Visualization")
    print("=" * 60)
    print()

    # Create mesh and data
    domain = DomainParams(xmin=0, xmax=400, ymin=0, ymax=200)
    generator = MeshGenerator(domain)
    mesh = generator.generate_uniform_mesh(40, 20)  # Coarser for vector display

    terrain, water_depth, u, v = create_synthetic_data(mesh, scenario='wave')

    # Visualize water surface
    engine = VisualizationEngine(offscreen=True, window_size=(1920, 1080))
    plotter = engine.visualize_water_surface(
        mesh, water_depth, terrain,
        cmap='Blues', opacity=0.7
    )

    # Add velocity vectors
    water_surface = terrain + water_depth
    engine.add_velocity_vectors(
        mesh, u, v,
        elevation=water_surface,
        scale_factor=20.0,
        color='red'
    )

    output_dir = "output/visualization"
    output_file = os.path.join(output_dir, "combined_visualization.png")
    engine.screenshot(output_file)
    print(f"[OK] Saved: {output_file}")

    engine.close()
    print()


def example_6_animation():
    """Example 6: Create animation"""
    print("=" * 60)
    print("Example 6: Animation Generation")
    print("=" * 60)
    print()

    # Create mesh
    domain = DomainParams(xmin=0, xmax=400, ymin=0, ymax=200)
    generator = MeshGenerator(domain)
    mesh = generator.generate_uniform_mesh(80, 40)

    # Create time series (simulate dam break wave propagation)
    terrain = 5.0 + 0.001 * mesh.x
    dam_x = 120

    print("Generating time series data...")

    animation_gen = AnimationGenerator(mesh)

    num_frames = 30
    for i in range(num_frames):
        t = i * 0.5  # Time step

        # Simulate wave front moving downstream
        wave_front = dam_x + 50 * t
        wave_decay = 1.0 / (1.0 + 0.1 * t)

        water_depth = np.zeros_like(mesh.x)

        # Upstream water
        water_depth[mesh.x < dam_x] = 10.0

        # Wave
        mask = (mesh.x >= dam_x) & (mesh.x < wave_front)
        water_depth[mask] = 10.0 * wave_decay * (1 - (mesh.x[mask] - dam_x) / (wave_front - dam_x))

        animation_gen.add_timestep(t, {'h': water_depth})

    print(f"Generated {num_frames} time steps")

    # Create animation
    output_dir = "output/visualization"
    output_file = os.path.join(output_dir, "flood_animation.gif")

    print(f"Creating animation (this may take a moment)...")

    try:
        animation_gen.create_water_surface_animation(
            terrain, output_file,
            fps=10, field='h', cmap='Blues'
        )
        print(f"[OK] Animation saved: {output_file}")
    except Exception as e:
        print(f"[WARN] Animation creation failed: {e}")
        print("  (ffmpeg may not be available)")

    print()


def example_7_colormap_guide():
    """Example 7: Display colormap guide"""
    print("=" * 60)
    print("Example 7: Colormap Guide")
    print("=" * 60)
    print()

    print_colormap_guide()

    print("\nExample usage:")
    print("-" * 60)

    examples = [
        ('water_depth', 'Blues'),
        ('velocity', 'jet'),
        ('elevation', 'terrain'),
        ('froude_number', 'RdYlBu_r')
    ]

    for field, expected_cmap in examples:
        cmap = get_colormap(field)
        print(f"  get_colormap('{field}') -> '{cmap}'")

    print()


def main():
    """Run all visualization examples"""
    print()
    print("=" * 60)
    print("HYDROSIS-2D VISUALIZATION EXAMPLES")
    print("=" * 60)
    print()

    examples = [
        ("Mesh Structure", example_1_mesh_visualization),
        ("Terrain", example_2_terrain_visualization),
        ("Water Surface", example_3_water_surface),
        ("Velocity Field", example_4_velocity_field),
        ("Combined Visualization", example_5_combined_visualization),
        ("Animation", example_6_animation),
        ("Colormap Guide", example_7_colormap_guide)
    ]

    for name, func in examples:
        try:
            func()
        except Exception as e:
            print(f"[WARN] {name} failed: {e}")
            import traceback
            traceback.print_exc()
            print()

    print("=" * 60)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 60)
    print()
    print("Output files saved to: output/visualization/")
    print()
    print("Next steps:")
    print("  1. Review the generated images in output/visualization/")
    print("  2. Try modifying parameters (colormaps, resolution, etc.)")
    print("  3. Use these tools with your own simulation results")
    print("  4. Create custom animations with real data")
    print()


if __name__ == "__main__":
    main()
