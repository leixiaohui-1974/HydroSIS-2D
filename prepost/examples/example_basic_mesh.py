#!/usr/bin/env python3
"""
Example 1: Basic Mesh Generation

Demonstrates basic uniform mesh generation for HydroSIS-2D.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from preprocessing.mesh_generation import (
    MeshGenerator,
    DomainParams,
    MeshQualityChecker,
    MeshIO
)


def main():
    print("=" * 60)
    print("Example 1: Basic Uniform Mesh Generation")
    print("=" * 60)
    print()

    # Define computational domain
    domain = DomainParams(
        xmin=0.0,
        xmax=200.0,  # 200 m in x-direction
        ymin=0.0,
        ymax=100.0   # 100 m in y-direction
    )

    print(f"Domain: {domain.length_x} m × {domain.length_y} m")
    print(f"Domain area: {domain.area} m²")
    print()

    # Create mesh generator
    generator = MeshGenerator(domain)

    # Method 1: Generate mesh with specified cell count
    print("Method 1: Mesh with specified cell count")
    mesh1 = generator.generate_uniform_mesh(nx=100, ny=50)
    print(f"  Grid: {mesh1.nx} × {mesh1.ny} = {mesh1.ncells} cells")
    print(f"  Cell size: dx={mesh1.dx:.3f} m, dy={mesh1.dy:.3f} m")
    print(f"  Cell area: {mesh1.get_cell_area():.3f} m²")
    print()

    # Method 2: Generate mesh with specified cell spacing
    print("Method 2: Mesh with specified cell spacing")
    mesh2 = generator.generate_mesh_with_spacing(dx=1.0, dy=1.0)
    print(f"  Grid: {mesh2.nx} × {mesh2.ny} = {mesh2.ncells} cells")
    print(f"  Cell size: dx={mesh2.dx:.3f} m, dy={mesh2.dy:.3f} m")
    print()

    # Method 3: Generate mesh with target resolution
    print("Method 3: Mesh with target cell area")
    mesh3 = generator.generate_mesh_from_resolution(target_cell_area=4.0)
    print(f"  Grid: {mesh3.nx} × {mesh3.ny} = {mesh3.ncells} cells")
    print(f"  Cell size: dx={mesh3.dx:.3f} m, dy={mesh3.dy:.3f} m")
    print(f"  Actual cell area: {mesh3.get_cell_area():.3f} m²")
    print()

    # Quality check
    print("=" * 60)
    print("Mesh Quality Assessment")
    print("=" * 60)
    checker = MeshQualityChecker(mesh1)
    checker.print_report()

    # CFL condition check
    print("\n" + "=" * 60)
    print("CFL Stability Check")
    print("=" * 60)
    cfl_result = checker.check_cfl_condition(max_velocity=5.0, dt=0.1)
    print(f"CFL number: {cfl_result['cfl_max']:.3f}")
    print(f"Stable: {cfl_result['is_stable']}")
    print(f"Recommended dt: {cfl_result['recommended_dt']:.4f} s")
    print()

    # Computational cost estimate
    print("=" * 60)
    print("Computational Cost Estimate")
    print("=" * 60)
    cost = checker.estimate_computational_cost(simulation_time=100.0, max_velocity=5.0)
    print(f"Estimated time step: {cost['estimated_dt']:.4f} s")
    print(f"Number of time steps: {cost['num_timesteps']:,}")
    print(f"Total GFLOPS: {cost['total_gflops']:.2f}")
    print(f"Memory required: {cost['memory_mb']:.2f} MB")
    print(f"Estimated wall time: {cost['walltime_estimate_minutes']:.1f} minutes")
    print()

    # Export mesh
    print("=" * 60)
    print("Exporting Mesh")
    print("=" * 60)

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Export to HydroSIS-2D INI format
    ini_file = os.path.join(output_dir, "mesh_basic.ini")
    MeshIO.export_to_ini(mesh1, ini_file, include_physics=True)
    print(f"✓ Exported INI configuration: {ini_file}")

    # Export to VTK format
    vtk_file = os.path.join(output_dir, "mesh_basic.vtk")
    MeshIO.export_to_vtk(mesh1, vtk_file)
    print(f"✓ Exported VTK mesh: {vtk_file}")

    # Export to JSON
    json_file = os.path.join(output_dir, "mesh_basic.json")
    MeshIO.export_to_json(mesh1, json_file)
    print(f"✓ Exported JSON metadata: {json_file}")

    # Export batch
    MeshIO.export_batch(mesh1, output_dir, base_name="mesh_complete")
    print(f"✓ Exported all formats to: {output_dir}/")

    # Visualize mesh
    print()
    print("=" * 60)
    print("Visualization")
    print("=" * 60)
    try:
        import matplotlib.pyplot as plt
        fig, ax = generator.visualize_mesh(show_every=10)
        plot_file = os.path.join(output_dir, "mesh_visualization.png")
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved mesh visualization: {plot_file}")
        # plt.show()  # Uncomment to display
    except ImportError:
        print("⚠ Matplotlib not available, skipping visualization")

    print()
    print("=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
