"""
Example: 1D Dam Break Problem using Shallow Water Solver

This example demonstrates the use of the Python shallow water solver
for a classical dam break problem.

Physical Setup:
    - Domain: 200m × 20m (quasi-1D in x-direction)
    - Dam at x = 100m
    - Upstream depth: 10m
    - Downstream depth: 1m
    - Flat bed (z = 0)
    - All walls boundaries

Expected Behavior:
    - Shock wave propagates downstream
    - Rarefaction wave propagates upstream
    - Can be compared with analytical solution

Author: HydroSIS-2D Development Team
Date: 2025-10-29
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from preprocessing.mesh_generation import MeshGenerator, DomainParams
from solver import ShallowWaterSolver, SolverConfig


def main():
    print("="*70)
    print("1D Dam Break Problem - Python Solver Test")
    print("="*70)

    # 1. Create mesh
    print("\n1. Creating mesh...")
    domain = DomainParams(xmin=0.0, xmax=200.0, ymin=0.0, ymax=20.0)
    mesh_gen = MeshGenerator(domain)
    mesh = mesh_gen.generate_uniform_mesh(nx=200, ny=20)

    print(f"   Grid: {mesh.nx} × {mesh.ny} = {mesh.ncells:,} cells")
    print(f"   Cell size: dx={mesh.dx:.2f}m, dy={mesh.dy:.2f}m")

    # 2. Setup terrain (flat bed)
    print("\n2. Setting up terrain...")
    terrain = np.zeros((mesh.nx, mesh.ny))
    print("   Flat bed at z = 0m")

    # 3. Create solver
    print("\n3. Initializing solver...")
    config = SolverConfig(
        t_end=10.0,
        cfl=0.5,
        output_interval=0.5,
        output_dir='output/solver_test',
        print_progress=True,
        progress_interval=50
    )

    solver = ShallowWaterSolver(mesh, terrain, config)

    # 4. Set initial conditions (dam break)
    print("\n4. Setting initial conditions (dam break)...")
    h0 = np.zeros((mesh.nx, mesh.ny))
    u0 = np.zeros((mesh.nx, mesh.ny))
    v0 = np.zeros((mesh.nx, mesh.ny))

    # Dam at x = 100m
    dam_position = 100.0
    upstream_depth = 10.0
    downstream_depth = 1.0

    X = mesh.x
    h0[X < dam_position] = upstream_depth
    h0[X >= dam_position] = downstream_depth

    solver.set_initial_conditions(h0, u0, v0)

    # 5. Run solver
    print("\n5. Running solver...")
    stats = solver.solve()

    # 6. Print results
    print("\n" + "="*70)
    print("SOLVER STATISTICS")
    print("="*70)
    print(f"Total time steps:     {stats['steps']}")
    print(f"Simulated time:       {stats['simulated_time']:.3f} s")
    print(f"Wall clock time:      {stats['wall_time']:.2f} s")
    print(f"Speed factor:         {stats['speed_factor']:.2f}x realtime")
    print(f"Final mass:           {stats['final_mass']:.4f} m³")
    print(f"Mass conservation:    {stats['mass_error']:+.6f}%")

    # 7. Visualize results
    print("\n6. Creating visualization...")
    create_visualization(solver, mesh)

    print("\n" + "="*70)
    print("Test completed successfully!")
    print("="*70)


def create_visualization(solver, mesh):
    """Create visualization of final state"""
    state = solver.get_state()
    h = state['h']
    u = state['u']
    z = state['z']
    t = state['t']

    # Extract centerline profile (j = ny//2)
    j_center = mesh.ny // 2
    x = mesh.x[:, j_center]
    h_profile = h[:, j_center]
    u_profile = u[:, j_center]
    z_profile = z[:, j_center]

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Plot 1: Water depth
    axes[0].plot(x, h_profile, 'b-', linewidth=2, label='Water depth')
    axes[0].fill_between(x, 0, h_profile, alpha=0.3, color='blue')
    axes[0].plot(x, z_profile, 'k-', linewidth=1.5, label='Bed')
    axes[0].set_ylabel('Depth [m]', fontsize=12)
    axes[0].set_title(f'1D Dam Break at t = {t:.2f} s', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 200)

    # Plot 2: Velocity
    axes[1].plot(x, u_profile, 'r-', linewidth=2)
    axes[1].set_ylabel('Velocity [m/s]', fontsize=12)
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 200)

    # Plot 3: Froude number
    Fr = np.abs(u_profile) / np.sqrt(9.81 * (h_profile + 1e-6))
    axes[2].plot(x, Fr, 'g-', linewidth=2)
    axes[2].axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Critical (Fr=1)')
    axes[2].set_ylabel('Froude Number', fontsize=12)
    axes[2].set_xlabel('Distance [m]', fontsize=12)
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(0, 200)
    axes[2].set_ylim(0, 2)

    plt.tight_layout()

    output_file = 'output/solver_test/dam_break_profile.png'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   Plot saved: {output_file}")

    # 2D contour plot
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    # Water depth contour
    levels_h = np.linspace(0, np.max(h), 20)
    cs1 = axes2[0].contourf(mesh.x, mesh.y, h, levels=levels_h, cmap='Blues')
    plt.colorbar(cs1, ax=axes2[0], label='Water Depth [m]')
    axes2[0].set_xlabel('X [m]')
    axes2[0].set_ylabel('Y [m]')
    axes2[0].set_title(f'Water Depth at t = {t:.2f} s')
    axes2[0].set_aspect('equal')

    # Velocity magnitude
    V = np.sqrt(u**2 + state['v']**2)
    levels_v = np.linspace(0, np.max(V), 20)
    cs2 = axes2[1].contourf(mesh.x, mesh.y, V, levels=levels_v, cmap='Reds')
    plt.colorbar(cs2, ax=axes2[1], label='Velocity [m/s]')
    axes2[1].set_xlabel('X [m]')
    axes2[1].set_ylabel('Y [m]')
    axes2[1].set_title(f'Velocity Magnitude at t = {t:.2f} s')
    axes2[1].set_aspect('equal')

    plt.tight_layout()
    output_file2 = 'output/solver_test/dam_break_2d.png'
    plt.savefig(output_file2, dpi=150, bbox_inches='tight')
    print(f"   Plot saved: {output_file2}")

    plt.close('all')


if __name__ == '__main__':
    main()
