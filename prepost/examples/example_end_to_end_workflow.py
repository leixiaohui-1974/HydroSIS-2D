# -*- coding: utf-8 -*-
"""
Complete End-to-End Workflow Example

This example demonstrates the complete HydroSIS-2D workflow:
1. Preprocessing: Create mesh, terrain, boundary conditions, initial conditions
2. Solving: Run the shallow water solver
3. Postprocessing: Visualize and analyze results

This showcases the integration of all modules:
- preprocessing (mesh generation, geometry, BC, IC)
- solver (shallow water equations)
- postprocessing (visualization, analysis)

Author: HydroSIS-2D Development Team
Date: 2025-10-29
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

# Import all modules
from simulation import create_dam_break_simulation
from solver import ShallowWaterSolver, SolverConfig
from postprocessing.analysis.result_analyzer import ResultAnalyzer
from postprocessing.visualization.visualization_engine import VisualizationEngine


def main():
    print("\n" + "="*80)
    print(" "*20 + "HYDROSIS-2D COMPLETE WORKFLOW DEMO")
    print("="*80)

    # ========================================================================
    # STEP 1: PREPROCESSING
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 1: PREPROCESSING")
    print("="*80)

    print("\n1.1 Creating simulation configuration...")
    # Using the factory function from simulation module
    config = create_dam_break_simulation(
        length=200.0,
        width=40.0,
        dam_position=0.5,
        upstream_depth=10.0,
        downstream_depth=1.0,
        nx=200,
        ny=40,
        simulation_time=10.0
    )

    print("   Configuration created with:")
    print(f"     Domain: 200m x 40m")
    print(f"     Grid: 200 x 40 = 8,000 cells")
    print(f"     Dam at 50% (x=100m)")
    print(f"     Upstream depth: 10m")
    print(f"     Downstream depth: 1m")

    # Validate configuration
    print("\n1.2 Validating configuration...")
    is_valid, errors = config.validate()
    if is_valid:
        print("   [OK] Configuration is valid")
    else:
        print("   [ERROR] Configuration errors:")
        for err in errors:
            print(f"     - {err}")
        return

    # Export configuration (optional)
    print("\n1.3 Exporting configuration...")
    config.export_configuration('output/end_to_end')
    print("   Configuration exported to: output/end_to_end/")

    # ========================================================================
    # STEP 2: SOLVING
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 2: SOLVING")
    print("="*80)

    print("\n2.1 Initializing solver...")
    # Get mesh and terrain from config
    mesh = config.mesh
    terrain = np.zeros((mesh.nx, mesh.ny))  # Flat bed

    # Create solver configuration
    solver_config = SolverConfig(
        t_end=10.0,
        cfl=0.5,
        output_interval=1.0,  # Output every 1 second
        output_dir='output/end_to_end/results',
        print_progress=True,
        progress_interval=100,
        manning_n=0.0  # No friction for this test
    )

    # Create solver
    solver = ShallowWaterSolver(mesh, terrain, solver_config)

    # Set initial conditions from preprocessing
    print("\n2.2 Setting initial conditions...")
    ic_manager = config.ic_manager
    h0 = ic_manager.depth
    u0 = ic_manager.velocity_x
    v0 = ic_manager.velocity_y

    solver.set_initial_conditions(h0, u0, v0)

    # Run solver
    print("\n2.3 Running solver...")
    print(f"   This will simulate {solver_config.t_end}s of physical time")
    print(f"   Expected ~{mesh.ncells} cells, ~0.04s per time step")
    print(f"   Estimated ~{int(solver_config.t_end/0.04)} time steps\n")

    stats = solver.solve()

    print("\n2.4 Solver completed!")
    print(f"   Time steps: {stats['steps']}")
    print(f"   Wall time: {stats['wall_time']:.1f}s")
    print(f"   Speed: {stats['speed_factor']:.2f}x realtime")
    print(f"   Mass conservation: {stats['mass_error']:+.6f}%")

    # ========================================================================
    # STEP 3: POSTPROCESSING
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 3: POSTPROCESSING")
    print("="*80)

    print("\n3.1 Loading results...")
    analyzer = ResultAnalyzer()
    result_pattern = 'output/end_to_end/results/result_*.vtk'

    # Note: Our simple VTK writer doesn't follow the exact format expected by ResultAnalyzer
    # So we'll work directly with the solver state instead
    print("   Using final solver state for visualization...")

    # Get final state from solver
    state = solver.get_state()

    print(f"   Final time: t = {state['t']:.2f}s")
    print(f"   Max depth: {np.max(state['h']):.2f}m")
    print(f"   Max velocity: {np.max(np.sqrt(state['u']**2 + state['v']**2)):.2f}m/s")

    # ========================================================================
    # STEP 4: VISUALIZATION
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 4: VISUALIZATION")
    print("="*80)

    print("\n4.1 Creating visualizations...")
    create_comprehensive_plots(mesh, state, stats)

    print("\n" + "="*80)
    print("WORKFLOW COMPLETED SUCCESSFULLY!")
    print("="*80)

    print("\nOutputs created:")
    print("  - Configuration: output/end_to_end/simulation_config.json")
    print("  - VTK results: output/end_to_end/results/result_*.vtk")
    print("  - Visualizations: output/end_to_end/analysis/*.png")

    print("\n" + "="*80 + "\n")


def create_comprehensive_plots(mesh, state, stats):
    """Create comprehensive visualization plots"""
    h = state['h']
    u = state['u']
    v = state['v']
    z = state['z']
    t = state['t']

    output_dir = 'output/end_to_end/analysis'
    os.makedirs(output_dir, exist_ok=True)

    # ========================================================================
    # Plot 1: 1D Centerline Profile
    # ========================================================================
    j_center = mesh.ny // 2
    x = mesh.x[:, j_center]
    h_profile = h[:, j_center]
    u_profile = u[:, j_center]
    z_profile = z[:, j_center]

    fig1, axes = plt.subplots(4, 1, figsize=(14, 12))

    # Depth profile
    axes[0].plot(x, h_profile, 'b-', linewidth=2.5, label='Water depth')
    axes[0].fill_between(x, 0, h_profile, alpha=0.3, color='blue')
    axes[0].plot(x, z_profile, 'k-', linewidth=2, label='Bed elevation')
    axes[0].axvline(x=100, color='r', linestyle='--', alpha=0.5, label='Initial dam position')
    axes[0].set_ylabel('Elevation [m]', fontsize=12, fontweight='bold')
    axes[0].set_title(f'1D Dam Break Profile at t = {t:.2f}s', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11, loc='upper right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 200)
    axes[0].set_ylim(0, 12)

    # Velocity profile
    axes[1].plot(x, u_profile, 'r-', linewidth=2.5)
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1].axvline(x=100, color='r', linestyle='--', alpha=0.5)
    axes[1].set_ylabel('Velocity [m/s]', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 200)

    # Froude number
    Fr = np.abs(u_profile) / np.sqrt(9.81 * (h_profile + 1e-6))
    axes[2].plot(x, Fr, 'g-', linewidth=2.5)
    axes[2].axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Critical (Fr=1)')
    axes[2].axvline(x=100, color='r', linestyle='--', alpha=0.5)
    axes[2].set_ylabel('Froude Number', fontsize=12, fontweight='bold')
    axes[2].legend(fontsize=11)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(0, 200)
    axes[2].set_ylim(0, 2.5)

    # Specific energy
    E = h_profile + u_profile**2 / (2 * 9.81)
    axes[3].plot(x, E, 'm-', linewidth=2.5)
    axes[3].axvline(x=100, color='r', linestyle='--', alpha=0.5)
    axes[3].set_xlabel('Distance [m]', fontsize=12, fontweight='bold')
    axes[3].set_ylabel('Specific Energy [m]', fontsize=12, fontweight='bold')
    axes[3].grid(True, alpha=0.3)
    axes[3].set_xlim(0, 200)

    plt.tight_layout()
    filename1 = os.path.join(output_dir, '1_centerline_profile.png')
    plt.savefig(filename1, dpi=150, bbox_inches='tight')
    print(f"   [OK] Saved: {filename1}")
    plt.close()

    # ========================================================================
    # Plot 2: 2D Contour Maps
    # ========================================================================
    fig2, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Water depth
    levels_h = np.linspace(0, np.max(h), 25)
    cs1 = axes[0, 0].contourf(mesh.x, mesh.y, h, levels=levels_h, cmap='Blues')
    axes[0, 0].contour(mesh.x, mesh.y, h, levels=10, colors='black', linewidths=0.5, alpha=0.3)
    plt.colorbar(cs1, ax=axes[0, 0], label='Water Depth [m]')
    axes[0, 0].axvline(x=100, color='r', linestyle='--', alpha=0.7, linewidth=2)
    axes[0, 0].set_xlabel('X [m]', fontweight='bold')
    axes[0, 0].set_ylabel('Y [m]', fontweight='bold')
    axes[0, 0].set_title(f'Water Depth at t = {t:.2f}s', fontweight='bold', fontsize=13)
    axes[0, 0].set_aspect('equal')

    # Velocity magnitude
    V = np.sqrt(u**2 + v**2)
    levels_v = np.linspace(0, np.max(V), 25)
    cs2 = axes[0, 1].contourf(mesh.x, mesh.y, V, levels=levels_v, cmap='Reds')
    axes[0, 1].contour(mesh.x, mesh.y, V, levels=10, colors='black', linewidths=0.5, alpha=0.3)
    plt.colorbar(cs2, ax=axes[0, 1], label='Velocity [m/s]')
    axes[0, 1].axvline(x=100, color='r', linestyle='--', alpha=0.7, linewidth=2)
    axes[0, 1].set_xlabel('X [m]', fontweight='bold')
    axes[0, 1].set_ylabel('Y [m]', fontweight='bold')
    axes[0, 1].set_title(f'Velocity Magnitude at t = {t:.2f}s', fontweight='bold', fontsize=13)
    axes[0, 1].set_aspect('equal')

    # Froude number
    Fr_2d = V / np.sqrt(9.81 * (h + 1e-6))
    levels_fr = np.linspace(0, np.min([np.max(Fr_2d), 2.0]), 25)
    cs3 = axes[1, 0].contourf(mesh.x, mesh.y, Fr_2d, levels=levels_fr, cmap='RdYlGn_r')
    axes[1, 0].contour(mesh.x, mesh.y, Fr_2d, levels=[1.0], colors='black', linewidths=2, linestyles='--')
    plt.colorbar(cs3, ax=axes[1, 0], label='Froude Number')
    axes[1, 0].axvline(x=100, color='r', linestyle='--', alpha=0.7, linewidth=2)
    axes[1, 0].set_xlabel('X [m]', fontweight='bold')
    axes[1, 0].set_ylabel('Y [m]', fontweight='bold')
    axes[1, 0].set_title(f'Froude Number at t = {t:.2f}s', fontweight='bold', fontsize=13)
    axes[1, 0].set_aspect('equal')

    # Unit discharge (qx = u*h)
    qx = u * h
    levels_q = np.linspace(np.min(qx), np.max(qx), 25)
    cs4 = axes[1, 1].contourf(mesh.x, mesh.y, qx, levels=levels_q, cmap='viridis')
    axes[1, 1].contour(mesh.x, mesh.y, qx, levels=10, colors='black', linewidths=0.5, alpha=0.3)
    plt.colorbar(cs4, ax=axes[1, 1], label='Unit Discharge [m^2/s]')
    axes[1, 1].axvline(x=100, color='r', linestyle='--', alpha=0.7, linewidth=2)
    axes[1, 1].set_xlabel('X [m]', fontweight='bold')
    axes[1, 1].set_ylabel('Y [m]', fontweight='bold')
    axes[1, 1].set_title(f'Unit Discharge (qx) at t = {t:.2f}s', fontweight='bold', fontsize=13)
    axes[1, 1].set_aspect('equal')

    plt.tight_layout()
    filename2 = os.path.join(output_dir, '2_contour_maps.png')
    plt.savefig(filename2, dpi=150, bbox_inches='tight')
    print(f"   [OK] Saved: {filename2}")
    plt.close()

    # ========================================================================
    # Plot 3: Solver Statistics
    # ========================================================================
    fig3, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Statistics text
    stats_text = f"""
SOLVER STATISTICS

Simulation Time:  {stats['simulated_time']:.3f} s
Wall Clock Time:  {stats['wall_time']:.2f} s
Speed Factor:     {stats['speed_factor']:.2f}x realtime

Time Steps:       {stats['steps']}
Average dt:       {stats['simulated_time']/stats['steps']:.4f} s

Mass Conservation:
  Final Mass:     {stats['final_mass']:.2f} m^3
  Error:          {stats['mass_error']:+.6f}%

Flow Characteristics:
  Max Depth:      {np.max(h):.2f} m
  Max Velocity:   {np.max(V):.2f} m/s
  Max Froude:     {np.max(Fr_2d):.2f}
    """

    axes[0, 0].text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                    verticalalignment='center', transform=axes[0, 0].transAxes)
    axes[0, 0].axis('off')

    # Depth histogram
    axes[0, 1].hist(h.ravel(), bins=50, color='blue', alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Water Depth [m]', fontweight='bold')
    axes[0, 1].set_ylabel('Frequency', fontweight='bold')
    axes[0, 1].set_title('Depth Distribution', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # Velocity histogram
    axes[1, 0].hist(V.ravel(), bins=50, color='red', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Velocity [m/s]', fontweight='bold')
    axes[1, 0].set_ylabel('Frequency', fontweight='bold')
    axes[1, 0].set_title('Velocity Distribution', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # Froude number histogram
    axes[1, 1].hist(Fr_2d.ravel(), bins=50, color='green', alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(x=1.0, color='k', linestyle='--', linewidth=2, label='Critical (Fr=1)')
    axes[1, 1].set_xlabel('Froude Number', fontweight='bold')
    axes[1, 1].set_ylabel('Frequency', fontweight='bold')
    axes[1, 1].set_title('Froude Number Distribution', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    filename3 = os.path.join(output_dir, '3_statistics.png')
    plt.savefig(filename3, dpi=150, bbox_inches='tight')
    print(f"   [OK] Saved: {filename3}")
    plt.close()


if __name__ == '__main__':
    main()
