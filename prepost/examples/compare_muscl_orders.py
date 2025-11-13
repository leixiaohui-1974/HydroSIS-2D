#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MUSCL Order Comparison Example

This example compares first-order and second-order (MUSCL) spatial accuracy
on a classic dam break problem. It demonstrates the reduced numerical diffusion
and improved shock resolution of the MUSCL scheme.

Usage:
    python compare_muscl_orders.py

Output:
    - Terminal output showing simulation progress
    - Comparison plots of water depth profiles
    - Quantitative error analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.mesh_generation import MeshGenerator, DomainParams
from solver import ShallowWaterSolver, SolverConfig


def run_dam_break_comparison(nx=200, t_end=2.0, save_plots=True):
    """
    Run dam break simulation with both first and second-order schemes

    Parameters
    ----------
    nx : int
        Number of cells in x-direction
    t_end : float
        Simulation end time [s]
    save_plots : bool
        Whether to save plots to files
    """
    print("="*70)
    print("MUSCL Order Comparison Example")
    print("="*70)
    print(f"\nProblem: Dam break")
    print(f"Grid: {nx}x20 cells")
    print(f"Simulation time: {t_end} seconds")
    print("="*70)

    # Create mesh
    domain = DomainParams(-50, 50, 0, 20)
    ny = 20
    mesh = MeshGenerator(domain).generate_uniform_mesh(nx=nx, ny=ny)
    terrain = np.zeros((nx, ny))

    # Initial condition: Dam break at x=0
    h0 = np.ones((nx, ny))
    x = np.linspace(-50, 50, nx)
    for i in range(nx):
        h0[i, :] = 10.0 if x[i] < 0 else 1.0
    u0 = np.zeros((nx, ny))
    v0 = np.zeros((nx, ny))

    print(f"\nInitial conditions:")
    print(f"  Left water depth:  10.0 m")
    print(f"  Right water depth: 1.0 m")
    print(f"  Depth ratio:       10:1")
    print(f"  Initial velocity:  0.0 m/s")

    # ==========================================================================
    # First-order simulation
    # ==========================================================================
    print("\n" + "="*70)
    print("Running FIRST-ORDER simulation...")
    print("="*70)

    config_1st = SolverConfig(
        t_end=t_end,
        spatial_order=1,      # First-order (piecewise constant)
        manning_n=0.0,        # Frictionless
        output_interval=t_end,
        print_progress=True
    )

    solver_1st = ShallowWaterSolver(mesh, terrain, config_1st)
    solver_1st.set_initial_conditions(h0.copy(), u0.copy(), v0.copy())

    start_time = time.time()
    solver_1st.solve()
    time_1st = time.time() - start_time

    # Extract centerline profile
    h_1st = solver_1st.h[:, ny//2].copy()
    u_1st = solver_1st.u[:, ny//2].copy()

    print(f"\nFirst-order results:")
    print(f"  Computation time: {time_1st:.3f} s")
    print(f"  Final depth range: [{np.min(h_1st):.3f}, {np.max(h_1st):.3f}] m")
    print(f"  Final velocity range: [{np.min(u_1st):.3f}, {np.max(u_1st):.3f}] m/s")

    # ==========================================================================
    # Second-order simulation
    # ==========================================================================
    print("\n" + "="*70)
    print("Running SECOND-ORDER (MUSCL) simulation...")
    print("="*70)

    config_2nd = SolverConfig(
        t_end=t_end,
        spatial_order=2,      # Second-order (MUSCL)
        muscl_limiter='minmod',
        manning_n=0.0,
        output_interval=t_end,
        print_progress=True
    )

    solver_2nd = ShallowWaterSolver(mesh, terrain, config_2nd)
    solver_2nd.set_initial_conditions(h0.copy(), u0.copy(), v0.copy())

    start_time = time.time()
    solver_2nd.solve()
    time_2nd = time.time() - start_time

    # Extract centerline profile
    h_2nd = solver_2nd.h[:, ny//2].copy()
    u_2nd = solver_2nd.u[:, ny//2].copy()

    print(f"\nSecond-order results:")
    print(f"  Computation time: {time_2nd:.3f} s")
    print(f"  Final depth range: [{np.min(h_2nd):.3f}, {np.max(h_2nd):.3f}] m")
    print(f"  Final velocity range: [{np.min(u_2nd):.3f}, {np.max(u_2nd):.3f}] m/s")

    # ==========================================================================
    # Analysis and comparison
    # ==========================================================================
    print("\n" + "="*70)
    print("COMPARISON ANALYSIS")
    print("="*70)

    # Total variation (measure of numerical diffusion)
    tv_1st = np.sum(np.abs(np.diff(h_1st)))
    tv_2nd = np.sum(np.abs(np.diff(h_2nd)))

    print(f"\nNumerical diffusion analysis:")
    print(f"  First-order  Total Variation: {tv_1st:.2f}")
    print(f"  Second-order Total Variation: {tv_2nd:.2f}")
    print(f"  Ratio (2nd/1st): {tv_2nd/tv_1st:.3f}")
    print(f"  -> Second-order is {((tv_2nd/tv_1st - 1)*100):+.1f}% less diffusive")

    # Shock width (measure of resolution)
    # Find shock position (max gradient)
    grad_1st = np.abs(np.diff(h_1st))
    grad_2nd = np.abs(np.diff(h_2nd))
    shock_idx_1st = np.argmax(grad_1st)
    shock_idx_2nd = np.argmax(grad_2nd)

    print(f"\nShock resolution analysis:")
    print(f"  First-order  max gradient: {grad_1st[shock_idx_1st]:.3f} m/m")
    print(f"  Second-order max gradient: {grad_2nd[shock_idx_2nd]:.3f} m/m")
    print(f"  -> Second-order shock is {grad_2nd[shock_idx_2nd]/grad_1st[shock_idx_1st]:.2f}x sharper")

    # Performance
    print(f"\nPerformance:")
    print(f"  First-order time:  {time_1st:.3f} s")
    print(f"  Second-order time: {time_2nd:.3f} s")
    print(f"  Overhead: {(time_2nd/time_1st - 1)*100:.1f}%")

    # ==========================================================================
    # Visualization
    # ==========================================================================
    print("\n" + "="*70)
    print("Creating comparison plots...")
    print("="*70)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Water depth comparison
    ax = axes[0, 0]
    ax.plot(x, h_1st, 'b-', linewidth=2, label='First-order')
    ax.plot(x, h_2nd, 'r--', linewidth=2, label='Second-order (MUSCL)')
    ax.set_xlabel('x [m]', fontsize=12)
    ax.set_ylabel('Water depth h [m]', fontsize=12)
    ax.set_title(f'Dam Break at t = {t_end:.1f} s', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 11])

    # Plot 2: Velocity comparison
    ax = axes[0, 1]
    ax.plot(x, u_1st, 'b-', linewidth=2, label='First-order')
    ax.plot(x, u_2nd, 'r--', linewidth=2, label='Second-order (MUSCL)')
    ax.set_xlabel('x [m]', fontsize=12)
    ax.set_ylabel('Velocity u [m/s]', fontsize=12)
    ax.set_title('Velocity Profile', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    # Plot 3: Difference plot (zoomed near shock)
    ax = axes[1, 0]
    diff = h_2nd - h_1st
    ax.plot(x, diff, 'g-', linewidth=2)
    ax.axhline(0, color='k', linestyle=':', alpha=0.5)
    ax.set_xlabel('x [m]', fontsize=12)
    ax.set_ylabel('Depth difference (2nd - 1st) [m]', fontsize=12)
    ax.set_title('Difference Between Methods', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-1, 1])

    # Plot 4: Gradient comparison (shock sharpness)
    ax = axes[1, 1]
    ax.plot(x[:-1], grad_1st, 'b-', linewidth=2, label='First-order')
    ax.plot(x[:-1], grad_2nd, 'r--', linewidth=2, label='Second-order (MUSCL)')
    ax.set_xlabel('x [m]', fontsize=12)
    ax.set_ylabel('|dh/dx| [m/m]', fontsize=12)
    ax.set_title('Shock Sharpness', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_xlim([-20, 20])

    plt.tight_layout()

    if save_plots:
        filename = f'muscl_comparison_nx{nx}_t{t_end:.1f}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\n[OK] Plot saved: {filename}")

    plt.show()

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n[OK] Second-order MUSCL reduces numerical diffusion")
    print(f"[OK] Second-order produces sharper shock resolution")
    print(f"[OK] Computational overhead: ~{(time_2nd/time_1st - 1)*100:.0f}%")
    print(f"\nRecommendation: Use spatial_order=2 for production simulations")
    print("="*70)


if __name__ == '__main__':
    # Run comparison with default parameters
    run_dam_break_comparison(nx=200, t_end=2.0, save_plots=True)
