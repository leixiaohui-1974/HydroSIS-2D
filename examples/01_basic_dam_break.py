#!/usr/bin/env python3
"""
Example 1: Basic Dam Break Simulation

This example demonstrates the simplest use case:
- Create a dam break scenario
- Run GPU simulation
- Visualize results

Requirements:
- GPU with CUDA support
- hydrosis2d_cuda module compiled
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add preprocessing to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'prepost'))

from preprocessing.utils import create_dam_break_simulation


def main():
    print("="*60)
    print("Example 1: Basic Dam Break")
    print("="*60)

    # Step 1: Create dam break configuration
    print("\n[1/4] Creating dam break configuration...")

    config = create_dam_break_simulation(
        length=200.0,           # Domain length (m)
        width=100.0,            # Domain width (m)
        nx=200,                 # Grid cells in x
        ny=100,                 # Grid cells in y
        dam_position=0.5,       # Dam at x = 100m (50%)
        upstream_depth=10.0,    # Upstream water depth (m)
        downstream_depth=1.0,   # Downstream water depth (m)
        simulation_time=10.0    # Simulation time (s)
    )

    mesh = config.mesh
    ic_manager = config.ic_manager

    print(f"  ✓ Mesh created: {mesh.nx}×{mesh.ny} = {mesh.ncells:,} cells")
    print(f"  ✓ Grid spacing: dx={mesh.dx}m, dy={mesh.dy}m")
    print(f"  ✓ Initial conditions set")

    # Step 2: Setup GPU solver
    print("\n[2/4] Setting up GPU solver...")

    try:
        import hydrosis2d_cuda
    except ImportError:
        print("  ✗ GPU solver not available (hydrosis2d_cuda not found)")
        print("  → Please compile GPU solver first:")
        print("    cd src/solver && mkdir build && cd build")
        print("    cmake .. -DCMAKE_CUDA_ARCHITECTURES=native")
        print("    make -j$(nproc)")
        sys.exit(1)

    # Create solver
    solver = hydrosis2d_cuda.Solver()

    # Configure solver
    solver_config = hydrosis2d_cuda.SolverConfig()
    solver_config.nx = mesh.nx
    solver_config.ny = mesh.ny
    solver_config.dx = mesh.dx
    solver_config.dy = mesh.dy
    solver_config.g = 9.81
    solver_config.cfl = 0.8
    solver_config.riemann_solver = hydrosis2d_cuda.RiemannSolver.HLLC
    solver_config.spatial_order = 1  # First-order for stability
    solver_config.time_integrator = hydrosis2d_cuda.TimeIntegrator.EULER

    solver.initialize(solver_config)

    print(f"  ✓ Solver configured:")
    print(f"    - Riemann solver: HLLC")
    print(f"    - Spatial order: 1 (Godunov)")
    print(f"    - Time integrator: Euler")
    print(f"    - CFL number: 0.8")

    # Step 3: Set initial conditions
    print("\n[3/4] Setting initial conditions...")

    h_init = ic_manager.depth
    u_init = ic_manager.velocity_x
    v_init = ic_manager.velocity_y
    z_init = np.zeros_like(h_init)  # Flat bed

    solver.set_initial_conditions(h_init, u_init, v_init, z_init)

    # Calculate initial mass
    mass_initial = np.sum(h_init) * mesh.dx * mesh.dy

    print(f"  ✓ Initial conditions loaded")
    print(f"  ✓ Initial mass: {mass_initial:.2f} m³")

    # Step 4: Run simulation
    print("\n[4/4] Running simulation...")

    def progress_callback(time, step, dt):
        """Progress callback"""
        if step % 100 == 0:
            print(f"  t={time:.2f}s, step={step}, dt={dt:.4f}s")

    solver.set_callback(progress_callback, interval=100)

    t_end = 10.0
    solver.run(t_end=t_end)

    print(f"  ✓ Simulation complete!")

    # Step 5: Get results
    print("\n[5/5] Analyzing results...")

    result = solver.get_solution()
    h_final = result['h']
    u_final = result['u']
    v_final = result['v']
    t_final = result['time']
    n_steps = result['steps']

    # Calculate final mass
    mass_final = np.sum(h_final) * mesh.dx * mesh.dy
    mass_error = abs(mass_final - mass_initial) / mass_initial

    print(f"  ✓ Final time: {t_final:.2f} s")
    print(f"  ✓ Total steps: {n_steps}")
    print(f"  ✓ Final mass: {mass_final:.2f} m³")
    print(f"  ✓ Mass error: {mass_error:.2e}")

    # Statistics
    print(f"\n  Water depth statistics:")
    print(f"    Max: {h_final.max():.2f} m")
    print(f"    Min: {h_final.min():.2f} m")
    print(f"    Mean: {h_final.mean():.2f} m")

    # Step 6: Visualize results
    print("\n[6/6] Creating visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Initial depth
    ax = axes[0, 0]
    im1 = ax.imshow(h_init.T, origin='lower', cmap='Blues',
                    extent=[0, mesh.nx*mesh.dx, 0, mesh.ny*mesh.dy])
    ax.set_title('Initial Water Depth')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    plt.colorbar(im1, ax=ax, label='Depth (m)')

    # Plot 2: Final depth
    ax = axes[0, 1]
    im2 = ax.imshow(h_final.T, origin='lower', cmap='Blues',
                    extent=[0, mesh.nx*mesh.dx, 0, mesh.ny*mesh.dy])
    ax.set_title(f'Final Water Depth (t={t_final:.1f}s)')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    plt.colorbar(im2, ax=ax, label='Depth (m)')

    # Plot 3: Velocity magnitude
    ax = axes[1, 0]
    vel_mag = np.sqrt(u_final**2 + v_final**2)
    im3 = ax.imshow(vel_mag.T, origin='lower', cmap='Reds',
                    extent=[0, mesh.nx*mesh.dx, 0, mesh.ny*mesh.dy])
    ax.set_title('Final Velocity Magnitude')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    plt.colorbar(im3, ax=ax, label='Velocity (m/s)')

    # Plot 4: Cross-section at y=50m
    ax = axes[1, 1]
    j_mid = mesh.ny // 2
    x_coords = np.linspace(0, mesh.nx*mesh.dx, mesh.nx)

    ax.plot(x_coords, h_init[:, j_mid], 'b--', label='Initial', linewidth=2)
    ax.plot(x_coords, h_final[:, j_mid], 'b-', label='Final', linewidth=2)
    ax.axvline(x=mesh.nx*mesh.dx/2, color='k', linestyle=':', alpha=0.5, label='Dam')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('Water Depth (m)')
    ax.set_title(f'Cross-section at y={j_mid*mesh.dy:.0f}m')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_file = Path(__file__).parent / 'output' / '01_dam_break_results.png'
    output_file.parent.mkdir(exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  ✓ Figure saved: {output_file}")

    plt.show()

    print("\n" + "="*60)
    print("Example completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
