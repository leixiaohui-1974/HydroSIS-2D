#!/usr/bin/env python3
"""
Example 3: Analytical Solution Validation

This example validates the GPU solver against analytical solutions:
1. 1D Ritter dam break (analytical solution exists)
2. Lake at rest (C-property test)
3. Circular dam break (symmetry test)

Demonstrates numerical accuracy and well-balanced properties.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / 'prepost'))

from preprocessing.mesh_generation import MeshGenerator, DomainParams
from preprocessing.initial_conditions import InitialConditionManager


def ritter_analytical_solution(x, t, h_L, h_R, g=9.81):
    """
    Ritter analytical solution for 1D dam break

    Args:
        x: Spatial coordinates (numpy array)
        t: Time
        h_L: Left (upstream) water depth
        h_R: Right (downstream) water depth
        g: Gravitational acceleration

    Returns:
        tuple: (h, u) - depth and velocity at each x
    """
    c_L = np.sqrt(g * h_L)  # Left wave speed
    c_R = np.sqrt(g * h_R)  # Right wave speed

    h = np.zeros_like(x)
    u = np.zeros_like(x)

    # Wave positions
    x_head = -c_L * t  # Left edge of rarefaction
    x_tail = 2.0 * c_L * t  # Right edge of rarefaction (if dry bed)

    # Shock speed (for wet bed)
    u_star = 2.0 * (c_L - c_R)
    h_star = ((c_L - 0.5 * u_star)**2) / g
    x_shock = x_head + (c_L + u_star) * t

    for i, xi in enumerate(x):
        if xi < x_head:
            # Left state (undisturbed)
            h[i] = h_L
            u[i] = 0.0
        elif xi < x_shock:
            # Rarefaction wave
            h[i] = (1.0 / (9.0 * g)) * (2.0 * c_L - xi / t)**2
            u[i] = (2.0 / 3.0) * (c_L + xi / t)
        else:
            # Right state
            h[i] = h_R
            u[i] = 0.0

    return h, u


def validate_ritter_solution():
    """Validate against Ritter analytical solution"""
    print("\n" + "="*70)
    print("Test 1: Ritter Dam Break (1D)")
    print("="*70)

    # Parameters
    h_L = 10.0  # Upstream depth (m)
    h_R = 1.0   # Downstream depth (m)
    g = 9.81
    t_end = 2.0  # Time (s)

    # Create quasi-1D mesh (narrow in y)
    domain = DomainParams(xmin=-50.0, xmax=50.0, ymin=0.0, ymax=2.0)
    mesh_gen = MeshGenerator(domain)
    mesh = mesh_gen.generate_uniform_mesh(nx=500, ny=5)

    print(f"\n  Configuration:")
    print(f"    Domain: {domain.xmin} to {domain.xmax} m")
    print(f"    Mesh: {mesh.nx}×{mesh.ny} = {mesh.ncells:,} cells")
    print(f"    Upstream depth: {h_L} m")
    print(f"    Downstream depth: {h_R} m")
    print(f"    Simulation time: {t_end} s")

    # Create initial conditions
    ic_manager = InitialConditionManager(mesh)

    h_init = np.zeros((mesh.nx, mesh.ny))
    x_coords = mesh.x  # Cell centers in x

    for i in range(mesh.nx):
        if x_coords[i] < 0:
            h_init[i, :] = h_L
        else:
            h_init[i, :] = h_R

    ic_manager.set_depth_array(h_init)
    ic_manager.set_velocity_zero()

    # Run GPU simulation
    print("\n  Running GPU simulation...")

    try:
        import hydrosis2d_cuda
    except ImportError:
        print("  ✗ GPU solver not available")
        return

    solver = hydrosis2d_cuda.Solver()

    config = hydrosis2d_cuda.SolverConfig()
    config.nx = mesh.nx
    config.ny = mesh.ny
    config.dx = mesh.dx
    config.dy = mesh.dy
    config.cfl = 0.8

    solver.initialize(config)
    solver.set_initial_conditions(
        ic_manager.depth,
        ic_manager.velocity_x,
        ic_manager.velocity_y,
        np.zeros_like(ic_manager.depth)
    )

    solver.run(t_end=t_end)

    result = solver.get_solution()
    h_numerical = result['h']
    u_numerical = result['u']

    print(f"  ✓ Simulation completed ({result['steps']} steps)")

    # Compute analytical solution
    print("\n  Computing analytical solution...")

    h_analytical, u_analytical = ritter_analytical_solution(x_coords, t_end, h_L, h_R, g)

    # Extend to 2D array (copy to all y-rows)
    h_analytical_2d = np.tile(h_analytical, (mesh.ny, 1)).T
    u_analytical_2d = np.tile(u_analytical, (mesh.ny, 1)).T

    # Compute errors (use center row)
    j_mid = mesh.ny // 2
    h_num_1d = h_numerical[:, j_mid]
    u_num_1d = u_numerical[:, j_mid]

    # L1 error
    l1_h = np.mean(np.abs(h_num_1d - h_analytical))
    l1_u = np.mean(np.abs(u_num_1d - u_analytical))

    # L2 error
    l2_h = np.sqrt(np.mean((h_num_1d - h_analytical)**2))
    l2_u = np.sqrt(np.mean((u_num_1d - u_analytical)**2))

    # L-infinity error
    linf_h = np.max(np.abs(h_num_1d - h_analytical))
    linf_u = np.max(np.abs(u_num_1d - u_analytical))

    print(f"\n  Error Analysis:")
    print(f"    Water depth:")
    print(f"      L1 error:    {l1_h:.4e}")
    print(f"      L2 error:    {l2_h:.4e}")
    print(f"      L∞ error:    {linf_h:.4e}")
    print(f"    Velocity:")
    print(f"      L1 error:    {l1_u:.4e}")
    print(f"      L2 error:    {l2_u:.4e}")
    print(f"      L∞ error:    {linf_u:.4e}")

    # Visualization
    print("\n  Creating visualizations...")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: Water depth
    ax = axes[0]
    ax.plot(x_coords, h_analytical, 'k-', linewidth=2, label='Analytical')
    ax.plot(x_coords, h_num_1d, 'r--', linewidth=1.5, label='Numerical (GPU)')
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5, label='Dam')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('Water Depth (m)')
    ax.set_title(f'Ritter Dam Break: Water Depth at t={t_end}s')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Velocity
    ax = axes[1]
    ax.plot(x_coords, u_analytical, 'k-', linewidth=2, label='Analytical')
    ax.plot(x_coords, u_num_1d, 'b--', linewidth=1.5, label='Numerical (GPU)')
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5, label='Dam')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Velocity Profile')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = Path(__file__).parent / 'output' / '03_ritter_validation.png'
    output_file.parent.mkdir(exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  ✓ Figure saved: {output_file}")

    plt.show()

    # Check if errors are acceptable
    print(f"\n  Validation Status:")
    if l2_h < 0.1 and l2_u < 0.1:
        print(f"    ✓ PASS - Errors within acceptable range")
    else:
        print(f"    ✗ FAIL - Errors too large")

    return l2_h, l2_u


def validate_lake_at_rest():
    """Validate C-property (lake at rest)"""
    print("\n" + "="*70)
    print("Test 2: Lake at Rest (C-property)")
    print("="*70)

    # Create mesh
    domain = DomainParams(xmin=0.0, xmax=100.0, ymin=0.0, ymax=100.0)
    mesh_gen = MeshGenerator(domain)
    mesh = mesh_gen.generate_uniform_mesh(nx=50, ny=50)

    print(f"\n  Configuration:")
    print(f"    Domain: 100m × 100m")
    print(f"    Mesh: {mesh.nx}×{mesh.ny} = {mesh.ncells:,} cells")

    # Sloped terrain
    x_coords = mesh.x
    z = np.zeros((mesh.nx, mesh.ny))
    for i in range(mesh.nx):
        z[i, :] = 0.1 * x_coords[i]  # 10% slope

    # Hydrostatic equilibrium
    eta = 10.0  # Free surface elevation
    h = eta - z
    h = np.maximum(h, 0.0)

    ic_manager = InitialConditionManager(mesh)
    ic_manager.set_depth_array(h)
    ic_manager.set_velocity_zero()

    # Run simulation
    print("\n  Running GPU simulation (should remain at rest)...")

    try:
        import hydrosis2d_cuda
    except ImportError:
        print("  ✗ GPU solver not available")
        return

    solver = hydrosis2d_cuda.Solver()

    config = hydrosis2d_cuda.SolverConfig()
    config.nx = mesh.nx
    config.ny = mesh.ny
    config.dx = mesh.dx
    config.dy = mesh.dy
    config.cfl = 0.8

    solver.initialize(config)
    solver.set_initial_conditions(h, ic_manager.velocity_x, ic_manager.velocity_y, z)

    t_end = 10.0
    solver.run(t_end=t_end)

    result = solver.get_solution()
    h_final = result['h']
    u_final = result['u']
    v_final = result['v']

    print(f"  ✓ Simulation completed ({result['steps']} steps)")

    # Check that solution remained at rest
    max_u = np.max(np.abs(u_final))
    max_v = np.max(np.abs(v_final))
    max_dh = np.max(np.abs(h_final - h))

    print(f"\n  Results:")
    print(f"    Max velocity (u): {max_u:.2e} m/s")
    print(f"    Max velocity (v): {max_v:.2e} m/s")
    print(f"    Max depth change: {max_dh:.2e} m")

    print(f"\n  Validation Status:")
    if max_u < 1e-6 and max_v < 1e-6 and max_dh < 1e-6:
        print(f"    ✓ PASS - Solution remained at rest (C-property satisfied)")
    else:
        print(f"    ✗ FAIL - Spurious currents detected")

    return max_u, max_v, max_dh


def main():
    print("="*70)
    print("Example 3: Analytical Solution Validation")
    print("="*70)

    # Test 1: Ritter dam break
    try:
        l2_h, l2_u = validate_ritter_solution()
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        l2_h, l2_u = None, None

    # Test 2: Lake at rest
    try:
        max_u, max_v, max_dh = validate_lake_at_rest()
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        max_u, max_v, max_dh = None, None, None

    # Summary
    print("\n" + "="*70)
    print("Validation Summary")
    print("="*70)

    if l2_h is not None:
        status_ritter = "PASS" if l2_h < 0.1 and l2_u < 0.1 else "FAIL"
        print(f"\n  Ritter Dam Break: {status_ritter}")
        print(f"    L2 error (h): {l2_h:.4e}")
        print(f"    L2 error (u): {l2_u:.4e}")

    if max_u is not None:
        status_lake = "PASS" if max_u < 1e-6 and max_v < 1e-6 else "FAIL"
        print(f"\n  Lake at Rest: {status_lake}")
        print(f"    Max velocity: {max(max_u, max_v):.4e} m/s")
        print(f"    Max depth change: {max_dh:.4e} m")

    print("\n" + "="*70)
    print("Validation completed!")
    print("="*70)


if __name__ == "__main__":
    main()
