#!/usr/bin/env python3
"""
Example 4: Urban Flood Simulation

This example demonstrates a real-world application:
- Rainfall on urban terrain with buildings
- Complex topography with streets and structures
- Manning roughness zones (streets, grass, buildings)
- Realistic boundary conditions
- Real-time monitoring of flood extent

This type of simulation is commonly used for:
- Flood risk assessment
- Emergency response planning
- Infrastructure design
- Insurance risk modeling

Similar to commercial software workflows:
- RiverFlow2D: Urban flooding module
- TUFLOW: 2D urban flood modeling
- InfoWorks ICM: Integrated urban drainage
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).parent.parent / 'prepost'))

from preprocessing.mesh_generation import MeshGenerator, DomainParams
from preprocessing.geometry import GeometryGenerator
from preprocessing.initial_conditions import InitialConditionManager
from preprocessing.boundary_conditions import BoundaryConditionManager


def create_urban_terrain(mesh):
    """
    Create urban terrain with buildings and streets

    Layout:
    - Street grid pattern
    - Buildings as elevated blocks
    - Drainage channels
    - Gentle terrain slope
    """
    nx, ny = mesh.nx, mesh.ny
    x_centers = mesh.x
    y_centers = mesh.y

    # Base terrain: gentle slope towards outlet
    z = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            # 0.2% slope in x-direction, 0.1% in y-direction
            z[i, j] = 50.0 - 0.002 * x_centers[i] - 0.001 * y_centers[j]

    # Add buildings (elevated terrain blocks)
    buildings = [
        # (x_min, x_max, y_min, y_max, height)
        (200, 280, 150, 250, 5.0),   # Building 1
        (350, 450, 150, 250, 5.0),   # Building 2
        (200, 280, 350, 450, 5.0),   # Building 3
        (350, 450, 350, 450, 5.0),   # Building 4
        (100, 150, 250, 320, 3.0),   # Small building
        (520, 580, 200, 280, 4.0),   # Commercial
    ]

    for x_min, x_max, y_min, y_max, height in buildings:
        for i in range(nx):
            for j in range(ny):
                if x_min <= x_centers[i] <= x_max and y_min <= y_centers[j] <= y_max:
                    z[i, j] += height

    # Add drainage channel
    for i in range(nx):
        for j in range(ny):
            # Channel along y = 100m
            if 90 <= y_centers[j] <= 110:
                z[i, j] -= 0.5  # 0.5m depression

    return z


def create_manning_roughness_zones(mesh):
    """
    Create spatially varying Manning roughness

    Roughness values:
    - Paved streets: n = 0.015
    - Grass/parks: n = 0.035
    - Buildings: n = 0.100 (high resistance)
    """
    nx, ny = mesh.nx, mesh.ny
    x_centers = mesh.x
    y_centers = mesh.y

    # Default: paved (streets)
    manning_n = np.full((nx, ny), 0.015)

    # Buildings (from terrain function)
    buildings = [
        (200, 280, 150, 250),
        (350, 450, 150, 250),
        (200, 280, 350, 450),
        (350, 450, 350, 450),
        (100, 150, 250, 320),
        (520, 580, 200, 280),
    ]

    for x_min, x_max, y_min, y_max in buildings:
        for i in range(nx):
            for j in range(ny):
                if x_min <= x_centers[i] <= x_max and y_min <= y_centers[j] <= y_max:
                    manning_n[i, j] = 0.100  # High resistance

    # Parks/green spaces
    green_spaces = [
        (50, 180, 50, 140),     # Park 1
        (480, 590, 400, 490),   # Park 2
    ]

    for x_min, x_max, y_min, y_max in green_spaces:
        for i in range(nx):
            for j in range(ny):
                if x_min <= x_centers[i] <= x_max and y_min <= y_centers[j] <= y_max:
                    manning_n[i, j] = 0.035  # Grass

    return manning_n


def create_rainfall_source(mesh, intensity_mm_hr, duration_s):
    """
    Create rainfall source term

    Args:
        intensity_mm_hr: Rainfall intensity (mm/hr)
        duration_s: Rainfall duration (seconds)

    Returns:
        Source term array (m/s)
    """
    # Convert mm/hr to m/s
    intensity_m_s = (intensity_mm_hr / 1000.0) / 3600.0

    # Uniform rainfall over entire domain
    source = np.full((mesh.nx, mesh.ny), intensity_m_s)

    return source


def calculate_flood_metrics(h, z, mesh, threshold_depth=0.05):
    """
    Calculate flood extent and depth statistics

    Args:
        h: Water depth array (m)
        z: Terrain elevation (m)
        threshold_depth: Minimum depth to consider as flooded (m)
    """
    # Flooded cells
    flooded = h > threshold_depth
    n_flooded = np.sum(flooded)

    # Total flooded area
    cell_area = mesh.dx * mesh.dy
    flooded_area = n_flooded * cell_area

    # Total water volume
    total_volume = np.sum(h) * cell_area

    # Max flood depth
    max_depth = h.max()
    max_elevation = (h + z).max()

    # Average depth in flooded areas
    if n_flooded > 0:
        avg_flood_depth = h[flooded].mean()
    else:
        avg_flood_depth = 0.0

    return {
        'flooded_cells': n_flooded,
        'flooded_area_m2': flooded_area,
        'flooded_area_ha': flooded_area / 10000.0,
        'total_volume_m3': total_volume,
        'max_depth_m': max_depth,
        'max_elevation_m': max_elevation,
        'avg_flood_depth_m': avg_flood_depth,
        'percent_flooded': 100.0 * n_flooded / (mesh.nx * mesh.ny)
    }


def main():
    print("="*70)
    print("Example 4: Urban Flood Simulation")
    print("="*70)
    print("\nScenario: 100mm/hr rainfall over urban area")
    print("Similar to: RiverFlow2D Urban, TUFLOW, InfoWorks ICM")

    # ========================================================================
    # Step 1: Setup urban domain
    # ========================================================================
    print("\n[1/6] Creating urban domain...")

    # Urban area: 600m x 500m
    domain = DomainParams(xmin=0.0, xmax=600.0, ymin=0.0, ymax=500.0)
    mesh_gen = MeshGenerator(domain)

    # Fine mesh for urban features (2m resolution)
    mesh = mesh_gen.generate_uniform_mesh(nx=300, ny=250)

    print(f"  ✓ Domain: {domain.xmax}m × {domain.ymax}m")
    print(f"  ✓ Mesh: {mesh.nx}×{mesh.ny} = {mesh.ncells:,} cells")
    print(f"  ✓ Resolution: {mesh.dx}m × {mesh.dy}m")

    # ========================================================================
    # Step 2: Create urban terrain
    # ========================================================================
    print("\n[2/6] Generating urban terrain...")

    z = create_urban_terrain(mesh)

    print(f"  ✓ Buildings: 6 structures")
    print(f"  ✓ Elevation range: {z.min():.2f} to {z.max():.2f} m")
    print(f"  ✓ Terrain slope: ~0.2% (drainage)")

    # ========================================================================
    # Step 3: Manning roughness zones
    # ========================================================================
    print("\n[3/6] Setting up roughness zones...")

    manning_n = create_manning_roughness_zones(mesh)

    print(f"  ✓ Streets (paved):   n = 0.015")
    print(f"  ✓ Parks (grass):     n = 0.035")
    print(f"  ✓ Buildings:         n = 0.100")

    # ========================================================================
    # Step 4: Initial conditions (dry)
    # ========================================================================
    print("\n[4/6] Setting initial conditions...")

    ic_manager = InitialConditionManager(mesh)

    # Start dry (or minimal initial depth for numerical stability)
    h_init = np.full((mesh.nx, mesh.ny), 0.001)  # 1mm initial
    ic_manager.set_depth_array(h_init)
    ic_manager.set_velocity_zero()

    print(f"  ✓ Initial depth: 0.001 m (dry start)")
    print(f"  ✓ Initial velocity: 0 m/s")

    # ========================================================================
    # Step 5: Rainfall event
    # ========================================================================
    print("\n[5/6] Configuring rainfall event...")

    # Design storm: 100mm/hr for 30 minutes (1-in-100 year event)
    rainfall_intensity = 100.0  # mm/hr
    rainfall_duration = 30.0 * 60.0  # 30 minutes in seconds

    rainfall_source = create_rainfall_source(mesh, rainfall_intensity, rainfall_duration)

    # Total rainfall volume
    total_rainfall_mm = rainfall_intensity * (rainfall_duration / 3600.0)
    total_rainfall_m = total_rainfall_mm / 1000.0
    domain_area = (domain.xmax - domain.xmin) * (domain.ymax - domain.ymin)
    total_volume = total_rainfall_m * domain_area

    print(f"  ✓ Intensity: {rainfall_intensity} mm/hr")
    print(f"  ✓ Duration: {rainfall_duration/60.0:.1f} minutes")
    print(f"  ✓ Total rainfall: {total_rainfall_mm:.1f} mm")
    print(f"  ✓ Total volume: {total_volume:.0f} m³")

    # ========================================================================
    # Step 6: Setup GPU simulation
    # ========================================================================
    print("\n[6/6] Setting up GPU simulation...")

    try:
        import hydrosis2d_cuda
    except ImportError:
        print("  ✗ GPU solver not available (hydrosis2d_cuda not found)")
        print("\n  → This example demonstrates the setup workflow")
        print("  → When GPU solver is compiled, it will:")
        print("     1. Apply rainfall source term each time step")
        print("     2. Route water through urban terrain")
        print("     3. Handle complex flow around buildings")
        print("     4. Calculate flood extent in real-time")
        print("\n  To compile GPU solver:")
        print("    cd src/solver && mkdir build && cd build")
        print("    cmake .. -DCMAKE_CUDA_ARCHITECTURES=native")
        print("    make -j$(nproc)")

        # Continue with visualization of setup
        create_visualization_without_simulation(mesh, z, manning_n, h_init)
        return

    # Create solver
    solver = hydrosis2d_cuda.Solver()

    # Configure
    solver_config = hydrosis2d_cuda.SolverConfig()
    solver_config.nx = mesh.nx
    solver_config.ny = mesh.ny
    solver_config.dx = mesh.dx
    solver_config.dy = mesh.dy
    solver_config.g = 9.81
    solver_config.cfl = 0.5  # Conservative for complex urban flow
    solver_config.riemann_solver = hydrosis2d_cuda.RiemannSolver.HLLC
    solver_config.spatial_order = 1
    solver_config.manning_friction = True
    solver_config.rainfall_source = True

    solver.initialize(solver_config)

    print(f"  ✓ Solver configured")
    print(f"  ✓ CFL: {solver_config.cfl}")
    print(f"  ✓ Manning friction: enabled")
    print(f"  ✓ Rainfall source: enabled")

    # Set initial conditions
    solver.set_initial_conditions(h_init, ic_manager.velocity_x, ic_manager.velocity_y, z)
    solver.set_manning_field(manning_n)
    solver.set_rainfall_source(rainfall_source, rainfall_duration)

    # Simulation time: rainfall duration + drainage time
    t_simulation = rainfall_duration + 1800.0  # +30 min drainage

    print(f"\n[7/7] Running simulation...")
    print(f"  Simulation time: {t_simulation/60.0:.1f} minutes")
    print(f"  (Rainfall: {rainfall_duration/60.0:.1f} min, Drainage: 30 min)")

    # Callback to monitor flood progression
    flood_history = []

    def progress_callback(time, step, dt):
        if step % 50 == 0:
            result = solver.get_solution()
            h_current = result['h']
            metrics = calculate_flood_metrics(h_current, z, mesh)

            flood_history.append({
                'time': time,
                'step': step,
                'flooded_area_ha': metrics['flooded_area_ha'],
                'max_depth_m': metrics['max_depth_m'],
                'total_volume_m3': metrics['total_volume_m3']
            })

            print(f"  t={time/60.0:.1f}min, flooded area={metrics['flooded_area_ha']:.2f}ha, "
                  f"max depth={metrics['max_depth_m']:.2f}m")

    solver.set_callback(progress_callback, interval=50)

    # Run simulation
    solver.run(t_end=t_simulation)

    print(f"  ✓ Simulation complete!")

    # ========================================================================
    # Step 8: Analyze results
    # ========================================================================
    print("\n[8/8] Analyzing flood results...")

    result = solver.get_solution()
    h_final = result['h']
    u_final = result['u']
    v_final = result['v']
    t_final = result['time']

    metrics = calculate_flood_metrics(h_final, z, mesh)

    print(f"\n  Final Flood Metrics:")
    print(f"    Flooded area:     {metrics['flooded_area_ha']:.2f} ha")
    print(f"    Percent flooded:  {metrics['percent_flooded']:.1f}%")
    print(f"    Total volume:     {metrics['total_volume_m3']:.0f} m³")
    print(f"    Max flood depth:  {metrics['max_depth_m']:.2f} m")
    print(f"    Avg flood depth:  {metrics['avg_flood_depth_m']:.2f} m")

    # ========================================================================
    # Visualization
    # ========================================================================
    print("\n[9/9] Creating visualizations...")

    create_visualization_with_simulation(
        mesh, z, manning_n, h_init, h_final, u_final, v_final,
        flood_history, rainfall_intensity, t_final
    )

    print("\n" + "="*70)
    print("Urban flood simulation completed!")
    print("="*70)


def create_visualization_without_simulation(mesh, z, manning_n, h_init):
    """Create visualization of setup only (no simulation results)"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    extent = [0, mesh.nx * mesh.dx, 0, mesh.ny * mesh.dy]

    # Plot 1: Terrain elevation
    ax = axes[0, 0]
    im1 = ax.imshow(z.T, origin='lower', cmap='terrain', extent=extent)
    ax.set_title('Urban Terrain Elevation', fontsize=12, fontweight='bold')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    plt.colorbar(im1, ax=ax, label='Elevation (m)')

    # Plot 2: Manning roughness
    ax = axes[0, 1]
    im2 = ax.imshow(manning_n.T, origin='lower', cmap='YlOrRd', extent=extent)
    ax.set_title('Manning Roughness Zones', fontsize=12, fontweight='bold')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    cbar = plt.colorbar(im2, ax=ax, label="Manning's n")
    cbar.ax.text(0.5, 0.95, 'Buildings', transform=cbar.ax.transAxes, ha='center')
    cbar.ax.text(0.5, 0.50, 'Parks', transform=cbar.ax.transAxes, ha='center')
    cbar.ax.text(0.5, 0.15, 'Streets', transform=cbar.ax.transAxes, ha='center')

    # Plot 3: Domain layout
    ax = axes[1, 0]
    ax.imshow(z.T, origin='lower', cmap='gray', alpha=0.3, extent=extent)

    # Highlight features
    x_centers = mesh.x
    y_centers = mesh.y

    # Mark buildings
    buildings_x = []
    buildings_y = []
    for i in range(mesh.nx):
        for j in range(mesh.ny):
            if manning_n[i, j] > 0.08:  # Building roughness
                buildings_x.append(x_centers[i])
                buildings_y.append(y_centers[j])

    ax.scatter(buildings_x, buildings_y, c='red', s=1, alpha=0.5, label='Buildings')

    # Mark parks
    parks_x = []
    parks_y = []
    for i in range(mesh.nx):
        for j in range(mesh.ny):
            if 0.03 < manning_n[i, j] < 0.04:  # Park roughness
                parks_x.append(x_centers[i])
                parks_y.append(y_centers[j])

    ax.scatter(parks_x, parks_y, c='green', s=1, alpha=0.3, label='Parks')

    ax.set_title('Urban Layout', fontsize=12, fontweight='bold')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.legend()

    # Plot 4: Simulation info
    ax = axes[1, 1]
    ax.axis('off')

    info_text = """
    URBAN FLOOD SIMULATION SETUP
    ═════════════════════════════

    Domain:
      • Size: 600m × 500m
      • Resolution: 2m × 2m
      • Total cells: {:,}

    Features:
      • 6 buildings (5m height)
      • 2 parks (grass)
      • Street network (paved)
      • Drainage channel

    Rainfall Event:
      • Intensity: 100 mm/hr
      • Duration: 30 minutes
      • Total: 50 mm
      • Return period: ~100 years

    Physics:
      • Manning friction (spatially varying)
      • Complex terrain (buildings)
      • 2D shallow water equations
      • HLLC Riemann solver

    Status:
      ⚠ GPU solver not compiled
      → Setup ready for simulation
      → See compilation instructions above

    Commercial Equivalents:
      • RiverFlow2D Urban
      • TUFLOW GPU
      • InfoWorks ICM
    """.format(mesh.ncells)

    ax.text(0.05, 0.95, info_text, transform=ax.transAxes,
            fontfamily='monospace', fontsize=9,
            verticalalignment='top')

    plt.tight_layout()

    # Save
    output_file = Path(__file__).parent / 'output' / '04_urban_flood_setup.png'
    output_file.parent.mkdir(exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  ✓ Figure saved: {output_file}")

    plt.show()


def create_visualization_with_simulation(mesh, z, manning_n, h_init, h_final,
                                        u_final, v_final, flood_history,
                                        rainfall_intensity, t_final):
    """Create comprehensive visualization with simulation results"""

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    extent = [0, mesh.nx * mesh.dx, 0, mesh.ny * mesh.dy]

    # Plot 1: Terrain
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(z.T, origin='lower', cmap='terrain', extent=extent)
    ax1.set_title('Terrain Elevation', fontweight='bold')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    plt.colorbar(im1, ax=ax1, label='Elevation (m)', shrink=0.8)

    # Plot 2: Final flood depth
    ax2 = fig.add_subplot(gs[0, 1])
    h_plot = np.ma.masked_where(h_final < 0.01, h_final)
    im2 = ax2.imshow(h_plot.T, origin='lower', cmap='Blues', extent=extent, vmin=0, vmax=1.0)
    ax2.set_title(f'Final Flood Depth (t={t_final/60:.0f}min)', fontweight='bold')
    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('y (m)')
    plt.colorbar(im2, ax=ax2, label='Depth (m)', shrink=0.8)

    # Plot 3: Velocity magnitude
    ax3 = fig.add_subplot(gs[0, 2])
    vel_mag = np.sqrt(u_final**2 + v_final**2)
    vel_plot = np.ma.masked_where(h_final < 0.01, vel_mag)
    im3 = ax3.imshow(vel_plot.T, origin='lower', cmap='Reds', extent=extent)
    ax3.set_title('Flow Velocity', fontweight='bold')
    ax3.set_xlabel('x (m)')
    ax3.set_ylabel('y (m)')
    plt.colorbar(im3, ax=ax3, label='Velocity (m/s)', shrink=0.8)

    # Plot 4: Flood extent over terrain
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(z.T, origin='lower', cmap='gray', alpha=0.4, extent=extent)
    flood_extent = h_final > 0.05
    ax4.contourf(flood_extent.T, levels=[0.5, 1.5], colors=['blue'], alpha=0.5,
                 extent=extent, origin='lower')
    ax4.set_title('Flood Extent Overlay', fontweight='bold')
    ax4.set_xlabel('x (m)')
    ax4.set_ylabel('y (m)')

    # Plot 5: Water surface elevation
    ax5 = fig.add_subplot(gs[1, 1])
    eta = h_final + z
    im5 = ax5.imshow(eta.T, origin='lower', cmap='viridis', extent=extent)
    ax5.set_title('Water Surface Elevation', fontweight='bold')
    ax5.set_xlabel('x (m)')
    ax5.set_ylabel('y (m)')
    plt.colorbar(im5, ax=ax5, label='Elevation (m)', shrink=0.8)

    # Plot 6: Flow vectors
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.imshow(h_plot.T, origin='lower', cmap='Blues', extent=extent, alpha=0.5)

    # Subsample for quiver
    skip = 15
    X, Y = np.meshgrid(mesh.x[::skip], mesh.y[::skip], indexing='ij')
    U = u_final[::skip, ::skip]
    V = v_final[::skip, ::skip]
    H = h_final[::skip, ::skip]

    # Only show vectors where water exists
    mask = H > 0.05
    ax6.quiver(X[mask], Y[mask], U[mask], V[mask],
               scale=20, width=0.003, color='red', alpha=0.7)
    ax6.set_title('Flow Vectors', fontweight='bold')
    ax6.set_xlabel('x (m)')
    ax6.set_ylabel('y (m)')

    # Plot 7: Flood progression
    ax7 = fig.add_subplot(gs[2, :2])
    times = [h['time']/60.0 for h in flood_history]
    flooded_areas = [h['flooded_area_ha'] for h in flood_history]
    max_depths = [h['max_depth_m'] for h in flood_history]

    ax7_twin = ax7.twinx()

    line1 = ax7.plot(times, flooded_areas, 'b-', linewidth=2, label='Flooded Area')
    ax7.axvline(x=30, color='gray', linestyle='--', alpha=0.5, label='Rainfall ends')
    ax7.set_xlabel('Time (minutes)', fontweight='bold')
    ax7.set_ylabel('Flooded Area (hectares)', color='b', fontweight='bold')
    ax7.tick_params(axis='y', labelcolor='b')
    ax7.grid(True, alpha=0.3)

    line2 = ax7_twin.plot(times, max_depths, 'r-', linewidth=2, label='Max Depth')
    ax7_twin.set_ylabel('Maximum Depth (m)', color='r', fontweight='bold')
    ax7_twin.tick_params(axis='y', labelcolor='r')

    ax7.set_title('Flood Progression Over Time', fontweight='bold')

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax7.legend(lines, labels, loc='upper left')

    # Plot 8: Statistics
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')

    metrics = calculate_flood_metrics(h_final, z, mesh)

    stats_text = f"""
FLOOD ANALYSIS RESULTS
═══════════════════════

Rainfall Event:
  Intensity: {rainfall_intensity} mm/hr
  Duration: 30 minutes
  Total: {rainfall_intensity * 0.5:.1f} mm

Flood Extent:
  Area: {metrics['flooded_area_ha']:.2f} ha
  Percentage: {metrics['percent_flooded']:.1f}%
  Volume: {metrics['total_volume_m3']:.0f} m³

Water Depth:
  Maximum: {metrics['max_depth_m']:.2f} m
  Average: {metrics['avg_flood_depth_m']:.2f} m

Surface Elevation:
  Maximum: {metrics['max_elevation_m']:.2f} m

Simulation:
  Final time: {t_final/60:.0f} min
  Grid cells: {mesh.ncells:,}
  Resolution: {mesh.dx:.1f}m
"""

    ax8.text(0.05, 0.95, stats_text, transform=ax8.transAxes,
             fontfamily='monospace', fontsize=9,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Save
    output_file = Path(__file__).parent / 'output' / '04_urban_flood_results.png'
    output_file.parent.mkdir(exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  ✓ Figure saved: {output_file}")

    plt.show()


if __name__ == "__main__":
    main()
