"""
Example: Boundary Conditions Setup for HydroSIS-2D

Demonstrates various boundary condition types and configurations.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt

from preprocessing.boundary_conditions import (
    BCLocation,
    WallBC,
    InflowBC,
    OutflowBC,
    PeriodicBC,
    TransmissiveBC,
    TimeSeriesBC,
    FunctionBC,
    BoundaryConditionManager
)
from preprocessing.mesh_generation import DomainParams, MeshGenerator


def example_1_basic_wall_boundaries():
    """Example 1: Enclosed domain with wall boundaries"""
    print("=" * 60)
    print("Example 1: Enclosed Domain (All Walls)")
    print("=" * 60)

    # Create domain and manager
    domain = DomainParams(xmin=0.0, xmax=100.0, ymin=0.0, ymax=50.0)
    manager = BoundaryConditionManager(domain)

    # Set all boundaries as walls
    manager.set_all_walls()

    # Print summary
    print(manager.summary())

    # Validate
    is_valid, errors = manager.validate()
    print(f"\nValidation: {'✓ PASSED' if is_valid else '✗ FAILED'}")

    # Export to JSON
    manager.export_to_json('bc_enclosed.json')
    print("\n✓ Exported to 'bc_enclosed.json'")


def example_2_channel_flow():
    """Example 2: Typical channel flow setup"""
    print("\n" + "=" * 60)
    print("Example 2: Channel Flow")
    print("=" * 60)

    # Create domain
    domain = DomainParams(xmin=0.0, xmax=500.0, ymin=0.0, ymax=100.0)
    manager = BoundaryConditionManager(domain)

    # Use convenience function for channel setup
    manager.set_channel_bcs(
        inflow_depth=5.0,
        inflow_velocity=2.0,
        outflow_type='zero_gradient'
    )

    print(manager.summary())

    # Export
    manager.export_to_json('bc_channel.json')
    print("\n✓ Exported to 'bc_channel.json'")


def example_3_time_series_inflow():
    """Example 3: Time-varying inflow boundary"""
    print("\n" + "=" * 60)
    print("Example 3: Time-Series Inflow")
    print("=" * 60)

    domain = DomainParams(xmin=0.0, xmax=200.0, ymin=0.0, ymax=50.0)
    manager = BoundaryConditionManager(domain)

    # Create time-series inflow (flood hydrograph)
    times = np.array([0, 600, 1200, 1800, 2400, 3000, 3600])  # seconds
    depths = np.array([1.0, 2.0, 4.0, 3.5, 2.5, 1.5, 1.0])    # meters
    velocities = np.array([0.5, 1.0, 1.5, 1.3, 1.0, 0.7, 0.5])  # m/s

    inflow = TimeSeriesBC(
        BCLocation.WEST,
        times=times,
        depths=depths,
        velocity_x=velocities
    )

    manager.set_boundary(inflow)
    manager.set_boundary(OutflowBC(BCLocation.EAST, outflow_type='zero_gradient'))
    manager.set_boundary(WallBC(BCLocation.NORTH))
    manager.set_boundary(WallBC(BCLocation.SOUTH))

    print(manager.summary())

    # Plot time series
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Depth time series
    ax1.plot(times / 60, depths, 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel('Time [min]')
    ax1.set_ylabel('Water Depth [m]')
    ax1.set_title('Inflow Hydrograph - Depth')
    ax1.grid(True, alpha=0.3)

    # Velocity time series
    ax2.plot(times / 60, velocities, 'r-o', linewidth=2, markersize=6)
    ax2.set_xlabel('Time [min]')
    ax2.set_ylabel('Velocity [m/s]')
    ax2.set_title('Inflow Hydrograph - Velocity')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('bc_time_series.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved time series plot to 'bc_time_series.png'")

    # Test interpolation
    print("\nInterpolation test:")
    test_time = 900.0  # 15 minutes
    h, u, v = inflow.interpolate(test_time)
    print(f"  At t={test_time}s: h={h:.2f}m, u={u:.2f}m/s")


def example_4_periodic_boundaries():
    """Example 4: Periodic boundaries"""
    print("\n" + "=" * 60)
    print("Example 4: Periodic Boundaries")
    print("=" * 60)

    domain = DomainParams(xmin=0.0, xmax=100.0, ymin=0.0, ymax=100.0)
    manager = BoundaryConditionManager(domain)

    # Set periodic boundaries in both directions
    manager.set_boundary(PeriodicBC(BCLocation.WEST, BCLocation.EAST))
    manager.set_boundary(PeriodicBC(BCLocation.EAST, BCLocation.WEST))
    manager.set_boundary(PeriodicBC(BCLocation.SOUTH, BCLocation.NORTH))
    manager.set_boundary(PeriodicBC(BCLocation.NORTH, BCLocation.SOUTH))

    print(manager.summary())

    # Validate
    is_valid, errors = manager.validate()
    print(f"\nValidation: {'✓ PASSED' if is_valid else '✗ FAILED'}")


def example_5_function_based_bc():
    """Example 5: Function-based boundary conditions"""
    print("\n" + "=" * 60)
    print("Example 5: Function-Based Boundary Conditions")
    print("=" * 60)

    domain = DomainParams(xmin=0.0, xmax=200.0, ymin=0.0, ymax=50.0)
    manager = BoundaryConditionManager(domain)

    # Sinusoidal inflow (tidal or wave)
    def tidal_depth(t):
        """Tidal depth variation"""
        mean_depth = 3.0
        amplitude = 1.5
        period = 12 * 3600  # 12 hours in seconds
        return mean_depth + amplitude * np.sin(2 * np.pi * t / period)

    def tidal_velocity(t):
        """Tidal velocity"""
        max_vel = 0.5
        period = 12 * 3600
        return max_vel * np.cos(2 * np.pi * t / period)

    # Create function BC
    inflow = FunctionBC(
        BCLocation.WEST,
        depth_func=tidal_depth,
        velocity_x_func=tidal_velocity
    )

    # Note: Manager expects specific BC types for validation
    # For demonstration, we'll create a time-series approximation
    times = np.linspace(0, 24 * 3600, 100)  # 24 hours
    depths = np.array([tidal_depth(t) for t in times])
    velocities = np.array([tidal_velocity(t) for t in times])

    inflow_ts = TimeSeriesBC(
        BCLocation.WEST,
        times=times,
        depths=depths,
        velocity_x=velocities
    )

    manager.set_boundary(inflow_ts)
    manager.set_boundary(OutflowBC(BCLocation.EAST, outflow_type='transmissive'))
    manager.set_boundary(WallBC(BCLocation.NORTH))
    manager.set_boundary(WallBC(BCLocation.SOUTH))

    # Plot tidal variation
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Depth
    ax1.plot(times / 3600, depths, 'b-', linewidth=2)
    ax1.set_xlabel('Time [hours]')
    ax1.set_ylabel('Water Depth [m]')
    ax1.set_title('Tidal Depth Variation (24 hours)')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=3.0, color='k', linestyle='--', alpha=0.5, label='Mean depth')
    ax1.legend()

    # Velocity
    ax2.plot(times / 3600, velocities, 'r-', linewidth=2)
    ax2.set_xlabel('Time [hours]')
    ax2.set_ylabel('Velocity [m/s]')
    ax2.set_title('Tidal Velocity Variation')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('bc_tidal.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved tidal variation plot to 'bc_tidal.png'")


def example_6_boundary_visualization():
    """Example 6: Visualize boundary conditions on mesh"""
    print("\n" + "=" * 60)
    print("Example 6: Boundary Visualization")
    print("=" * 60)

    # Setup domain and BCs
    domain = DomainParams(xmin=0.0, xmax=100.0, ymin=0.0, ymax=50.0)
    manager = BoundaryConditionManager(domain)

    manager.set_boundary(InflowBC(BCLocation.WEST, depth=5.0, velocity_x=2.0))
    manager.set_boundary(OutflowBC(BCLocation.EAST, outflow_type='zero_gradient'))
    manager.set_boundary(WallBC(BCLocation.NORTH))
    manager.set_boundary(WallBC(BCLocation.SOUTH))

    # Create mesh
    generator = MeshGenerator(domain)
    mesh = generator.generate_uniform_mesh(50, 25)

    # Get boundary masks
    masks = manager.apply_to_mesh(mesh)

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    locations = ['west', 'east', 'south', 'north']
    colors = ['blue', 'red', 'green', 'orange']
    titles = ['West (Inflow)', 'East (Outflow)', 'South (Wall)', 'North (Wall)']

    for idx, (loc, color, title) in enumerate(zip(locations, colors, titles)):
        ax = axes[idx]

        # Plot mesh
        ax.plot(mesh.x, mesh.y, 'k.', markersize=1, alpha=0.3)

        # Highlight boundary
        mask = masks[loc]
        boundary_x = mesh.x[mask]
        boundary_y = mesh.y[mask]
        ax.plot(boundary_x, boundary_y, 'o', color=color, markersize=4,
               label=f'{title} boundary')

        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title(title)
        ax.legend()
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('bc_visualization.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved boundary visualization to 'bc_visualization.png'")

    # Print boundary cell counts
    print("\nBoundary cell counts:")
    for loc in locations:
        count = np.sum(masks[loc])
        print(f"  {loc.upper()}: {count} cells")


def example_7_complex_scenario():
    """Example 7: Complex multi-boundary scenario"""
    print("\n" + "=" * 60)
    print("Example 7: Complex Scenario")
    print("=" * 60)

    domain = DomainParams(xmin=0.0, xmax=1000.0, ymin=0.0, ymax=500.0)
    manager = BoundaryConditionManager(domain)

    # Multiple time-varying inflow
    times1 = np.array([0, 1800, 3600, 5400, 7200])
    depths1 = np.array([2.0, 4.0, 5.0, 3.0, 2.0])
    vel1 = np.array([1.0, 2.0, 2.5, 1.5, 1.0])

    manager.set_boundary(TimeSeriesBC(
        BCLocation.WEST,
        times=times1,
        depths=depths1,
        velocity_x=vel1
    ))

    # Fixed depth outflow
    manager.set_boundary(OutflowBC(
        BCLocation.EAST,
        outflow_type='fixed_depth',
        depth=1.5
    ))

    # Wall with roughness
    manager.set_boundary(WallBC(BCLocation.NORTH, roughness=0.03))
    manager.set_boundary(WallBC(BCLocation.SOUTH, roughness=0.025))

    print(manager.summary())

    # Validate
    is_valid, errors = manager.validate()
    if is_valid:
        print("\n✓ Configuration is valid and ready for simulation")
    else:
        print("\n✗ Configuration has errors:")
        for error in errors:
            print(f"  - {error}")

    # Export
    manager.export_to_json('bc_complex.json')
    print("\n✓ Exported to 'bc_complex.json'")


def example_8_import_export():
    """Example 8: Import/Export boundary conditions"""
    print("\n" + "=" * 60)
    print("Example 8: Import/Export")
    print("=" * 60)

    # Create and export
    domain = DomainParams(xmin=0.0, xmax=200.0, ymin=0.0, ymax=100.0)
    manager1 = BoundaryConditionManager(domain)
    manager1.set_channel_bcs(inflow_depth=3.5, inflow_velocity=1.8)
    manager1.export_to_json('bc_export_test.json')
    print("✓ Exported boundary conditions")

    # Import
    manager2 = BoundaryConditionManager.import_from_json('bc_export_test.json')
    print("✓ Imported boundary conditions")

    # Verify
    print("\nVerifying import:")
    print(f"  Domain matches: {manager2.domain.xmin == domain.xmin}")
    print(f"  All boundaries present: {len(manager2.boundaries) == 4}")

    # Compare summaries
    print("\nOriginal configuration:")
    print(manager1.summary())


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("BOUNDARY CONDITIONS EXAMPLES")
    print("HydroSIS-2D Pre/Post-Processing Toolkit")
    print("=" * 60)

    example_1_basic_wall_boundaries()
    example_2_channel_flow()
    example_3_time_series_inflow()
    example_4_periodic_boundaries()
    example_5_function_based_bc()
    example_6_boundary_visualization()
    example_7_complex_scenario()
    example_8_import_export()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - bc_enclosed.json")
    print("  - bc_channel.json")
    print("  - bc_complex.json")
    print("  - bc_export_test.json")
    print("  - bc_time_series.png")
    print("  - bc_tidal.png")
    print("  - bc_visualization.png")


if __name__ == '__main__':
    main()
