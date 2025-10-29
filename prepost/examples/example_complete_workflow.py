"""
Example: Complete Simulation Workflow for HydroSIS-2D

Demonstrates end-to-end simulation setup integrating all modules.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from simulation import SimulationConfig, create_dam_break_simulation, create_channel_flow_simulation
from preprocessing.mesh_generation import DomainParams, MeshGenerator
from preprocessing.geometry import GeometryGenerator, TerrainProcessor
from preprocessing.boundary_conditions import InflowBC, OutflowBC, WallBC, TimeSeriesBC, BCLocation
from preprocessing.initial_conditions import UniformIC, DamBreakIC, GaussianHumpIC


def example_1_dam_break_complete():
    """Example 1: Complete dam break setup"""
    print("=" * 70)
    print("Example 1: Dam Break - Complete Setup")
    print("=" * 70)

    # Use factory function
    config = create_dam_break_simulation(
        length=200.0,
        width=100.0,
        dam_position=0.5,
        upstream_depth=10.0,
        downstream_depth=0.0,
        nx=100,
        ny=50,
        simulation_time=10.0
    )

    # Print summary
    print(config.summary())

    # Export configuration
    config.export_configuration('output/dam_break')
    print("\n✓ Configuration exported to 'output/dam_break/'")

    # Visualize setup
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Initial depth
    im0 = axes[0, 0].imshow(config.ic_manager.depth.T, origin='lower', cmap='Blues', vmin=0, vmax=10)
    axes[0, 0].set_title('Initial Water Depth [m]')
    axes[0, 0].set_xlabel('X [cells]')
    axes[0, 0].set_ylabel('Y [cells]')
    plt.colorbar(im0, ax=axes[0, 0])

    # Terrain
    im1 = axes[0, 1].imshow(config.terrain.T, origin='lower', cmap='terrain')
    axes[0, 1].set_title('Bed Elevation [m]')
    axes[0, 1].set_xlabel('X [cells]')
    axes[0, 1].set_ylabel('Y [cells]')
    plt.colorbar(im1, ax=axes[0, 1])

    # Centerline profile
    centerline = config.ic_manager.depth[:, 25]
    x_coords = np.linspace(0, 200, config.mesh.nx)
    axes[1, 0].plot(x_coords, centerline, 'b-', linewidth=2)
    axes[1, 0].axvline(x=100, color='r', linestyle='--', linewidth=2, label='Dam')
    axes[1, 0].set_xlabel('X [m]')
    axes[1, 0].set_ylabel('Water Depth [m]')
    axes[1, 0].set_title('Initial Depth Profile')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # Configuration summary
    stats = config.get_statistics()
    summary_text = f"""Configuration Summary:

Domain: {stats['domain']['xmin']:.0f}m × {stats['domain']['ymin']:.0f}m
        to {stats['domain']['xmax']:.0f}m × {stats['domain']['ymax']:.0f}m

Mesh: {stats['mesh']['nx']} × {stats['mesh']['ny']} = {stats['mesh']['ncells']} cells
Cell size: {stats['mesh']['dx']:.2f}m × {stats['mesh']['dy']:.2f}m

Simulation: {stats['simulation']['time']:.1f}s
Outputs: {stats['simulation']['num_outputs']} files
CFL: {stats['simulation']['cfl']}

Volume: {stats['initial_conditions']['total_volume']:.0f} m³"""

    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
                   family='monospace')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig('workflow_dam_break.png', dpi=150, bbox_inches='tight')
    print("✓ Saved visualization to 'workflow_dam_break.png'")


def example_2_channel_flow_complete():
    """Example 2: Complete channel flow setup"""
    print("\n" + "=" * 70)
    print("Example 2: Channel Flow - Complete Setup")
    print("=" * 70)

    # Use factory function
    config = create_channel_flow_simulation(
        length=500.0,
        width=100.0,
        inflow_depth=5.0,
        inflow_velocity=2.0,
        slope=0.001,
        nx=100,
        ny=50,
        simulation_time=100.0
    )

    # Print summary
    print(config.summary())

    # Export configuration
    config.export_configuration('output/channel_flow')
    print("\n✓ Configuration exported to 'output/channel_flow/'")


def example_3_custom_scenario():
    """Example 3: Custom scenario with terrain"""
    print("\n" + "=" * 70)
    print("Example 3: Custom Scenario with Terrain")
    print("=" * 70)

    # Create configuration
    config = SimulationConfig(
        name="Flood over Complex Terrain",
        description="Flood propagation over hilly terrain"
    )

    # Setup domain and mesh
    config.set_domain_and_mesh(
        xmin=0.0, xmax=1000.0,
        ymin=0.0, ymax=500.0,
        nx=200, ny=100
    )

    # Create complex terrain
    terrain_gen = GeometryGenerator()
    terrain = terrain_gen.composite_terrain(
        features=[
            ('inclined_plane', {'slope_x': 0.002, 'base_elevation': 0.0}),
            ('gaussian_hill', {'center_x': 500.0, 'center_y': 250.0, 'height': 20.0, 'width': 100.0}),
            ('gaussian_hill', {'center_x': 700.0, 'center_y': 150.0, 'height': 15.0, 'width': 80.0})
        ],
        ncols=200,
        nrows=100,
        xllcorner=0.0,
        yllcorner=0.0,
        cellsize=5.0
    )

    # Apply terrain
    config.set_terrain(terrain.data)

    # Setup boundary conditions
    bc_manager = config.setup_boundary_conditions()

    # Time-varying inflow
    times = np.array([0, 30, 60, 90, 120, 150])
    depths = np.array([2.0, 4.0, 6.0, 5.0, 3.0, 2.0])
    velocities = np.array([1.0, 2.0, 3.0, 2.5, 1.5, 1.0])

    inflow = TimeSeriesBC(
        BCLocation.WEST,
        times=times,
        depths=depths,
        velocity_x=velocities
    )

    bc_manager.set_boundary(inflow)
    bc_manager.set_boundary(OutflowBC(BCLocation.EAST, outflow_type='zero_gradient'))
    bc_manager.set_boundary(WallBC(BCLocation.NORTH))
    bc_manager.set_boundary(WallBC(BCLocation.SOUTH))

    # Setup initial conditions (dry bed)
    from preprocessing.initial_conditions import DryBedIC
    ic_manager = config.setup_initial_conditions()
    ic_manager.set_initial_condition(DryBedIC())

    # Set simulation parameters
    config.set_simulation_parameters(
        simulation_time=200.0,
        output_interval=10.0,
        cfl_number=0.5
    )

    # Print summary
    print(config.summary())

    # Validate
    is_valid, errors = config.validate()
    if is_valid:
        print("\n✓ Configuration is valid and ready!")

        # Export
        config.export_configuration('output/custom_scenario')
        print("✓ Configuration exported to 'output/custom_scenario/'")
    else:
        print("\n✗ Configuration has errors:")
        for error in errors:
            print(f"  - {error}")

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Terrain
    im0 = axes[0, 0].imshow(config.terrain.T, origin='lower', cmap='terrain')
    axes[0, 0].set_title('Terrain Elevation [m]')
    axes[0, 0].set_xlabel('X [cells]')
    axes[0, 0].set_ylabel('Y [cells]')
    plt.colorbar(im0, ax=axes[0, 0])

    # Initial depth (dry bed)
    im1 = axes[0, 1].imshow(config.ic_manager.depth.T, origin='lower', cmap='Blues')
    axes[0, 1].set_title('Initial Water Depth [m]')
    axes[0, 1].set_xlabel('X [cells]')
    axes[0, 1].set_ylabel('Y [cells]')
    plt.colorbar(im1, ax=axes[0, 1])

    # Terrain profile
    centerline_terrain = config.terrain[:, 50]
    x_coords = np.linspace(0, 1000, config.mesh.nx)
    axes[1, 0].plot(x_coords, centerline_terrain, 'brown', linewidth=2)
    axes[1, 0].set_xlabel('X [m]')
    axes[1, 0].set_ylabel('Elevation [m]')
    axes[1, 0].set_title('Terrain Profile')
    axes[1, 0].grid(True, alpha=0.3)

    # Inflow hydrograph
    axes[1, 1].plot(times / 60, depths, 'b-o', linewidth=2, markersize=6, label='Depth')
    axes[1, 1].set_xlabel('Time [min]')
    axes[1, 1].set_ylabel('Inflow Depth [m]', color='b')
    axes[1, 1].tick_params(axis='y', labelcolor='b')
    axes[1, 1].grid(True, alpha=0.3)

    ax2 = axes[1, 1].twinx()
    ax2.plot(times / 60, velocities, 'r-s', linewidth=2, markersize=6, label='Velocity')
    ax2.set_ylabel('Inflow Velocity [m/s]', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    axes[1, 1].set_title('Inflow Hydrograph')

    plt.tight_layout()
    plt.savefig('workflow_custom_scenario.png', dpi=150, bbox_inches='tight')
    print("✓ Saved visualization to 'workflow_custom_scenario.png'")


def example_4_step_by_step():
    """Example 4: Step-by-step custom setup"""
    print("\n" + "=" * 70)
    print("Example 4: Step-by-Step Custom Setup")
    print("=" * 70)

    # Step 1: Create configuration
    print("\nStep 1: Creating simulation configuration...")
    config = SimulationConfig(
        name="Tutorial Example",
        description="Step-by-step setup demonstration"
    )

    # Step 2: Setup domain and mesh
    print("Step 2: Setting up domain and mesh...")
    config.set_domain_and_mesh(
        xmin=0.0, xmax=100.0,
        ymin=0.0, ymax=50.0,
        nx=50, ny=25
    )
    print(f"  ✓ Created {config.mesh.nx}×{config.mesh.ny} mesh")

    # Step 3: Setup terrain
    print("Step 3: Setting up terrain...")
    config.set_flat_terrain(elevation=0.0)
    print(f"  ✓ Set flat terrain at 0.0 m")

    # Step 4: Setup boundary conditions
    print("Step 4: Setting up boundary conditions...")
    bc_manager = config.setup_boundary_conditions()
    bc_manager.set_boundary(InflowBC(BCLocation.WEST, depth=3.0, velocity_x=1.0))
    bc_manager.set_boundary(OutflowBC(BCLocation.EAST, outflow_type='zero_gradient'))
    bc_manager.set_boundary(WallBC(BCLocation.NORTH))
    bc_manager.set_boundary(WallBC(BCLocation.SOUTH))
    print("  ✓ Set inflow, outflow, and wall boundaries")

    # Step 5: Setup initial conditions
    print("Step 5: Setting up initial conditions...")
    ic_manager = config.setup_initial_conditions()
    ic_manager.set_initial_condition(UniformIC(depth=2.0, velocity_x=0.5))
    print("  ✓ Set uniform initial condition")

    # Step 6: Set simulation parameters
    print("Step 6: Setting simulation parameters...")
    config.set_simulation_parameters(
        simulation_time=50.0,
        output_interval=2.0,
        cfl_number=0.5
    )
    print("  ✓ Set simulation time: 50s, outputs every 2s")

    # Step 7: Validate
    print("\nStep 7: Validating configuration...")
    is_valid, errors = config.validate()
    if is_valid:
        print("  ✓ Configuration is valid!")
    else:
        print("  ✗ Configuration has errors:")
        for error in errors:
            print(f"    - {error}")

    # Step 8: Export
    print("\nStep 8: Exporting configuration...")
    config.export_configuration('output/tutorial')
    print("  ✓ Exported to 'output/tutorial/'")

    # Step 9: Print summary
    print("\n" + "=" * 70)
    print("FINAL CONFIGURATION")
    print("=" * 70)
    print(config.summary())


def example_5_compare_scenarios():
    """Example 5: Compare multiple scenarios"""
    print("\n" + "=" * 70)
    print("Example 5: Compare Multiple Scenarios")
    print("=" * 70)

    scenarios = []

    # Scenario 1: Small dam
    config1 = create_dam_break_simulation(
        length=100.0, width=50.0,
        upstream_depth=5.0, downstream_depth=0.0,
        nx=50, ny=25,
        simulation_time=5.0
    )
    config1.name = "Small Dam Break"
    scenarios.append(config1)

    # Scenario 2: Large dam
    config2 = create_dam_break_simulation(
        length=200.0, width=100.0,
        upstream_depth=15.0, downstream_depth=0.0,
        nx=100, ny=50,
        simulation_time=10.0
    )
    config2.name = "Large Dam Break"
    scenarios.append(config2)

    # Scenario 3: Partial dam break
    config3 = create_dam_break_simulation(
        length=150.0, width=75.0,
        upstream_depth=10.0, downstream_depth=2.0,
        nx=75, ny=37,
        simulation_time=8.0
    )
    config3.name = "Partial Dam Break"
    scenarios.append(config3)

    # Compare scenarios
    print("\nScenario Comparison:")
    print("-" * 70)
    print(f"{'Name':<20} {'Mesh':<15} {'Volume [m³]':<15} {'Time [s]':<10}")
    print("-" * 70)

    for config in scenarios:
        stats = config.get_statistics()
        print(f"{config.name:<20} "
              f"{stats['mesh']['nx']}×{stats['mesh']['ny']:<10} "
              f"{stats['initial_conditions']['total_volume']:<15.0f} "
              f"{stats['simulation']['time']:<10.1f}")

    print("-" * 70)


def main():
    """Run all examples"""
    print("\n" + "=" * 70)
    print("COMPLETE SIMULATION WORKFLOW EXAMPLES")
    print("HydroSIS-2D Pre/Post-Processing Toolkit")
    print("=" * 70)

    example_1_dam_break_complete()
    example_2_channel_flow_complete()
    example_3_custom_scenario()
    example_4_step_by_step()
    example_5_compare_scenarios()

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  Visualizations:")
    print("    - workflow_dam_break.png")
    print("    - workflow_custom_scenario.png")
    print("")
    print("  Configurations:")
    print("    - output/dam_break/")
    print("    - output/channel_flow/")
    print("    - output/custom_scenario/")
    print("    - output/tutorial/")


if __name__ == '__main__':
    main()
