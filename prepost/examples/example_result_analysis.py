#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example 4: Result Analysis and Statistics

Demonstrates result analysis capabilities for HydroSIS-2D simulation outputs.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from preprocessing.mesh_generation import MeshGenerator, DomainParams
from postprocessing.analysis import ResultAnalyzer, StatisticsCalculator, ProfileExtractor
from postprocessing.analysis.result_analyzer import SimulationResult


def create_synthetic_results(mesh, num_timesteps=10):
    """Create synthetic simulation results for demonstration"""
    results = []

    for i in range(num_timesteps):
        t = i * 1.0

        # Simulate dam break wave propagation
        wave_front = 100 + 20 * t
        water_depth = np.zeros((mesh.nx, mesh.ny))

        # Upstream water
        water_depth[mesh.x < 100] = 10.0

        # Wave front
        mask = (mesh.x >= 100) & (mesh.x < wave_front)
        decay = np.exp(-0.1 * t)
        water_depth[mask] = 10.0 * decay * (1 - (mesh.x[mask] - 100) / (wave_front - 100))

        # Velocities
        u = np.where(water_depth > 0.01, 3.0 * decay, 0.0)
        v = np.zeros_like(u)

        result = SimulationResult(
            time=t,
            data={'h': water_depth, 'u': u, 'v': v},
            mesh=mesh
        )
        results.append(result)

    return results


def example_1_load_and_summarize():
    """Example 1: Load results and print summary"""
    print("=" * 60)
    print("Example 1: Load and Summarize Results")
    print("=" * 60)
    print()

    # Create mesh
    domain = DomainParams(xmin=0, xmax=400, ymin=0, ymax=200)
    generator = MeshGenerator(domain)
    mesh = generator.generate_uniform_mesh(80, 40)

    # Create analyzer
    analyzer = ResultAnalyzer(mesh=mesh)

    # Create synthetic results
    print("Creating synthetic results...")
    results = create_synthetic_results(mesh, num_timesteps=20)

    for result in results:
        analyzer.add_result(result)

    print(f"Loaded {len(analyzer.results)} timesteps")
    print()

    # Print summary
    print(analyzer.summary())
    print()

    return analyzer


def example_2_temporal_statistics(analyzer):
    """Example 2: Compute temporal statistics"""
    print("=" * 60)
    print("Example 2: Temporal Statistics")
    print("=" * 60)
    print()

    # Get maximum water depth over time
    print("Computing maximum water depth...")
    max_depth = analyzer.compute_max_field('h')

    print(f"  Maximum depth: {np.max(max_depth):.3f} m")
    print(f"  Mean maximum depth: {np.mean(max_depth[max_depth > 0.01]):.3f} m")
    print()

    # Get time of maximum
    print("Computing time of maximum depth...")
    time_of_max = analyzer.compute_time_of_max('h')

    wet_mask = max_depth > 0.01
    print(f"  Earliest arrival: {np.min(time_of_max[wet_mask]):.2f} s")
    print(f"  Latest arrival: {np.max(time_of_max[wet_mask]):.2f} s")
    print()

    # Inundation duration
    print("Computing inundation duration...")
    duration = analyzer.compute_inundation_duration(depth_threshold=0.05)

    print(f"  Maximum duration: {np.max(duration):.2f} s")
    print(f"  Mean duration (wet areas): {np.mean(duration[duration > 0]):.2f} s")
    print()


def example_3_mass_conservation(analyzer):
    """Example 3: Check mass conservation"""
    print("=" * 60)
    print("Example 3: Mass Conservation Check")
    print("=" * 60)
    print()

    conservation = analyzer.check_mass_conservation()

    print(f"Reference volume: {conservation['reference_volume']:.2f} m^3")
    print(f"Final volume: {conservation['final_volume']:.2f} m^3")
    print(f"Volume change: {conservation['volume_change']:.2f} m^3")
    print(f"Maximum relative error: {conservation['max_error']*100:.4f}%")
    print(f"Mean relative error: {conservation['mean_error']*100:.4f}%")
    print()

    if conservation['max_error'] < 0.001:
        print("[OK] Mass is well conserved (error < 0.1%)")
    else:
        print("[WARN] Mass conservation may have issues")
    print()


def example_4_field_statistics(analyzer):
    """Example 4: Field statistics"""
    print("=" * 60)
    print("Example 4: Field Statistics")
    print("=" * 60)
    print()

    fields = ['h', 'u', 'v']

    for field in fields:
        print(f"\nStatistics for '{field}':")
        stats = analyzer.get_statistics(field)
        print(f"  Min: {stats['min']:.4f}")
        print(f"  Max: {stats['max']:.4f}")
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Std: {stats['std']:.4f}")
    print()


def example_5_profile_extraction(analyzer):
    """Example 5: Extract profiles"""
    print("=" * 60)
    print("Example 5: Profile Extraction")
    print("=" * 60)
    print()

    # Extract profile at centerline
    extractor = ProfileExtractor(analyzer.mesh)

    # Get final result
    final_result = analyzer.results[-1]

    # Extract x-profile at y=100m
    print("Extracting profile along x-axis at y=100m...")
    x_coords, h_values = extractor.extract_x_profile(final_result.data['h'], y=100.0)

    print(f"  Profile length: {len(x_coords)} points")
    print(f"  Depth range: [{np.min(h_values):.3f}, {np.max(h_values):.3f}] m")
    print()

    # Extract y-profile at x=200m
    print("Extracting profile along y-axis at x=200m...")
    y_coords, h_values = extractor.extract_y_profile(final_result.data['h'], x=200.0)

    print(f"  Profile length: {len(y_coords)} points")
    print(f"  Depth range: [{np.min(h_values):.3f}, {np.max(h_values):.3f}] m")
    print()


def example_6_statistics_calculator():
    """Example 6: Statistics Calculator"""
    print("=" * 60)
    print("Example 6: Advanced Statistics")
    print("=" * 60)
    print()

    calc = StatisticsCalculator()

    # Create sample data
    field = np.random.rand(50, 25) * 10.0

    print("Computing spatial statistics...")
    stats = calc.spatial_stats(field)

    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
    print()

    print("Computing percentiles...")
    percentiles = calc.compute_percentiles(field)

    for key, value in percentiles.items():
        print(f"  {key}: {value:.4f}")
    print()


def main():
    """Run all examples"""
    print()
    print("=" * 60)
    print("HYDROSIS-2D RESULT ANALYSIS EXAMPLES")
    print("=" * 60)
    print()

    # Example 1: Load and summarize
    analyzer = example_1_load_and_summarize()

    # Example 2: Temporal statistics
    example_2_temporal_statistics(analyzer)

    # Example 3: Mass conservation
    example_3_mass_conservation(analyzer)

    # Example 4: Field statistics
    example_4_field_statistics(analyzer)

    # Example 5: Profile extraction
    example_5_profile_extraction(analyzer)

    # Example 6: Statistics calculator
    example_6_statistics_calculator()

    print("=" * 60)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 60)
    print()
    print("Key capabilities demonstrated:")
    print("  [OK] Loading simulation results")
    print("  [OK] Computing temporal statistics (max, time of max, duration)")
    print("  [OK] Checking mass conservation")
    print("  [OK] Field statistics and percentiles")
    print("  [OK] Profile extraction along lines")
    print()


if __name__ == "__main__":
    main()
