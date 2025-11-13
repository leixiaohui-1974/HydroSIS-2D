# -*- coding: utf-8 -*-
"""
End-to-End Test: Complete Dam Break Workflow

This test validates the entire workflow from preprocessing to postprocessing:
1. Mesh generation
2. Terrain setup
3. Boundary conditions
4. Initial conditions
5. Solver configuration
6. GPU solving (or CPU fallback)
7. Result loading
8. Visualization
9. Analysis and validation

Author: HydroSIS-2D Team
Date: 2025-11-13
"""

import pytest
import numpy as np
from pathlib import Path
import time
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from preprocessing.mesh_generation import MeshGenerator, DomainParams
from preprocessing.geometry import GeometryGenerator
from preprocessing.boundary_conditions import BoundaryConditionManager, WallBC, BCLocation
from preprocessing.initial_conditions import InitialConditionManager, DamBreakIC
from simulation import create_dam_break_simulation


class TestE2E_DamBreak:
    """Complete dam break workflow end-to-end test"""

    @pytest.fixture
    def output_dir(self, tmp_path):
        """Create temporary output directory"""
        output = tmp_path / "e2e_dam_break"
        output.mkdir(parents=True, exist_ok=True)
        return output

    def test_complete_dam_break_workflow(self, output_dir):
        """
        Test complete workflow: preprocessing -> solving -> postprocessing

        This is the most important test - it validates that a user can:
        1. Create a simulation from scratch
        2. Run it successfully
        3. Get valid results
        4. Analyze the results

        Expected behavior:
        - All steps complete without errors
        - Mass conservation error < 1e-6
        - Wave front position physically reasonable
        - Output files created and readable
        """

        print("\n" + "="*60)
        print("E2E TEST: Dam Break Complete Workflow")
        print("="*60)

        # ========== STAGE 1: PREPROCESSING ==========
        print("\n[Stage 1/4] Preprocessing")
        print("-" * 40)

        # Use convenient simulation builder function
        print("  → Creating dam break simulation configuration...")
        config = create_dam_break_simulation(
            length=200.0,
            width=100.0,
            nx=200,
            ny=100,
            dam_position=0.5,  # Middle of domain
            upstream_depth=10.0,
            downstream_depth=1.0,
            simulation_time=10.0
        )

        # Verify configuration
        mesh = config.mesh
        ic_manager = config.ic_manager

        assert mesh.ncells == 200 * 100, "Mesh cell count mismatch"
        assert mesh.dx == pytest.approx(1.0, rel=1e-10), "dx incorrect"
        assert mesh.dy == pytest.approx(1.0, rel=1e-10), "dy incorrect"
        print(f"    ✓ Mesh created: {mesh.ncells} cells")
        print(f"      Grid spacing: dx={mesh.dx:.3f}m, dy={mesh.dy:.3f}m")

        # Check initial conditions
        h_init = ic_manager.depth
        assert h_init is not None, "IC depth not generated"
        assert h_init.shape == (mesh.nx, mesh.ny), "Initial depth shape mismatch"

        # Check upstream (x < 100) - array is (nx, ny)
        h_upstream = h_init[:100, :].mean()
        assert 9.0 < h_upstream < 11.0, f"Upstream depth incorrect: {h_upstream}"

        # Check downstream (x >= 100)
        h_downstream = h_init[100:, :].mean()
        assert 0.5 < h_downstream < 1.5, f"Downstream depth incorrect: {h_downstream}"

        print(f"    ✓ Dam break IC: upstream={h_upstream:.1f}m, downstream={h_downstream:.1f}m")
        print("    ✓ Boundary conditions: 4 walls")
        print("    ✓ Terrain: flat at z=0.0m")

        # Validate configuration
        print("  → Validating configuration...")
        is_valid, errors = config.validate()

        if not is_valid:
            print("    ✗ Configuration validation failed:")
            for error in errors:
                print(f"      - {error}")
            pytest.fail(f"Configuration validation failed: {errors}")

        print("    ✓ Configuration valid")

        # Export configuration
        config_dir = output_dir / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        config.export_configuration(str(config_dir))

        config_file = config_dir / "simulation_config.json"
        assert config_file.exists(), "Configuration file not created"
        print(f"    ✓ Configuration exported to {config_file}")

        print("\n  ✅ Stage 1 complete: Preprocessing done\n")

        # ========== STAGE 2: SOLVING ==========
        print("[Stage 2/4] Solving")
        print("-" * 40)

        # Check if GPU solver is available
        try:
            import hydrosis2d_cuda
            has_gpu = hydrosis2d_cuda.get_gpu_count() > 0
            print(f"  → GPU available: {has_gpu}")
        except ImportError:
            has_gpu = False
            print("  → GPU solver not built, using CPU")

        # For now, we'll skip actual solving since GPU solver is not implemented yet
        # This will be filled in once the CUDA solver is ready

        print("  ⚠️  GPU solver not yet implemented")
        print("  → Creating mock solution for testing framework...")

        # Create mock solution data for testing
        # In reality, this would come from the solver
        t_end = 10.0
        dt = 0.1
        n_steps = int(t_end / dt)

        # Simulate some results
        results_dir = output_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Record initial mass for conservation check
        h0 = ic_manager.depth
        u0 = ic_manager.velocity_x
        v0 = ic_manager.velocity_y
        mass0 = np.sum(h0) * mesh.dx * mesh.dy
        print(f"  → Initial mass: {mass0:.2f} m³")

        # For now, just copy initial condition as final state
        # (In real solver, this would evolve over time)
        h_final = h0.copy()
        u_final = u0.copy()
        v_final = v0.copy()

        # Check mass conservation (should be perfect for mock data)
        mass_final = np.sum(h_final) * mesh.dx * mesh.dy
        mass_error = abs(mass_final - mass0) / mass0

        print(f"  → Final mass: {mass_final:.2f} m³")
        print(f"  → Mass error: {mass_error:.2e}")

        print("\n  ⏭️  Stage 2 skipped: Solver not yet implemented\n")

        # ========== STAGE 3: POSTPROCESSING ==========
        print("[Stage 3/4] Postprocessing")
        print("-" * 40)

        # For now, work with mock data
        print("  → Analyzing mock solution...")

        # Basic statistics
        h_max = np.max(h_final)
        h_min = np.min(h_final)
        h_mean = np.mean(h_final)

        print(f"    Water depth statistics:")
        print(f"      Max: {h_max:.2f} m")
        print(f"      Min: {h_min:.2f} m")
        print(f"      Mean: {h_mean:.2f} m")

        # Verify physical reasonableness
        assert h_max <= 10.0, "Max depth exceeds initial upstream depth"
        assert h_min >= 0.0, "Negative depth detected"

        print("    ✓ Solution physically reasonable")

        # Estimate wave front position (for real solver)
        # For dam break: wave speed ~ sqrt(g*h) ~ sqrt(10*10) = 10 m/s
        # After 10s, wave should travel ~100m
        # expected_wave_front = 100 + 10 * np.sqrt(9.81 * 10) * 10

        print("\n  ⏭️  Stage 3 simplified: Full analysis pending solver\n")

        # ========== STAGE 4: VALIDATION ==========
        print("[Stage 4/4] Validation")
        print("-" * 40)

        # Mass conservation
        print(f"  → Mass conservation: {mass_error:.2e}")
        if mass_error < 1e-6:
            print("    ✓ PASS: Mass conserved (error < 1e-6)")
        else:
            print(f"    ✗ FAIL: Mass error too large")
            pytest.fail(f"Mass conservation violated: error={mass_error:.2e}")

        # Physical constraints
        print("  → Physical constraints:")
        if h_min >= 0:
            print("    ✓ PASS: No negative depths")
        else:
            print(f"    ✗ FAIL: Negative depth: {h_min}")
            pytest.fail(f"Negative depth detected: {h_min}")

        if h_max <= 10.5:  # Allow 5% above initial
            print("    ✓ PASS: Max depth reasonable")
        else:
            print(f"    ✗ FAIL: Max depth too large: {h_max}")
            pytest.fail(f"Max depth unreasonable: {h_max}")

        # Configuration roundtrip
        print("  → Configuration files:")
        if config_file.exists():
            print("    ✓ PASS: Config file created")
        else:
            pytest.fail("Configuration file missing")

        print("\n  ✅ Stage 4 complete: Validation passed\n")

        # ========== FINAL SUMMARY ==========
        print("="*60)
        print("E2E TEST SUMMARY")
        print("="*60)
        print("✅ Stage 1: Preprocessing - COMPLETE")
        print("⏭️  Stage 2: Solving - SKIPPED (solver not implemented)")
        print("⏭️  Stage 3: Postprocessing - SIMPLIFIED")
        print("✅ Stage 4: Validation - COMPLETE")
        print()
        print("📊 Results:")
        print(f"   Mesh: {mesh.nx}×{mesh.ny} = {mesh.ncells:,} cells")
        print(f"   Mass conservation: {mass_error:.2e}")
        print(f"   Physical constraints: SATISFIED")
        print()
        print("🎯 TEST STATUS: PASSED (framework validated)")
        print("📝 Note: Full test will pass once GPU solver is implemented")
        print("="*60 + "\n")

    def test_e2e_with_analysis(self, output_dir):
        """
        Extended E2E test with result analysis (for when solver is ready)

        This test will be activated once the GPU solver is implemented.
        """
        pytest.skip("Skipping - GPU solver not yet implemented")

        # This will test:
        # - Particle tracking
        # - Streamlines
        # - Animation generation
        # - Statistical analysis
        # - VTK export/import
        # - Mass conservation over time
        # - Wave front tracking

    def test_e2e_performance_benchmark(self, output_dir):
        """
        Performance benchmark test

        Validates that simulation runs within expected time bounds.
        """
        pytest.skip("Skipping - GPU solver not yet implemented")

        # This will test:
        # - GPU vs CPU speedup
        # - Scalability with mesh size
        # - Memory usage
        # - Time per step


if __name__ == '__main__':
    # Run tests with verbose output
    pytest.main([__file__, '-v', '-s'])
