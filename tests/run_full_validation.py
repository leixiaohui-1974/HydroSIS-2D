#!/usr/bin/env python3
"""
Comprehensive Validation Test Runner

This script runs the complete validation suite for HydroSIS-2D:
1. Unit tests (preprocessing, mesh, IC, BC)
2. GPU-CPU consistency tests
3. Analytical solution validation
4. MacDonald benchmark suite
5. Performance benchmarks
6. End-to-end workflow tests

Usage:
    python run_full_validation.py                    # Run all tests
    python run_full_validation.py --quick            # Quick test suite only
    python run_full_validation.py --validation       # Validation tests only
    python run_full_validation.py --performance      # Performance tests only
    python run_full_validation.py --report           # Generate HTML report

Requirements:
    - GPU solver compiled (hydrosis2d_cuda)
    - All dependencies installed
    - pytest, pytest-html (for reports)
"""

import argparse
import subprocess
import sys
from pathlib import Path
import time
import json
from datetime import datetime
import platform


class TestRunner:
    """Orchestrate all validation tests"""

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.prepost_tests = root_dir / "prepost" / "tests"
        self.results = {}
        self.start_time = None
        self.end_time = None

    def check_gpu_solver(self) -> bool:
        """Check if GPU solver is available"""
        try:
            import hydrosis2d_cuda
            print("✓ GPU solver available (hydrosis2d_cuda)")
            return True
        except ImportError:
            print("✗ GPU solver not available")
            print("  → Compile GPU solver first:")
            print("    cd src/solver && mkdir build && cd build")
            print("    cmake .. -DCMAKE_CUDA_ARCHITECTURES=native")
            print("    make -j$(nproc)")
            return False

    def run_unit_tests(self) -> dict:
        """Run unit tests (preprocessing, mesh generation, etc.)"""
        print("\n" + "="*70)
        print("UNIT TESTS: Preprocessing and Core Functionality")
        print("="*70)

        test_files = [
            "test_mesh_generation.py",
            "test_geometry.py",
            "test_initial_conditions.py",
            "test_boundary_conditions.py",
            "test_integration.py",
            "test_visualization.py",
        ]

        results = {}
        for test_file in test_files:
            test_path = self.prepost_tests / test_file
            if not test_path.exists():
                continue

            print(f"\nRunning {test_file}...")
            result = self._run_pytest(test_path)
            results[test_file] = result

        return results

    def run_gpu_consistency_tests(self, gpu_available: bool) -> dict:
        """Run GPU-CPU consistency tests"""
        print("\n" + "="*70)
        print("GPU-CPU CONSISTENCY TESTS")
        print("="*70)

        if not gpu_available:
            print("⊗ Skipped (GPU solver not available)")
            return {"status": "skipped"}

        test_path = self.prepost_tests / "test_gpu_cpu_consistency.py"

        if not test_path.exists():
            print("⊗ Test file not found")
            return {"status": "not_found"}

        print(f"\nRunning GPU-CPU consistency tests...")
        result = self._run_pytest(test_path)

        return result

    def run_analytical_validation(self, gpu_available: bool) -> dict:
        """Run analytical solution validation"""
        print("\n" + "="*70)
        print("ANALYTICAL SOLUTION VALIDATION")
        print("="*70)

        if not gpu_available:
            print("⊗ Skipped (GPU solver not available)")
            return {"status": "skipped"}

        test_path = self.prepost_tests / "validation" / "test_analytical_solutions.py"

        if not test_path.exists():
            print("⊗ Test file not found")
            return {"status": "not_found"}

        print(f"\nRunning analytical validation tests...")
        print("  - Ritter dam break (1D exact solution)")
        print("  - Lake at rest (C-property)")
        print("  - Circular dam break (symmetry)")
        print("  - Steady flow over bump")

        result = self._run_pytest(test_path)

        return result

    def run_macdonald_suite(self, gpu_available: bool) -> dict:
        """Run MacDonald benchmark suite"""
        print("\n" + "="*70)
        print("MACDONALD BENCHMARK SUITE")
        print("="*70)

        if not gpu_available:
            print("⊗ Skipped (GPU solver not available)")
            return {"status": "skipped"}

        test_path = self.prepost_tests / "validation" / "test_macdonald_suite.py"

        if not test_path.exists():
            print("⊗ Test file not found")
            return {"status": "not_found"}

        print(f"\nRunning MacDonald benchmarks...")
        print("  - Test 1: Uniform flow (friction balance)")
        print("  - Test 2: Transcritical flow (shock capturing)")
        print("  - Test 4: 2D flow over hump")
        print("  - Test 5: Partial dam break")

        result = self._run_pytest(test_path)

        return result

    def run_performance_benchmarks(self, gpu_available: bool) -> dict:
        """Run performance benchmarks"""
        print("\n" + "="*70)
        print("PERFORMANCE BENCHMARKS")
        print("="*70)

        if not gpu_available:
            print("⊗ Skipped (GPU solver not available)")
            return {"status": "skipped"}

        test_path = self.prepost_tests / "performance" / "test_performance_benchmarks.py"

        if not test_path.exists():
            print("⊗ Test file not found")
            return {"status": "not_found"}

        print(f"\nRunning performance benchmarks...")
        print("  - Mesh scaling tests (2.5k to 80k cells)")
        print("  - GPU speedup measurement")
        print("  - Throughput analysis (Mcups)")
        print("  - Commercial software comparison")

        result = self._run_pytest(test_path)

        return result

    def run_e2e_tests(self, gpu_available: bool) -> dict:
        """Run end-to-end workflow tests"""
        print("\n" + "="*70)
        print("END-TO-END WORKFLOW TESTS")
        print("="*70)

        if not gpu_available:
            print("⊗ Skipped (GPU solver not available)")
            return {"status": "skipped"}

        test_path = self.prepost_tests / "e2e" / "test_e2e_dam_break.py"

        if not test_path.exists():
            print("⊗ Test file not found")
            return {"status": "not_found"}

        print(f"\nRunning end-to-end tests...")
        print("  - Complete workflow: preprocessing → GPU → postprocessing")

        result = self._run_pytest(test_path)

        return result

    def run_examples(self, gpu_available: bool) -> dict:
        """Run example scripts to verify they work"""
        print("\n" + "="*70)
        print("EXAMPLE SCRIPTS VALIDATION")
        print("="*70)

        if not gpu_available:
            print("⊗ Skipped (GPU solver not available)")
            return {"status": "skipped"}

        examples_dir = self.root_dir / "examples"
        example_files = [
            "01_basic_dam_break.py",
            "02_performance_benchmark.py",
            "03_analytical_validation.py",
            "04_urban_flood.py",
        ]

        results = {}
        for example_file in example_files:
            example_path = examples_dir / example_file
            if not example_path.exists():
                continue

            print(f"\nRunning {example_file}...")
            result = self._run_python_script(example_path)
            results[example_file] = result

        return results

    def _run_pytest(self, test_path: Path) -> dict:
        """Run pytest on a specific test file"""
        start = time.time()

        try:
            result = subprocess.run(
                ["pytest", str(test_path), "-v", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )

            elapsed = time.time() - start

            return {
                "status": "passed" if result.returncode == 0 else "failed",
                "returncode": result.returncode,
                "duration": elapsed,
                "stdout": result.stdout,
                "stderr": result.stderr
            }

        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            return {
                "status": "timeout",
                "returncode": -1,
                "duration": elapsed,
                "stdout": "",
                "stderr": "Test timed out after 10 minutes"
            }

        except Exception as e:
            elapsed = time.time() - start
            return {
                "status": "error",
                "returncode": -1,
                "duration": elapsed,
                "stdout": "",
                "stderr": str(e)
            }

    def _run_python_script(self, script_path: Path) -> dict:
        """Run a Python script directly"""
        start = time.time()

        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )

            elapsed = time.time() - start

            return {
                "status": "passed" if result.returncode == 0 else "failed",
                "returncode": result.returncode,
                "duration": elapsed,
                "stdout": result.stdout,
                "stderr": result.stderr
            }

        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            return {
                "status": "timeout",
                "returncode": -1,
                "duration": elapsed,
                "stdout": "",
                "stderr": "Script timed out after 10 minutes"
            }

        except Exception as e:
            elapsed = time.time() - start
            return {
                "status": "error",
                "returncode": -1,
                "duration": elapsed,
                "stdout": "",
                "stderr": str(e)
            }

    def generate_summary_report(self):
        """Generate summary report"""
        print("\n" + "="*70)
        print("VALIDATION SUMMARY REPORT")
        print("="*70)

        total_duration = self.end_time - self.start_time

        print(f"\nTest Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Duration: {total_duration:.1f} seconds")
        print(f"Platform: {platform.system()} {platform.release()}")
        print(f"Python: {platform.python_version()}")

        # Count results
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        skipped_tests = 0

        for category, result in self.results.items():
            if isinstance(result, dict):
                if result.get("status") == "passed":
                    passed_tests += 1
                    total_tests += 1
                elif result.get("status") == "failed":
                    failed_tests += 1
                    total_tests += 1
                elif result.get("status") == "skipped":
                    skipped_tests += 1

        print(f"\nTest Categories:")
        print(f"  Total:   {total_tests}")
        print(f"  Passed:  {passed_tests} ✓")
        print(f"  Failed:  {failed_tests} ✗")
        print(f"  Skipped: {skipped_tests} ⊗")

        # Detailed results
        print(f"\nDetailed Results:")
        print(f"{'Category':<30} {'Status':<15} {'Duration':<15}")
        print("-" * 70)

        for category, result in self.results.items():
            if isinstance(result, dict):
                status = result.get("status", "unknown")
                duration = result.get("duration", 0)

                status_symbol = {
                    "passed": "✓ PASSED",
                    "failed": "✗ FAILED",
                    "skipped": "⊗ SKIPPED",
                    "timeout": "⊗ TIMEOUT",
                    "error": "✗ ERROR",
                    "not_found": "⊗ NOT FOUND"
                }.get(status, status)

                print(f"{category:<30} {status_symbol:<15} {duration:>6.1f}s")

        # Overall status
        print("\n" + "="*70)
        if failed_tests == 0 and total_tests > 0:
            print("✓ ALL TESTS PASSED")
        elif failed_tests > 0:
            print(f"✗ {failed_tests} TEST(S) FAILED")
        else:
            print("⊗ NO TESTS RUN (GPU solver may not be available)")
        print("="*70)

        return {
            "total": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "skipped": skipped_tests,
            "duration": total_duration
        }

    def save_json_report(self, output_path: Path):
        """Save results as JSON"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "python": platform.python_version()
            },
            "duration": self.end_time - self.start_time,
            "results": self.results
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n✓ JSON report saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive validation suite for HydroSIS-2D"
    )

    parser.add_argument("--quick", action="store_true",
                       help="Run quick test suite only (unit tests)")
    parser.add_argument("--validation", action="store_true",
                       help="Run validation tests only")
    parser.add_argument("--performance", action="store_true",
                       help="Run performance benchmarks only")
    parser.add_argument("--examples", action="store_true",
                       help="Run example scripts only")
    parser.add_argument("--report", type=str, default=None,
                       help="Generate JSON report (specify output file)")
    parser.add_argument("--no-gpu-check", action="store_true",
                       help="Skip GPU availability check")

    args = parser.parse_args()

    # Get project root
    root_dir = Path(__file__).parent.parent

    print("="*70)
    print("HydroSIS-2D COMPREHENSIVE VALIDATION SUITE")
    print("="*70)
    print(f"Project root: {root_dir}")

    runner = TestRunner(root_dir)

    # Check GPU solver availability
    if not args.no_gpu_check:
        gpu_available = runner.check_gpu_solver()
    else:
        gpu_available = True
        print("⊗ GPU check skipped")

    runner.start_time = time.time()

    # Run test suites based on arguments
    if args.quick:
        runner.results["unit_tests"] = runner.run_unit_tests()

    elif args.validation:
        runner.results["analytical_validation"] = runner.run_analytical_validation(gpu_available)
        runner.results["macdonald_suite"] = runner.run_macdonald_suite(gpu_available)

    elif args.performance:
        runner.results["performance_benchmarks"] = runner.run_performance_benchmarks(gpu_available)

    elif args.examples:
        runner.results["examples"] = runner.run_examples(gpu_available)

    else:
        # Run all tests
        runner.results["unit_tests"] = runner.run_unit_tests()
        runner.results["gpu_consistency"] = runner.run_gpu_consistency_tests(gpu_available)
        runner.results["analytical_validation"] = runner.run_analytical_validation(gpu_available)
        runner.results["macdonald_suite"] = runner.run_macdonald_suite(gpu_available)
        runner.results["performance_benchmarks"] = runner.run_performance_benchmarks(gpu_available)
        runner.results["e2e_tests"] = runner.run_e2e_tests(gpu_available)
        runner.results["examples"] = runner.run_examples(gpu_available)

    runner.end_time = time.time()

    # Generate summary
    summary = runner.generate_summary_report()

    # Save JSON report if requested
    if args.report:
        report_path = Path(args.report)
        runner.save_json_report(report_path)

    # Exit with appropriate code
    sys.exit(0 if summary["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
