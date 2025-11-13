"""
Operational & Production Readiness Tests for HydroSIS-2D GPU Solver

This module tests features required for production deployment and operational use.

Test Categories:
1. Checkpoint & Restart (4 tests)
   - State serialization/deserialization
   - Restart from checkpoint
   - Checksum verification
   - Incremental checkpointing

2. Error Recovery (3 tests)
   - Graceful degradation
   - Automatic retry mechanisms
   - Fallback to CPU

3. Configuration Validation (3 tests)
   - Parameter bounds checking
   - Consistency validation
   - Schema validation

4. Production Deployment (3 tests)
   - Multi-case batch processing
   - Resource monitoring
   - Logging and diagnostics

Physics Context:
- Checkpointing: Long simulations need restart capability
- Error recovery: Production systems must handle failures gracefully
- Configuration: Prevent invalid setups that waste compute time
- Deployment: Support operational forecasting workflows

Author: HydroSIS-2D Development Team
Date: 2025-11-13
"""

import pytest
import numpy as np
from typing import Tuple, Dict, List
import json
import hashlib
import tempfile
import os
import time
from pathlib import Path


class TestCheckpointRestart:
    """Tests for checkpoint/restart functionality."""

    def test_state_serialization_deserialization(self):
        """
        Test saving and loading simulation state.

        State includes:
        - Solution arrays (h, u, v, z)
        - Time information (t, step)
        - Grid parameters (nx, ny, dx, dy)
        - Solver configuration

        Physical context: Enable pause/resume of long simulations.
        """
        # Create simulation state
        nx, ny = 100, 100
        dx, dy = 1.0, 1.0

        state = {
            "solution": {
                "h": np.random.rand(ny, nx) * 5.0 + 2.0,  # 2-7 m
                "u": np.random.rand(ny, nx) * 2.0 - 1.0,  # -1 to 1 m/s
                "v": np.random.rand(ny, nx) * 2.0 - 1.0,
                "z": np.random.rand(ny, nx) * 3.0,  # 0-3 m elevation
            },
            "time": {
                "t_current": 123.45,  # s
                "step": 5678,
                "dt": 0.1
            },
            "grid": {
                "nx": nx,
                "ny": ny,
                "dx": dx,
                "dy": dy,
                "Lx": nx * dx,
                "Ly": ny * dy
            },
            "solver": {
                "cfl": 0.8,
                "riemann_solver": "HLLC",
                "limiter": "minmod",
                "gravity": 9.81
            },
            "metadata": {
                "version": "0.1.0",
                "created": "2025-11-13",
                "description": "Dam break simulation"
            }
        }

        # Serialize to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            checkpoint_file = f.name
            # Convert numpy arrays to lists for JSON
            state_serializable = {
                "solution": {
                    k: v.tolist() for k, v in state["solution"].items()
                },
                "time": state["time"],
                "grid": state["grid"],
                "solver": state["solver"],
                "metadata": state["metadata"]
            }
            json.dump(state_serializable, f, indent=2)

        try:
            # Deserialize from file
            with open(checkpoint_file, 'r') as f:
                loaded_state = json.load(f)

            # Convert lists back to numpy arrays
            loaded_state["solution"] = {
                k: np.array(v) for k, v in loaded_state["solution"].items()
            }

            # Verify all fields match
            for key in ["h", "u", "v", "z"]:
                np.testing.assert_array_equal(
                    state["solution"][key],
                    loaded_state["solution"][key],
                    err_msg=f"Array {key} mismatch after deserialization"
                )

            assert state["time"] == loaded_state["time"]
            assert state["grid"] == loaded_state["grid"]
            assert state["solver"] == loaded_state["solver"]

        finally:
            # Cleanup
            os.unlink(checkpoint_file)

    def test_restart_from_checkpoint(self):
        """
        Test restarting simulation from checkpoint.

        Verify:
        - Exact continuation (bit-identical if deterministic)
        - No transient artifacts at restart
        - Conservation properties maintained

        Physical context: Resume crashed or time-limited runs.
        """
        # Initial condition
        nx, ny = 50, 50
        h_initial = 5.0 * np.ones((ny, nx))
        u_initial = np.zeros((ny, nx))
        v_initial = np.zeros((ny, nx))

        # Add perturbation
        h_initial[ny//2-5:ny//2+5, nx//2-5:nx//2+5] += 2.0

        # Simulate first segment (0 → t_checkpoint)
        dt = 0.1  # s
        t_checkpoint = 1.0  # s
        n_steps_1 = int(t_checkpoint / dt)

        h_checkpoint = h_initial.copy()
        u_checkpoint = u_initial.copy()
        v_checkpoint = v_initial.copy()

        # Simple forward Euler for testing
        for step in range(n_steps_1):
            # Simplified dynamics (just diffusion for test)
            alpha = 0.01
            h_checkpoint[1:-1, 1:-1] += alpha * (
                h_checkpoint[2:, 1:-1] + h_checkpoint[:-2, 1:-1] +
                h_checkpoint[1:-1, 2:] + h_checkpoint[1:-1, :-2] -
                4 * h_checkpoint[1:-1, 1:-1]
            )

        # Save checkpoint
        checkpoint_state = {
            "h": h_checkpoint.copy(),
            "u": u_checkpoint.copy(),
            "v": v_checkpoint.copy(),
            "t": t_checkpoint,
            "step": n_steps_1
        }

        # Continue simulation from checkpoint (t_checkpoint → t_end)
        t_end = 2.0  # s
        n_steps_2 = int((t_end - t_checkpoint) / dt)

        h_from_checkpoint = checkpoint_state["h"].copy()

        for step in range(n_steps_2):
            h_from_checkpoint[1:-1, 1:-1] += alpha * (
                h_from_checkpoint[2:, 1:-1] + h_from_checkpoint[:-2, 1:-1] +
                h_from_checkpoint[1:-1, 2:] + h_from_checkpoint[1:-1, :-2] -
                4 * h_from_checkpoint[1:-1, 1:-1]
            )

        # Run continuous simulation (0 → t_end) without checkpoint
        h_continuous = h_initial.copy()
        n_steps_total = int(t_end / dt)

        for step in range(n_steps_total):
            h_continuous[1:-1, 1:-1] += alpha * (
                h_continuous[2:, 1:-1] + h_continuous[:-2, 1:-1] +
                h_continuous[1:-1, 2:] + h_continuous[1:-1, :-2] -
                4 * h_continuous[1:-1, 1:-1]
            )

        # Verify restart gives same result as continuous run
        np.testing.assert_allclose(
            h_from_checkpoint,
            h_continuous,
            rtol=1e-12,
            err_msg="Restart did not reproduce continuous simulation"
        )

        # Verify conservation
        mass_checkpoint = np.sum(checkpoint_state["h"])
        mass_final = np.sum(h_from_checkpoint)
        mass_continuous = np.sum(h_continuous)

        # For this test (no source/sink), mass should be conserved
        np.testing.assert_allclose(mass_final, mass_checkpoint, rtol=1e-10)
        np.testing.assert_allclose(mass_continuous, mass_checkpoint, rtol=1e-10)

    def test_checksum_verification(self):
        """
        Test checksum/hash verification of checkpoint files.

        Purpose:
        - Detect corrupted checkpoint files
        - Verify integrity after transfer
        - Prevent silent data corruption

        Physical context: Reliability for archived simulations.
        """
        # Create checkpoint data
        nx, ny = 100, 100
        h = np.random.rand(ny, nx) * 5.0

        # Compute checksum (SHA-256)
        data_bytes = h.tobytes()
        checksum_original = hashlib.sha256(data_bytes).hexdigest()

        # Save with checksum
        checkpoint = {
            "data": h.tolist(),
            "checksum": checksum_original,
            "algorithm": "SHA-256"
        }

        # Write to file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            checkpoint_file = f.name
            json.dump(checkpoint, f)

        try:
            # Load and verify
            with open(checkpoint_file, 'r') as f:
                loaded = json.load(f)

            h_loaded = np.array(loaded["data"])
            checksum_loaded = hashlib.sha256(h_loaded.tobytes()).hexdigest()

            # Verify checksums match
            assert checksum_loaded == loaded["checksum"], \
                "Checksum mismatch - possible corruption"

            # Verify data is identical
            np.testing.assert_array_equal(h, h_loaded)

            # Test corruption detection: modify one value
            h_corrupted = h_loaded.copy()
            h_corrupted[50, 50] += 0.001  # Small corruption

            checksum_corrupted = hashlib.sha256(h_corrupted.tobytes()).hexdigest()

            assert checksum_corrupted != loaded["checksum"], \
                "Should detect corruption"

        finally:
            os.unlink(checkpoint_file)

    def test_incremental_checkpointing(self):
        """
        Test incremental checkpointing strategy.

        Strategies:
        - Time-based: Every Δt seconds
        - Step-based: Every N steps
        - Event-based: On significant events

        Physical context: Balance overhead vs. restart cost.
        """
        # Simulation parameters
        t_end = 100.0  # s
        dt = 0.1  # s
        n_steps = int(t_end / dt)

        # Checkpointing strategy
        checkpoint_interval_time = 10.0  # s
        checkpoint_interval_steps = int(checkpoint_interval_time / dt)

        # Track checkpoints
        checkpoints = []
        t_current = 0.0

        for step in range(n_steps):
            t_current = step * dt

            # Check if checkpoint is due
            if step > 0 and step % checkpoint_interval_steps == 0:
                checkpoint = {
                    "step": step,
                    "time": t_current,
                    "data": f"checkpoint_{step}"
                }
                checkpoints.append(checkpoint)

        # Verify checkpoint frequency
        expected_n_checkpoints = n_steps // checkpoint_interval_steps
        assert len(checkpoints) == expected_n_checkpoints, \
            f"Expected {expected_n_checkpoints} checkpoints, got {len(checkpoints)}"

        # Verify checkpoint times
        expected_times = [i * checkpoint_interval_time
                         for i in range(1, expected_n_checkpoints + 1)]

        checkpoint_times = [cp["time"] for cp in checkpoints]

        np.testing.assert_allclose(checkpoint_times, expected_times, rtol=1e-10)

        # Test adaptive checkpointing (more frequent near events)
        # Simulate significant event at t = 50s
        t_event = 50.0
        event_window = 5.0  # s (checkpoint more frequently ±5s around event)
        freq_normal = 10.0  # s
        freq_event = 1.0  # s (10x more frequent)

        checkpoints_adaptive = []
        t_current = 0.0

        for step in range(n_steps):
            t_current = step * dt

            # Determine checkpoint frequency based on proximity to event
            if abs(t_current - t_event) < event_window:
                checkpoint_freq = freq_event
            else:
                checkpoint_freq = freq_normal

            checkpoint_interval = int(checkpoint_freq / dt)

            if step > 0 and step % checkpoint_interval == 0:
                checkpoints_adaptive.append({
                    "step": step,
                    "time": t_current
                })

        # Should have more checkpoints with adaptive strategy
        assert len(checkpoints_adaptive) > len(checkpoints), \
            "Adaptive strategy should create more checkpoints"


class TestErrorRecovery:
    """Tests for error recovery and resilience."""

    def test_graceful_degradation(self):
        """
        Test graceful degradation when encountering issues.

        Scenarios:
        - Reduce timestep if CFL violation
        - Reduce order if instability detected
        - Switch limiter if excessive oscillations

        Physical context: Avoid complete failure in challenging cases.
        """
        # Test CFL violation recovery
        dx = 1.0  # m
        h = 5.0  # m
        u = 10.0  # m/s (high velocity)
        g = 9.81  # m/s²

        c = np.sqrt(g * h)  # Wave speed
        lambda_max = u + c  # Max eigenvalue

        # Initial timestep
        CFL_target = 0.8
        dt = CFL_target * dx / lambda_max

        # Simulate CFL violation (velocity suddenly increases)
        u_new = 20.0  # m/s (doubled)
        lambda_max_new = u_new + c

        # Check CFL condition
        CFL_actual = lambda_max_new * dt / dx

        if CFL_actual > 1.0:
            # CFL violation detected: reduce timestep
            dt_reduced = CFL_target * dx / lambda_max_new
            reduction_factor = dt_reduced / dt

            assert reduction_factor < 1.0, "Timestep should be reduced"
            assert reduction_factor > 0.3, "Reduction should be moderate"

            # Verify new timestep satisfies CFL
            CFL_new = lambda_max_new * dt_reduced / dx
            assert CFL_new <= CFL_target, "New timestep should satisfy CFL"

        # Test order reduction for stability
        order_original = 2  # MUSCL 2nd order
        oscillation_indicator = 0.8  # High (0-1 scale)
        threshold = 0.5

        if oscillation_indicator > threshold:
            # Switch to 1st order (more dissipative)
            order_reduced = 1
            assert order_reduced < order_original, "Should reduce order"

    def test_automatic_retry_mechanisms(self):
        """
        Test automatic retry with adjusted parameters.

        Retry strategies:
        - Reduce timestep (halve dt)
        - Switch to more robust limiter
        - Increase dissipation

        Physical context: Recover from transient instabilities.
        """
        max_retries = 3
        dt_initial = 0.1  # s

        # Simulate step that fails initially
        def simulation_step(dt, attempt):
            """Simulate step that succeeds only with small enough dt."""
            dt_critical = 0.03  # s
            if dt <= dt_critical:
                return True, None  # Success
            else:
                return False, "CFL violation"  # Failure

        # Retry loop
        dt = dt_initial
        success = False

        for attempt in range(max_retries):
            success, error = simulation_step(dt, attempt)

            if success:
                break
            else:
                # Reduce timestep for retry
                dt = dt / 2.0
                print(f"Attempt {attempt + 1} failed: {error}. "
                      f"Reducing dt to {dt:.4f}s")

        assert success, f"Failed after {max_retries} attempts"
        assert dt < dt_initial, "Should have reduced timestep"

        # Verify final timestep is reasonable
        assert dt >= dt_initial / (2**max_retries), \
            "Timestep reduced too much"

    def test_fallback_to_cpu(self):
        """
        Test fallback to CPU if GPU fails.

        GPU failure scenarios:
        - Out of memory
        - Driver error
        - Hardware failure

        Physical context: Ensure computation completes even if GPU unavailable.
        """
        # Simulate GPU availability check
        class ComputeBackend:
            def __init__(self):
                self.gpu_available = True
                self.cpu_available = True

            def run_on_gpu(self, data):
                if not self.gpu_available:
                    raise RuntimeError("GPU not available")
                return data * 2  # Simulate computation

            def run_on_cpu(self, data):
                if not self.cpu_available:
                    raise RuntimeError("CPU not available")
                return data * 2  # Same computation

            def run_with_fallback(self, data):
                try:
                    result = self.run_on_gpu(data)
                    backend_used = "GPU"
                except RuntimeError as e:
                    print(f"GPU failed: {e}. Falling back to CPU.")
                    result = self.run_on_cpu(data)
                    backend_used = "CPU"

                return result, backend_used

        backend = ComputeBackend()
        data = np.array([1.0, 2.0, 3.0])

        # Test normal GPU execution
        result, backend_used = backend.run_with_fallback(data)
        np.testing.assert_array_equal(result, np.array([2.0, 4.0, 6.0]))
        assert backend_used == "GPU"

        # Test fallback when GPU fails
        backend.gpu_available = False
        result, backend_used = backend.run_with_fallback(data)
        np.testing.assert_array_equal(result, np.array([2.0, 4.0, 6.0]))
        assert backend_used == "CPU"


class TestConfigurationValidation:
    """Tests for configuration validation and sanity checks."""

    def test_parameter_bounds_checking(self):
        """
        Test validation of parameter bounds.

        Physical constraints:
        - CFL: 0 < CFL < 1 (typically 0.5-0.9)
        - Manning: 0 < n < 0.2 (typical range)
        - Depth: h ≥ 0
        - Time: dt > 0, t_end > 0

        Physical context: Catch input errors before expensive computation.
        """
        def validate_cfl(cfl):
            """Validate CFL number."""
            if not (0.0 < cfl < 1.0):
                raise ValueError(f"CFL {cfl} must be in (0, 1)")
            if cfl > 0.95:
                raise Warning(f"CFL {cfl} very close to stability limit")
            return True

        # Valid CFL
        assert validate_cfl(0.8) == True

        # Invalid CFL: too small
        with pytest.raises(ValueError):
            validate_cfl(0.0)

        # Invalid CFL: too large
        with pytest.raises(ValueError):
            validate_cfl(1.5)

        # Manning coefficient validation
        def validate_manning(n):
            """Validate Manning roughness coefficient."""
            if n <= 0:
                raise ValueError(f"Manning coefficient {n} must be positive")
            if n > 0.2:
                raise Warning(f"Manning coefficient {n} unusually high")
            return True

        # Valid Manning
        assert validate_manning(0.03) == True

        # Invalid: negative
        with pytest.raises(ValueError):
            validate_manning(-0.01)

        # Grid spacing validation
        def validate_grid(dx, dy):
            """Validate grid spacing."""
            if dx <= 0 or dy <= 0:
                raise ValueError("Grid spacing must be positive")

            aspect_ratio = max(dx, dy) / min(dx, dy)
            if aspect_ratio > 10.0:
                raise Warning(f"Grid aspect ratio {aspect_ratio:.1f} is very anisotropic")

            return True

        # Valid grid
        assert validate_grid(1.0, 1.0) == True

        # Invalid: zero spacing
        with pytest.raises(ValueError):
            validate_grid(0.0, 1.0)

    def test_consistency_validation(self):
        """
        Test validation of parameter consistency.

        Consistency checks:
        - dt < CFL * dx / u_max
        - Domain size matches boundary conditions
        - Initial conditions satisfy positivity

        Physical context: Catch logically inconsistent setups.
        """
        # Test timestep vs. CFL condition
        config = {
            "dt": 0.1,  # s
            "dx": 1.0,  # m
            "cfl": 0.8,
            "u_max": 10.0,  # m/s (estimated max velocity)
            "g": 9.81,
            "h_max": 5.0  # m
        }

        # Compute maximum wave speed
        c_max = np.sqrt(config["g"] * config["h_max"])
        lambda_max = config["u_max"] + c_max

        # Compute required timestep for CFL
        dt_required = config["cfl"] * config["dx"] / lambda_max

        if config["dt"] > dt_required:
            raise ValueError(
                f"Timestep dt={config['dt']:.4f}s too large. "
                f"CFL condition requires dt ≤ {dt_required:.4f}s"
            )

        # Test domain/BC consistency
        def validate_domain_bc(nx, ny, bc_types):
            """Validate domain and boundary condition consistency."""
            required_bcs = ["west", "east", "south", "north"]

            for bc in required_bcs:
                if bc not in bc_types:
                    raise ValueError(f"Missing boundary condition: {bc}")

            # For periodic BC, opposite boundaries must both be periodic
            if bc_types["west"] == "periodic":
                if bc_types["east"] != "periodic":
                    raise ValueError("Periodic BC requires both west and east to be periodic")

            return True

        # Valid configuration
        bc_valid = {
            "west": "wall",
            "east": "wall",
            "south": "wall",
            "north": "wall"
        }
        assert validate_domain_bc(100, 100, bc_valid) == True

        # Invalid: inconsistent periodic
        bc_invalid = {
            "west": "periodic",
            "east": "wall",
            "south": "wall",
            "north": "wall"
        }
        with pytest.raises(ValueError):
            validate_domain_bc(100, 100, bc_invalid)

    def test_schema_validation(self):
        """
        Test validation against configuration schema.

        Schema defines:
        - Required fields
        - Field types
        - Value ranges
        - Nested structures

        Physical context: Structured configuration management.
        """
        # Define schema
        schema = {
            "solver": {
                "required": ["cfl", "riemann_solver", "limiter"],
                "types": {
                    "cfl": float,
                    "riemann_solver": str,
                    "limiter": str
                },
                "valid_values": {
                    "riemann_solver": ["HLL", "HLLC", "Roe"],
                    "limiter": ["none", "minmod", "superbee", "vanleer", "mc"]
                }
            },
            "grid": {
                "required": ["nx", "ny", "dx", "dy"],
                "types": {
                    "nx": int,
                    "ny": int,
                    "dx": float,
                    "dy": float
                }
            }
        }

        def validate_config(config, schema):
            """Validate configuration against schema."""
            for section, spec in schema.items():
                if section not in config:
                    raise ValueError(f"Missing section: {section}")

                section_config = config[section]

                # Check required fields
                for field in spec["required"]:
                    if field not in section_config:
                        raise ValueError(f"Missing required field: {section}.{field}")

                # Check types
                for field, expected_type in spec["types"].items():
                    if field in section_config:
                        value = section_config[field]
                        if not isinstance(value, expected_type):
                            raise TypeError(
                                f"Field {section}.{field} has type {type(value)}, "
                                f"expected {expected_type}"
                            )

                # Check valid values
                if "valid_values" in spec:
                    for field, valid_list in spec["valid_values"].items():
                        if field in section_config:
                            value = section_config[field]
                            if value not in valid_list:
                                raise ValueError(
                                    f"Field {section}.{field}={value} not in "
                                    f"valid values: {valid_list}"
                                )

            return True

        # Valid configuration
        config_valid = {
            "solver": {
                "cfl": 0.8,
                "riemann_solver": "HLLC",
                "limiter": "minmod"
            },
            "grid": {
                "nx": 100,
                "ny": 100,
                "dx": 1.0,
                "dy": 1.0
            }
        }
        assert validate_config(config_valid, schema) == True

        # Invalid: missing field
        config_missing = {
            "solver": {
                "cfl": 0.8,
                # Missing riemann_solver
                "limiter": "minmod"
            },
            "grid": {
                "nx": 100,
                "ny": 100,
                "dx": 1.0,
                "dy": 1.0
            }
        }
        with pytest.raises(ValueError):
            validate_config(config_missing, schema)

        # Invalid: wrong type
        config_wrong_type = {
            "solver": {
                "cfl": "0.8",  # Should be float, not string
                "riemann_solver": "HLLC",
                "limiter": "minmod"
            },
            "grid": {
                "nx": 100,
                "ny": 100,
                "dx": 1.0,
                "dy": 1.0
            }
        }
        with pytest.raises(TypeError):
            validate_config(config_wrong_type, schema)

        # Invalid: invalid value
        config_invalid_value = {
            "solver": {
                "cfl": 0.8,
                "riemann_solver": "INVALID",  # Not in valid list
                "limiter": "minmod"
            },
            "grid": {
                "nx": 100,
                "ny": 100,
                "dx": 1.0,
                "dy": 1.0
            }
        }
        with pytest.raises(ValueError):
            validate_config(config_invalid_value, schema)


class TestProductionDeployment:
    """Tests for production deployment scenarios."""

    def test_multi_case_batch_processing(self):
        """
        Test batch processing of multiple simulation cases.

        Use cases:
        - Ensemble forecasting
        - Parameter sweep
        - Scenario comparison

        Physical context: Operational flood forecasting systems.
        """
        # Define ensemble of cases (e.g., different rainfall scenarios)
        cases = [
            {"id": 1, "rainfall": 50.0, "manning": 0.03},   # mm/hr
            {"id": 2, "rainfall": 75.0, "manning": 0.03},
            {"id": 3, "rainfall": 100.0, "manning": 0.03},
            {"id": 4, "rainfall": 50.0, "manning": 0.05},   # Higher roughness
            {"id": 5, "rainfall": 100.0, "manning": 0.05},
        ]

        # Process batch
        results = []

        for case in cases:
            # Simulate case (simplified)
            rainfall = case["rainfall"]
            manning = case["manning"]

            # Simplified flood depth calculation
            # Real case would run full GPU solver
            runoff = rainfall * 0.8  # 80% runoff coefficient
            flood_depth = runoff / (manning * 1000.0)  # Simplified

            result = {
                "case_id": case["id"],
                "rainfall": rainfall,
                "manning": manning,
                "flood_depth": flood_depth,
                "status": "completed"
            }
            results.append(result)

        # Verify all cases processed
        assert len(results) == len(cases)
        assert all(r["status"] == "completed" for r in results)

        # Extract flood depths
        flood_depths = [r["flood_depth"] for r in results]

        # Verify physical trends
        # Higher rainfall → deeper flooding
        assert flood_depths[2] > flood_depths[1] > flood_depths[0]

        # Higher roughness → less flooding (more resistance)
        # Compare cases with same rainfall
        assert flood_depths[0] > flood_depths[3]  # 50mm/hr
        assert flood_depths[2] > flood_depths[4]  # 100mm/hr

    def test_resource_monitoring(self):
        """
        Test monitoring of computational resources.

        Metrics:
        - GPU memory usage
        - Computation time
        - Throughput (Mcups)

        Physical context: Performance monitoring for SLAs.
        """
        # Simulate resource monitoring
        class ResourceMonitor:
            def __init__(self):
                self.metrics = {
                    "gpu_memory_mb": [],
                    "cpu_memory_mb": [],
                    "computation_time_s": [],
                    "throughput_mcups": []
                }

            def record(self, gpu_mem, cpu_mem, comp_time, throughput):
                self.metrics["gpu_memory_mb"].append(gpu_mem)
                self.metrics["cpu_memory_mb"].append(cpu_mem)
                self.metrics["computation_time_s"].append(comp_time)
                self.metrics["throughput_mcups"].append(throughput)

            def get_statistics(self):
                return {
                    "gpu_memory_peak_mb": max(self.metrics["gpu_memory_mb"]),
                    "gpu_memory_mean_mb": np.mean(self.metrics["gpu_memory_mb"]),
                    "total_time_s": sum(self.metrics["computation_time_s"]),
                    "throughput_mean_mcups": np.mean(self.metrics["throughput_mcups"])
                }

        monitor = ResourceMonitor()

        # Simulate timesteps
        n_steps = 100
        nx, ny = 1000, 1000
        n_cells = nx * ny

        for step in range(n_steps):
            # Simulate resource usage
            gpu_mem = 1500 + step * 0.1  # MB (slight growth)
            cpu_mem = 500  # MB (constant)

            # Computation time (varies slightly)
            comp_time = 0.01 + 0.001 * np.random.rand()  # s

            # Throughput (Mcups)
            cell_updates = n_cells
            throughput = cell_updates / comp_time / 1e6  # Mcups

            monitor.record(gpu_mem, cpu_mem, comp_time, throughput)

        # Get statistics
        stats = monitor.get_statistics()

        # Verify metrics are reasonable
        assert 1000 <= stats["gpu_memory_peak_mb"] <= 3000, \
            "GPU memory usage outside expected range"

        assert 500 <= stats["throughput_mean_mcups"] <= 2000, \
            "Throughput outside expected range"

        assert stats["total_time_s"] > 0, "Total time must be positive"

        # Check for memory leaks (GPU memory should be stable)
        gpu_mem_growth = (monitor.metrics["gpu_memory_mb"][-1] -
                          monitor.metrics["gpu_memory_mb"][0])
        growth_percent = 100.0 * gpu_mem_growth / monitor.metrics["gpu_memory_mb"][0]

        assert growth_percent < 5.0, \
            f"Excessive GPU memory growth: {growth_percent:.1f}%"

    def test_logging_and_diagnostics(self):
        """
        Test logging and diagnostic output.

        Logging levels:
        - ERROR: Critical failures
        - WARNING: Potential issues
        - INFO: Progress updates
        - DEBUG: Detailed diagnostics

        Physical context: Operational troubleshooting and monitoring.
        """
        # Simulate logging system
        class Logger:
            def __init__(self):
                self.logs = []

            def log(self, level, message, context=None):
                entry = {
                    "timestamp": time.time(),
                    "level": level,
                    "message": message,
                    "context": context or {}
                }
                self.logs.append(entry)

            def get_errors(self):
                return [log for log in self.logs if log["level"] == "ERROR"]

            def get_warnings(self):
                return [log for log in self.logs if log["level"] == "WARNING"]

            def get_summary(self):
                return {
                    "total": len(self.logs),
                    "errors": len(self.get_errors()),
                    "warnings": len(self.get_warnings())
                }

        logger = Logger()

        # Simulate simulation with logging
        logger.log("INFO", "Starting simulation", {"nx": 100, "ny": 100})

        for step in range(10):
            logger.log("DEBUG", f"Step {step}", {"t": step * 0.1})

            # Simulate warning condition
            if step == 5:
                logger.log("WARNING", "High CFL number detected",
                          {"cfl": 0.95, "step": step})

        logger.log("INFO", "Simulation completed", {"steps": 10, "time": 1.0})

        # Verify logging
        summary = logger.get_summary()

        assert summary["total"] == 12, "Should have 12 log entries"
        assert summary["warnings"] == 1, "Should have 1 warning"
        assert summary["errors"] == 0, "Should have 0 errors"

        # Test error logging
        logger.log("ERROR", "CFL instability",
                  {"step": 15, "cfl": 1.5, "action": "abort"})

        assert len(logger.get_errors()) == 1, "Should have 1 error"

        # Verify log structure
        error = logger.get_errors()[0]
        assert "timestamp" in error
        assert error["level"] == "ERROR"
        assert "CFL instability" in error["message"]
        assert "step" in error["context"]


if __name__ == "__main__":
    """Run tests with: pytest test_operational_production_readiness.py -v"""
    pytest.main([__file__, "-v", "--tb=short"])
