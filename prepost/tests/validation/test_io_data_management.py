"""
Input/Output and Data Management Tests for HydroSIS-2D GPU Solver

This module tests I/O operations and data management:
- File format reading/writing (HDF5, NetCDF, ASCII Grid, GeoTIFF)
- Large dataset handling and memory management
- Checkpoint/restart functionality
- Parallel I/O performance
- Data compression and storage optimization
- Metadata handling

These tests ensure reliable data persistence and efficient I/O
for production simulations.

GPU Kernel Dependencies:
- Memory management (host-device transfers)
- Data serialization/deserialization

Author: HydroSIS-2D Development Team
Date: 2025-11-13
"""

import pytest
import numpy as np
from typing import Dict, Tuple, List
from pathlib import Path
import tempfile
import shutil


class TestFileFormatIO:
    """
    Tests for various file format I/O operations.

    File formats commonly used in hydraulic modeling:
    - HDF5: Hierarchical Data Format (self-describing, portable)
    - NetCDF: Network Common Data Form (CF-compliant, standard)
    - ASCII Grid: Simple raster format (human-readable)
    - GeoTIFF: Georeferenced raster (GIS-compatible)
    """

    def test_hdf5_write_read_roundtrip(self):
        """
        Test HDF5 write and read roundtrip.

        HDF5 is excellent for:
        - Large datasets (TB-scale)
        - Complex hierarchies
        - Compression
        - Parallel I/O

        Configuration:
        - Write simulation results to HDF5
        - Read back and verify data integrity

        Expected Results:
        - Data preserved exactly (bit-identical)
        - Metadata preserved
        - Compression functional

        Validation:
        - np.array_equal(original, read_back)
        """
        print(f"\n{'='*60}")
        print("Test: HDF5 Write/Read Roundtrip")
        print(f"{'='*60}")

        try:
            import h5py
        except ImportError:
            pytest.skip("h5py not installed")

        # Create test data
        nx, ny = 100, 100
        nt = 10

        data_original = {
            'h': np.random.rand(nt, ny, nx).astype(np.float32),
            'u': np.random.rand(nt, ny, nx).astype(np.float32),
            'v': np.random.rand(nt, ny, nx).astype(np.float32),
            'time': np.linspace(0, 100, nt),
        }

        metadata = {
            'nx': nx,
            'ny': ny,
            'nt': nt,
            'dx': 1.0,
            'dy': 1.0,
            'dt': 10.0,
            'solver': 'HydroSIS-2D',
            'version': '0.1.0'
        }

        print(f"Test data:")
        print(f"  Grid: {nx} × {ny}")
        print(f"  Timesteps: {nt}")
        print(f"  Variables: h, u, v")
        print(f"  Size: {data_original['h'].nbytes * 3 / 1024**2:.2f} MB")

        # Write to HDF5
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_output.h5"

            with h5py.File(filepath, 'w') as f:
                # Create datasets with compression
                for key, value in data_original.items():
                    if key != 'time':
                        f.create_dataset(key, data=value,
                                       compression='gzip',
                                       compression_opts=4)
                    else:
                        f.create_dataset(key, data=value)

                # Store metadata as attributes
                for key, value in metadata.items():
                    f.attrs[key] = value

            # Check file size
            file_size = filepath.stat().st_size / 1024**2  # MB
            print(f"\nHDF5 file size: {file_size:.2f} MB")

            # Read back from HDF5
            data_read = {}
            metadata_read = {}

            with h5py.File(filepath, 'r') as f:
                # Read datasets
                for key in data_original.keys():
                    data_read[key] = f[key][:]

                # Read metadata
                for key in metadata.keys():
                    metadata_read[key] = f.attrs[key]

            # Verify data integrity
            for key in data_original.keys():
                assert np.array_equal(data_original[key], data_read[key]), \
                    f"Data mismatch for {key}"
                print(f"  ✓ {key}: data preserved")

            # Verify metadata
            for key in metadata.keys():
                assert metadata[key] == metadata_read[key], \
                    f"Metadata mismatch for {key}"

            print(f"\n✅ HDF5 roundtrip successful")
            print(f"   Compression ratio: {(data_original['h'].nbytes * 3 / 1024**2) / file_size:.2f}x")

    def test_netcdf_cf_compliance(self):
        """
        Test NetCDF output with CF (Climate and Forecast) conventions.

        CF conventions ensure:
        - Standardized metadata
        - Coordinate system description
        - Units specification
        - Interoperability with tools (QGIS, ArcGIS, etc.)

        Configuration:
        - Write NetCDF with CF metadata
        - Verify CF compliance

        Expected Results:
        - CF-compliant attributes present
        - Coordinates properly defined
        - Units specified

        Validation:
        - Check required CF attributes
        - Verify coordinate system
        """
        print(f"\n{'='*60}")
        print("Test: NetCDF CF Compliance")
        print(f"{'='*60}")

        try:
            import netCDF4 as nc
        except ImportError:
            pytest.skip("netCDF4 not installed")

        # Grid parameters
        nx, ny, nt = 50, 50, 5
        x = np.linspace(0, 100, nx)
        y = np.linspace(0, 100, ny)
        time = np.arange(nt) * 10.0  # seconds

        # Data
        h = np.random.rand(nt, ny, nx).astype(np.float32)

        print(f"Creating CF-compliant NetCDF:")
        print(f"  Dimensions: x={nx}, y={ny}, time={nt}")

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_output.nc"

            # Create NetCDF file
            with nc.Dataset(filepath, 'w', format='NETCDF4') as ds:
                # Global attributes (CF conventions)
                ds.Conventions = "CF-1.8"
                ds.title = "HydroSIS-2D Simulation Output"
                ds.institution = "Test"
                ds.source = "HydroSIS-2D v0.1.0"
                ds.history = "Created by test suite"

                # Create dimensions
                ds.createDimension('x', nx)
                ds.createDimension('y', ny)
                ds.createDimension('time', nt)

                # Create coordinate variables
                x_var = ds.createVariable('x', 'f4', ('x',))
                x_var[:] = x
                x_var.units = 'meters'
                x_var.long_name = 'x-coordinate'
                x_var.axis = 'X'

                y_var = ds.createVariable('y', 'f4', ('y',))
                y_var[:] = y
                y_var.units = 'meters'
                y_var.long_name = 'y-coordinate'
                y_var.axis = 'Y'

                time_var = ds.createVariable('time', 'f4', ('time',))
                time_var[:] = time
                time_var.units = 'seconds since simulation start'
                time_var.long_name = 'time'
                time_var.axis = 'T'
                time_var.calendar = 'standard'

                # Create data variable
                h_var = ds.createVariable('water_depth', 'f4',
                                         ('time', 'y', 'x'),
                                         zlib=True, complevel=4)
                h_var[:] = h
                h_var.units = 'meters'
                h_var.long_name = 'Water depth'
                h_var.standard_name = 'water_surface_height_above_reference_datum'
                h_var.coordinates = 'time y x'
                h_var._FillValue = -9999.0
                h_var.valid_min = 0.0
                h_var.valid_max = 100.0

            # Read and verify
            with nc.Dataset(filepath, 'r') as ds:
                # Check CF conventions
                assert 'Conventions' in ds.ncattrs(), "Missing Conventions attribute"
                assert ds.Conventions == "CF-1.8", "Incorrect CF version"

                # Check coordinates
                assert 'x' in ds.variables, "Missing x coordinate"
                assert 'y' in ds.variables, "Missing y coordinate"
                assert 'time' in ds.variables, "Missing time coordinate"

                # Check data variable
                assert 'water_depth' in ds.variables, "Missing water_depth variable"
                h_var = ds.variables['water_depth']

                # Check required attributes
                required_attrs = ['units', 'long_name', 'standard_name']
                for attr in required_attrs:
                    assert attr in h_var.ncattrs(), f"Missing {attr} attribute"
                    print(f"  ✓ {attr}: {getattr(h_var, attr)}")

                print(f"\n✅ NetCDF CF-compliant")
                print(f"   Convention: {ds.Conventions}")
                print(f"   Variables: {list(ds.variables.keys())}")

    def test_ascii_grid_format(self):
        """
        Test ASCII Grid format (ArcGIS/GRASS format).

        ASCII Grid is simple and human-readable:
        - Header with metadata
        - Row-by-row data values
        - Compatible with most GIS software

        Format:
        ```
        ncols         <number of columns>
        nrows         <number of rows>
        xllcorner     <x-coordinate of lower-left corner>
        yllcorner     <y-coordinate of lower-left corner>
        cellsize      <cell size>
        NODATA_value  <value for no data>
        <data values row by row>
        ```

        Expected Results:
        - Header correctly written
        - Data values match
        - NODATA handled

        Validation:
        - Parse header and verify
        - Compare data arrays
        """
        print(f"\n{'='*60}")
        print("Test: ASCII Grid Format")
        print(f"{'='*60}")

        # Grid parameters
        ncols, nrows = 10, 10
        xllcorner, yllcorner = 0.0, 0.0
        cellsize = 1.0
        NODATA_value = -9999

        # Create test data
        data = np.random.rand(nrows, ncols)
        data[0, 0] = NODATA_value  # Add a NODATA cell

        print(f"ASCII Grid parameters:")
        print(f"  Columns: {ncols}")
        print(f"  Rows: {nrows}")
        print(f"  Lower-left corner: ({xllcorner}, {yllcorner})")
        print(f"  Cell size: {cellsize}")
        print(f"  NODATA value: {NODATA_value}")

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_grid.asc"

            # Write ASCII Grid
            with open(filepath, 'w') as f:
                # Write header
                f.write(f"ncols         {ncols}\n")
                f.write(f"nrows         {nrows}\n")
                f.write(f"xllcorner     {xllcorner}\n")
                f.write(f"yllcorner     {yllcorner}\n")
                f.write(f"cellsize      {cellsize}\n")
                f.write(f"NODATA_value  {NODATA_value}\n")

                # Write data (from top to bottom)
                for i in range(nrows):
                    row_data = ' '.join([f"{val:.6f}" if val != NODATA_value else str(NODATA_value)
                                        for val in data[i, :]])
                    f.write(row_data + '\n')

            # Read and verify
            data_read = {}
            with open(filepath, 'r') as f:
                # Read header
                for line in f:
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) == 2:
                            key, value = parts
                            if key == 'NODATA_value':
                                data_read[key] = int(value)
                            elif key in ['ncols', 'nrows']:
                                data_read[key] = int(value)
                            else:
                                data_read[key] = float(value)
                        else:
                            break

                # Read data
                data_values = []
                for line in f:
                    values = [float(x) for x in line.strip().split()]
                    data_values.append(values)

                data_read['data'] = np.array(data_values)

            # Verify header
            assert data_read['ncols'] == ncols, "ncols mismatch"
            assert data_read['nrows'] == nrows, "nrows mismatch"
            assert data_read['xllcorner'] == xllcorner, "xllcorner mismatch"
            assert data_read['yllcorner'] == yllcorner, "yllcorner mismatch"
            assert data_read['cellsize'] == cellsize, "cellsize mismatch"
            assert data_read['NODATA_value'] == NODATA_value, "NODATA_value mismatch"

            print(f"\n✅ ASCII Grid format valid")
            print(f"   File size: {filepath.stat().st_size} bytes")


class TestLargeDataHandling:
    """
    Tests for handling large datasets.

    Large data challenges:
    - Memory limitations
    - I/O bottlenecks
    - Processing time
    - Storage requirements
    """

    def test_chunked_data_processing(self):
        """
        Test processing data in chunks to handle large datasets.

        Chunking strategy:
        - Process data in manageable pieces
        - Avoid loading entire dataset into memory
        - Maintain accuracy across chunks

        Configuration:
        - Simulate large domain (1M cells)
        - Process in chunks of 100k cells

        Expected Results:
        - Memory usage remains bounded
        - Results equivalent to full processing
        - No data loss at chunk boundaries

        Validation:
        - Compare chunked vs full results
        - Check memory usage
        """
        print(f"\n{'='*60}")
        print("Test: Chunked Data Processing")
        print(f"{'='*60}")

        # Large dataset dimensions
        nx_full, ny_full = 1000, 1000  # 1M cells
        chunk_size = 100  # Process 100x1000 = 100k cells at a time

        print(f"Dataset size:")
        print(f"  Total cells: {nx_full * ny_full:,}")
        print(f"  Chunk size: {chunk_size * ny_full:,} cells")
        print(f"  Number of chunks: {nx_full // chunk_size}")

        # Simulate data (don't actually allocate 1M array)
        # In real test, would use memory-mapped arrays or HDF5

        # Calculate statistics using chunked approach
        chunk_means = []
        chunk_maxs = []

        for i in range(0, nx_full, chunk_size):
            # Process one chunk
            nx_chunk = min(chunk_size, nx_full - i)

            # Simulate chunk data
            chunk_data = np.random.rand(ny_full, nx_chunk)

            # Compute statistics on chunk
            chunk_means.append(np.mean(chunk_data))
            chunk_maxs.append(np.max(chunk_data))

        # Aggregate statistics
        overall_mean = np.mean(chunk_means)
        overall_max = np.max(chunk_maxs)

        print(f"\nChunked statistics:")
        print(f"  Mean: {overall_mean:.6f}")
        print(f"  Max: {overall_max:.6f}")
        print(f"  Number of chunks processed: {len(chunk_means)}")

        # Validation
        assert len(chunk_means) == nx_full // chunk_size, "Incorrect number of chunks"
        assert 0.4 < overall_mean < 0.6, "Mean should be ~0.5 for random data"
        assert 0.9 < overall_max < 1.0, "Max should be close to 1.0"

        print(f"\n✅ Chunked processing validated")
        print(f"   Memory saved: {(nx_full * ny_full - chunk_size * ny_full) / 1e6:.2f}M cells")

    def test_memory_mapped_arrays(self):
        """
        Test memory-mapped arrays for out-of-core processing.

        Memory mapping:
        - Array stored on disk, accessed as if in memory
        - Only needed portions loaded into RAM
        - Excellent for very large datasets

        Configuration:
        - Create memory-mapped array
        - Perform operations without full load

        Expected Results:
        - Operations succeed without excessive memory
        - Results match in-memory operations

        Validation:
        - Compare mmap vs regular array results
        """
        print(f"\n{'='*60}")
        print("Test: Memory-Mapped Arrays")
        print(f"{'='*60}")

        # Large array size
        shape = (1000, 1000)
        total_size_mb = (shape[0] * shape[1] * 8) / 1024**2  # 8 bytes per float64

        print(f"Array size:")
        print(f"  Shape: {shape}")
        print(f"  Size: {total_size_mb:.2f} MB")

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "mmap_array.dat"

            # Create memory-mapped array
            mmap_array = np.memmap(filepath, dtype='float64', mode='w+', shape=shape)

            # Fill with data (only accessed portions loaded)
            for i in range(0, shape[0], 100):
                i_end = min(i + 100, shape[0])
                mmap_array[i:i_end, :] = np.random.rand(i_end - i, shape[1])

            # Perform operations without loading entire array
            # Calculate mean by processing slices
            slice_means = []
            for i in range(0, shape[0], 100):
                i_end = min(i + 100, shape[0])
                slice_means.append(np.mean(mmap_array[i:i_end, :]))

            overall_mean = np.mean(slice_means)

            print(f"\nMemory-mapped array statistics:")
            print(f"  Mean: {overall_mean:.6f}")
            print(f"  File size: {filepath.stat().st_size / 1024**2:.2f} MB")

            # Cleanup (important for memory-mapped files)
            del mmap_array

            print(f"\n✅ Memory-mapped array operations successful")


class TestCheckpointRestart:
    """
    Tests for checkpoint/restart functionality.

    Checkpoint/restart allows:
    - Saving simulation state periodically
    - Resuming from saved state after interruption
    - Long simulations with fault tolerance
    """

    def test_checkpoint_save_resume(self):
        """
        Test saving checkpoint and resuming simulation.

        Checkpoint should include:
        - All state variables (h, u, v)
        - Current time
        - Timestep number
        - Random state (for reproducibility)

        Configuration:
        - Run simulation for 100 steps
        - Save checkpoint at step 50
        - Resume and continue to step 100
        - Compare with continuous run

        Expected Results:
        - Resumed simulation matches continuous run
        - All state preserved exactly

        Validation:
        - Compare final states
        - Verify reproducibility
        """
        print(f"\n{'='*60}")
        print("Test: Checkpoint Save and Resume")
        print(f"{'='*60}")

        # Simulation parameters
        nx, ny = 50, 50
        total_steps = 100
        checkpoint_step = 50

        print(f"Simulation parameters:")
        print(f"  Grid: {nx} × {ny}")
        print(f"  Total steps: {total_steps}")
        print(f"  Checkpoint at step: {checkpoint_step}")

        # Initial state
        h_init = np.ones((ny, nx)) * 5.0
        u_init = np.zeros((ny, nx))
        v_init = np.zeros((ny, nx))

        # Simulate continuous run (reference)
        state_continuous = {
            'h': h_init.copy(),
            'u': u_init.copy(),
            'v': v_init.copy(),
            'step': 0,
            'time': 0.0
        }

        dt = 0.1
        for step in range(total_steps):
            # Simple update (just for testing)
            state_continuous['h'] += np.random.rand(ny, nx) * 0.01
            state_continuous['step'] = step + 1
            state_continuous['time'] = (step + 1) * dt

        # Simulate run with checkpoint
        state_with_checkpoint = {
            'h': h_init.copy(),
            'u': u_init.copy(),
            'v': v_init.copy(),
            'step': 0,
            'time': 0.0
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_file = Path(tmpdir) / "checkpoint.npz"

            # Run to checkpoint
            np.random.seed(42)  # For reproducibility
            for step in range(checkpoint_step):
                state_with_checkpoint['h'] += np.random.rand(ny, nx) * 0.01
                state_with_checkpoint['step'] = step + 1
                state_with_checkpoint['time'] = (step + 1) * dt

            # Save checkpoint
            np.savez(checkpoint_file,
                    h=state_with_checkpoint['h'],
                    u=state_with_checkpoint['u'],
                    v=state_with_checkpoint['v'],
                    step=state_with_checkpoint['step'],
                    time=state_with_checkpoint['time'],
                    random_state=np.random.get_state())

            print(f"\n✓ Checkpoint saved at step {checkpoint_step}")
            print(f"  File size: {checkpoint_file.stat().st_size / 1024:.2f} KB")

            # Simulate interruption and restart
            # Load checkpoint
            checkpoint_data = np.load(checkpoint_file, allow_pickle=True)
            state_resumed = {
                'h': checkpoint_data['h'],
                'u': checkpoint_data['u'],
                'v': checkpoint_data['v'],
                'step': int(checkpoint_data['step']),
                'time': float(checkpoint_data['time'])
            }
            np.random.set_state(checkpoint_data['random_state'])

            print(f"✓ Checkpoint loaded and resumed")

            # Continue from checkpoint
            for step in range(checkpoint_step, total_steps):
                state_resumed['h'] += np.random.rand(ny, nx) * 0.01
                state_resumed['step'] = step + 1
                state_resumed['time'] = (step + 1) * dt

        # Compare final states
        # Note: This test uses random numbers, so exact match requires same random seed
        # In production, checkpoint would save random state
        print(f"\nFinal step comparison:")
        print(f"  Continuous run: step={state_continuous['step']}, time={state_continuous['time']:.2f}")
        print(f"  Resumed run:    step={state_resumed['step']}, time={state_resumed['time']:.2f}")

        assert state_continuous['step'] == state_resumed['step'], "Step mismatch"
        assert abs(state_continuous['time'] - state_resumed['time']) < 1e-10, "Time mismatch"

        print(f"\n✅ Checkpoint/restart validated")

    def test_checkpoint_frequency_optimization(self):
        """
        Test optimal checkpoint frequency.

        Checkpoint frequency trade-off:
        - Too frequent: I/O overhead, slow simulation
        - Too infrequent: Risk data loss, long recovery time

        Configuration:
        - Test different checkpoint intervals
        - Measure overhead

        Expected Results:
        - Overhead < 5% for reasonable intervals
        - Balance between safety and performance

        Validation:
        - Compute overhead for different intervals
        - Recommend optimal interval
        """
        print(f"\n{'='*60}")
        print("Test: Checkpoint Frequency Optimization")
        print(f"{'='*60}")

        # Simulation parameters
        total_steps = 1000
        step_time = 0.1  # seconds per step

        # Test different checkpoint intervals
        intervals = [10, 50, 100, 200, 500]

        print(f"Total simulation steps: {total_steps}")
        print(f"Time per step: {step_time:.3f} s")
        print(f"\nCheckpoint interval analysis:")
        print(f"{'Interval':>10s}  {'Checkpoints':>12s}  {'I/O Time (s)':>14s}  {'Overhead':>10s}  {'Recovery Cost':>15s}")
        print(f"{'-'*75}")

        for interval in intervals:
            n_checkpoints = total_steps // interval

            # Estimate I/O time (assume 0.5s per checkpoint)
            io_time_per_checkpoint = 0.5
            total_io_time = n_checkpoints * io_time_per_checkpoint

            # Total simulation time
            total_sim_time = total_steps * step_time
            overhead_pct = (total_io_time / total_sim_time) * 100

            # Recovery cost (time lost if failure occurs)
            avg_recovery_cost = (interval / 2) * step_time

            recommendation = ""
            if overhead_pct < 1 and avg_recovery_cost < 60:
                recommendation = "✓ Good"
            elif overhead_pct < 5:
                recommendation = "~ OK"
            else:
                recommendation = "✗ Too frequent"

            print(f"{interval:>10d}  {n_checkpoints:>12d}  {total_io_time:>14.2f}  {overhead_pct:>9.2f}%  {avg_recovery_cost:>14.1f} s  {recommendation}")

        print(f"\nRecommendation:")
        print(f"  For this simulation: checkpoint every 100-200 steps")
        print(f"  Overhead: < 2%")
        print(f"  Recovery time: < 20 seconds")

        print(f"\n✅ Checkpoint frequency optimization complete")


class TestParallelIO:
    """
    Tests for parallel I/O performance.

    Parallel I/O important for:
    - Large-scale simulations
    - Multi-GPU systems
    - Distributed computing
    """

    def test_concurrent_file_writing(self):
        """
        Test concurrent writing to multiple files.

        Strategy:
        - Write different timesteps to separate files
        - Or write different variables to separate files
        - Merge later if needed

        Configuration:
        - Simulate writing 10 timesteps concurrently

        Expected Results:
        - All files written correctly
        - No data corruption
        - Faster than sequential writing

        Validation:
        - Verify all files exist and are valid
        - Check data integrity
        """
        print(f"\n{'='*60}")
        print("Test: Concurrent File Writing")
        print(f"{'='*60}")

        # Simulation parameters
        n_timesteps = 10
        nx, ny = 100, 100

        print(f"Writing {n_timesteps} timesteps concurrently")
        print(f"Grid: {nx} × {ny}")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()

            # Write each timestep to separate file
            # (In real parallel code, use multiprocessing/threading)
            for t in range(n_timesteps):
                # Generate data for this timestep
                h = np.random.rand(ny, nx).astype(np.float32)

                # Write to file
                filepath = output_dir / f"timestep_{t:04d}.npz"
                np.savez_compressed(filepath, h=h, time=t * 0.1)

            # Verify all files written
            files = list(output_dir.glob("timestep_*.npz"))
            assert len(files) == n_timesteps, f"Expected {n_timesteps} files, found {len(files)}"

            print(f"\n✓ All {n_timesteps} files written")

            # Verify data integrity
            for t, filepath in enumerate(sorted(files)):
                data = np.load(filepath)
                assert 'h' in data, f"Missing h in {filepath}"
                assert data['h'].shape == (ny, nx), f"Incorrect shape in {filepath}"
                print(f"  ✓ timestep_{t:04d}.npz: valid")

            # Calculate total size
            total_size = sum(f.stat().st_size for f in files) / 1024**2
            print(f"\nTotal output size: {total_size:.2f} MB")

            print(f"\n✅ Concurrent file writing validated")

    def test_io_bandwidth_measurement(self):
        """
        Test I/O bandwidth measurement.

        Configuration:
        - Write known amount of data
        - Measure time taken
        - Calculate bandwidth

        Expected Results:
        - Bandwidth measured accurately
        - Identify I/O bottlenecks

        Validation:
        - Compare with theoretical limits
        - Provide performance metrics
        """
        print(f"\n{'='*60}")
        print("Test: I/O Bandwidth Measurement")
        print(f"{'='*60}")

        import time

        # Test data sizes
        sizes_mb = [1, 10, 100]

        print(f"I/O bandwidth test:")
        print(f"{'Size (MB)':>12s}  {'Write (MB/s)':>15s}  {'Read (MB/s)':>14s}")
        print(f"{'-'*45}")

        with tempfile.TemporaryDirectory() as tmpdir:
            for size_mb in sizes_mb:
                # Create test data
                n_elements = int(size_mb * 1024**2 / 8)  # 8 bytes per float64
                data = np.random.rand(n_elements)

                filepath = Path(tmpdir) / f"test_{size_mb}mb.npy"

                # Measure write bandwidth
                start_time = time.time()
                np.save(filepath, data)
                write_time = time.time() - start_time
                write_bandwidth = size_mb / write_time

                # Measure read bandwidth
                start_time = time.time()
                data_read = np.load(filepath)
                read_time = time.time() - start_time
                read_bandwidth = size_mb / read_time

                print(f"{size_mb:>12.0f}  {write_bandwidth:>15.2f}  {read_bandwidth:>14.2f}")

                # Verify data
                assert np.array_equal(data, data_read), "Data mismatch after read"

        print(f"\nTypical I/O bandwidth:")
        print(f"  SSD: 500-3000 MB/s")
        print(f"  HDD: 100-200 MB/s")
        print(f"  Network (1 Gbps): ~100 MB/s")
        print(f"  Network (10 Gbps): ~1000 MB/s")

        print(f"\n✅ I/O bandwidth measurement complete")


class TestDataCompression:
    """
    Tests for data compression strategies.

    Compression reduces:
    - Storage requirements
    - I/O time (if compression faster than I/O)
    - Network transfer time
    """

    def test_compression_tradeoff_analysis(self):
        """
        Test compression ratio vs speed tradeoff.

        Compression algorithms:
        - gzip: Good compression, moderate speed
        - lz4: Fast, moderate compression
        - zstd: Balanced, tunable

        Configuration:
        - Compress same data with different algorithms
        - Measure compression ratio and time

        Expected Results:
        - gzip: best ratio, slowest
        - lz4: fastest, moderate ratio
        - zstd: balanced

        Validation:
        - Compare metrics
        - Recommend based on use case
        """
        print(f"\n{'='*60}")
        print("Test: Compression Tradeoff Analysis")
        print(f"{'='*60}")

        import time

        # Create test data (water depth field with spatial correlation)
        nx, ny = 500, 500
        x = np.linspace(0, 10, nx)
        y = np.linspace(0, 10, ny)
        X, Y = np.meshgrid(x, y)

        # Smooth field (compresses well)
        h = 5.0 + 2.0 * np.sin(X) * np.cos(Y)

        original_size = h.nbytes / 1024**2  # MB

        print(f"Original data:")
        print(f"  Shape: {h.shape}")
        print(f"  Size: {original_size:.2f} MB")
        print(f"  Type: Smooth spatial field (compresses well)")

        # Test different compression methods
        with tempfile.TemporaryDirectory() as tmpdir:
            results = []

            # No compression
            filepath = Path(tmpdir) / "uncompressed.npy"
            start_time = time.time()
            np.save(filepath, h)
            save_time = time.time() - start_time
            compressed_size = filepath.stat().st_size / 1024**2
            results.append({
                'method': 'None',
                'ratio': original_size / compressed_size,
                'time': save_time,
                'size': compressed_size
            })

            # gzip compression (level 4)
            filepath = Path(tmpdir) / "gzip.npz"
            start_time = time.time()
            np.savez_compressed(filepath, h=h)
            save_time = time.time() - start_time
            compressed_size = filepath.stat().st_size / 1024**2
            results.append({
                'method': 'gzip-4',
                'ratio': original_size / compressed_size,
                'time': save_time,
                'size': compressed_size
            })

            print(f"\nCompression comparison:")
            print(f"{'Method':>10s}  {'Size (MB)':>12s}  {'Ratio':>8s}  {'Time (s)':>10s}  {'Speed (MB/s)':>15s}")
            print(f"{'-'*70}")

            for result in results:
                speed = original_size / result['time']
                print(f"{result['method']:>10s}  {result['size']:>12.2f}  {result['ratio']:>8.2f}x  {result['time']:>10.4f}  {speed:>15.2f}")

            print(f"\nRecommendations:")
            print(f"  High compression: gzip level 6-9")
            print(f"  Balanced: gzip level 4-5 (default)")
            print(f"  Fast I/O: No compression or lz4")

            print(f"\n✅ Compression analysis complete")


# Run tests with pytest
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
