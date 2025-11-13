# -*- coding: utf-8 -*-
"""
Unit tests for geometry processing module
"""

import pytest
import numpy as np
import tempfile
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.geometry import (
    TerrainData,
    TerrainReader,
    ASCIIGridReader,
    TerrainProcessor,
    GeometryGenerator,
    create_test_terrain,
    load_terrain,
    save_terrain,
    process_terrain_for_simulation
)
from preprocessing.mesh_generation import DomainParams, MeshGenerator


class TestTerrainData:
    """Test TerrainData class"""

    def test_creation(self):
        """Test TerrainData creation"""
        data = np.random.rand(10, 20)
        terrain = TerrainData(
            data=data,
            ncols=10,
            nrows=20,
            xllcorner=0.0,
            yllcorner=0.0,
            cellsize=1.0
        )

        assert terrain.ncols == 10
        assert terrain.nrows == 20
        assert np.array_equal(terrain.data, data)

    def test_extent_properties(self):
        """Test extent calculation"""
        terrain = TerrainData(
            data=np.zeros((10, 20)),
            ncols=10,
            nrows=20,
            xllcorner=100.0,
            yllcorner=200.0,
            cellsize=2.0
        )

        assert terrain.xmax == 100.0 + 10 * 2.0
        assert terrain.ymax == 200.0 + 20 * 2.0

        extent = terrain.extent
        assert extent == (100.0, 120.0, 200.0, 240.0)

    def test_statistics(self):
        """Test terrain statistics"""
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, -9999]])
        terrain = TerrainData(
            data=data,
            ncols=3,
            nrows=3,
            xllcorner=0.0,
            yllcorner=0.0,
            cellsize=1.0,
            nodata_value=-9999
        )

        stats = terrain.get_statistics()
        assert stats['min'] == 1.0
        assert stats['max'] == 8.0
        assert stats['mean'] == pytest.approx(4.5)


class TestASCIIGridReader:
    """Test ASCII Grid reader/writer"""

    def test_write_read_roundtrip(self):
        """Test writing and reading ASCII Grid"""
        # Create test terrain
        data = np.random.rand(5, 10)
        terrain = TerrainData(
            data=data,
            ncols=5,
            nrows=10,
            xllcorner=100.0,
            yllcorner=200.0,
            cellsize=2.5,
            nodata_value=-9999
        )

        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.asc', delete=False) as f:
            filepath = f.name

        try:
            ASCIIGridReader.write(terrain, filepath)

            # Read back
            terrain_read = ASCIIGridReader.read(filepath)

            # Check metadata
            assert terrain_read.ncols == terrain.ncols
            assert terrain_read.nrows == terrain.nrows
            assert terrain_read.xllcorner == pytest.approx(terrain.xllcorner)
            assert terrain_read.yllcorner == pytest.approx(terrain.yllcorner)
            assert terrain_read.cellsize == pytest.approx(terrain.cellsize)

            # Check data (within floating point precision)
            np.testing.assert_allclose(terrain_read.data, terrain.data, rtol=1e-6)

        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_read_nonexistent_file(self):
        """Test reading nonexistent file"""
        with pytest.raises(FileNotFoundError):
            ASCIIGridReader.read('/nonexistent/path/file.asc')


class TestTerrainReader:
    """Test TerrainReader interface"""

    def test_from_array(self):
        """Test creating terrain from array"""
        data = np.random.rand(20, 30)
        terrain = TerrainReader.from_array(
            data,
            xllcorner=50.0,
            yllcorner=75.0,
            cellsize=1.5
        )

        assert terrain.ncols == 20
        assert terrain.nrows == 30
        assert terrain.xllcorner == 50.0
        assert terrain.yllcorner == 75.0
        assert terrain.cellsize == 1.5


class TestGeometryGenerator:
    """Test geometry generator"""

    def test_flat_surface(self):
        """Test flat surface generation"""
        terrain = GeometryGenerator.flat_surface(
            ncols=10,
            nrows=20,
            elevation=5.0
        )

        assert terrain.ncols == 10
        assert terrain.nrows == 20
        assert np.all(terrain.data == 5.0)

    def test_inclined_plane(self):
        """Test inclined plane generation"""
        terrain = GeometryGenerator.inclined_plane(
            ncols=10,
            nrows=10,
            slope_x=0.1,
            slope_y=0.0,
            base_elevation=0.0,
            cellsize=1.0
        )

        # Check that elevation increases with x
        assert terrain.data[9, 0] > terrain.data[0, 0]
        # Slope should be approximately 0.1
        expected_diff = 0.1 * 9  # 9 cells * 0.1 slope
        actual_diff = terrain.data[9, 0] - terrain.data[0, 0]
        assert actual_diff == pytest.approx(expected_diff)

    def test_gaussian_hill(self):
        """Test Gaussian hill generation"""
        terrain = GeometryGenerator.gaussian_hill(
            ncols=50,
            nrows=50,
            center_x=25.0,
            center_y=25.0,
            height=10.0,
            width=10.0,
            cellsize=1.0
        )

        # Maximum should be at center
        center_idx = 25
        max_elevation = np.max(terrain.data)
        center_elevation = terrain.data[center_idx, center_idx]

        assert center_elevation == pytest.approx(max_elevation, rel=0.01)

    def test_valley(self):
        """Test valley generation"""
        terrain = GeometryGenerator.valley(
            ncols=50,
            nrows=50,
            orientation='x',
            depth=10.0,
            base_elevation=20.0,
            cellsize=1.0
        )

        # Minimum should be at center row
        center_j = 25
        min_elevation = np.min(terrain.data[:, center_j])
        max_elevation = np.max(terrain.data[:, 0])

        assert min_elevation < max_elevation

    def test_random_terrain(self):
        """Test random terrain generation"""
        terrain1 = GeometryGenerator.random_terrain(
            ncols=50,
            nrows=50,
            amplitude=5.0,
            wavelength=20.0,
            seed=42
        )

        terrain2 = GeometryGenerator.random_terrain(
            ncols=50,
            nrows=50,
            amplitude=5.0,
            wavelength=20.0,
            seed=42
        )

        # Same seed should produce same terrain
        np.testing.assert_array_equal(terrain1.data, terrain2.data)

    def test_dam_break_channel(self):
        """Test dam break channel generation"""
        terrain = GeometryGenerator.dam_break_channel(
            length=100.0,
            width=50.0,
            dam_position=50.0,
            upstream_elevation=10.0,
            downstream_elevation=0.0,
            cellsize=1.0
        )

        # Check dimensions
        assert terrain.ncols == 100
        assert terrain.nrows == 50

        # Check step at dam
        assert terrain.data[40, 25] == pytest.approx(10.0)  # Upstream
        assert terrain.data[60, 25] == pytest.approx(0.0)   # Downstream

    def test_composite_terrain(self):
        """Test composite terrain generation"""
        features = [
            ('inclined_plane', {'slope_x': 0.01, 'base_elevation': 0.0}),
            ('gaussian_hill', {'center_x': 50.0, 'center_y': 25.0, 'height': 10.0, 'width': 20.0})
        ]

        terrain = GeometryGenerator.composite_terrain(
            features=features,
            ncols=100,
            nrows=50,
            cellsize=1.0
        )

        assert terrain.ncols == 100
        assert terrain.nrows == 50

        # Should have both slope and hill
        # Check that terrain varies
        assert np.std(terrain.data) > 0

    def test_create_test_terrain(self):
        """Test convenience function"""
        terrain = create_test_terrain('hill', ncols=50, nrows=50, height=15.0)

        assert terrain.ncols == 50
        assert terrain.nrows == 50
        assert np.max(terrain.data) > 10.0  # Hill should have some height


class TestTerrainProcessor:
    """Test terrain processor"""

    def setup_method(self):
        """Setup for each test"""
        # Create simple test terrain
        self.terrain = GeometryGenerator.gaussian_hill(
            ncols=50,
            nrows=50,
            height=10.0,
            width=20.0,
            cellsize=1.0
        )
        self.processor = TerrainProcessor(self.terrain)

    def test_interpolate_to_mesh(self):
        """Test terrain interpolation to mesh"""
        # Create mesh
        domain = DomainParams(xmin=0.0, xmax=49.0, ymin=0.0, ymax=49.0)
        generator = MeshGenerator(domain)
        mesh = generator.generate_uniform_mesh(nx=25, ny=25)

        # Interpolate
        elevation = self.processor.interpolate_to_mesh(mesh, method='linear')

        assert elevation.shape == (25, 25)
        # Should have reasonable values
        assert np.max(elevation) > 0
        assert np.max(elevation) <= 11.0  # Hill height + buffer

    def test_smooth_gaussian(self):
        """Test Gaussian smoothing"""
        terrain_smooth = self.processor.smooth_gaussian(sigma=2.0)

        # Smoothed terrain should have lower variance
        assert np.std(terrain_smooth.data) < np.std(self.terrain.data)

    def test_smooth_uniform(self):
        """Test uniform smoothing"""
        terrain_smooth = self.processor.smooth_uniform(size=3)

        # Smoothed terrain should exist
        assert terrain_smooth.data.shape == self.terrain.data.shape

    def test_compute_gradient(self):
        """Test gradient computation"""
        grad_x, grad_y = self.processor.compute_gradient()

        assert grad_x.shape == self.terrain.data.shape
        assert grad_y.shape == self.terrain.data.shape

        # Gradients should not all be zero for hill
        assert np.max(np.abs(grad_x)) > 0
        assert np.max(np.abs(grad_y)) > 0

    def test_compute_slope(self):
        """Test slope computation"""
        slope = self.processor.compute_slope()

        assert slope.shape == self.terrain.data.shape
        # Slope should be positive
        assert np.all(slope >= 0)
        # Maximum slope should be at hill sides
        assert np.max(slope) > 0

    def test_compute_aspect(self):
        """Test aspect computation"""
        aspect = self.processor.compute_aspect()

        assert aspect.shape == self.terrain.data.shape
        # Aspect should be in [0, 360] range (or -1 for nodata)
        valid_mask = aspect >= 0
        assert np.all(aspect[valid_mask] >= 0)
        assert np.all(aspect[valid_mask] <= 360)

    def test_compute_curvature(self):
        """Test curvature computation"""
        profile_curv, planform_curv = self.processor.compute_curvature()

        assert profile_curv.shape == self.terrain.data.shape
        assert planform_curv.shape == self.terrain.data.shape

    def test_identify_refinement_zones(self):
        """Test refinement zone identification"""
        refinement = self.processor.identify_refinement_zones(slope_threshold=0.1)

        assert refinement.shape == self.terrain.data.shape
        assert refinement.dtype == bool
        # Should identify some zones
        assert np.any(refinement)

    def test_resample(self):
        """Test terrain resampling"""
        terrain_coarse = self.processor.resample(new_cellsize=2.0)

        # Coarser terrain should have fewer cells
        assert terrain_coarse.ncols < self.terrain.ncols
        assert terrain_coarse.nrows < self.terrain.nrows
        assert terrain_coarse.cellsize == 2.0

    def test_clip_to_extent(self):
        """Test terrain clipping"""
        terrain_clipped = self.processor.clip_to_extent(
            xmin=10.0, xmax=40.0,
            ymin=10.0, ymax=40.0
        )

        # Clipped terrain should be smaller
        assert terrain_clipped.ncols < self.terrain.ncols
        assert terrain_clipped.nrows < self.terrain.nrows

    def test_fill_nodata(self):
        """Test nodata filling"""
        # Create terrain with nodata values
        data = np.random.rand(20, 20)
        data[5:7, 5:7] = -9999  # Add nodata region

        terrain = TerrainData(
            data=data,
            ncols=20,
            nrows=20,
            xllcorner=0.0,
            yllcorner=0.0,
            cellsize=1.0,
            nodata_value=-9999
        )

        processor = TerrainProcessor(terrain)
        terrain_filled = processor.fill_nodata(method='nearest')

        # Should have no more nodata values
        assert not np.any(terrain_filled.data == -9999)


class TestConvenienceFunctions:
    """Test convenience functions"""

    def test_process_terrain_for_simulation(self):
        """Test full terrain processing pipeline"""
        # Create terrain
        terrain = GeometryGenerator.gaussian_hill(
            ncols=100,
            nrows=100,
            height=10.0,
            cellsize=1.0
        )

        # Create mesh
        domain = DomainParams(xmin=0.0, xmax=99.0, ymin=0.0, ymax=99.0)
        generator = MeshGenerator(domain)
        mesh = generator.generate_uniform_mesh(nx=50, ny=50)

        # Process terrain
        elevation = process_terrain_for_simulation(
            terrain,
            mesh,
            smooth=True,
            smooth_sigma=1.0
        )

        assert elevation.shape == (50, 50)
        assert np.max(elevation) > 0

    def test_load_save_terrain(self):
        """Test load/save convenience functions"""
        # Create test terrain
        terrain = create_test_terrain('hill', ncols=20, nrows=30)

        # Save
        with tempfile.NamedTemporaryFile(mode='w', suffix='.asc', delete=False) as f:
            filepath = f.name

        try:
            save_terrain(terrain, filepath)
            assert os.path.exists(filepath)

            # Load back
            terrain_loaded = load_terrain(filepath)

            assert terrain_loaded.ncols == terrain.ncols
            assert terrain_loaded.nrows == terrain.nrows

        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_load_unsupported_format(self):
        """Test loading unsupported format"""
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_terrain('test.xyz')

    def test_save_unsupported_format(self):
        """Test saving unsupported format"""
        terrain = create_test_terrain('flat')
        with pytest.raises(ValueError, match="Unsupported file format"):
            save_terrain(terrain, 'test.xyz')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
