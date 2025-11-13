# -*- coding: utf-8 -*-
"""
Unit tests for visualization module

Note: Some tests require PyVista with rendering support (OSMesa/EGL).
Tests are designed to work without actual rendering when in headless mode.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.mesh_generation import MeshGenerator, DomainParams
from postprocessing.visualization import get_colormap, available_colormaps

# Try to import visualization engine
try:
    from postprocessing.visualization import VisualizationEngine
    VIS_ENGINE_AVAILABLE = True
except (ImportError, OSError):
    VIS_ENGINE_AVAILABLE = False
    VisualizationEngine = None


class TestColormaps:
    """Test colormap utilities"""

    def test_get_colormap_with_field_type(self):
        cmap = get_colormap('water_depth')
        assert cmap == 'Blues'

        cmap = get_colormap('velocity')
        assert cmap == 'jet'

        cmap = get_colormap('elevation')
        assert cmap == 'terrain'

    def test_get_colormap_with_explicit_name(self):
        cmap = get_colormap(colormap_name='viridis')
        assert cmap == 'viridis'

        # Explicit name overrides field type
        cmap = get_colormap(field_type='water_depth', colormap_name='jet')
        assert cmap == 'jet'

    def test_get_colormap_default(self):
        cmap = get_colormap()
        assert cmap == 'viridis'

        cmap = get_colormap(field_type='unknown_field')
        assert cmap == 'viridis'

    def test_available_colormaps(self):
        cmaps = available_colormaps()
        assert isinstance(cmaps, list)
        assert len(cmaps) > 0
        assert 'Blues' in cmaps
        assert 'viridis' in cmaps
        assert 'terrain' in cmaps


@pytest.mark.skipif(not VIS_ENGINE_AVAILABLE,
                   reason="VisualizationEngine not available (PyVista required)")
class TestVisualizationEngine:
    """Test visualization engine"""

    def test_engine_initialization(self):
        engine = VisualizationEngine(offscreen=True)
        assert engine.offscreen is True
        assert engine.window_size == (1920, 1080)
        engine.close()

    def test_create_plotter(self):
        engine = VisualizationEngine(offscreen=True)
        plotter = engine.create_plotter()
        assert plotter is not None
        assert engine.plotter is not None
        engine.close()

    def test_create_structured_grid(self):
        # Create simple mesh
        domain = DomainParams(xmin=0, xmax=100, ymin=0, ymax=50)
        generator = MeshGenerator(domain)
        mesh = generator.generate_uniform_mesh(10, 5)

        engine = VisualizationEngine(offscreen=True)

        # Create grid without elevation
        grid = engine.create_structured_grid(mesh)
        assert grid is not None

        # Create grid with elevation
        elevation = np.ones((mesh.nx, mesh.ny)) * 5.0
        grid = engine.create_structured_grid(mesh, elevation=elevation)
        assert grid is not None

        engine.close()

    def test_create_structured_grid_wrong_shape(self):
        domain = DomainParams(xmin=0, xmax=100, ymin=0, ymax=50)
        generator = MeshGenerator(domain)
        mesh = generator.generate_uniform_mesh(10, 5)

        engine = VisualizationEngine(offscreen=True)

        # Wrong elevation shape should raise error
        elevation = np.ones((5, 10))  # Swapped dimensions
        with pytest.raises(ValueError):
            grid = engine.create_structured_grid(mesh, elevation=elevation)

        engine.close()


class TestSyntheticData:
    """Test synthetic data generation for examples"""

    def test_synthetic_data_shapes(self):
        domain = DomainParams(xmin=0, xmax=100, ymin=0, ymax=50)
        generator = MeshGenerator(domain)
        mesh = generator.generate_uniform_mesh(20, 10)

        # Create simple terrain
        terrain = np.ones((mesh.nx, mesh.ny)) * 5.0
        assert terrain.shape == (mesh.nx, mesh.ny)

        # Create water depth
        water_depth = np.ones((mesh.nx, mesh.ny)) * 2.0
        assert water_depth.shape == (mesh.nx, mesh.ny)

        # Create velocities
        u = np.ones((mesh.nx, mesh.ny)) * 1.0
        v = np.zeros((mesh.nx, mesh.ny))
        assert u.shape == (mesh.nx, mesh.ny)
        assert v.shape == (mesh.nx, mesh.ny)

        # Calculate velocity magnitude
        vel_mag = np.sqrt(u**2 + v**2)
        assert vel_mag.shape == (mesh.nx, mesh.ny)
        assert np.allclose(vel_mag, 1.0)


class TestVisualizationIntegration:
    """Integration tests for visualization"""

    def test_mesh_and_data_compatibility(self):
        """Test that mesh and data arrays are compatible"""
        domain = DomainParams(xmin=0, xmax=200, ymin=0, ymax=100)
        generator = MeshGenerator(domain)
        mesh = generator.generate_uniform_mesh(40, 20)

        # Create data arrays
        terrain = np.zeros((mesh.nx, mesh.ny))
        water_depth = np.ones((mesh.nx, mesh.ny)) * 3.0
        u = np.ones((mesh.nx, mesh.ny)) * 2.0
        v = np.zeros((mesh.nx, mesh.ny))

        # Check compatibility
        assert terrain.shape == (mesh.nx, mesh.ny)
        assert water_depth.shape == (mesh.nx, mesh.ny)
        assert u.shape == (mesh.nx, mesh.ny)
        assert v.shape == (mesh.nx, mesh.ny)

        # Water surface calculation
        water_surface = terrain + water_depth
        assert water_surface.shape == (mesh.nx, mesh.ny)
        assert np.allclose(water_surface, 3.0)

    def test_data_range_validation(self):
        """Test data range validation"""
        domain = DomainParams(xmin=0, xmax=100, ymin=0, ymax=50)
        generator = MeshGenerator(domain)
        mesh = generator.generate_uniform_mesh(20, 10)

        # Create terrain with features
        x = mesh.x
        y = mesh.y

        terrain = 5.0 + 0.01 * x  # Sloped terrain

        assert np.min(terrain) >= 5.0
        assert np.max(terrain) <= 6.0

        # Water depth (should be non-negative)
        water_depth = np.maximum(0, 10.0 - 0.02 * x)

        assert np.min(water_depth) >= 0
        assert np.max(water_depth) <= 10.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
