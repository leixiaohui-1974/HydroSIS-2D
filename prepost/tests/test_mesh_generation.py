# -*- coding: utf-8 -*-
"""
Unit tests for mesh generation module
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.mesh_generation import (
    MeshGenerator,
    AdaptiveMeshGenerator,
    MeshQualityChecker,
    MeshIO,
    DomainParams,
    MeshParams,
    StructuredMesh,
    RefinementZone
)


class TestDomainParams:
    """Test domain parameter class"""

    def test_valid_domain(self):
        domain = DomainParams(xmin=0, xmax=100, ymin=0, ymax=50)
        assert domain.length_x == 100
        assert domain.length_y == 50
        assert domain.area == 5000

    def test_invalid_domain(self):
        with pytest.raises(ValueError):
            DomainParams(xmin=100, xmax=0, ymin=0, ymax=50)

        with pytest.raises(ValueError):
            DomainParams(xmin=0, xmax=100, ymin=50, ymax=0)


class TestMeshParams:
    """Test mesh parameter class"""

    def test_valid_params(self):
        params = MeshParams(nx=100, ny=50)
        assert params.nx == 100
        assert params.ny == 50

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            MeshParams(nx=-10, ny=50)

        with pytest.raises(ValueError):
            MeshParams(nx=100, ny=0)


class TestStructuredMesh:
    """Test structured mesh class"""

    def test_mesh_creation(self):
        domain = DomainParams(xmin=0, xmax=100, ymin=0, ymax=50)
        mesh_params = MeshParams(nx=10, ny=5)
        mesh = StructuredMesh(domain, mesh_params)

        assert mesh.nx == 10
        assert mesh.ny == 5
        assert mesh.ncells == 50
        assert mesh.dx == 10.0
        assert mesh.dy == 10.0

    def test_cell_coordinates(self):
        domain = DomainParams(xmin=0, xmax=100, ymin=0, ymax=100)
        mesh_params = MeshParams(nx=10, ny=10)
        mesh = StructuredMesh(domain, mesh_params)

        # First cell center should be at (5, 5)
        x, y = mesh.get_cell_coordinates(0, 0)
        assert abs(x - 5.0) < 1e-10
        assert abs(y - 5.0) < 1e-10

        # Last cell center
        x, y = mesh.get_cell_coordinates(9, 9)
        assert abs(x - 95.0) < 1e-10
        assert abs(y - 95.0) < 1e-10

    def test_find_cell(self):
        domain = DomainParams(xmin=0, xmax=100, ymin=0, ymax=100)
        mesh_params = MeshParams(nx=10, ny=10)
        mesh = StructuredMesh(domain, mesh_params)

        i, j = mesh.find_cell(5.0, 5.0)
        assert i == 0
        assert j == 0

        i, j = mesh.find_cell(95.0, 95.0)
        assert i == 9
        assert j == 9

    def test_mesh_info(self):
        domain = DomainParams(xmin=0, xmax=100, ymin=0, ymax=50)
        mesh_params = MeshParams(nx=10, ny=5)
        mesh = StructuredMesh(domain, mesh_params)

        info = mesh.get_info()
        assert info['nx'] == 10
        assert info['ny'] == 5
        assert info['ncells'] == 50
        assert abs(info['cell_area'] - 100.0) < 1e-10


class TestMeshGenerator:
    """Test basic mesh generator"""

    def test_uniform_mesh(self):
        domain = DomainParams(xmin=0, xmax=100, ymin=0, ymax=50)
        generator = MeshGenerator(domain)
        mesh = generator.generate_uniform_mesh(20, 10)

        assert mesh.nx == 20
        assert mesh.ny == 10
        assert abs(mesh.dx - 5.0) < 1e-10
        assert abs(mesh.dy - 5.0) < 1e-10

    def test_mesh_with_spacing(self):
        domain = DomainParams(xmin=0, xmax=100, ymin=0, ymax=100)
        generator = MeshGenerator(domain)
        mesh = generator.generate_mesh_with_spacing(dx=2.0, dy=2.0)

        assert mesh.dx == 2.0
        assert mesh.dy == 2.0
        assert mesh.nx == 50
        assert mesh.ny == 50

    def test_mesh_from_resolution(self):
        domain = DomainParams(xmin=0, xmax=100, ymin=0, ymax=100)
        generator = MeshGenerator(domain)
        mesh = generator.generate_mesh_from_resolution(target_cell_area=4.0)

        # Cell area should be close to 4.0 (2x2)
        assert abs(mesh.get_cell_area() - 4.0) < 0.1


class TestRefinementZone:
    """Test refinement zone class"""

    def test_zone_creation(self):
        zone = RefinementZone(xmin=10, xmax=20, ymin=10, ymax=20, refinement_level=2)
        assert zone.get_refinement_factor() == 4

    def test_point_containment(self):
        zone = RefinementZone(xmin=10, xmax=20, ymin=10, ymax=20, refinement_level=1)
        assert zone.contains_point(15, 15) is True
        assert zone.contains_point(5, 5) is False
        assert zone.contains_point(10, 10) is True  # On boundary
        assert zone.contains_point(20, 20) is True


class TestAdaptiveMeshGenerator:
    """Test adaptive mesh generator"""

    def test_add_refinement_zone(self):
        domain = DomainParams(xmin=0, xmax=100, ymin=0, ymax=100)
        generator = AdaptiveMeshGenerator(domain)

        generator.add_refinement_zone(xmin=40, xmax=60, ymin=40, ymax=60, refinement_level=1)
        assert len(generator.refinement_zones) == 1

    def test_refinement_map_generation(self):
        domain = DomainParams(xmin=0, xmax=100, ymin=0, ymax=100)
        generator = AdaptiveMeshGenerator(domain)

        generator.add_refinement_zone(xmin=40, xmax=60, ymin=40, ymax=60, refinement_level=2)
        refinement_map = generator.generate_refinement_map_from_zones(10, 10)

        assert refinement_map.shape == (10, 10)
        assert np.max(refinement_map) == 2
        assert np.sum(refinement_map > 0) > 0  # Some cells should be refined

    def test_terrain_based_refinement(self):
        domain = DomainParams(xmin=0, xmax=100, ymin=0, ymax=100)
        generator = AdaptiveMeshGenerator(domain)

        # Create synthetic terrain with gradient
        terrain = np.zeros((10, 10))
        terrain[5:, :] = 10.0  # Step change

        refinement_map = generator.generate_refinement_map_from_terrain(
            terrain, gradient_threshold=0.1, max_refinement_level=2
        )

        assert refinement_map.shape == (10, 10)
        assert np.sum(refinement_map > 0) > 0  # Should detect gradient

    def test_multiresolution_mesh(self):
        domain = DomainParams(xmin=0, xmax=100, ymin=0, ymax=100)
        generator = AdaptiveMeshGenerator(domain)

        generator.add_refinement_zone(xmin=40, xmax=60, ymin=40, ymax=60, refinement_level=1)
        mesh = generator.generate_multiresolution_mesh(10, 10)

        # Mesh should be 2x finer than base
        assert mesh.nx == 20
        assert mesh.ny == 20


class TestMeshQualityChecker:
    """Test mesh quality assessment"""

    def test_compute_metrics(self):
        domain = DomainParams(xmin=0, xmax=100, ymin=0, ymax=100)
        mesh_params = MeshParams(nx=10, ny=10)
        mesh = StructuredMesh(domain, mesh_params)

        checker = MeshQualityChecker(mesh)
        metrics = checker.compute_metrics()

        assert metrics.total_cells == 100
        assert abs(metrics.aspect_ratio_mean - 1.0) < 1e-10  # Square cells
        assert metrics.uniformity_score == 1.0  # Uniform mesh

    def test_cfl_check(self):
        domain = DomainParams(xmin=0, xmax=100, ymin=0, ymax=100)
        mesh_params = MeshParams(nx=10, ny=10)
        mesh = StructuredMesh(domain, mesh_params)

        checker = MeshQualityChecker(mesh)
        cfl_result = checker.check_cfl_condition(max_velocity=5.0, dt=0.1)

        assert 'cfl_max' in cfl_result
        assert 'is_stable' in cfl_result
        assert 'recommended_dt' in cfl_result

    def test_cost_estimate(self):
        domain = DomainParams(xmin=0, xmax=100, ymin=0, ymax=100)
        mesh_params = MeshParams(nx=50, ny=50)
        mesh = StructuredMesh(domain, mesh_params)

        checker = MeshQualityChecker(mesh)
        cost = checker.estimate_computational_cost(simulation_time=100.0)

        assert 'num_timesteps' in cost
        assert 'total_gflops' in cost
        assert 'memory_mb' in cost
        assert cost['num_timesteps'] > 0


class TestMeshIO:
    """Test mesh I/O operations"""

    def test_export_to_ini(self, tmp_path):
        domain = DomainParams(xmin=0, xmax=100, ymin=0, ymax=50)
        mesh_params = MeshParams(nx=10, ny=5)
        mesh = StructuredMesh(domain, mesh_params)

        output_file = tmp_path / "test_mesh.ini"
        MeshIO.export_to_ini(mesh, str(output_file))

        assert output_file.exists()

        # Check file content
        content = output_file.read_text()
        assert "nx = 10" in content
        assert "ny = 5" in content

    def test_export_to_json(self, tmp_path):
        domain = DomainParams(xmin=0, xmax=100, ymin=0, ymax=50)
        mesh_params = MeshParams(nx=10, ny=5)
        mesh = StructuredMesh(domain, mesh_params)

        output_file = tmp_path / "test_mesh.json"
        MeshIO.export_to_json(mesh, str(output_file))

        assert output_file.exists()

    def test_import_from_json(self, tmp_path):
        # Create and export mesh
        domain = DomainParams(xmin=0, xmax=100, ymin=0, ymax=50)
        mesh_params = MeshParams(nx=10, ny=5)
        mesh1 = StructuredMesh(domain, mesh_params)

        json_file = tmp_path / "test_mesh.json"
        MeshIO.export_to_json(mesh1, str(json_file))

        # Import mesh
        mesh2 = MeshIO.import_from_json(str(json_file))

        assert mesh2.nx == mesh1.nx
        assert mesh2.ny == mesh1.ny
        assert abs(mesh2.dx - mesh1.dx) < 1e-10

    def test_export_batch(self, tmp_path):
        domain = DomainParams(xmin=0, xmax=100, ymin=0, ymax=50)
        mesh_params = MeshParams(nx=10, ny=5)
        mesh = StructuredMesh(domain, mesh_params)

        outputs = MeshIO.export_batch(mesh, str(tmp_path), base_name="test")

        assert 'ini' in outputs
        assert 'vtk' in outputs
        assert 'json' in outputs
        assert 'npz' in outputs

        # Check all files exist
        for filepath in outputs.values():
            assert os.path.exists(filepath)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
