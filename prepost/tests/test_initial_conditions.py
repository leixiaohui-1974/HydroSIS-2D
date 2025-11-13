# -*- coding: utf-8 -*-
"""
Unit tests for initial conditions module
"""

import pytest
import numpy as np
import tempfile
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.initial_conditions import (
    ICType,
    InitialCondition,
    UniformIC,
    DamBreakIC,
    DryBedIC,
    GaussianHumpIC,
    ParabolicBowlIC,
    CustomFieldIC,
    InitialConditionManager,
    create_ic_from_dict
)
from preprocessing.mesh_generation import DomainParams, MeshGenerator


class TestICEnum:
    """Test initial condition enum"""

    def test_ic_type_enum(self):
        """Test ICType enumeration"""
        assert ICType.UNIFORM.value == "uniform"
        assert ICType.DAM_BREAK.value == "dam_break"
        assert ICType.DRY_BED.value == "dry_bed"
        assert ICType.CUSTOM.value == "custom"


class TestUniformIC:
    """Test uniform initial condition"""

    def test_creation(self):
        """Test uniform IC creation"""
        ic = UniformIC(depth=2.0, velocity_x=1.0, velocity_y=0.5)

        assert ic.ic_type == ICType.UNIFORM
        assert ic.depth == 2.0
        assert ic.velocity_x == 1.0
        assert ic.velocity_y == 0.5

    def test_generate(self):
        """Test field generation"""
        ic = UniformIC(depth=3.0, velocity_x=0.5)

        h, u, v = ic.generate(10, 20)

        assert h.shape == (10, 20)
        assert u.shape == (10, 20)
        assert v.shape == (10, 20)

        assert np.all(h == 3.0)
        assert np.all(u == 0.5)
        assert np.all(v == 0.0)

    def test_to_dict(self):
        """Test dictionary export"""
        ic = UniformIC(depth=2.5, velocity_x=1.5)
        data = ic.to_dict()

        assert data['type'] == 'uniform'
        assert data['depth'] == 2.5
        assert data['velocity_x'] == 1.5


class TestDamBreakIC:
    """Test dam break initial condition"""

    def test_creation(self):
        """Test dam break IC creation"""
        ic = DamBreakIC(
            dam_position=0.5,
            upstream_depth=10.0,
            downstream_depth=1.0,
            orientation='x'
        )

        assert ic.ic_type == ICType.DAM_BREAK
        assert ic.dam_position == 0.5
        assert ic.upstream_depth == 10.0
        assert ic.downstream_depth == 1.0

    def test_invalid_position(self):
        """Test invalid dam position"""
        with pytest.raises(ValueError, match="Dam position must be between 0 and 1"):
            DamBreakIC(dam_position=1.5, upstream_depth=10.0)

    def test_generate_x_orientation(self):
        """Test generation with x-orientation"""
        ic = DamBreakIC(
            dam_position=0.5,
            upstream_depth=10.0,
            downstream_depth=0.0,
            orientation='x'
        )

        h, u, v = ic.generate(20, 10)

        # Check upstream (first half)
        assert np.all(h[:10, :] == 10.0)
        # Check downstream (second half)
        assert np.all(h[10:, :] == 0.0)
        # Velocities should be zero
        assert np.all(u == 0.0)
        assert np.all(v == 0.0)

    def test_generate_y_orientation(self):
        """Test generation with y-orientation"""
        ic = DamBreakIC(
            dam_position=0.5,
            upstream_depth=5.0,
            downstream_depth=1.0,
            orientation='y'
        )

        h, u, v = ic.generate(10, 20)

        # Check upstream (first half in y)
        assert np.all(h[:, :10] == 5.0)
        # Check downstream (second half in y)
        assert np.all(h[:, 10:] == 1.0)


class TestDryBedIC:
    """Test dry bed initial condition"""

    def test_creation(self):
        """Test dry bed IC creation"""
        ic = DryBedIC()
        assert ic.ic_type == ICType.DRY_BED

    def test_generate(self):
        """Test field generation"""
        ic = DryBedIC()

        h, u, v = ic.generate(15, 25)

        assert np.all(h == 0.0)
        assert np.all(u == 0.0)
        assert np.all(v == 0.0)


class TestGaussianHumpIC:
    """Test Gaussian hump initial condition"""

    def test_creation(self):
        """Test Gaussian hump IC creation"""
        ic = GaussianHumpIC(
            center_x=0.5,
            center_y=0.5,
            amplitude=2.0,
            width=0.1
        )

        assert ic.amplitude == 2.0
        assert ic.width == 0.1

    def test_generate(self):
        """Test field generation"""
        ic = GaussianHumpIC(
            center_x=0.5,
            center_y=0.5,
            amplitude=1.0,
            width=0.1,
            base_depth=0.0
        )

        h, u, v = ic.generate(50, 50)

        # Maximum should be at center
        max_idx = np.unravel_index(np.argmax(h), h.shape)
        # Should be near center (index 25)
        assert abs(max_idx[0] - 25) <= 1
        assert abs(max_idx[1] - 25) <= 1

        # Maximum should be approximately amplitude
        assert h[max_idx] == pytest.approx(1.0, rel=0.1)

        # Velocities should be zero
        assert np.all(u == 0.0)
        assert np.all(v == 0.0)


class TestParabolicBowlIC:
    """Test parabolic bowl initial condition"""

    def test_creation(self):
        """Test parabolic bowl IC creation"""
        ic = ParabolicBowlIC(bowl_depth=1.0, water_depth=0.5)

        assert ic.bowl_depth == 1.0
        assert ic.water_depth == 0.5

    def test_generate(self):
        """Test field generation"""
        ic = ParabolicBowlIC(bowl_depth=1.0, water_depth=0.5)

        h, u, v = ic.generate(50, 50)

        # Water depth should be non-negative
        assert np.all(h >= 0)

        # Maximum depth at center
        center = 25
        assert h[center, center] == pytest.approx(0.5, rel=0.1)

        # Velocities should be zero
        assert np.all(u == 0.0)
        assert np.all(v == 0.0)


class TestCustomFieldIC:
    """Test custom field initial condition"""

    def test_creation(self):
        """Test custom field IC creation"""
        depth = np.random.rand(10, 20)
        vel_x = np.random.rand(10, 20)

        ic = CustomFieldIC(depth=depth, velocity_x=vel_x)

        assert ic.depth_field.shape == (10, 20)
        assert ic.velocity_x_field.shape == (10, 20)

    def test_shape_mismatch(self):
        """Test error on shape mismatch"""
        depth = np.random.rand(10, 20)
        vel_x = np.random.rand(15, 25)  # Wrong shape

        with pytest.raises(ValueError, match="shape must match"):
            CustomFieldIC(depth=depth, velocity_x=vel_x)

    def test_generate(self):
        """Test field generation"""
        depth = np.ones((10, 20)) * 2.0
        vel_x = np.ones((10, 20)) * 0.5

        ic = CustomFieldIC(depth=depth, velocity_x=vel_x)

        h, u, v = ic.generate(10, 20)

        np.testing.assert_array_equal(h, depth)
        np.testing.assert_array_equal(u, vel_x)


class TestInitialConditionManager:
    """Test initial condition manager"""

    def setup_method(self):
        """Setup for each test"""
        domain = DomainParams(xmin=0.0, xmax=100.0, ymin=0.0, ymax=50.0)
        generator = MeshGenerator(domain)
        self.mesh = generator.generate_uniform_mesh(20, 10)
        self.manager = InitialConditionManager(self.mesh)

    def test_creation(self):
        """Test manager creation"""
        assert self.manager.mesh == self.mesh
        assert self.manager.terrain.shape == (20, 10)
        assert self.manager.ic is None

    def test_set_initial_condition(self):
        """Test setting initial condition"""
        ic = UniformIC(depth=5.0)
        self.manager.set_initial_condition(ic)

        assert self.manager.ic == ic
        assert self.manager.depth is not None
        assert self.manager.depth.shape == (20, 10)

    def test_generate(self):
        """Test field generation"""
        ic = DamBreakIC(dam_position=0.5, upstream_depth=10.0)
        self.manager.set_initial_condition(ic)

        assert self.manager.depth is not None
        assert np.max(self.manager.depth) == 10.0

    def test_validation_valid(self):
        """Test validation with valid IC"""
        ic = UniformIC(depth=2.0)
        self.manager.set_initial_condition(ic)

        is_valid, errors = self.manager.validate()

        assert is_valid
        assert len(errors) == 0

    def test_validation_negative_depth(self):
        """Test validation with negative depth"""
        # Create custom IC with negative values
        depth = -np.ones((20, 10))
        ic = CustomFieldIC(depth=depth)
        self.manager.set_initial_condition(ic)

        is_valid, errors = self.manager.validate()

        assert not is_valid
        assert any("Negative depths" in err for err in errors)

    def test_apply_terrain(self):
        """Test applying terrain"""
        terrain = np.random.rand(20, 10) * 5.0
        self.manager.apply_terrain(terrain)

        np.testing.assert_array_equal(self.manager.terrain, terrain)

    def test_water_surface_elevation(self):
        """Test water surface elevation computation"""
        ic = UniformIC(depth=3.0)
        self.manager.set_initial_condition(ic)

        terrain = np.ones((20, 10)) * 10.0
        self.manager.apply_terrain(terrain)

        wse = self.manager.get_water_surface_elevation()

        expected = 3.0 + 10.0
        assert np.all(wse == expected)

    def test_total_volume(self):
        """Test total volume computation"""
        ic = UniformIC(depth=2.0)
        self.manager.set_initial_condition(ic)

        volume = self.manager.get_total_volume()

        # Expected volume = depth * area
        area = (self.mesh.domain.xmax - self.mesh.domain.xmin) * \
               (self.mesh.domain.ymax - self.mesh.domain.ymin)
        expected_volume = 2.0 * area

        assert volume == pytest.approx(expected_volume)

    def test_statistics(self):
        """Test statistics computation"""
        ic = DamBreakIC(dam_position=0.5, upstream_depth=10.0)
        self.manager.set_initial_condition(ic)

        stats = self.manager.get_statistics()

        assert 'num_cells' in stats
        assert 'num_wet_cells' in stats
        assert 'depth' in stats
        assert stats['depth']['max'] == 10.0
        assert stats['depth']['min'] == 0.0

    def test_export_import_numpy(self):
        """Test NumPy array export/import"""
        ic = UniformIC(depth=3.5, velocity_x=1.2)
        self.manager.set_initial_condition(ic)

        # Export
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_ic')
            self.manager.export_arrays_to_numpy(filepath)

            # Import
            manager_loaded = InitialConditionManager.import_arrays_from_numpy(
                f"{filepath}.npz",
                self.mesh
            )

            # Verify
            np.testing.assert_array_almost_equal(
                manager_loaded.depth,
                self.manager.depth
            )
            np.testing.assert_array_almost_equal(
                manager_loaded.velocity_x,
                self.manager.velocity_x
            )

    def test_export_json(self):
        """Test JSON export"""
        ic = DamBreakIC(dam_position=0.4, upstream_depth=8.0)
        self.manager.set_initial_condition(ic)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            self.manager.export_to_json(filepath)

            # Verify file exists and is valid JSON
            assert os.path.exists(filepath)

            import json
            with open(filepath, 'r') as f:
                data = json.load(f)

            assert 'mesh' in data
            assert 'initial_condition' in data
            assert data['initial_condition']['type'] == 'dam_break'

        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_summary(self):
        """Test summary generation"""
        ic = UniformIC(depth=4.0)
        self.manager.set_initial_condition(ic)

        summary = self.manager.summary()

        assert "INITIAL CONDITIONS SUMMARY" in summary
        assert "uniform" in summary
        assert "VALID" in summary


class TestICFactory:
    """Test IC factory function"""

    def test_create_uniform_from_dict(self):
        """Test creating uniform IC from dict"""
        data = {
            'type': 'uniform',
            'depth': 3.0,
            'velocity_x': 1.5,
            'velocity_y': 0.5
        }

        ic = create_ic_from_dict(data)

        assert isinstance(ic, UniformIC)
        assert ic.depth == 3.0
        assert ic.velocity_x == 1.5

    def test_create_dam_break_from_dict(self):
        """Test creating dam break IC from dict"""
        data = {
            'type': 'dam_break',
            'dam_position': 0.5,
            'upstream_depth': 10.0,
            'downstream_depth': 0.0,
            'orientation': 'x'
        }

        ic = create_ic_from_dict(data)

        assert isinstance(ic, DamBreakIC)
        assert ic.dam_position == 0.5
        assert ic.upstream_depth == 10.0

    def test_create_dry_bed_from_dict(self):
        """Test creating dry bed IC from dict"""
        data = {
            'type': 'dry_bed'
        }

        ic = create_ic_from_dict(data)

        assert isinstance(ic, DryBedIC)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
