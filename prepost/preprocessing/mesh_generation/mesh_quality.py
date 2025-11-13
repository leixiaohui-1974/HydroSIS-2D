# -*- coding: utf-8 -*-
"""
Mesh quality assessment tools for HydroSIS-2D

Provides metrics and checks for mesh quality evaluation.
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import logging

from .mesh_generator import StructuredMesh

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Container for mesh quality metrics"""
    min_cell_size: float
    max_cell_size: float
    mean_cell_size: float
    aspect_ratio_min: float
    aspect_ratio_max: float
    aspect_ratio_mean: float
    uniformity_score: float  # 0-1, 1 = perfectly uniform
    total_cells: int
    passed_checks: bool

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'min_cell_size': self.min_cell_size,
            'max_cell_size': self.max_cell_size,
            'mean_cell_size': self.mean_cell_size,
            'aspect_ratio_min': self.aspect_ratio_min,
            'aspect_ratio_max': self.aspect_ratio_max,
            'aspect_ratio_mean': self.aspect_ratio_mean,
            'uniformity_score': self.uniformity_score,
            'total_cells': self.total_cells,
            'passed_checks': self.passed_checks
        }


class MeshQualityChecker:
    """
    Mesh quality assessment tool

    Evaluates mesh quality based on:
        - Cell size distribution
        - Aspect ratios
        - Uniformity
        - Resolution adequacy
    """

    def __init__(self, mesh: StructuredMesh):
        """
        Initialize quality checker

        Args:
            mesh: StructuredMesh to evaluate
        """
        self.mesh = mesh
        self.metrics = None
        self.warnings: List[str] = []

    def compute_metrics(self) -> QualityMetrics:
        """
        Compute comprehensive quality metrics

        Returns:
            QualityMetrics object
        """
        # For structured mesh, cells are uniform
        cell_area = self.mesh.get_cell_area()
        cell_size = np.sqrt(cell_area)

        aspect_ratio = self.mesh.aspect_ratio

        # Uniformity score (1.0 for uniform structured mesh)
        uniformity_score = 1.0

        # Check quality criteria
        passed_checks = self._run_quality_checks()

        self.metrics = QualityMetrics(
            min_cell_size=cell_size,
            max_cell_size=cell_size,
            mean_cell_size=cell_size,
            aspect_ratio_min=aspect_ratio,
            aspect_ratio_max=aspect_ratio,
            aspect_ratio_mean=aspect_ratio,
            uniformity_score=uniformity_score,
            total_cells=self.mesh.ncells,
            passed_checks=passed_checks
        )

        return self.metrics

    def _run_quality_checks(self) -> bool:
        """
        Run quality checks and generate warnings

        Returns:
            True if all checks passed
        """
        self.warnings.clear()
        all_passed = True

        # Check 1: Aspect ratio
        if self.mesh.aspect_ratio > 5.0 or self.mesh.aspect_ratio < 0.2:
            self.warnings.append(
                f"[WARN]️  Aspect ratio {self.mesh.aspect_ratio:.2f} is outside "
                f"recommended range [0.2, 5.0]"
            )
            all_passed = False

        # Check 2: Resolution
        if self.mesh.dx < 0.1:
            self.warnings.append(
                f"[WARN]️  Cell size dx={self.mesh.dx:.3f} m may be too small "
                f"(computational cost warning)"
            )

        if self.mesh.dx > 100:
            self.warnings.append(
                f"[WARN]️  Cell size dx={self.mesh.dx:.3f} m may be too large "
                f"(accuracy warning)"
            )

        # Check 3: Total cell count
        if self.mesh.ncells > 10_000_000:
            self.warnings.append(
                f"[WARN]️  Large mesh ({self.mesh.ncells:,} cells) may require "
                f"significant computational resources"
            )

        if self.mesh.ncells < 100:
            self.warnings.append(
                f"[WARN]️  Very coarse mesh ({self.mesh.ncells} cells) may have "
                f"limited accuracy"
            )

        # Check 4: Domain aspect ratio
        domain_aspect = self.mesh.domain.length_y / self.mesh.domain.length_x
        if domain_aspect > 10 or domain_aspect < 0.1:
            self.warnings.append(
                f"[WARN]️  Domain aspect ratio {domain_aspect:.2f} is extreme, "
                f"consider reviewing domain definition"
            )

        return all_passed

    def check_cfl_condition(self, max_velocity: float, dt: float, gravity: float = 9.81) -> Dict:
        """
        Check CFL condition for stability

        Args:
            max_velocity: Maximum expected flow velocity [m/s]
            dt: Time step [s]
            gravity: Gravitational acceleration [m/s^2]

        Returns:
            Dictionary with CFL analysis
        """
        # Shallow water wave speed: c = sqrt(g*h)
        # For safety, check with reasonable water depth
        h_check = 10.0  # meters

        wave_speed = np.sqrt(gravity * h_check)
        max_signal_speed = max_velocity + wave_speed

        # CFL number
        cfl_x = max_signal_speed * dt / self.mesh.dx
        cfl_y = max_signal_speed * dt / self.mesh.dy
        cfl_max = max(cfl_x, cfl_y)

        # Recommended CFL < 0.5 for stability
        is_stable = cfl_max < 0.5

        result = {
            'cfl_x': cfl_x,
            'cfl_y': cfl_y,
            'cfl_max': cfl_max,
            'is_stable': is_stable,
            'recommended_dt': 0.5 * min(self.mesh.dx, self.mesh.dy) / max_signal_speed,
            'wave_speed': wave_speed,
            'max_signal_speed': max_signal_speed
        }

        if not is_stable:
            self.warnings.append(
                f"[WARN]️  CFL condition violated: CFL={cfl_max:.3f} > 0.5. "
                f"Reduce time step to dt < {result['recommended_dt']:.4f} s"
            )

        return result

    def estimate_computational_cost(self, simulation_time: float,
                                    cfl_number: float = 0.5,
                                    max_velocity: float = 5.0,
                                    gravity: float = 9.81) -> Dict:
        """
        Estimate computational cost

        Args:
            simulation_time: Total simulation time [s]
            cfl_number: Target CFL number
            max_velocity: Maximum expected velocity [m/s]
            gravity: Gravitational acceleration [m/s^2]

        Returns:
            Dictionary with cost estimates
        """
        h_typical = 5.0  # meters
        wave_speed = np.sqrt(gravity * h_typical)
        max_signal_speed = max_velocity + wave_speed

        # Estimate time step
        dt = cfl_number * min(self.mesh.dx, self.mesh.dy) / max_signal_speed

        # Number of time steps
        num_timesteps = int(np.ceil(simulation_time / dt))

        # Computational operations per timestep (rough estimate)
        ops_per_cell_per_step = 500  # FLOPS
        total_ops = self.mesh.ncells * num_timesteps * ops_per_cell_per_step

        # Memory estimate (bytes per cell for conservative vars + temp arrays)
        bytes_per_cell = 8 * 10  # 10 double precision values
        memory_mb = self.mesh.ncells * bytes_per_cell / (1024**2)

        result = {
            'estimated_dt': dt,
            'num_timesteps': num_timesteps,
            'total_gflops': total_ops / 1e9,
            'memory_mb': memory_mb,
            'walltime_estimate_minutes': num_timesteps * 0.001,  # rough estimate
            'output_files_estimate_mb': num_timesteps * memory_mb / 100  # if saving every 100 steps
        }

        return result

    def generate_quality_report(self) -> str:
        """
        Generate comprehensive quality report

        Returns:
            Formatted report string
        """
        if self.metrics is None:
            self.compute_metrics()

        report = []
        report.append("=" * 60)
        report.append("MESH QUALITY REPORT")
        report.append("=" * 60)
        report.append("")

        # Basic info
        report.append("Basic Information:")
        report.append(f"  Total cells: {self.metrics.total_cells:,}")
        report.append(f"  Grid: {self.mesh.nx} x {self.mesh.ny}")
        report.append(f"  Domain: [{self.mesh.domain.xmin:.2f}, {self.mesh.domain.xmax:.2f}] x "
                     f"[{self.mesh.domain.ymin:.2f}, {self.mesh.domain.ymax:.2f}] m")
        report.append("")

        # Cell properties
        report.append("Cell Properties:")
        report.append(f"  Cell size (dx): {self.mesh.dx:.4f} m")
        report.append(f"  Cell size (dy): {self.mesh.dy:.4f} m")
        report.append(f"  Cell area: {self.mesh.get_cell_area():.6f} m^2")
        report.append(f"  Aspect ratio: {self.mesh.aspect_ratio:.4f}")
        report.append("")

        # Quality metrics
        report.append("Quality Metrics:")
        report.append(f"  Uniformity score: {self.metrics.uniformity_score:.3f}")
        report.append(f"  Quality check: {'[OK] PASSED' if self.metrics.passed_checks else '[ERROR] WARNINGS'}")
        report.append("")

        # Warnings
        if self.warnings:
            report.append("Warnings:")
            for warning in self.warnings:
                report.append(f"  {warning}")
            report.append("")

        report.append("=" * 60)

        return "\n".join(report)

    def print_report(self) -> None:
        """Print quality report to console"""
        print(self.generate_quality_report())

    def export_report(self, filename: str) -> None:
        """
        Export quality report to file

        Args:
            filename: Output file path
        """
        report = self.generate_quality_report()
        with open(filename, 'w') as f:
            f.write(report)
        logger.info(f"Quality report exported to {filename}")
