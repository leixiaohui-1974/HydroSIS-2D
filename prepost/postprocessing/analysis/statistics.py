"""
Statistical analysis tools for HydroSIS-2D results
"""

import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class StatisticsCalculator:
    """
    Calculate various statistics from simulation results

    Provides methods for computing temporal and spatial statistics.
    """

    def __init__(self):
        """Initialize statistics calculator"""
        pass

    @staticmethod
    def temporal_stats(field_series: List[np.ndarray], times: List[float]) -> Dict[str, np.ndarray]:
        """
        Compute temporal statistics for each cell

        Args:
            field_series: List of 2D arrays for each timestep
            times: List of corresponding times

        Returns:
            Dictionary with statistics arrays
        """
        if not field_series:
            raise ValueError("Empty field series")

        # Stack along time dimension
        field_3d = np.stack(field_series, axis=-1)

        return {
            'mean': np.mean(field_3d, axis=-1),
            'max': np.max(field_3d, axis=-1),
            'min': np.min(field_3d, axis=-1),
            'std': np.std(field_3d, axis=-1),
            'time_of_max': np.array(times)[np.argmax(field_3d, axis=-1)]
        }

    @staticmethod
    def spatial_stats(field: np.ndarray) -> Dict[str, float]:
        """
        Compute spatial statistics for a single field

        Args:
            field: 2D field array

        Returns:
            Dictionary with scalar statistics
        """
        return {
            'mean': float(np.mean(field)),
            'max': float(np.max(field)),
            'min': float(np.min(field)),
            'std': float(np.std(field)),
            'median': float(np.median(field)),
            'sum': float(np.sum(field))
        }

    @staticmethod
    def compute_percentiles(field: np.ndarray, percentiles: List[float] = [25, 50, 75, 90, 95, 99]) -> Dict[str, float]:
        """
        Compute percentiles of field values

        Args:
            field: 2D field array
            percentiles: List of percentiles to compute

        Returns:
            Dictionary mapping percentile to value
        """
        flat_field = field.flatten()
        result = {}
        for p in percentiles:
            result[f'p{p}'] = float(np.percentile(flat_field, p))
        return result
