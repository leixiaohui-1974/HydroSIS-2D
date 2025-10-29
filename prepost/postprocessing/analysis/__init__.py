"""
Analysis module for HydroSIS-2D postprocessing

Provides tools for result analysis, statistics, and validation.
"""

from .result_analyzer import ResultAnalyzer
from .statistics import StatisticsCalculator
from .profile_extractor import ProfileExtractor

__all__ = [
    'ResultAnalyzer',
    'StatisticsCalculator',
    'ProfileExtractor'
]
