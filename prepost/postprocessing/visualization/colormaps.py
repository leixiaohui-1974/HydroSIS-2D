# -*- coding: utf-8 -*-
"""
Colormap utilities for HydroSIS-2D visualizations

Provides scientifically appropriate colormaps for different field types.
"""

from typing import List, Dict

# Recommended colormaps for different field types
FIELD_COLORMAPS = {
    'water_depth': 'Blues',
    'elevation': 'terrain',
    'velocity': 'jet',
    'velocity_magnitude': 'plasma',
    'froude_number': 'RdYlBu_r',
    'vorticity': 'seismic',
    'divergence': 'PiYG',
    'pressure': 'viridis',
    'energy': 'inferno',
    'discharge': 'YlGnBu'
}

# Available PyVista/Matplotlib colormaps
AVAILABLE_COLORMAPS = [
    # Sequential (single hue)
    'Blues', 'Greens', 'Greys', 'Oranges', 'Purples', 'Reds',

    # Sequential (multi-hue)
    'viridis', 'plasma', 'inferno', 'magma', 'cividis',
    'YlGn', 'YlGnBu', 'GnBu', 'BuGn', 'PuBuGn', 'PuBu', 'BuPu', 'RdPu',

    # Diverging
    'RdBu', 'RdYlBu', 'RdYlGn', 'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy',
    'seismic', 'coolwarm', 'bwr',

    # Cyclic
    'twilight', 'twilight_shifted', 'hsv',

    # Qualitative
    'Paired', 'Set1', 'Set2', 'Set3', 'tab10', 'tab20',

    # Specialized
    'terrain', 'ocean', 'gist_earth', 'jet', 'rainbow'
]


def get_colormap(field_type: str = None, colormap_name: str = None) -> str:
    """
    Get appropriate colormap for a field type

    Args:
        field_type: Type of field (e.g., 'water_depth', 'velocity')
        colormap_name: Explicit colormap name (overrides field_type)

    Returns:
        Colormap name string
    """
    if colormap_name is not None:
        return colormap_name

    if field_type in FIELD_COLORMAPS:
        return FIELD_COLORMAPS[field_type]

    # Default to viridis
    return 'viridis'


def available_colormaps() -> List[str]:
    """
    Get list of available colormap names

    Returns:
        List of colormap names
    """
    return AVAILABLE_COLORMAPS.copy()


def colormap_recommendations() -> Dict[str, str]:
    """
    Get recommended colormaps for field types

    Returns:
        Dictionary mapping field types to colormap names
    """
    return FIELD_COLORMAPS.copy()


def print_colormap_guide() -> None:
    """Print colormap guide to console"""
    print("=" * 60)
    print("COLORMAP GUIDE FOR HYDROSIS-2D")
    print("=" * 60)
    print()
    print("Recommended Colormaps by Field Type:")
    print("-" * 60)

    for field_type, cmap in FIELD_COLORMAPS.items():
        print(f"  {field_type:<20} -> {cmap}")

    print()
    print("Available Colormaps:")
    print("-" * 60)

    categories = {
        'Sequential (Single Hue)': ['Blues', 'Greens', 'Greys', 'Oranges', 'Purples', 'Reds'],
        'Sequential (Multi-Hue)': ['viridis', 'plasma', 'inferno', 'magma', 'cividis'],
        'Diverging': ['RdBu', 'RdYlBu', 'seismic', 'coolwarm', 'bwr'],
        'Specialized': ['terrain', 'ocean', 'jet', 'rainbow']
    }

    for category, cmaps in categories.items():
        print(f"\n{category}:")
        for cmap in cmaps:
            print(f"  - {cmap}")

    print()
    print("=" * 60)
    print()
    print("Usage:")
    print("  from postprocessing.visualization import get_colormap")
    print("  cmap = get_colormap('water_depth')  # Returns 'Blues'")
    print("=" * 60)
