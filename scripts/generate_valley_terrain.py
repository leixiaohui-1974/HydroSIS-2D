#!/usr/bin/env python3
"""
Generate V-shaped valley terrain for dam break simulation
Valley slopes downward in x-direction and upward toward sides in y-direction
"""

import numpy as np
import sys

def generate_valley_terrain(nx, ny, dx, dy, output_file):
    """
    Generate V-shaped valley terrain

    Parameters:
    - nx, ny: Grid dimensions
    - dx, dy: Cell spacing (m)
    - output_file: Output ASCII Grid file path
    """

    # Domain dimensions
    Lx = nx * dx
    Ly = ny * dy

    # Valley parameters
    valley_center_y = Ly / 2.0
    valley_width = Ly * 0.6  # Valley bottom width
    slope_x = 0.01  # 1% downward slope in x-direction
    slope_side = 0.15  # 15% side slope

    # Elevation at upstream end (where dam is)
    z_upstream = 50.0  # Dam crest at 50m elevation

    # Create elevation grid
    elevation = np.zeros((ny, nx))

    for j in range(ny):
        for i in range(nx):
            x = i * dx
            y = j * dy

            # Base elevation: slopes down in x-direction
            z_base = z_upstream - slope_x * x

            # V-shape in y-direction
            dy_center = abs(y - valley_center_y)

            if dy_center < valley_width / 2.0:
                # Valley bottom (flat)
                z_side = 0.0
            else:
                # Side slopes
                z_side = slope_side * (dy_center - valley_width / 2.0)

            elevation[j, i] = z_base + z_side

    # Write ASCII Grid format (top to bottom)
    with open(output_file, 'w') as f:
        f.write(f"ncols         {nx}\n")
        f.write(f"nrows         {ny}\n")
        f.write(f"xllcorner     0.0\n")
        f.write(f"yllcorner     0.0\n")
        f.write(f"cellsize      {dx}\n")
        f.write(f"NODATA_value  -9999\n")

        # Write elevation data (top to bottom)
        for j in range(ny-1, -1, -1):
            row_data = " ".join([f"{elevation[j, i]:.2f}" for i in range(nx)])
            f.write(row_data + "\n")

    # Print statistics
    print(f"Valley terrain generated: {output_file}")
    print(f"Grid dimensions: {nx} × {ny}")
    print(f"Cell size: {dx} m")
    print(f"Domain size: {Lx} m × {Ly} m")
    print(f"Elevation range: {elevation.min():.2f} - {elevation.max():.2f} m")
    print(f"Valley centerline: y = {valley_center_y} m")
    print(f"Valley width: {valley_width} m")

if __name__ == "__main__":
    # Valley terrain for dam break case
    # 2km × 1km domain, 5m resolution
    nx = 400
    ny = 200
    dx = 5.0
    dy = 5.0

    output_file = "examples/applications/valley_terrain.asc"

    generate_valley_terrain(nx, ny, dx, dy, output_file)
    print("\nValley terrain created successfully!")
