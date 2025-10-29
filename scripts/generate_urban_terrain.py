#!/usr/bin/env python3
"""
Generate urban terrain with buildings and streets
Buildings are represented as elevated areas that water cannot enter
"""

import numpy as np

def generate_urban_terrain(nx, ny, dx, dy, output_file):
    """
    Generate urban terrain with street grid and buildings

    Parameters:
    - nx, ny: Grid dimensions
    - dx, dy: Cell spacing (m)
    - output_file: Output ASCII Grid file path
    """

    # Domain dimensions
    Lx = nx * dx
    Ly = ny * dy

    # Urban parameters
    base_elevation = 100.0  # Base street elevation (m)
    building_height = 10.0  # Building height above street (m)
    street_width = 100.0    # Street width (m) = 20 cells
    block_size = 200.0      # City block size (m) = 40 cells
    slope = 0.001           # Very gentle slope for drainage (0.1%)

    # Create elevation grid
    elevation = np.zeros((ny, nx))

    for j in range(ny):
        for i in range(nx):
            x = i * dx
            y = j * dy

            # Base elevation with slight slope for drainage
            z_base = base_elevation + slope * (x + y) / 2.0

            # Determine if location is street or building
            x_mod = x % block_size
            y_mod = y % block_size

            # Streets run in grid pattern
            # Horizontal streets
            is_street_x = (x_mod < street_width/2.0) or (x_mod > block_size - street_width/2.0)
            # Vertical streets
            is_street_y = (y_mod < street_width/2.0) or (y_mod > block_size - street_width/2.0)

            if is_street_x or is_street_y:
                # Street - base elevation
                elevation[j, i] = z_base
            else:
                # Building - elevated above street
                elevation[j, i] = z_base + building_height

    # Write ASCII Grid format
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
    print(f"Urban terrain generated: {output_file}")
    print(f"Grid dimensions: {nx} × {ny}")
    print(f"Cell size: {dx} m")
    print(f"Domain size: {Lx} m × {Ly} m")
    print(f"Elevation range: {elevation.min():.2f} - {elevation.max():.2f} m")
    print(f"Building height: {building_height} m")
    print(f"Street width: {street_width} m")
    print(f"Block size: {block_size} m")

    # Calculate street area percentage
    street_cells = np.sum(elevation < (base_elevation + building_height / 2.0))
    total_cells = nx * ny
    street_percentage = 100.0 * street_cells / total_cells
    print(f"Street area: {street_percentage:.1f}%")

if __name__ == "__main__":
    # Urban terrain for flooding case
    # 1km × 1km domain, 5m resolution
    nx = 200
    ny = 200
    dx = 5.0
    dy = 5.0

    output_file = "examples/applications/urban_terrain.asc"

    generate_urban_terrain(nx, ny, dx, dy, output_file)
    print("\nUrban terrain created successfully!")
