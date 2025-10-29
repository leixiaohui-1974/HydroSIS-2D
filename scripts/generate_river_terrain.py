#!/usr/bin/env python3
"""
Generate river channel with floodplain terrain
River flows in x-direction, channel is parabolic in cross-section
"""

import numpy as np

def generate_river_terrain(nx, ny, dx, dy, output_file):
    """
    Generate river channel with floodplain

    Parameters:
    - nx, ny: Grid dimensions
    - dx, dy: Cell spacing (m)
    - output_file: Output ASCII Grid file path
    """

    # Domain dimensions
    Lx = nx * dx
    Ly = ny * dy

    # River parameters
    channel_center_y = Ly / 2.0
    channel_width = 100.0       # Channel width at bankfull (m)
    channel_depth = 5.0         # Channel depth (m)
    floodplain_width = Ly       # Full domain width
    floodplain_slope = 0.01     # Floodplain cross-slope (1%)
    streamwise_slope = 0.0005   # Longitudinal slope (0.05%)

    # Elevation at upstream end
    z_upstream = 50.0

    # Create elevation grid
    elevation = np.zeros((ny, nx))

    for j in range(ny):
        for i in range(nx):
            x = i * dx
            y = j * dy

            # Base elevation: slopes down in x-direction (streamwise)
            z_base = z_upstream - streamwise_slope * x

            # Cross-section profile
            dy_center = abs(y - channel_center_y)

            if dy_center < channel_width / 2.0:
                # In channel: parabolic cross-section
                normalized_y = 2.0 * dy_center / channel_width  # 0 at center, 1 at edge
                z_channel = -channel_depth * (1.0 - normalized_y * normalized_y)
            else:
                # On floodplain: gentle slope away from channel
                z_channel = floodplain_slope * (dy_center - channel_width / 2.0)

            elevation[j, i] = z_base + z_channel

    # Add some meandering to make it more realistic
    # Sinusoidal channel centerline shift
    meander_amplitude = 50.0  # meters
    meander_wavelength = 1000.0  # meters

    for j in range(ny):
        for i in range(nx):
            x = i * dx
            y = j * dy

            # Calculate meandering centerline
            y_shift = meander_amplitude * np.sin(2.0 * np.pi * x / meander_wavelength)
            channel_center_meander = channel_center_y + y_shift

            # Recalculate elevation with meandering
            z_base = z_upstream - streamwise_slope * x
            dy_center = abs(y - channel_center_meander)

            if dy_center < channel_width / 2.0:
                normalized_y = 2.0 * dy_center / channel_width
                z_channel = -channel_depth * (1.0 - normalized_y * normalized_y)
            else:
                z_channel = floodplain_slope * (dy_center - channel_width / 2.0)

            elevation[j, i] = z_base + z_channel

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
    print(f"River terrain generated: {output_file}")
    print(f"Grid dimensions: {nx} × {ny}")
    print(f"Cell size: {dx} m")
    print(f"Domain size: {Lx} m × {Ly} m")
    print(f"Elevation range: {elevation.min():.2f} - {elevation.max():.2f} m")
    print(f"Channel width: {channel_width} m")
    print(f"Channel depth: {channel_depth} m")
    print(f"Streamwise slope: {streamwise_slope * 100:.3f}%")
    print(f"Meander amplitude: {meander_amplitude} m")
    print(f"Meander wavelength: {meander_wavelength} m")

if __name__ == "__main__":
    # River terrain for flooding case
    # 3km × 1km domain, 5m resolution
    nx = 600
    ny = 200
    dx = 5.0
    dy = 5.0

    output_file = "examples/applications/river_terrain.asc"

    generate_river_terrain(nx, ny, dx, dy, output_file)
    print("\nRiver terrain created successfully!")
