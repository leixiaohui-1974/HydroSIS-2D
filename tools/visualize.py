#!/usr/bin/env python3
"""
HydroSIS-2D Visualization Tool

This script provides visualization capabilities for HydroSIS-2D output files.
Supports both ASCII and VTK formats.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
import glob
import os

class HydroSISVisualizer:
    def __init__(self, output_dir='.'):
        self.output_dir = output_dir
        self.data_files = []
        self.times = []

    def load_ascii_data(self, filename):
        """Load ASCII output file"""
        try:
            data = np.loadtxt(filename)

            # Extract time from header if present
            with open(filename, 'r') as f:
                first_line = f.readline()
                if 'Time:' in first_line:
                    time_str = first_line.split('Time:')[1].split('s')[0].strip()
                    time = float(time_str)
                else:
                    time = 0.0

            # Columns: x, y, h, u, v, z
            return {
                'x': data[:, 0],
                'y': data[:, 1],
                'h': data[:, 2],
                'u': data[:, 3],
                'v': data[:, 4],
                'z': data[:, 5],
                'time': time
            }
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return None

    def find_data_files(self, pattern='output_*.dat'):
        """Find all data files matching pattern"""
        files = sorted(glob.glob(os.path.join(self.output_dir, pattern)))
        self.data_files = files
        print(f"Found {len(files)} data files")
        return files

    def plot_snapshot(self, filename, field='h', save=None):
        """Plot a single snapshot"""
        data = self.load_ascii_data(filename)
        if data is None:
            return

        # Reshape to 2D grid
        x_unique = np.unique(data['x'])
        y_unique = np.unique(data['y'])
        nx, ny = len(x_unique), len(y_unique)

        X = data['x'].reshape(ny, nx)
        Y = data['y'].reshape(ny, nx)

        # Select field to plot
        field_map = {
            'h': ('Water Depth', data['h'], 'm'),
            'u': ('X-Velocity', data['u'], 'm/s'),
            'v': ('Y-Velocity', data['v'], 'm/s'),
            'z': ('Bed Elevation', data['z'], 'm'),
            'vel': ('Velocity Magnitude', np.sqrt(data['u']**2 + data['v']**2), 'm/s'),
            'eta': ('Water Surface', data['h'] + data['z'], 'm')
        }

        if field not in field_map:
            print(f"Unknown field: {field}")
            return

        title, values, unit = field_map[field]
        Z = values.reshape(ny, nx)

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.pcolormesh(X, Y, Z, shading='auto', cmap='viridis')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'{title} at t = {data["time"]:.2f} s')
        ax.set_aspect('equal')

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(f'{title} ({unit})')

        if save:
            plt.savefig(save, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {save}")
        else:
            plt.show()

        plt.close()

    def plot_1d_profile(self, filename, direction='x', position=0.5, save=None):
        """Plot 1D profile along x or y direction"""
        data = self.load_ascii_data(filename)
        if data is None:
            return

        # Get unique coordinates
        x_unique = np.unique(data['x'])
        y_unique = np.unique(data['y'])

        if direction == 'x':
            # Extract data at y = position * max(y)
            y_target = position * np.max(y_unique)
            idx = np.argmin(np.abs(data['y'] - y_target))

            # Get all points with this y
            mask = np.abs(data['y'] - data['y'][idx]) < 1e-6
            x_profile = data['x'][mask]
            sort_idx = np.argsort(x_profile)

            x_profile = x_profile[sort_idx]
            h_profile = data['h'][mask][sort_idx]
            z_profile = data['z'][mask][sort_idx]
            eta_profile = h_profile + z_profile

            xlabel = 'X (m)'
            title_pos = f'y = {data["y"][idx]:.2f} m'

        else:  # direction == 'y'
            # Extract data at x = position * max(x)
            x_target = position * np.max(x_unique)
            idx = np.argmin(np.abs(data['x'] - x_target))

            mask = np.abs(data['x'] - data['x'][idx]) < 1e-6
            y_profile = data['y'][mask]
            sort_idx = np.argsort(y_profile)

            x_profile = y_profile[sort_idx]
            h_profile = data['h'][mask][sort_idx]
            z_profile = data['z'][mask][sort_idx]
            eta_profile = h_profile + z_profile

            xlabel = 'Y (m)'
            title_pos = f'x = {data["x"][idx]:.2f} m'

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.fill_between(x_profile, z_profile, eta_profile,
                        alpha=0.5, label='Water', color='blue')
        ax.fill_between(x_profile, 0, z_profile,
                        alpha=0.7, label='Bed', color='brown')
        ax.plot(x_profile, eta_profile, 'b-', linewidth=2, label='Water Surface')
        ax.plot(x_profile, z_profile, 'k-', linewidth=2, label='Bed')

        ax.set_xlabel(xlabel)
        ax.set_ylabel('Elevation (m)')
        ax.set_title(f'Profile at {title_pos}, t = {data["time"]:.2f} s')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save:
            plt.savefig(save, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {save}")
        else:
            plt.show()

        plt.close()

    def create_animation(self, field='h', output_file='animation.mp4', fps=10):
        """Create animation from time series"""
        if not self.data_files:
            self.find_data_files()

        if len(self.data_files) == 0:
            print("No data files found!")
            return

        print(f"Creating animation with {len(self.data_files)} frames...")

        # Load first frame to setup plot
        data0 = self.load_ascii_data(self.data_files[0])
        x_unique = np.unique(data0['x'])
        y_unique = np.unique(data0['y'])
        nx, ny = len(x_unique), len(y_unique)

        X = data0['x'].reshape(ny, nx)
        Y = data0['y'].reshape(ny, nx)

        fig, ax = plt.subplots(figsize=(10, 8))

        # Initial plot
        field_map = {
            'h': ('Water Depth', 'm'),
            'vel': ('Velocity Magnitude', 'm/s'),
            'eta': ('Water Surface', 'm')
        }
        title, unit = field_map.get(field, ('Field', ''))

        if field == 'h':
            Z0 = data0['h'].reshape(ny, nx)
        elif field == 'vel':
            Z0 = np.sqrt(data0['u']**2 + data0['v']**2).reshape(ny, nx)
        elif field == 'eta':
            Z0 = (data0['h'] + data0['z']).reshape(ny, nx)

        im = ax.pcolormesh(X, Y, Z0, shading='auto', cmap='viridis')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_aspect('equal')

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(f'{title} ({unit})')

        title_text = ax.set_title(f'{title} at t = 0.00 s')

        def update(frame):
            data = self.load_ascii_data(self.data_files[frame])

            if field == 'h':
                Z = data['h'].reshape(ny, nx)
            elif field == 'vel':
                Z = np.sqrt(data['u']**2 + data['v']**2).reshape(ny, nx)
            elif field == 'eta':
                Z = (data['h'] + data['z']).reshape(ny, nx)

            im.set_array(Z.ravel())
            im.set_clim(Z.min(), Z.max())
            title_text.set_text(f'{title} at t = {data["time"]:.2f} s')

            return [im, title_text]

        anim = FuncAnimation(fig, update, frames=len(self.data_files),
                           interval=1000/fps, blit=True)

        try:
            anim.save(output_file, writer='ffmpeg', fps=fps, dpi=150)
            print(f"Animation saved to {output_file}")
        except Exception as e:
            print(f"Error saving animation: {e}")
            print("Make sure ffmpeg is installed: sudo apt-get install ffmpeg")

        plt.close()

def main():
    parser = argparse.ArgumentParser(description='HydroSIS-2D Visualization Tool')
    parser.add_argument('--dir', type=str, default='.', help='Output directory')
    parser.add_argument('--file', type=str, help='Specific file to plot')
    parser.add_argument('--field', type=str, default='h',
                       choices=['h', 'u', 'v', 'z', 'vel', 'eta'],
                       help='Field to visualize')
    parser.add_argument('--profile', action='store_true', help='Plot 1D profile')
    parser.add_argument('--direction', type=str, default='x', choices=['x', 'y'],
                       help='Direction for profile plot')
    parser.add_argument('--animate', action='store_true', help='Create animation')
    parser.add_argument('--output', type=str, help='Output filename')
    parser.add_argument('--fps', type=int, default=10, help='Animation FPS')

    args = parser.parse_args()

    vis = HydroSISVisualizer(args.dir)

    if args.animate:
        output_file = args.output or 'animation.mp4'
        vis.create_animation(field=args.field, output_file=output_file, fps=args.fps)
    elif args.profile:
        file = args.file or vis.find_data_files()[-1]
        output = args.output
        vis.plot_1d_profile(file, direction=args.direction, save=output)
    else:
        file = args.file or vis.find_data_files()[-1]
        output = args.output
        vis.plot_snapshot(file, field=args.field, save=output)

if __name__ == '__main__':
    main()
