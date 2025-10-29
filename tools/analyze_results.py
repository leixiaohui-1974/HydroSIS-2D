#!/usr/bin/env python3
"""
HydroSIS-2D Results Analysis Tool

Analyze simulation results: mass conservation, statistics, convergence, etc.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import glob
import os

class ResultsAnalyzer:
    def __init__(self, output_dir='.'):
        self.output_dir = output_dir

    def load_data(self, filename):
        """Load ASCII data file"""
        try:
            data = np.loadtxt(filename)

            # Parse time from header
            with open(filename, 'r') as f:
                first_line = f.readline()
                if 'Time:' in first_line:
                    time = float(first_line.split('Time:')[1].split('s')[0].strip())
                else:
                    time = 0.0

            return {
                'x': data[:, 0],
                'y': data[:, 1],
                'h': data[:, 2],
                'u': data[:, 3],
                'v': data[:, 4],
                'z': data[:, 5],
                'time': time
            }
        except:
            return None

    def compute_mass(self, data):
        """Compute total water mass"""
        # Approximate cell size
        x_unique = np.unique(data['x'])
        y_unique = np.unique(data['y'])

        if len(x_unique) > 1 and len(y_unique) > 1:
            dx = x_unique[1] - x_unique[0]
            dy = y_unique[1] - y_unique[0]
        else:
            dx = dy = 1.0

        mass = np.sum(data['h']) * dx * dy
        return mass

    def analyze_mass_conservation(self):
        """Analyze mass conservation over time"""
        files = sorted(glob.glob(os.path.join(self.output_dir, 'output_*.dat')))

        if len(files) == 0:
            print("No data files found!")
            return

        times = []
        masses = []

        print("Analyzing mass conservation...")
        for i, file in enumerate(files):
            data = self.load_data(file)
            if data is None:
                continue

            mass = self.compute_mass(data)
            times.append(data['time'])
            masses.append(mass)

            if i % 10 == 0:
                print(f"  Step {i}: t = {data['time']:.2f} s, Mass = {mass:.6f} m³")

        times = np.array(times)
        masses = np.array(masses)

        # Compute conservation error
        initial_mass = masses[0]
        rel_error = (masses - initial_mass) / initial_mass

        print(f"\nMass Conservation Analysis:")
        print(f"  Initial mass: {initial_mass:.6f} m³")
        print(f"  Final mass:   {masses[-1]:.6f} m³")
        print(f"  Absolute error: {abs(masses[-1] - initial_mass):.2e} m³")
        print(f"  Relative error: {abs(rel_error[-1]):.2e} ({abs(rel_error[-1])*100:.4f}%)")
        print(f"  Max error:      {abs(rel_error).max():.2e} ({abs(rel_error).max()*100:.4f}%)")

        # Plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        ax1.plot(times, masses, 'b-', linewidth=2)
        ax1.axhline(initial_mass, color='r', linestyle='--', label='Initial')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Total Mass (m³)')
        ax1.set_title('Mass Conservation')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(times, rel_error * 100, 'r-', linewidth=2)
        ax2.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Relative Error (%)')
        ax2.set_title('Mass Conservation Error')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('mass_conservation.png', dpi=150)
        print("\nPlot saved to mass_conservation.png")
        plt.close()

    def compute_statistics(self, filename):
        """Compute field statistics"""
        data = self.load_data(filename)
        if data is None:
            print(f"Error loading {filename}")
            return

        vel_mag = np.sqrt(data['u']**2 + data['v']**2)

        print(f"\n=== Statistics at t = {data['time']:.2f} s ===")
        print(f"\nWater Depth (h):")
        print(f"  Min:    {data['h'].min():.6f} m")
        print(f"  Max:    {data['h'].max():.6f} m")
        print(f"  Mean:   {data['h'].mean():.6f} m")
        print(f"  Std:    {data['h'].std():.6f} m")

        print(f"\nVelocity Magnitude:")
        print(f"  Min:    {vel_mag.min():.6f} m/s")
        print(f"  Max:    {vel_mag.max():.6f} m/s")
        print(f"  Mean:   {vel_mag.mean():.6f} m/s")
        print(f"  Std:    {vel_mag.std():.6f} m/s")

        # Froude number
        Fr = vel_mag / (np.sqrt(9.81 * data['h']) + 1e-10)
        print(f"\nFroude Number:")
        print(f"  Min:    {Fr.min():.6f}")
        print(f"  Max:    {Fr.max():.6f}")
        print(f"  Mean:   {Fr.mean():.6f}")

        # Wet/dry
        wet_cells = np.sum(data['h'] > 1e-4)
        dry_cells = len(data['h']) - wet_cells
        print(f"\nWet/Dry:")
        print(f"  Wet cells:  {wet_cells} ({wet_cells/len(data['h'])*100:.1f}%)")
        print(f"  Dry cells:  {dry_cells} ({dry_cells/len(data['h'])*100:.1f}%)")

    def compare_analytical(self, filename, test_case='dam_break'):
        """Compare with analytical solution"""
        data = self.load_data(filename)
        if data is None:
            return

        if test_case == 'dam_break':
            # 1D Dam Break (Ritter's solution)
            x_dam = (data['x'].max() + data['x'].min()) / 2.0
            h_L = 10.0
            h_R = 1.0
            g = 9.81
            t = data['time']

            if t < 1e-6:
                print("Warning: t too small for comparison")
                return

            # Extract centerline
            y_center = (data['y'].max() + data['y'].min()) / 2.0
            # Find all unique y values
            y_unique = np.unique(data['y'])
            # Find the y value closest to center
            y_closest_idx = np.argmin(np.abs(y_unique - y_center))
            y_target = y_unique[y_closest_idx]

            # Use larger tolerance or exact match
            dy = y_unique[1] - y_unique[0] if len(y_unique) > 1 else 1.0
            mask = np.abs(data['y'] - y_target) < dy * 0.1

            x_profile = data['x'][mask]
            h_numerical = data['h'][mask]

            if len(x_profile) == 0:
                print("Error: No data points found on centerline")
                return

            # Sort by x
            sort_idx = np.argsort(x_profile)
            x_profile = x_profile[sort_idx]
            h_numerical = h_numerical[sort_idx]

            # Analytical solution
            c_L = np.sqrt(g * h_L)
            h_analytical = np.zeros_like(x_profile)

            for i, x in enumerate(x_profile):
                xi = (x - x_dam) / t

                if xi <= -c_L:
                    h_analytical[i] = h_L
                elif xi >= 2.0 * c_L:
                    h_analytical[i] = h_R
                else:
                    h_analytical[i] = (4.0 / (9.0 * g)) * (c_L - 0.5 * xi)**2

            # Compute errors
            L1 = np.mean(np.abs(h_numerical - h_analytical))
            L2 = np.sqrt(np.mean((h_numerical - h_analytical)**2))
            Linf = np.max(np.abs(h_numerical - h_analytical))

            print(f"\n=== Comparison with Analytical Solution ===")
            print(f"Time: {t:.2f} s")
            print(f"L1 error:   {L1:.6f}")
            print(f"L2 error:   {L2:.6f}")
            print(f"Linf error: {Linf:.6f}")

            # Plot
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(x_profile, h_numerical, 'b-', linewidth=2, label='Numerical')
            ax.plot(x_profile, h_analytical, 'r--', linewidth=2, label='Analytical')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Water Depth (m)')
            ax.set_title(f'1D Dam Break Comparison at t = {t:.2f} s')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.savefig('analytical_comparison.png', dpi=150)
            print("\nPlot saved to analytical_comparison.png")
            plt.close()

def main():
    parser = argparse.ArgumentParser(description='HydroSIS-2D Results Analysis')
    parser.add_argument('--dir', type=str, default='.', help='Output directory')
    parser.add_argument('--mass', action='store_true', help='Analyze mass conservation')
    parser.add_argument('--stats', action='store_true', help='Compute statistics')
    parser.add_argument('--file', type=str, help='File for statistics')
    parser.add_argument('--analytical', action='store_true', help='Compare with analytical')
    parser.add_argument('--test', type=str, default='dam_break', help='Test case for comparison')

    args = parser.parse_args()

    analyzer = ResultsAnalyzer(args.dir)

    if args.mass:
        analyzer.analyze_mass_conservation()

    if args.stats:
        file = args.file or sorted(glob.glob(os.path.join(args.dir, 'output_*.dat')))[-1]
        analyzer.compute_statistics(file)

    if args.analytical:
        file = args.file or sorted(glob.glob(os.path.join(args.dir, 'output_*.dat')))[-1]
        analyzer.compare_analytical(file, test_case=args.test)

if __name__ == '__main__':
    main()
