#!/usr/bin/env python3
"""
HydroSIS-2D Performance Visualization
Generates plots from benchmarking results
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

def load_results(csv_file):
    """Load benchmark results from CSV file"""
    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} benchmark results from {csv_file}")
        return df
    except Exception as e:
        print(f"Error loading {csv_file}: {e}")
        sys.exit(1)

def plot_scaling(df, output_dir):
    """Plot strong scaling efficiency"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Group by problem size
    for size in df['size_nx'].unique():
        subset = df[df['size_nx'] == size].sort_values('num_gpus')

        if len(subset) < 2:
            continue

        gpus = subset['num_gpus'].values
        speedup = subset['speedup'].values
        efficiency = subset['efficiency'].values

        # Speedup plot
        ax1.plot(gpus, speedup, 'o-', label=f'{size}×{size}', linewidth=2, markersize=8)

        # Efficiency plot
        ax2.plot(gpus, efficiency * 100, 's-', label=f'{size}×{size}', linewidth=2, markersize=8)

    # Ideal scaling reference
    max_gpus = df['num_gpus'].max()
    ideal_gpus = np.arange(1, max_gpus + 1)
    ax1.plot(ideal_gpus, ideal_gpus, 'k--', alpha=0.5, label='Ideal', linewidth=1.5)
    ax2.plot(ideal_gpus, np.ones_like(ideal_gpus) * 100, 'k--', alpha=0.5, label='Ideal', linewidth=1.5)

    # Speedup plot formatting
    ax1.set_xlabel('Number of GPUs', fontsize=12)
    ax1.set_ylabel('Speedup', fontsize=12)
    ax1.set_title('Strong Scaling: Speedup', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0.5)
    ax1.set_ylim(bottom=0)

    # Efficiency plot formatting
    ax2.set_xlabel('Number of GPUs', fontsize=12)
    ax2.set_ylabel('Parallel Efficiency (%)', fontsize=12)
    ax2.set_title('Strong Scaling: Efficiency', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0.5)
    ax2.set_ylim([0, 105])

    plt.tight_layout()
    output_file = output_dir / 'scaling_efficiency.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved scaling plot: {output_file}")
    plt.close()

def plot_performance_vs_size(df, output_dir):
    """Plot performance vs problem size"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by GPU count
    for ngpus in sorted(df['num_gpus'].unique()):
        subset = df[df['num_gpus'] == ngpus].sort_values('num_cells')

        cells = subset['num_cells'].values / 1e6  # Convert to millions
        perf = subset['cell_updates_per_sec'].values / 1e6  # Convert to M cells/s

        ax.plot(cells, perf, 'o-', label=f'{ngpus} GPU{"s" if ngpus > 1 else ""}',
                linewidth=2, markersize=8)

    ax.set_xlabel('Problem Size (M cells)', fontsize=12)
    ax.set_ylabel('Performance (M cells/s)', fontsize=12)
    ax.set_title('Performance vs Problem Size', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.tight_layout()
    output_file = output_dir / 'performance_vs_size.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved performance plot: {output_file}")
    plt.close()

def plot_wall_time_breakdown(df, output_dir):
    """Plot wall time for different configurations"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create labels
    labels = []
    times = []
    colors = []

    color_map = plt.cm.viridis(np.linspace(0, 1, len(df['num_gpus'].unique())))
    gpu_colors = {ngpus: color_map[i] for i, ngpus in enumerate(sorted(df['num_gpus'].unique()))}

    for idx, row in df.iterrows():
        label = f"{row['case']}\n{row['size_nx']}×{row['size_ny']}\n{row['num_gpus']}GPU"
        labels.append(label)
        times.append(row['wall_time'])
        colors.append(gpu_colors[row['num_gpus']])

    bars = ax.bar(range(len(labels)), times, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Wall Time (seconds)', fontsize=12)
    ax.set_title('Wall Time Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s',
                ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    output_file = output_dir / 'wall_time_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved wall time plot: {output_file}")
    plt.close()

def plot_memory_usage(df, output_dir):
    """Plot memory usage vs problem size"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Memory per GPU
    for ngpus in sorted(df['num_gpus'].unique()):
        subset = df[df['num_gpus'] == ngpus].sort_values('num_cells')

        cells = subset['num_cells'].values / 1e6
        memory_per_gpu = subset['memory_mb'].values / ngpus

        ax.plot(cells, memory_per_gpu, 'o-', label=f'{ngpus} GPU{"s" if ngpus > 1 else ""}',
                linewidth=2, markersize=8)

    ax.set_xlabel('Problem Size (M cells)', fontsize=12)
    ax.set_ylabel('Memory per GPU (MB)', fontsize=12)
    ax.set_title('Memory Usage per GPU', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.tight_layout()
    output_file = output_dir / 'memory_usage.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved memory usage plot: {output_file}")
    plt.close()

def generate_summary_table(df, output_dir):
    """Generate summary statistics table"""
    summary_file = output_dir / 'benchmark_summary.txt'

    with open(summary_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("HydroSIS-2D Performance Benchmark Summary\n")
        f.write("=" * 80 + "\n\n")

        # Overall statistics
        f.write("Overall Statistics:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total benchmark runs: {len(df)}\n")
        f.write(f"Problem sizes tested: {sorted(df['size_nx'].unique())}\n")
        f.write(f"GPU configurations: {sorted(df['num_gpus'].unique())}\n")
        f.write(f"\n")

        # Performance by configuration
        f.write("Performance by Configuration:\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Size':<12} {'GPUs':<6} {'Cells/s':<15} {'Wall Time':<12} {'Efficiency':<12}\n")
        f.write("-" * 40 + "\n")

        for idx, row in df.iterrows():
            size_str = f"{row['size_nx']}×{row['size_ny']}"
            perf_str = f"{row['cell_updates_per_sec']/1e6:.2f} M"
            time_str = f"{row['wall_time']:.2f} s"
            eff_str = f"{row['efficiency']*100:.1f}%"

            f.write(f"{size_str:<12} {row['num_gpus']:<6} {perf_str:<15} {time_str:<12} {eff_str:<12}\n")

        f.write("\n")

        # Best performance
        best_perf = df.loc[df['cell_updates_per_sec'].idxmax()]
        f.write("Best Performance:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Configuration: {best_perf['size_nx']}×{best_perf['size_ny']} on {best_perf['num_gpus']} GPU(s)\n")
        f.write(f"Performance: {best_perf['cell_updates_per_sec']/1e6:.2f} M cells/s\n")
        f.write(f"Wall time: {best_perf['wall_time']:.2f} s\n")
        f.write("\n")

        # Scaling efficiency
        if len(df['num_gpus'].unique()) > 1:
            f.write("Multi-GPU Scaling:\n")
            f.write("-" * 40 + "\n")
            for size in sorted(df['size_nx'].unique()):
                subset = df[df['size_nx'] == size].sort_values('num_gpus')
                if len(subset) > 1:
                    max_gpus = subset.iloc[-1]
                    f.write(f"{size}×{size}: {max_gpus['efficiency']*100:.1f}% efficiency on {max_gpus['num_gpus']} GPUs\n")

        f.write("\n")
        f.write("=" * 80 + "\n")

    print(f"Saved summary table: {summary_file}")

    # Print to console as well
    with open(summary_file, 'r') as f:
        print("\n" + f.read())

def main():
    parser = argparse.ArgumentParser(description='HydroSIS-2D Performance Visualization')
    parser.add_argument('--input', '-i', required=True, help='Input CSV file from benchmark')
    parser.add_argument('--output', '-o', default='results/performance/plots', help='Output directory for plots')
    parser.add_argument('--format', '-f', default='png', choices=['png', 'pdf', 'svg'], help='Output format')

    args = parser.parse_args()

    # Load data
    df = load_results(args.input)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating plots in {output_dir}...")

    # Generate all plots
    if len(df['num_gpus'].unique()) > 1:
        plot_scaling(df, output_dir)

    plot_performance_vs_size(df, output_dir)
    plot_wall_time_breakdown(df, output_dir)
    plot_memory_usage(df, output_dir)

    # Generate summary
    generate_summary_table(df, output_dir)

    print(f"\n✓ All plots generated successfully!")
    print(f"  Output directory: {output_dir}")

if __name__ == '__main__':
    main()
