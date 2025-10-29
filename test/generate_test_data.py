#!/usr/bin/env python3
"""
生成标准测试案例的数据，用于验证分析工具
Generate test data for standard benchmark cases to verify analysis tools
"""

import numpy as np
import os

def generate_dam_break_1d(nx=200, ny=50, t=0.0, output_dir='.'):
    """生成1D溃坝问题的精确解 / Generate exact solution for 1D dam break"""

    # 区域设置 / Domain setup
    x_min, x_max = 0.0, 200.0
    y_min, y_max = 0.0, 50.0

    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(x, y)

    # 初始条件 / Initial conditions
    x_dam = 100.0
    h_L = 10.0  # 左侧水深
    h_R = 1.0   # 右侧水深
    g = 9.81

    # Ritter解析解 / Ritter's analytical solution
    if t < 1e-6:
        # 初始状态
        H = np.where(X < x_dam, h_L, h_R)
        U = np.zeros_like(X)
        V = np.zeros_like(X)
    else:
        c_L = np.sqrt(g * h_L)
        H = np.zeros_like(X)
        U = np.zeros_like(X)
        V = np.zeros_like(X)

        for i in range(ny):
            for j in range(nx):
                xi = (X[i, j] - x_dam) / t

                if xi <= -c_L:
                    H[i, j] = h_L
                    U[i, j] = 0.0
                elif xi >= 2.0 * c_L:
                    H[i, j] = h_R
                    U[i, j] = 0.0
                else:
                    H[i, j] = (4.0 / (9.0 * g)) * (c_L - 0.5 * xi)**2
                    U[i, j] = (2.0 / 3.0) * (xi + c_L)

    Z = np.zeros_like(X)  # 平底

    # 输出文件
    filename = os.path.join(output_dir, f'output_{int(t*10):05d}.dat')
    with open(filename, 'w') as f:
        f.write(f'# Time: {t:.6f} s\n')
        f.write(f'# Columns: x y h u v z\n')
        for i in range(ny):
            for j in range(nx):
                f.write(f'{X[i,j]:.6f} {Y[i,j]:.6f} {H[i,j]:.6f} ')
                f.write(f'{U[i,j]:.6f} {V[i,j]:.6f} {Z[i,j]:.6f}\n')

    print(f'Generated: {filename}')
    return filename

def generate_lake_at_rest(nx=100, ny=100, t=0.0, output_dir='.'):
    """生成静水平衡测试 / Generate lake at rest test"""

    x = np.linspace(0, 100, nx)
    y = np.linspace(0, 100, ny)
    X, Y = np.meshgrid(x, y)

    # 抛物线形河床
    Z = 0.2 * (X - 50)**2 / 100.0

    # 水位恒定
    eta = 10.0
    H = eta - Z
    H = np.maximum(H, 0.0)  # 确保非负

    U = np.zeros_like(X)
    V = np.zeros_like(X)

    # 添加小扰动测试稳定性
    if t > 0:
        perturbation = 0.001 * np.sin(2*np.pi*X/100.0) * np.sin(2*np.pi*Y/100.0)
        H += perturbation
        H = np.maximum(H, 0.0)

    filename = os.path.join(output_dir, f'output_{int(t*10):05d}.dat')
    with open(filename, 'w') as f:
        f.write(f'# Time: {t:.6f} s\n')
        f.write(f'# Columns: x y h u v z\n')
        for i in range(ny):
            for j in range(nx):
                f.write(f'{X[i,j]:.6f} {Y[i,j]:.6f} {H[i,j]:.6f} ')
                f.write(f'{U[i,j]:.6f} {V[i,j]:.6f} {Z[i,j]:.6f}\n')

    print(f'Generated: {filename}')
    return filename

def generate_circular_dam_break(nx=100, ny=100, t=0.0, output_dir='.'):
    """生成2D圆形溃坝 / Generate 2D circular dam break"""

    x = np.linspace(0, 100, nx)
    y = np.linspace(0, 100, ny)
    X, Y = np.meshgrid(x, y)

    # 中心圆形高水位
    x_c, y_c = 50.0, 50.0
    r_dam = 20.0

    R = np.sqrt((X - x_c)**2 + (Y - y_c)**2)

    if t < 1e-6:
        H = np.where(R <= r_dam, 10.0, 1.0)
        U = np.zeros_like(X)
        V = np.zeros_like(X)
    else:
        # 简化的径向流动解
        g = 9.81
        h_inner = 10.0
        c = np.sqrt(g * h_inner)
        r_wave = c * t

        H = np.zeros_like(X)
        U = np.zeros_like(X)
        V = np.zeros_like(X)

        # 使用简化模型
        for i in range(ny):
            for j in range(nx):
                r = R[i, j]
                if r < r_dam:
                    H[i, j] = h_inner - (h_inner - 1.0) * (r / r_dam) * (t / 2.0)
                elif r < r_dam + r_wave:
                    ratio = (r - r_dam) / r_wave
                    H[i, j] = h_inner * (1.0 - ratio) + 1.0 * ratio
                    if r > 0:
                        vel = c * (1.0 - ratio)
                        U[i, j] = vel * (X[i, j] - x_c) / r
                        V[i, j] = vel * (Y[i, j] - y_c) / r
                else:
                    H[i, j] = 1.0

        H = np.maximum(H, 0.1)

    Z = np.zeros_like(X)

    filename = os.path.join(output_dir, f'output_{int(t*10):05d}.dat')
    with open(filename, 'w') as f:
        f.write(f'# Time: {t:.6f} s\n')
        f.write(f'# Columns: x y h u v z\n')
        for i in range(ny):
            for j in range(nx):
                f.write(f'{X[i,j]:.6f} {Y[i,j]:.6f} {H[i,j]:.6f} ')
                f.write(f'{U[i,j]:.6f} {V[i,j]:.6f} {Z[i,j]:.6f}\n')

    print(f'Generated: {filename}')
    return filename

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='生成测试数据')
    parser.add_argument('--case', type=str, default='dam_break',
                       choices=['dam_break', 'lake_at_rest', 'circular'],
                       help='测试案例')
    parser.add_argument('--dir', type=str, default='test_output',
                       help='输出目录')
    parser.add_argument('--times', type=str, default='0,1,2,3,4,5',
                       help='时间点列表（逗号分隔）')
    parser.add_argument('--nx', type=int, default=200, help='X方向网格数')
    parser.add_argument('--ny', type=int, default=50, help='Y方向网格数')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.dir, exist_ok=True)

    # 解析时间点
    times = [float(t) for t in args.times.split(',')]

    print(f'\n生成测试案例: {args.case}')
    print(f'输出目录: {args.dir}')
    print(f'网格: {args.nx} x {args.ny}')
    print(f'时间点: {times}\n')

    # 生成数据
    for t in times:
        if args.case == 'dam_break':
            generate_dam_break_1d(args.nx, args.ny, t, args.dir)
        elif args.case == 'lake_at_rest':
            generate_lake_at_rest(args.nx, args.ny, t, args.dir)
        elif args.case == 'circular':
            generate_circular_dam_break(args.nx, args.ny, t, args.dir)

    print(f'\n✓ 完成! 生成了 {len(times)} 个时间步的数据')
    print(f'\n验证命令:')
    print(f'  python3 tools/analyze_results.py --dir {args.dir} --mass')
    print(f'  python3 tools/analyze_results.py --dir {args.dir} --stats')
    print(f'  python3 tools/visualize.py --dir {args.dir} --animate --output {args.case}.mp4')
