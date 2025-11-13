#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HydroSIS-2D 快速性能测试
对比NumPy vs Numba JIT加速性能 (简化版，更快完成)

作者: HydroSIS-2D Team
版本: 1.0
"""

import sys
import os
import time
import numpy as np

# 设置控制台编码
if sys.platform == 'win32':
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleOutputCP(65001)
        kernel32.SetConsoleCP(65001)
    except:
        pass
    
    try:
        import io
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True
        )
    except:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing.mesh_generation import MeshGenerator, DomainParams
from solver import ShallowWaterSolver, SolverConfig

def run_test(grid_size, use_numba, duration=2.0):
    """运行快速测试"""
    nx, ny = grid_size
    
    # 创建网格
    domain = DomainParams(xmin=0.0, xmax=200.0, ymin=0.0, ymax=20.0)
    mesh_gen = MeshGenerator(domain)
    mesh = mesh_gen.generate_uniform_mesh(nx=nx, ny=ny)
    
    # 地形
    terrain = np.zeros((mesh.nx, mesh.ny))
    
    # 配置
    config = SolverConfig(
        t_end=duration,
        cfl=0.5,
        output_interval=999.0,  # 不输出
        output_dir=None,
        print_progress=False,
        use_numba=use_numba
    )
    
    solver = ShallowWaterSolver(mesh, terrain, config)
    
    # 初始条件
    h0 = np.zeros((mesh.nx, mesh.ny))
    u0 = np.zeros((mesh.nx, mesh.ny))
    v0 = np.zeros((mesh.nx, mesh.ny))
    
    X = mesh.x
    h0[X < 100.0] = 10.0
    h0[X >= 100.0] = 1.0
    
    solver.set_initial_conditions(h0, u0, v0)
    
    # 计时
    start = time.perf_counter()
    result = solver.solve()
    elapsed = time.perf_counter() - start
    
    return {
        'elapsed': elapsed,
        'steps': result['steps'],
        'sim_time': result['simulated_time']
    }

def main():
    print("\n" + "=" * 70)
    print("  HydroSIS-2D 快速性能测试")
    print("=" * 70)
    
    # 检查Numba
    try:
        import numba
        print(f"\nNumba版本: {numba.__version__}")
        numba_available = True
    except ImportError:
        print("\nNumba未安装")
        numba_available = False
        return
    
    # 测试配置 (更小的网格，更短的时间)
    tests = [
        (50, 50, 2.0, "小规模"),
        (100, 50, 2.0, "中等规模"),
        (150, 50, 2.0, "大规模"),
    ]
    
    print("\n" + "=" * 70)
    print("开始测试 (每个测试模拟2秒物理时间)")
    print("=" * 70)
    
    results = []
    
    for nx, ny, duration, desc in tests:
        print(f"\n{desc}: {nx}x{ny} 网格")
        print("-" * 70)
        
        # NumPy测试
        print(f"  NumPy版本运行中...", end='', flush=True)
        try:
            r_numpy = run_test((nx, ny), False, duration)
            print(f" 完成 ({r_numpy['elapsed']:.2f}s, {r_numpy['steps']}步)")
            results.append(('NumPy', desc, nx, ny, r_numpy))
        except Exception as e:
            print(f" 错误: {e}")
            r_numpy = None
        
        # Numba测试  
        print(f"  Numba版本运行中...", end='', flush=True)
        try:
            r_numba = run_test((nx, ny), True, duration)
            print(f" 完成 ({r_numba['elapsed']:.2f}s, {r_numba['steps']}步)")
            results.append(('Numba', desc, nx, ny, r_numba))
            
            if r_numpy:
                speedup = r_numpy['elapsed'] / r_numba['elapsed']
                print(f"  加速比: {speedup:.2f}x")
        except Exception as e:
            print(f" 错误: {e}")
    
    # 总结
    print("\n" + "=" * 70)
    print("  测试结果总结")
    print("=" * 70)
    
    print(f"\n{'模式':<10} {'规模':<12} {'网格':<12} {'时间(s)':<10} {'步数':<8}")
    print("-" * 70)
    
    for mode, desc, nx, ny, result in results:
        grid = f"{nx}x{ny}"
        print(f"{mode:<10} {desc:<12} {grid:<12} {result['elapsed']:<10.2f} {result['steps']:<8}")
    
    # 计算加速比
    numpy_dict = {(desc, nx, ny): r for m, desc, nx, ny, r in results if m == 'NumPy'}
    numba_dict = {(desc, nx, ny): r for m, desc, nx, ny, r in results if m == 'Numba'}
    
    if numpy_dict and numba_dict:
        print("\n加速比总结:")
        print("-" * 70)
        
        speedups = []
        for key in numpy_dict:
            if key in numba_dict:
                desc, nx, ny = key
                speedup = numpy_dict[key]['elapsed'] / numba_dict[key]['elapsed']
                speedups.append(speedup)
                print(f"  {desc:<12} ({nx}x{ny}):  {speedup:.2f}x")
        
        if speedups:
            avg = sum(speedups) / len(speedups)
            print(f"\n  平均加速比: {avg:.2f}x")
            
            print("\n" + "=" * 70)
            print("  结论")
            print("=" * 70)
            
            if avg > 5:
                print(f"\n  Numba提供了显著的性能提升 ({avg:.1f}倍加速)")
                print("  强烈建议在生产环境中启用Numba加速")
            elif avg > 2:
                print(f"\n  Numba提供了良好的性能提升 ({avg:.1f}倍加速)")
                print("  建议在生产环境中启用Numba加速")
            else:
                print(f"\n  Numba提供了一定的性能提升 ({avg:.1f}倍加速)")
                print("  可以选择启用Numba加速")
            
            print("\n  配置方法:")
            print("    config = SolverConfig(use_numba=True)")
    
    print("\n" + "=" * 70 + "\n")

if __name__ == '__main__':
    main()

