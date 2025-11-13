#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HydroSIS-2D 性能测试工具
对比NumPy vs Numba JIT加速性能

作者: HydroSIS-2D Team
版本: 1.0
"""

import sys
import os
import platform
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
            sys.stdout.buffer,
            encoding='utf-8',
            errors='replace',
            line_buffering=True
        )
    except:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing.mesh_generation import MeshGenerator, DomainParams
from solver import ShallowWaterSolver, SolverConfig

def print_header(text):
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def print_section(text):
    print(f"\n{text}")
    print("-" * 70)

def run_performance_test(grid_size, use_numba=False, test_name=""):
    """运行性能测试"""
    nx, ny = grid_size
    
    print(f"\n  网格规模: {nx} x {ny} = {nx*ny:,} 单元")
    print(f"  加速模式: {'Numba JIT' if use_numba else 'NumPy向量化'}")
    
    # 创建网格
    domain = DomainParams(xmin=0.0, xmax=200.0, ymin=0.0, ymax=20.0)
    mesh_gen = MeshGenerator(domain)
    mesh = mesh_gen.generate_uniform_mesh(nx=nx, ny=ny)
    
    # 创建地形
    terrain = np.zeros((mesh.nx, mesh.ny))
    
    # 配置求解器
    config = SolverConfig(
        t_end=5.0,
        cfl=0.5,
        output_interval=10.0,  # 不输出文件
        output_dir=None,
        print_progress=False,
        use_numba=use_numba
    )
    
    # 初始化求解器
    solver = ShallowWaterSolver(mesh, terrain, config)
    
    # 设置溃坝初始条件
    h0 = np.zeros((mesh.nx, mesh.ny))
    u0 = np.zeros((mesh.nx, mesh.ny))
    v0 = np.zeros((mesh.nx, mesh.ny))
    
    dam_position = 100.0
    X = mesh.x
    h0[X < dam_position] = 10.0
    h0[X >= dam_position] = 1.0
    
    solver.set_initial_conditions(h0, u0, v0)
    
    # 运行模拟并计时
    start_time = time.perf_counter()
    result = solver.solve()
    elapsed_time = time.perf_counter() - start_time
    
    # 计算性能指标
    simulated_time = result['simulation_time']
    n_steps = result['n_steps']
    speed_factor = simulated_time / elapsed_time if elapsed_time > 0 else 0
    time_per_step = elapsed_time / n_steps if n_steps > 0 else 0
    cells_per_second = (nx * ny * n_steps) / elapsed_time if elapsed_time > 0 else 0
    
    return {
        'elapsed_time': elapsed_time,
        'simulated_time': simulated_time,
        'n_steps': n_steps,
        'speed_factor': speed_factor,
        'time_per_step': time_per_step,
        'cells_per_second': cells_per_second,
        'grid_size': (nx, ny)
    }

def main():
    """主函数"""
    print_header("HydroSIS-2D 性能测试")
    
    print(f"\n系统信息:")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  操作系统: {platform.system()} {platform.release()}")
    print(f"  处理器: {platform.processor()}")
    
    # 检查Numba
    try:
        import numba
        print(f"  Numba: {numba.__version__}")
        numba_available = True
    except ImportError:
        print(f"  Numba: 未安装")
        numba_available = False
    
    # 测试配置
    test_configs = [
        (50, 50, "小规模"),
        (100, 100, "中等规模"),
        (200, 200, "大规模"),
    ]
    
    print_section("性能测试开始")
    print("\n注: 每个测试模拟5秒物理时间")
    
    results = []
    
    for nx, ny, desc in test_configs:
        print_section(f"测试 {desc}: {nx}x{ny} 网格")
        
        # NumPy版本
        print("\n  [1/2] NumPy向量化版本...")
        try:
            result_numpy = run_performance_test((nx, ny), use_numba=False, test_name=desc)
            results.append(('NumPy', desc, result_numpy))
            print(f"      完成时间: {result_numpy['elapsed_time']:.2f} 秒")
            print(f"      时间步数: {result_numpy['n_steps']}")
            print(f"      速度因子: {result_numpy['speed_factor']:.2f}x 实时")
        except Exception as e:
            print(f"      错误: {str(e)}")
            result_numpy = None
        
        # Numba版本
        if numba_available:
            print("\n  [2/2] Numba JIT加速版本...")
            try:
                result_numba = run_performance_test((nx, ny), use_numba=True, test_name=desc)
                results.append(('Numba', desc, result_numba))
                print(f"      完成时间: {result_numba['elapsed_time']:.2f} 秒")
                print(f"      时间步数: {result_numba['n_steps']}")
                print(f"      速度因子: {result_numba['speed_factor']:.2f}x 实时")
                
                # 计算加速比
                if result_numpy:
                    speedup = result_numpy['elapsed_time'] / result_numba['elapsed_time']
                    print(f"\n      加速比: {speedup:.2f}x (Numba相对NumPy)")
            except Exception as e:
                print(f"      错误: {str(e)}")
        else:
            print("\n  [2/2] 跳过Numba测试 (未安装)")
    
    # 生成总结报告
    print_header("性能测试总结")
    
    print("\n详细结果:")
    print(f"\n{'模式':<10} {'规模':<12} {'网格':<12} {'时间(s)':<10} {'步数':<8} {'速度':<12}")
    print("-" * 70)
    
    for mode, desc, result in results:
        nx, ny = result['grid_size']
        grid_str = f"{nx}x{ny}"
        print(f"{mode:<10} {desc:<12} {grid_str:<12} "
              f"{result['elapsed_time']:<10.2f} {result['n_steps']:<8} "
              f"{result['speed_factor']:<12.2f}x")
    
    # 计算加速比总结
    numpy_results = {}
    numba_results = {}
    
    if numba_available and len(results) >= 2:
        print("\n加速比总结 (Numba / NumPy):")
        print("-" * 70)
        
        numpy_results = {r[1]: r[2] for r in results if r[0] == 'NumPy'}
        numba_results = {r[1]: r[2] for r in results if r[0] == 'Numba'}
        
        for desc in numpy_results.keys():
            if desc in numba_results:
                speedup = numpy_results[desc]['elapsed_time'] / numba_results[desc]['elapsed_time']
                nx, ny = numpy_results[desc]['grid_size']
                print(f"  {desc:<12} ({nx}x{ny}):  {speedup:.2f}x 加速")
    
    # 建议
    print_header("性能建议")
    
    if numba_available and len(numpy_results) > 0 and len(numba_results) > 0:
        avg_speedup = 0
        count = 0
        for desc in numpy_results.keys():
            if desc in numba_results:
                speedup = numpy_results[desc]['elapsed_time'] / numba_results[desc]['elapsed_time']
                avg_speedup += speedup
                count += 1
        
        if count > 0:
            avg_speedup /= count
            print(f"\n  平均加速比: {avg_speedup:.2f}x")
            
            if avg_speedup > 3:
                print("\n  [推荐] 强烈建议启用 Numba JIT 加速")
                print("  在求解器配置中设置: use_numba = True")
            elif avg_speedup > 1.5:
                print("\n  [推荐] 建议启用 Numba JIT 加速")
                print("  在求解器配置中设置: use_numba = True")
            else:
                print("\n  [建议] Numba 加速效果有限，可选择启用")
    else:
        print("\n  [建议] 安装 Numba 以获得更好性能")
        print("  安装命令: conda install numba 或 pip install numba")
    
    print("\n" + "=" * 70)
    print()

if __name__ == '__main__':
    main()

