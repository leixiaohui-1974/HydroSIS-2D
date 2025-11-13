#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HydroSIS-2D 可视化功能测试

作者: HydroSIS-2D Team
版本: 1.0
"""

import sys
import os

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

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def test_matplotlib_visualization():
    """测试Matplotlib 2D可视化"""
    print("\n测试 Matplotlib 2D可视化...")
    
    try:
        # 创建测试数据
        x = np.linspace(0, 200, 100)
        y = np.linspace(0, 20, 20)
        X, Y = np.meshgrid(x, y)  # X.shape = (20, 100), Y.shape = (20, 100)
        
        # 模拟溃坝水深分布
        h = np.zeros_like(X)  # h.shape = (20, 100)
        h[X < 100] = 10.0 - 0.05 * X[X < 100]
        h[X >= 100] = 1.0 + 0.02 * (X[X >= 100] - 100)
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 1. 等高线图
        ax = axes[0, 0]
        contour = ax.contourf(X, Y, h, levels=20, cmap='Blues')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title('水深等高线图')
        plt.colorbar(contour, ax=ax, label='水深 [m]')
        
        # 2. 3D表面图（伪3D）
        ax = axes[0, 1]
        im = ax.imshow(h, extent=[0, 200, 0, 20], aspect='auto', 
                       cmap='Blues', origin='lower', interpolation='bilinear')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title('水深热力图')
        plt.colorbar(im, ax=ax, label='水深 [m]')
        
        # 3. 沿中心线剖面
        ax = axes[1, 0]
        profile = h[h.shape[0]//2, :]  # 取中间行
        ax.plot(x, profile, 'b-', linewidth=2)
        ax.fill_between(x, 0, profile, alpha=0.3)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('水深 [m]')
        ax.set_title('中心线剖面')
        ax.grid(True, alpha=0.3)
        
        # 4. 统计信息
        ax = axes[1, 1]
        ax.axis('off')
        stats_text = f"""
        可视化测试统计
        ─────────────────────
        
        数据网格: {h.shape[0]} x {h.shape[1]}
        
        水深统计:
          最小值: {h.min():.2f} m
          最大值: {h.max():.2f} m
          平均值: {h.mean():.2f} m
          标准差: {h.std():.2f} m
        
        总水量: {np.sum(h) * 2.0 * 1.0:.0f} m^3
        
        状态: [OK] 成功
        """
        ax.text(0.1, 0.5, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='center',
                family='monospace')
        
        plt.tight_layout()
        
        # 保存
        output_dir = Path('output/viz_test')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / 'matplotlib_test.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  [OK] Matplotlib测试成功")
        print(f"       图片已保存: {output_file}")
        return True
        
    except Exception as e:
        print(f"  [ERROR] Matplotlib测试失败: {e}")
        return False

def test_pyvista_visualization():
    """测试PyVista 3D可视化"""
    print("\n测试 PyVista 3D可视化...")
    
    try:
        import pyvista as pv
        
        # 创建测试数据
        x = np.linspace(0, 200, 50)
        y = np.linspace(0, 20, 10)
        X, Y = np.meshgrid(x, y)
        
        # 模拟地形和水深
        terrain = -2.0 * np.sin(X / 50.0) * np.sin(Y / 10.0)
        h = np.zeros_like(X)
        h[X < 100] = 10.0
        h[X >= 100] = 1.0
        water_surface = terrain + h
        
        # 创建结构化网格
        grid = pv.StructuredGrid(X, Y, terrain)
        grid['terrain'] = terrain.ravel(order='F')
        grid['water_depth'] = h.ravel(order='F')
        grid['water_surface'] = water_surface.ravel(order='F')
        
        # 创建离屏绘图器
        plotter = pv.Plotter(off_screen=True, window_size=[1200, 800])
        
        # 添加地形
        plotter.add_mesh(grid, scalars='terrain', cmap='terrain',
                        opacity=1.0, show_edges=False,
                        scalar_bar_args={'title': '地形高程 [m]'})
        
        # 添加水面
        water_grid = pv.StructuredGrid(X, Y, water_surface)
        plotter.add_mesh(water_grid, color='blue', opacity=0.6,
                        show_edges=False)
        
        # 设置视角
        plotter.camera_position = 'iso'
        plotter.add_text('HydroSIS-2D 3D可视化测试', 
                        position='upper_edge', font_size=12)
        
        # 保存截图
        output_dir = Path('output/viz_test')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / 'pyvista_test.png'
        plotter.screenshot(str(output_file))
        plotter.close()
        
        print(f"  [OK] PyVista测试成功")
        print(f"       图片已保存: {output_file}")
        return True
        
    except ImportError:
        print(f"  [SKIP] PyVista未安装，跳过3D可视化测试")
        return None
    except Exception as e:
        print(f"  [ERROR] PyVista测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vtk_output():
    """测试VTK文件输出"""
    print("\n测试 VTK文件输出...")
    
    try:
        # 检查之前生成的VTK文件
        vtk_dir = Path('output/solver_test')
        if vtk_dir.exists():
            vtk_files = list(vtk_dir.glob('result_*.vtk'))
            if vtk_files:
                print(f"  [OK] 找到 {len(vtk_files)} 个VTK文件")
                
                # 读取第一个文件验证
                first_file = vtk_files[0]
                with open(first_file, 'r') as f:
                    header = f.readline()
                    if header.startswith('# vtk'):
                        print(f"  [OK] VTK文件格式正确")
                        print(f"       示例: {first_file.name}")
                        return True
                    else:
                        print(f"  [ERROR] VTK文件格式错误")
                        return False
            else:
                print(f"  [WARN] 未找到VTK文件")
                return None
        else:
            print(f"  [WARN] 输出目录不存在")
            return None
            
    except Exception as e:
        print(f"  [ERROR] VTK测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("\n" + "=" * 70)
    print("  HydroSIS-2D 可视化功能测试")
    print("=" * 70)
    
    results = []
    
    # 测试Matplotlib
    results.append(('Matplotlib 2D', test_matplotlib_visualization()))
    
    # 测试PyVista
    results.append(('PyVista 3D', test_pyvista_visualization()))
    
    # 测试VTK输出
    results.append(('VTK文件输出', test_vtk_output()))
    
    # 总结
    print("\n" + "=" * 70)
    print("  测试结果总结")
    print("=" * 70)
    
    for name, result in results:
        if result is True:
            status = "[OK]"
        elif result is False:
            status = "[ERROR]"
        else:
            status = "[SKIP]"
        print(f"  {status:<8} {name}")
    
    # 统计
    success = sum(1 for _, r in results if r is True)
    failed = sum(1 for _, r in results if r is False)
    skipped = sum(1 for _, r in results if r is None)
    
    print(f"\n  成功: {success}  失败: {failed}  跳过: {skipped}")
    
    if failed == 0:
        print("\n  [结论] 所有可视化功能测试通过!")
    else:
        print("\n  [结论] 部分测试失败，请检查错误信息")
    
    print("\n" + "=" * 70 + "\n")

if __name__ == '__main__':
    main()

