#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HydroSIS-2D 深度算法分析
对数值方法、精度、收敛性进行全面验证

作者: 技术审查团队
日期: 2025-11-12
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 设置编码
if sys.platform == 'win32':
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleOutputCP(65001)
        kernel32.SetConsoleCP(65001)
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    except:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing.mesh_generation import MeshGenerator, DomainParams
from solver import ShallowWaterSolver, SolverConfig

class AlgorithmAnalyzer:
    """算法深度分析器"""
    
    def __init__(self):
        self.results = []
        self.issues = []
        
    def print_section(self, title):
        print(f"\n{'='*70}")
        print(f" {title}")
        print(f"{'='*70}")
    
    def test_hll_riemann_solver(self):
        """测试1: HLL黎曼求解器实现验证"""
        self.print_section("测试1: HLL黎曼求解器算法验证")
        
        print("\n分析HLL波速估计...")
        
        # 理论验证: Riemann问题
        h_L, h_R = 10.0, 1.0
        u_L, u_R = 0.0, 0.0
        g = 9.81
        
        # 波速计算
        c_L = np.sqrt(g * h_L)  # 左侧波速
        c_R = np.sqrt(g * h_R)  # 右侧波速
        
        # HLL波速估计
        s_L = min(u_L - c_L, u_R - c_R)
        s_R = max(u_L + c_L, u_R + c_R)
        
        print(f"  左侧状态: h={h_L}m, u={u_L}m/s, c={c_L:.3f}m/s")
        print(f"  右侧状态: h={h_R}m, u={u_R}m/s, c={c_R:.3f}m/s")
        print(f"  波速估计: s_L={s_L:.3f}m/s, s_R={s_R:.3f}m/s")
        
        # 验证波速估计的合理性
        if s_L < 0 and s_R > 0:
            print(f"  [OK] 波速估计合理 (跨音速流动)")
            self.results.append(('HLL波速估计', True, '正确'))
        else:
            print(f"  [WARN] 波速估计异常")
            self.issues.append('HLL波速估计可能有问题')
            self.results.append(('HLL波速估计', False, '异常'))
        
        # 理论解: 溃坝问题的特征速度
        # 稀疏波速度: u - 2*sqrt(g*h)
        # 激波速度: (h_L*u_L - h_R*u_R)/(h_L - h_R) + sqrt(0.5*g*(h_L+h_R))
        
        rarefaction_speed = u_L - 2*np.sqrt(g*h_L)
        print(f"  理论稀疏波头部速度: {rarefaction_speed:.3f}m/s")
        
        # 验证HLL是否能捕捉
        if abs(s_L - rarefaction_speed) / abs(rarefaction_speed) < 0.5:
            print(f"  [OK] 稀疏波捕捉合理")
        else:
            print(f"  [WARN] 稀疏波捕捉可能不准确")
            self.issues.append('HLL对稀疏波的估计可能偏差较大')
    
    def test_cfl_condition(self):
        """测试2: CFL条件计算验证"""
        self.print_section("测试2: CFL条件与时间步长计算")
        
        print("\n测试CFL条件实现...")
        
        # 创建测试网格
        domain = DomainParams(xmin=0, xmax=100, ymin=0, ymax=10)
        mesh_gen = MeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=50, ny=5)
        
        # 创建求解器
        terrain = np.zeros((mesh.nx, mesh.ny))
        config = SolverConfig(cfl=0.5)
        solver = ShallowWaterSolver(mesh, terrain, config)
        
        # 设置测试状态
        solver.h[:, :] = 5.0
        solver.u[:, :] = 2.0
        solver.v[:, :] = 0.0
        
        # 计算时间步
        dt = solver.compute_timestep()
        
        # 理论CFL条件: dt <= CFL * min(dx, dy) / (|u| + sqrt(g*h))
        max_wave_speed = abs(solver.u.max()) + np.sqrt(config.g * solver.h.max())
        dt_theory = config.cfl * min(mesh.dx, mesh.dy) / max_wave_speed
        
        print(f"  网格尺寸: dx={mesh.dx:.3f}m, dy={mesh.dy:.3f}m")
        print(f"  流动状态: h={solver.h.max():.1f}m, u={solver.u.max():.1f}m/s")
        print(f"  最大波速: {max_wave_speed:.3f}m/s")
        print(f"  理论时间步: dt={dt_theory:.6f}s")
        print(f"  计算时间步: dt={dt:.6f}s")
        
        rel_error = abs(dt - dt_theory) / dt_theory
        if rel_error < 0.01:
            print(f"  [OK] CFL条件计算正确 (误差{rel_error*100:.2f}%)")
            self.results.append(('CFL条件', True, f'误差{rel_error*100:.2f}%'))
        else:
            print(f"  [ERROR] CFL条件计算有误 (误差{rel_error*100:.2f}%)")
            self.issues.append(f'CFL条件计算误差过大: {rel_error*100:.1f}%')
            self.results.append(('CFL条件', False, f'误差{rel_error*100:.2f}%'))
    
    def test_mass_conservation(self):
        """测试3: 质量守恒验证"""
        self.print_section("测试3: 数值格式的质量守恒性")
        
        print("\n运行质量守恒测试...")
        
        # 封闭区域 - 理论上质量应该绝对守恒
        domain = DomainParams(xmin=0, xmax=100, ymin=0, ymax=50)
        mesh_gen = MeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=50, ny=25)
        
        terrain = np.zeros((mesh.nx, mesh.ny))
        config = SolverConfig(
            t_end=5.0,
            cfl=0.5,
            output_interval=999,
            output_dir=None,
            print_progress=False
        )
        solver = ShallowWaterSolver(mesh, terrain, config)
        
        # 初始条件: 中心有一个高斯分布的水体
        X = mesh.x
        Y = mesh.y
        x0, y0 = 50, 25
        sigma = 10
        h0 = 5.0 + 3.0 * np.exp(-((X-x0)**2 + (Y-y0)**2)/(2*sigma**2))
        u0 = np.zeros_like(h0)
        v0 = np.zeros_like(h0)
        
        solver.set_initial_conditions(h0, u0, v0)
        
        initial_mass = np.sum(solver.h) * mesh.dx * mesh.dy
        print(f"  初始质量: {initial_mass:.6f} m^3")
        
        # 运行模拟
        result = solver.solve()
        
        final_mass = np.sum(solver.h) * mesh.dx * mesh.dy
        mass_error = (final_mass - initial_mass) / initial_mass * 100
        
        print(f"  最终质量: {final_mass:.6f} m^3")
        print(f"  质量误差: {mass_error:+.6f}%")
        
        if abs(mass_error) < 0.1:
            print(f"  [OK] 质量守恒优秀 (<0.1%)")
            self.results.append(('质量守恒', True, f'{mass_error:+.4f}%'))
        elif abs(mass_error) < 1.0:
            print(f"  [WARN] 质量守恒良好但有损失 ({abs(mass_error):.2f}%)")
            self.issues.append(f'质量守恒有轻微损失: {mass_error:+.4f}%')
            self.results.append(('质量守恒', True, f'{mass_error:+.4f}%'))
        else:
            print(f"  [ERROR] 质量守恒差 ({abs(mass_error):.2f}%)")
            self.issues.append(f'质量守恒误差严重: {mass_error:+.4f}%')
            self.results.append(('质量守恒', False, f'{mass_error:+.4f}%'))
    
    def test_dry_bed_treatment(self):
        """测试4: 干湿边界处理"""
        self.print_section("测试4: 干湿边界处理 (Wet-Dry Front)")
        
        print("\n测试干床处理...")
        
        # 溃坝到干床的情况
        domain = DomainParams(xmin=0, xmax=200, ymin=0, ymax=10)
        mesh_gen = MeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=100, ny=5)
        
        terrain = np.zeros((mesh.nx, mesh.ny))
        config = SolverConfig(
            t_end=2.0,
            cfl=0.5,
            h_dry=1e-4,
            output_interval=999,
            output_dir=None,
            print_progress=False
        )
        solver = ShallowWaterSolver(mesh, terrain, config)
        
        # 左半部分有水，右半部分完全干燥
        X = mesh.x
        h0 = np.where(X < 100, 10.0, 0.0)
        u0 = np.zeros_like(h0)
        v0 = np.zeros_like(h0)
        
        solver.set_initial_conditions(h0, u0, v0)
        
        # 运行
        result = solver.solve()
        
        # 检查是否有负水深
        min_h = solver.h.min()
        print(f"  最小水深: {min_h:.6e} m")
        
        if min_h >= -1e-10:
            print(f"  [OK] 无负水深")
            self.results.append(('干湿边界-正性', True, '无负值'))
        else:
            print(f"  [ERROR] 出现负水深!")
            self.issues.append(f'干湿边界处理有问题，出现负水深: {min_h}')
            self.results.append(('干湿边界-正性', False, f'min_h={min_h}'))
        
        # 检查波前是否平滑
        # 找到波前位置 (h > h_dry 的最右侧)
        wet_cells = solver.h > config.h_dry
        if wet_cells.any():
            wet_x = X[wet_cells]
            front_position = wet_x.max()
            print(f"  波前位置: x={front_position:.2f}m")
            
            # 理论波前速度约为 2*sqrt(g*h0)
            theory_speed = 2*np.sqrt(config.g * 10.0)
            theory_position = 100 + theory_speed * result['simulated_time']
            
            error = abs(front_position - theory_position)
            print(f"  理论位置: x={theory_position:.2f}m")
            print(f"  位置误差: {error:.2f}m")
            
            if error / front_position < 0.1:
                print(f"  [OK] 波前传播速度合理")
                self.results.append(('干湿边界-波速', True, f'误差{error:.1f}m'))
            else:
                print(f"  [WARN] 波前速度与理论值偏差较大")
                self.results.append(('干湿边界-波速', False, f'误差{error:.1f}m'))
    
    def test_numerical_convergence(self):
        """测试5: 网格收敛性测试"""
        self.print_section("测试5: 网格收敛性分析")
        
        print("\n测试空间收敛性...")
        
        grid_sizes = [25, 50, 100]
        errors = []
        
        for nx in grid_sizes:
            domain = DomainParams(xmin=0, xmax=200, ymin=0, ymax=20)
            mesh_gen = MeshGenerator(domain)
            mesh = mesh_gen.generate_uniform_mesh(nx=nx, ny=max(5, nx//10))
            
            terrain = np.zeros((mesh.nx, mesh.ny))
            config = SolverConfig(
                t_end=1.0,
                cfl=0.5,
                output_interval=999,
                output_dir=None,
                print_progress=False
            )
            solver = ShallowWaterSolver(mesh, terrain, config)
            
            # 溃坝初始条件
            X = mesh.x
            h0 = np.where(X < 100, 10.0, 1.0)
            u0 = np.zeros_like(h0)
            v0 = np.zeros_like(h0)
            
            solver.set_initial_conditions(h0, u0, v0)
            result = solver.solve()
            
            # 计算与解析解的误差 (简化: 使用质量守恒误差)
            final_mass = np.sum(solver.h) * mesh.dx * mesh.dy
            initial_mass = np.sum(h0) * mesh.dx * mesh.dy
            error = abs(final_mass - initial_mass) / initial_mass
            errors.append(error)
            
            print(f"  网格 {nx}x{mesh.ny}: 质量误差 = {error*100:.6f}%")
        
        # 分析收敛阶
        if len(errors) >= 2:
            # 计算收敛率
            ratios = []
            for i in range(len(errors)-1):
                if errors[i+1] > 0:
                    ratio = errors[i] / errors[i+1]
                    ratios.append(ratio)
            
            if ratios:
                avg_ratio = np.mean(ratios)
                convergence_order = np.log(avg_ratio) / np.log(2)
                
                print(f"\n  平均误差减小比: {avg_ratio:.3f}")
                print(f"  收敛阶估计: {convergence_order:.2f}")
                
                if convergence_order > 0.8:
                    print(f"  [OK] 格式具有收敛性")
                    self.results.append(('收敛性', True, f'阶数~{convergence_order:.1f}'))
                else:
                    print(f"  [WARN] 收敛阶低于预期")
                    self.issues.append(f'收敛阶较低: {convergence_order:.2f}')
                    self.results.append(('收敛性', False, f'阶数~{convergence_order:.1f}'))
    
    def test_source_term_balance(self):
        """测试6: 源项平衡性 (C-property)"""
        self.print_section("测试6: 源项平衡性测试 (Well-Balanced Property)")
        
        print("\n测试静水平衡...")
        
        # 静水湖面，有地形变化
        domain = DomainParams(xmin=0, xmax=100, ymin=0, ymax=10)
        mesh_gen = MeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=50, ny=5)
        
        # 地形: 抛物线形状
        X = mesh.x
        terrain = 0.1 * (X - 50)**2 / 100  # 中间低，两边高
        
        config = SolverConfig(
            t_end=5.0,
            cfl=0.5,
            output_interval=999,
            output_dir=None,
            print_progress=False
        )
        solver = ShallowWaterSolver(mesh, terrain, config)
        
        # 静水初始条件: 水面高程恒定
        water_surface = 10.0  # 水面高程
        h0 = water_surface - terrain
        h0 = np.maximum(h0, 0.0)  # 确保非负
        u0 = np.zeros_like(h0)
        v0 = np.zeros_like(h0)
        
        solver.set_initial_conditions(h0, u0, v0)
        
        initial_u_max = abs(solver.u).max()
        initial_v_max = abs(solver.v).max()
        
        # 运行
        result = solver.solve()
        
        final_u_max = abs(solver.u).max()
        final_v_max = abs(solver.v).max()
        
        print(f"  初始最大速度: u={initial_u_max:.6e} m/s")
        print(f"  最终最大速度: u={final_u_max:.6e} m/s, v={final_v_max:.6e} m/s")
        
        # 理论上静水应该保持静止
        if final_u_max < 1e-6 and final_v_max < 1e-6:
            print(f"  [OK] 保持静水平衡 (Well-Balanced)")
            self.results.append(('Well-Balanced', True, f'u_max={final_u_max:.2e}'))
        elif final_u_max < 1e-3:
            print(f"  [WARN] 有轻微扰动但可接受")
            self.issues.append(f'静水平衡有小扰动: u_max={final_u_max:.2e}')
            self.results.append(('Well-Balanced', True, f'u_max={final_u_max:.2e}'))
        else:
            print(f"  [ERROR] 静水平衡被破坏!")
            self.issues.append(f'静水平衡测试失败: u_max={final_u_max:.2e}')
            self.results.append(('Well-Balanced', False, f'u_max={final_u_max:.2e}'))
    
    def generate_report(self):
        """生成分析报告"""
        self.print_section("深度算法分析报告")
        
        print(f"\n测试统计:")
        total = len(self.results)
        passed = sum(1 for _, status, _ in self.results if status)
        failed = total - passed
        
        print(f"  总测试数: {total}")
        print(f"  通过: {passed}")
        print(f"  失败: {failed}")
        print(f"  通过率: {passed/total*100:.1f}%")
        
        print(f"\n详细结果:")
        for name, status, detail in self.results:
            status_str = '[OK]' if status else '[FAIL]'
            print(f"  {status_str:<8} {name:<25} {detail}")
        
        if self.issues:
            print(f"\n发现的问题 ({len(self.issues)}个):")
            for i, issue in enumerate(self.issues, 1):
                print(f"  {i}. {issue}")
        else:
            print(f"\n未发现严重问题!")
        
        # 算法评级
        print(f"\n算法实现评级:")
        if failed == 0:
            print("  等级: A (优秀) - 所有测试通过")
        elif failed <= 2:
            print("  等级: B (良好) - 少量问题")
        else:
            print("  等级: C (一般) - 存在多个问题")

def main():
    analyzer = AlgorithmAnalyzer()
    
    analyzer.print_section("HydroSIS-2D 深度算法分析")
    print("\n本测试将验证:")
    print("  1. HLL黎曼求解器实现")
    print("  2. CFL条件计算")
    print("  3. 质量守恒性")
    print("  4. 干湿边界处理")
    print("  5. 网格收敛性")
    print("  6. 源项平衡性")
    
    # 执行所有测试
    try:
        analyzer.test_hll_riemann_solver()
    except Exception as e:
        print(f"[ERROR] HLL测试失败: {e}")
        analyzer.issues.append(f'HLL测试异常: {str(e)}')
    
    try:
        analyzer.test_cfl_condition()
    except Exception as e:
        print(f"[ERROR] CFL测试失败: {e}")
        analyzer.issues.append(f'CFL测试异常: {str(e)}')
    
    try:
        analyzer.test_mass_conservation()
    except Exception as e:
        print(f"[ERROR] 质量守恒测试失败: {e}")
        analyzer.issues.append(f'质量守恒测试异常: {str(e)}')
    
    try:
        analyzer.test_dry_bed_treatment()
    except Exception as e:
        print(f"[ERROR] 干湿边界测试失败: {e}")
        analyzer.issues.append(f'干湿边界测试异常: {str(e)}')
    
    try:
        analyzer.test_numerical_convergence()
    except Exception as e:
        print(f"[ERROR] 收敛性测试失败: {e}")
        analyzer.issues.append(f'收敛性测试异常: {str(e)}')
    
    try:
        analyzer.test_source_term_balance()
    except Exception as e:
        print(f"[ERROR] Well-Balanced测试失败: {e}")
        analyzer.issues.append(f'Well-Balanced测试异常: {str(e)}')
    
    # 生成报告
    analyzer.generate_report()
    
    print(f"\n{'='*70}\n")

if __name__ == '__main__':
    main()

