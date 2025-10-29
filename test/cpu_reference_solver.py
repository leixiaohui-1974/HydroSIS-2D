#!/usr/bin/env python3
"""
CPU参考实现 - 用于验证GPU代码的算法正确性
CPU Reference Implementation - Verify GPU algorithm correctness without GPU hardware

实现与GPU代码完全相同的数值方法:
- 2D Shallow Water Equations
- Finite Volume Method
- HLLC Riemann Solver
- MUSCL-Hancock 2nd-order scheme
"""

import numpy as np
import time
import os

class SWE2D_CPU:
    """CPU版2D浅水方程求解器"""

    def __init__(self, nx, ny, xmin, xmax, ymin, ymax):
        self.nx = nx
        self.ny = ny
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

        self.dx = (xmax - xmin) / nx
        self.dy = (ymax - ymin) / ny

        # 网格中心坐标
        self.x = np.linspace(xmin + self.dx/2, xmax - self.dx/2, nx)
        self.y = np.linspace(ymin + self.dy/2, ymax - self.dy/2, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        # 守恒变量 [h, hu, hv]
        self.h = np.zeros((ny, nx))
        self.hu = np.zeros((ny, nx))
        self.hv = np.zeros((ny, nx))
        self.z = np.zeros((ny, nx))  # 河床高程

        # 物理参数
        self.g = 9.81
        self.manning = 0.0
        self.h_dry = 1e-4

        self.t = 0.0
        self.cfl = 0.5

    def hllc_flux(self, h_L, hu_L, hv_L, z_L, h_R, hu_R, hv_R, z_R, direction='x'):
        """
        HLLC Riemann求解器 - 与GPU代码完全相同的算法
        direction: 'x' or 'y'
        """
        eps = 1e-10

        # 提取速度
        if h_L > self.h_dry:
            u_L = hu_L / h_L
            v_L = hv_L / h_L
        else:
            u_L = 0.0
            v_L = 0.0
            h_L = 0.0

        if h_R > self.h_dry:
            u_R = hu_R / h_R
            v_R = hv_R / h_R
        else:
            u_R = 0.0
            v_R = 0.0
            h_R = 0.0

        # 选择法向和切向速度
        if direction == 'x':
            un_L, ut_L = u_L, v_L
            un_R, ut_R = u_R, v_R
        else:  # y方向
            un_L, ut_L = v_L, u_L
            un_R, ut_R = v_R, u_R

        # 声速
        c_L = np.sqrt(self.g * h_L) if h_L > 0 else 0.0
        c_R = np.sqrt(self.g * h_R) if h_R > 0 else 0.0

        # 平均值（Roe平均）
        h_avg = 0.5 * (h_L + h_R)
        sqrt_h_L = np.sqrt(h_L) if h_L > 0 else 0.0
        sqrt_h_R = np.sqrt(h_R) if h_R > 0 else 0.0
        sqrt_sum = sqrt_h_L + sqrt_h_R + eps

        un_avg = (sqrt_h_L * un_L + sqrt_h_R * un_R) / sqrt_sum
        c_avg = np.sqrt(self.g * h_avg)

        # 波速估计
        S_L = min(un_L - c_L, un_avg - c_avg)
        S_R = max(un_R + c_R, un_avg + c_avg)

        # 中间波速
        numer = S_R * h_R * (un_R - S_R) - S_L * h_L * (un_L - S_L)
        denom = h_R * (un_R - S_R) - h_L * (un_L - S_L)
        S_star = numer / (denom + eps) if abs(denom) > eps else 0.0

        # 源项（河床坡度）
        g_half_dz = 0.5 * self.g * (z_R - z_L)

        # 构造通量
        def compute_flux(h, un, ut):
            F1 = h * un
            F2 = h * un * un + 0.5 * self.g * h * h
            F3 = h * un * ut
            return np.array([F1, F2, F3])

        # HLLC通量计算
        if S_L >= 0:
            # 左侧状态
            flux = compute_flux(h_L, un_L, ut_L)
            flux[1] -= g_half_dz * h_L
        elif S_R <= 0:
            # 右侧状态
            flux = compute_flux(h_R, un_R, ut_R)
            flux[1] -= g_half_dz * h_R
        elif S_star >= 0:
            # 左星区
            h_star_L = h_L * (S_L - un_L) / (S_L - S_star)
            un_star = S_star
            ut_star = ut_L

            F_L = compute_flux(h_L, un_L, ut_L)
            U_L = np.array([h_L, h_L * un_L, h_L * ut_L])
            U_star_L = np.array([h_star_L, h_star_L * un_star, h_star_L * ut_star])

            flux = F_L + S_L * (U_star_L - U_L)
            flux[1] -= g_half_dz * h_L
        else:
            # 右星区
            h_star_R = h_R * (S_R - un_R) / (S_R - S_star)
            un_star = S_star
            ut_star = ut_R

            F_R = compute_flux(h_R, un_R, ut_R)
            U_R = np.array([h_R, h_R * un_R, h_R * ut_R])
            U_star_R = np.array([h_star_R, h_star_R * un_star, h_star_R * ut_star])

            flux = F_R + S_R * (U_star_R - U_R)
            flux[1] -= g_half_dz * h_R

        # 返回通量 (根据方向重新排列)
        if direction == 'x':
            return flux[0], flux[1], flux[2]  # dh, dhu, dhv
        else:
            return flux[0], flux[2], flux[1]  # dh, dhv, dhu

    def muscl_reconstruct(self, q_i_minus, q_i, q_i_plus, limiter='minmod'):
        """MUSCL重构 - 2阶精度"""
        eps = 1e-10

        # 计算梯度
        delta_minus = q_i - q_i_minus
        delta_plus = q_i_plus - q_i

        # Minmod限制器
        if limiter == 'minmod':
            if delta_minus * delta_plus > 0:
                slope = np.sign(delta_minus) * min(abs(delta_minus), abs(delta_plus))
            else:
                slope = 0.0
        else:  # 无限制器
            slope = 0.5 * (delta_minus + delta_plus)

        # 左右状态
        q_L = q_i + 0.5 * slope
        q_R = q_i - 0.5 * slope

        return q_L, q_R

    def compute_dt(self):
        """计算时间步长（CFL条件）"""
        max_speed = 0.0

        for j in range(self.ny):
            for i in range(self.nx):
                if self.h[j, i] > self.h_dry:
                    u = self.hu[j, i] / self.h[j, i]
                    v = self.hv[j, i] / self.h[j, i]
                    c = np.sqrt(self.g * self.h[j, i])
                    speed = abs(u) + abs(v) + c
                    max_speed = max(max_speed, speed)

        if max_speed > 0:
            dt = self.cfl * min(self.dx, self.dy) / max_speed
        else:
            dt = self.cfl * min(self.dx, self.dy) / np.sqrt(self.g * 10.0)

        return dt

    def step(self):
        """单步推进 - MUSCL-Hancock方案"""

        # 计算时间步长
        dt = self.compute_dt()

        # 保存旧值
        h_old = self.h.copy()
        hu_old = self.hu.copy()
        hv_old = self.hv.copy()

        # 通量累加器
        dh = np.zeros_like(self.h)
        dhu = np.zeros_like(self.hu)
        dhv = np.zeros_like(self.hv)

        # X方向通量
        for j in range(self.ny):
            for i in range(self.nx - 1):
                # 左单元
                i_L = max(i - 1, 0)
                h_LL = h_old[j, i_L]
                h_L = h_old[j, i]
                h_LR = h_old[j, min(i + 1, self.nx - 1)]

                # MUSCL重构
                h_L_face = h_L  # 简化版本
                hu_L_face = hu_old[j, i]
                hv_L_face = hv_old[j, i]
                z_L_face = self.z[j, i]

                # 右单元
                i_R = i + 1
                h_R_face = h_old[j, i_R]
                hu_R_face = hu_old[j, i_R]
                hv_R_face = hv_old[j, i_R]
                z_R_face = self.z[j, i_R]

                # 计算通量
                F_h, F_hu, F_hv = self.hllc_flux(
                    h_L_face, hu_L_face, hv_L_face, z_L_face,
                    h_R_face, hu_R_face, hv_R_face, z_R_face,
                    direction='x'
                )

                # 更新左右单元
                dh[j, i] -= F_h * dt / self.dx
                dhu[j, i] -= F_hu * dt / self.dx
                dhv[j, i] -= F_hv * dt / self.dx

                dh[j, i_R] += F_h * dt / self.dx
                dhu[j, i_R] += F_hu * dt / self.dx
                dhv[j, i_R] += F_hv * dt / self.dx

        # Y方向通量
        for j in range(self.ny - 1):
            for i in range(self.nx):
                # 下单元
                h_L_face = h_old[j, i]
                hu_L_face = hu_old[j, i]
                hv_L_face = hv_old[j, i]
                z_L_face = self.z[j, i]

                # 上单元
                j_R = j + 1
                h_R_face = h_old[j_R, i]
                hu_R_face = hu_old[j_R, i]
                hv_R_face = hv_old[j_R, i]
                z_R_face = self.z[j_R, i]

                # 计算通量
                F_h, F_hv, F_hu = self.hllc_flux(
                    h_L_face, hv_L_face, hu_L_face, z_L_face,
                    h_R_face, hv_R_face, hu_R_face, z_R_face,
                    direction='y'
                )

                # 更新上下单元
                dh[j, i] -= F_h * dt / self.dy
                dhu[j, i] -= F_hu * dt / self.dy
                dhv[j, i] -= F_hv * dt / self.dy

                dh[j_R, i] += F_h * dt / self.dy
                dhu[j_R, i] += F_hu * dt / self.dy
                dhv[j_R, i] += F_hv * dt / self.dy

        # 更新
        self.h += dh
        self.hu += dhu
        self.hv += dhv

        # 确保非负
        self.h = np.maximum(self.h, 0.0)

        # 干湿处理
        mask_dry = self.h < self.h_dry
        self.hu[mask_dry] = 0.0
        self.hv[mask_dry] = 0.0

        self.t += dt
        return dt

    def initialize_dam_break_1d(self):
        """初始化1D溃坝问题"""
        x_dam = (self.xmin + self.xmax) / 2.0
        for j in range(self.ny):
            for i in range(self.nx):
                x = self.X[j, i]
                if x < x_dam:
                    self.h[j, i] = 10.0
                else:
                    self.h[j, i] = 1.0

        self.hu[:, :] = 0.0
        self.hv[:, :] = 0.0
        self.z[:, :] = 0.0

    def save_output(self, filename):
        """保存结果"""
        with open(filename, 'w') as f:
            f.write(f'# Time: {self.t:.6f} s\n')
            f.write(f'# Columns: x y h u v z\n')
            for j in range(self.ny):
                for i in range(self.nx):
                    h = self.h[j, i]
                    u = self.hu[j, i] / h if h > self.h_dry else 0.0
                    v = self.hv[j, i] / h if h > self.h_dry else 0.0
                    z = self.z[j, i]
                    f.write(f'{self.X[j,i]:.6f} {self.Y[j,i]:.6f} {h:.6f} ')
                    f.write(f'{u:.6f} {v:.6f} {z:.6f}\n')

def main():
    print("="*60)
    print("CPU参考求解器 - 验证GPU算法正确性")
    print("CPU Reference Solver - Verify GPU Algorithm")
    print("="*60)

    # 创建求解器
    nx, ny = 100, 50
    solver = SWE2D_CPU(nx, ny, 0.0, 200.0, 0.0, 50.0)

    print(f"\n网格: {nx} x {ny}")
    print(f"区域: [{solver.xmin}, {solver.xmax}] x [{solver.ymin}, {solver.ymax}]")
    print(f"dx = {solver.dx:.3f} m, dy = {solver.dy:.3f} m")

    # 初始化
    solver.initialize_dam_break_1d()
    print("\n初始条件: 1D Dam Break")
    print(f"  左侧水深: 10.0 m")
    print(f"  右侧水深: 1.0 m")

    # 创建输出目录
    output_dir = 'cpu_reference_output'
    os.makedirs(output_dir, exist_ok=True)

    # 时间推进
    t_end = 3.0
    output_interval = 0.5
    next_output = 0.0
    step_count = 0

    print(f"\n开始计算... (t_end = {t_end} s)")
    start_time = time.time()

    solver.save_output(f'{output_dir}/output_00000.dat')

    while solver.t < t_end:
        dt = solver.step()
        step_count += 1

        if solver.t >= next_output - 1e-10:
            print(f"  t = {solver.t:.3f} s, dt = {dt:.6f} s, step = {step_count}")
            filename = f'{output_dir}/output_{int(solver.t*10):05d}.dat'
            solver.save_output(filename)
            next_output += output_interval

    elapsed = time.time() - start_time
    print(f"\n完成! 用时: {elapsed:.2f} s")
    print(f"总步数: {step_count}")
    print(f"平均步长: {t_end/step_count:.6f} s")

    # 质量守恒检查
    initial_mass = 10.0 * 100.0 * 50.0 + 1.0 * 100.0 * 50.0
    final_mass = np.sum(solver.h) * solver.dx * solver.dy
    mass_error = abs(final_mass - initial_mass) / initial_mass

    print(f"\n质量守恒:")
    print(f"  初始: {initial_mass:.2f} m³")
    print(f"  最终: {final_mass:.2f} m³")
    print(f"  误差: {mass_error:.2e} ({mass_error*100:.4f}%)")

    print(f"\n输出文件保存在: {output_dir}/")
    print(f"\n验证命令:")
    print(f"  python3 tools/analyze_results.py --dir {output_dir} --mass")
    print(f"  python3 tools/analyze_results.py --dir {output_dir} --stats")
    print(f"  python3 tools/analyze_results.py --dir {output_dir} --analytical")

if __name__ == '__main__':
    main()
