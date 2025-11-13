# -*- coding: utf-8 -*-
"""
HydroSIS-2D 完整工作流演示
从网格剖分 → 二维模拟 → 流场可视化

对标: MIKE FLOOD, HEC-RAS 2D
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import time
import os

# HydroSIS-2D模块
from preprocessing.mesh_generation import MeshGenerator, AdaptiveMeshGenerator, DomainParams, MeshQualityChecker
from preprocessing.geometry import GeometryGenerator
from solver.shallow_water_solver import ShallowWaterSolver, SolverConfig

print("="*80)
print("  HydroSIS-2D 完整工作流演示")
print("  Full Workflow: Mesh Generation -> 2D Simulation -> Flow Visualization")
print("="*80)

# 创建输出目录
output_base = "output/full_workflow"
for subdir in ['mesh', 'simulation', 'visualization', 'animation']:
    os.makedirs(f"{output_base}/{subdir}", exist_ok=True)

# ============================================================================
# PHASE 1: 网格剖分 (Mesh Generation)
# ============================================================================
print("\n" + "="*80)
print("PHASE 1: 网格剖分 (Mesh Generation)")
print("="*80)

# 定义计算域：溃坝场景
domain = DomainParams(xmin=0, xmax=500, ymin=0, ymax=200)
print(f"\n[1.1] 计算域定义")
print(f"  X方向: [{domain.xmin}, {domain.xmax}] m")
print(f"  Y方向: [{domain.ymin}, {domain.ymax}] m")
print(f"  总面积: {domain.area/1000:.1f} km^2")

# 创建自适应网格生成器
mesh_gen = AdaptiveMeshGenerator(domain)

# 添加溃坝口加密区域
dam_location = 200  # 溃坝位置 x=200m
mesh_gen.add_refinement_zone(
    xmin=dam_location-40,
    xmax=dam_location+40,
    ymin=0,
    ymax=200,
    refinement_level=1,  # 2倍加密
    priority=1
)
print(f"\n[1.2] 加密区域设置")
print(f"  溃坝位置: x={dam_location}m")
print(f"  加密区域: [{dam_location-40}, {dam_location+40}] x [0, 200]")
print(f"  加密级别: 1 (2倍加密)")

# 生成基础网格
mesh = mesh_gen.generate_uniform_mesh(nx=200, ny=80)
print(f"\n[1.3] 网格生成完成")
print(f"  网格规模: {mesh.nx} × {mesh.ny} = {mesh.ncells:,} 单元")
print(f"  单元尺寸: dx={mesh.dx:.2f}m, dy={mesh.dy:.2f}m")

# 网格质量检查
checker = MeshQualityChecker(mesh)
metrics = checker.compute_metrics()
print(f"\n[1.4] 网格质量")
print(f"  长宽比: {metrics.aspect_ratio_mean:.3f}")
print(f"  均匀性: {metrics.uniformity_score:.3f}")

# ============================================================================
# PHASE 2: 地形生成 (Terrain Generation)
# ============================================================================
print("\n" + "="*80)
print("PHASE 2: 地形生成 (Terrain Generation)")
print("="*80)

# 创建渐变地形：上游平台 → 下游平原
terrain = np.zeros((mesh.nx, mesh.ny))
X = mesh.x
Y = mesh.y

# 上游水库区域 (x < dam_location)
terrain[X < dam_location] = 0.0

# 下游渐变坡度 (x > dam_location)
downstream_mask = X >= dam_location
x_downstream = X[downstream_mask] - dam_location
max_downstream = domain.xmax - dam_location
terrain[downstream_mask] = -3.0 * (x_downstream / max_downstream)  # 向下游下降3m

# 两侧略微抬高（模拟河岸）
side_margin = 20  # m
left_side = Y < side_margin
right_side = Y > (domain.ymax - side_margin)
terrain[left_side] += 0.5
terrain[right_side] += 0.5

print(f"\n[2.1] 地形特征")
print(f"  上游水库: z = 0.0 m (平坦)")
print(f"  下游平原: z = 0 ~ -3.0 m (渐变)")
print(f"  两侧河岸: 抬高 0.5 m")
print(f"  地形范围: [{terrain.min():.2f}, {terrain.max():.2f}] m")

# ============================================================================
# PHASE 3: 初始条件设置 (Initial Conditions)
# ============================================================================
print("\n" + "="*80)
print("PHASE 3: 初始条件设置 (Initial Conditions)")
print("="*80)

# 溃坝初始条件
h0 = np.zeros((mesh.nx, mesh.ny))
u0 = np.zeros((mesh.nx, mesh.ny))
v0 = np.zeros((mesh.nx, mesh.ny))

# 上游水深10m，下游水深0.5m
h0[X < dam_location] = 10.0
h0[X >= dam_location] = 0.5

initial_volume = np.sum(h0) * mesh.dx * mesh.dy

print(f"\n[3.1] 溃坝初始条件")
print(f"  上游水深: 10.0 m")
print(f"  下游水深: 0.5 m")
print(f"  初始水量: {initial_volume/1000:.2f} thousand m^3")
print(f"  初始流速: 0.0 m/s (静止)")

# ============================================================================
# PHASE 4: 二维水动力模拟 (2D Hydrodynamic Simulation)
# ============================================================================
print("\n" + "="*80)
print("PHASE 4: 二维水动力模拟 (2D Simulation)")
print("="*80)

# 求解器配置
config = SolverConfig(
    t_end=30.0,  # 模拟30秒
    cfl=0.5,
    output_interval=0.5,  # 每0.5秒输出一次 (60帧)
    output_dir=f"{output_base}/simulation",
    manning_n=0.03,
    print_progress=True,
    use_numba=False  # 中等规模用NumPy即可
)

print(f"\n[4.1] 求解器配置")
print(f"  模拟时间: {config.t_end} s")
print(f"  CFL数: {config.cfl}")
print(f"  输出间隔: {config.output_interval} s")
print(f"  Manning系数: {config.manning_n}")

# 初始化求解器
solver = ShallowWaterSolver(mesh, terrain, config)
solver.set_initial_conditions(h0, u0, v0)

print(f"\n[4.2] 开始模拟...")
start_time = time.time()

# 运行模拟
result = solver.solve()

end_time = time.time()
computation_time = end_time - start_time

print(f"\n[4.3] 模拟完成!")
print(f"  时间步数: {result['steps']}")
print(f"  模拟时间: {result['simulated_time']:.2f} s")
print(f"  计算耗时: {computation_time:.2f} s")
print(f"  速度因子: {result['simulated_time']/computation_time:.2f}x 实时")
print(f"  质量守恒: {result['mass_error']:+.6f}%")

# ============================================================================
# PHASE 5: 高级可视化 (Advanced Visualization)
# ============================================================================
print("\n" + "="*80)
print("PHASE 5: 高级可视化 (Advanced Visualization)")
print("="*80)

# 准备最终时刻数据
h_final = solver.h
u_final = solver.u
v_final = solver.v
V_final = np.sqrt(u_final**2 + v_final**2)  # 流速大小
eta_final = h_final + terrain  # 水位

# 创建自定义颜色映射
def create_water_colormap():
    """创建专业水深颜色映射"""
    colors = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', 
              '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b']
    return LinearSegmentedColormap.from_list('water_depth', colors)

def create_velocity_colormap():
    """创建流速颜色映射"""
    return plt.cm.jet

水深色 = create_water_colormap()
流速色 = create_velocity_colormap()

# ============================================================================
# 图1: 水深场 (Water Depth Field)
# ============================================================================
print(f"\n[5.1] 绘制水深场...")
fig1, ax1 = plt.subplots(figsize=(16, 8))

# 填充等值线
levels_h = np.linspace(0, h_final.max(), 20)
cf1 = ax1.contourf(X, Y, h_final, levels=levels_h, cmap=水深色, extend='max')

# 等值线
contours = ax1.contour(X, Y, h_final, levels=10, colors='black', 
                        linewidths=0.8, alpha=0.4)
ax1.clabel(contours, inline=True, fontsize=9, fmt='%.1f')

# 标注溃坝位置
ax1.axvline(x=dam_location, color='red', linestyle='--', linewidth=2, 
            label='Dam Location', alpha=0.7)

ax1.set_xlabel('X (m)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
ax1.set_title(f'Water Depth Field at t={config.t_end:.1f}s', 
              fontsize=14, fontweight='bold')
ax1.legend(loc='upper right', fontsize=11)
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

cbar1 = plt.colorbar(cf1, ax=ax1, orientation='vertical', pad=0.02)
cbar1.set_label('Water Depth (m)', fontsize=12, fontweight='bold')

plt.tight_layout()
output_file = f"{output_base}/visualization/water_depth_field.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"  已保存: {output_file}")
plt.close()

# ============================================================================
# 图2: 流速大小场 (Velocity Magnitude)
# ============================================================================
print(f"\n[5.2] 绘制流速大小场...")
fig2, ax2 = plt.subplots(figsize=(16, 8))

# 只显示有水的区域
V_plot = V_final.copy()
V_plot[h_final < 0.01] = np.nan

levels_v = np.linspace(0, V_plot[~np.isnan(V_plot)].max(), 20)
cf2 = ax2.contourf(X, Y, V_plot, levels=levels_v, cmap=流速色, extend='max')

# 溃坝位置
ax2.axvline(x=dam_location, color='white', linestyle='--', linewidth=2, 
            label='Dam Location', alpha=0.8)

ax2.set_xlabel('X (m)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
ax2.set_title(f'Velocity Magnitude at t={config.t_end:.1f}s', 
              fontsize=14, fontweight='bold')
ax2.legend(loc='upper right', fontsize=11)
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.5, color='white')

cbar2 = plt.colorbar(cf2, ax=ax2, orientation='vertical', pad=0.02)
cbar2.set_label('Velocity (m/s)', fontsize=12, fontweight='bold')

plt.tight_layout()
output_file = f"{output_base}/visualization/velocity_magnitude.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"  已保存: {output_file}")
plt.close()

# ============================================================================
# 图3: 流速矢量场 (Velocity Vectors)
# ============================================================================
print(f"\n[5.3] 绘制流速矢量场...")
fig3, ax3 = plt.subplots(figsize=(16, 8))

# 背景：水深
cf3 = ax3.contourf(X, Y, h_final, levels=20, cmap=水深色, alpha=0.7)

# 矢量箭头（降采样以提高可读性）
step = 10  # 每10个单元一个箭头
X_quiver = X[::step, ::step]
Y_quiver = Y[::step, ::step]
U_quiver = u_final[::step, ::step]
V_quiver = v_final[::step, ::step]
V_quiver_mag = V_final[::step, ::step]

# 根据流速大小着色的箭头
quiver = ax3.quiver(X_quiver, Y_quiver, U_quiver, V_quiver, V_quiver_mag,
                     cmap=流速色, scale=100, width=0.003, alpha=0.8,
                     headwidth=4, headlength=5)

# 溃坝位置
ax3.axvline(x=dam_location, color='red', linestyle='--', linewidth=2, 
            label='Dam Location', alpha=0.7)

ax3.set_xlabel('X (m)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
ax3.set_title(f'Velocity Vector Field at t={config.t_end:.1f}s', 
              fontsize=14, fontweight='bold')
ax3.legend(loc='upper right', fontsize=11)
ax3.set_aspect('equal')
ax3.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

cbar3 = plt.colorbar(quiver, ax=ax3, orientation='vertical', pad=0.02)
cbar3.set_label('Velocity Magnitude (m/s)', fontsize=12, fontweight='bold')

plt.tight_layout()
output_file = f"{output_base}/visualization/velocity_vectors.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"  已保存: {output_file}")
plt.close()

# ============================================================================
# 图4: Froude数场 (Froude Number)
# ============================================================================
print(f"\n[5.4] 绘制Froude数场...")
fig4, ax4 = plt.subplots(figsize=(16, 8))

# 计算Froude数
g = 9.81
Fr = np.zeros_like(h_final)
wet = h_final > 0.01
Fr[wet] = V_final[wet] / np.sqrt(g * h_final[wet])
Fr[~wet] = np.nan

# 按流态分类着色
# Fr < 0.5: 缓流, 0.5-1.5: 临界, Fr > 1.5: 急流
colors_fr = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0',  # 缓流（蓝）
             '#fddbc7', '#f4a582', '#d6604d', '#b2182b']  # 急流（红）
froude_cmap = LinearSegmentedColormap.from_list('froude', colors_fr)

levels_fr = [0, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 2.0, 3.0]
cf4 = ax4.contourf(X, Y, Fr, levels=levels_fr, cmap=froude_cmap, extend='max')

# 临界线 Fr=1
ax4.contour(X, Y, Fr, levels=[1.0], colors='black', linewidths=2, 
            linestyles='--', alpha=0.8)

ax4.axvline(x=dam_location, color='white', linestyle='--', linewidth=2, 
            label='Dam Location', alpha=0.8)

ax4.set_xlabel('X (m)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
ax4.set_title(f'Froude Number Field at t={config.t_end:.1f}s', 
              fontsize=14, fontweight='bold')
ax4.legend(loc='upper right', fontsize=11)
ax4.set_aspect('equal')
ax4.grid(True, alpha=0.3, linestyle=':', linewidth=0.5, color='white')

cbar4 = plt.colorbar(cf4, ax=ax4, orientation='vertical', pad=0.02)
cbar4.set_label('Froude Number', fontsize=12, fontweight='bold')
cbar4.ax.axhline(y=1.0, color='black', linewidth=2, linestyle='--')
cbar4.ax.text(1.5, 1.0, 'Critical', fontsize=10, va='center')

plt.tight_layout()
output_file = f"{output_base}/visualization/froude_number.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"  已保存: {output_file}")
plt.close()

# ============================================================================
# 图5: 洪水危险度 (Flood Hazard)
# ============================================================================
print(f"\n[5.5] 绘制洪水危险度...")
fig5, ax5 = plt.subplots(figsize=(16, 8))

# 计算危险度 H = h × V
hazard = h_final * V_final
hazard[h_final < 0.01] = np.nan

# 危险度分级颜色
colors_hazard = ['#1a9850', '#91cf60', '#d9ef8b', '#fee08b',  # 低危（绿-黄）
                 '#fc8d59', '#e66101', '#d73027', '#a50026']  # 高危（橙-红）
hazard_cmap = LinearSegmentedColormap.from_list('hazard', colors_hazard)

levels_hazard = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0]
cf5 = ax5.contourf(X, Y, hazard, levels=levels_hazard, cmap=hazard_cmap, extend='max')

ax5.axvline(x=dam_location, color='white', linestyle='--', linewidth=2, 
            label='Dam Location', alpha=0.8)

ax5.set_xlabel('X (m)', fontsize=12, fontweight='bold')
ax5.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
ax5.set_title(f'Flood Hazard (h×V) at t={config.t_end:.1f}s', 
              fontsize=14, fontweight='bold')
ax5.legend(loc='upper right', fontsize=11)
ax5.set_aspect('equal')
ax5.grid(True, alpha=0.3, linestyle=':', linewidth=0.5, color='white')

cbar5 = plt.colorbar(cf5, ax=ax5, orientation='vertical', pad=0.02)
cbar5.set_label('Hazard Level (m^2/s)', fontsize=12, fontweight='bold')

# 标注危险等级
hazard_levels_text = [
    (0.25, 'Low', 'green'),
    (1.25, 'Medium', 'orange'),
    (2.5, 'High', 'red'),
    (5.0, 'Extreme', 'darkred')
]
for val, text, color in hazard_levels_text:
    cbar5.ax.text(2.5, val, text, fontsize=9, color=color, 
                  fontweight='bold', va='center')

plt.tight_layout()
output_file = f"{output_base}/visualization/flood_hazard.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"  已保存: {output_file}")
plt.close()

# ============================================================================
# 图6: 综合对比图 (Comprehensive Comparison)
# ============================================================================
print(f"\n[5.6] 绘制综合对比图...")
fig6, axes = plt.subplots(2, 3, figsize=(20, 12))

# 子图1: 初始水深
ax = axes[0, 0]
cf = ax.contourf(X, Y, h0, levels=20, cmap=水深色)
ax.set_title('Initial Condition (t=0s)', fontsize=12, fontweight='bold')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_aspect('equal')
plt.colorbar(cf, ax=ax, label='Depth (m)')

# 子图2: 最终水深
ax = axes[0, 1]
cf = ax.contourf(X, Y, h_final, levels=20, cmap=水深色)
ax.set_title(f'Final Depth (t={config.t_end}s)', fontsize=12, fontweight='bold')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_aspect('equal')
plt.colorbar(cf, ax=ax, label='Depth (m)')

# 子图3: 地形
ax = axes[0, 2]
cf = ax.contourf(X, Y, terrain, levels=20, cmap='terrain')
ax.set_title('Terrain Elevation', fontsize=12, fontweight='bold')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_aspect('equal')
plt.colorbar(cf, ax=ax, label='Elevation (m)')

# 子图4: 流速大小
ax = axes[1, 0]
V_plot = V_final.copy()
V_plot[h_final < 0.01] = np.nan
cf = ax.contourf(X, Y, V_plot, levels=20, cmap=流速色)
ax.set_title('Velocity Magnitude', fontsize=12, fontweight='bold')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_aspect('equal')
plt.colorbar(cf, ax=ax, label='Velocity (m/s)')

# 子图5: Froude数
ax = axes[1, 1]
cf = ax.contourf(X, Y, Fr, levels=levels_fr, cmap=froude_cmap, extend='max')
ax.contour(X, Y, Fr, levels=[1.0], colors='black', linewidths=2, linestyles='--')
ax.set_title('Froude Number', fontsize=12, fontweight='bold')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_aspect('equal')
plt.colorbar(cf, ax=ax, label='Fr')

# 子图6: 洪水危险度
ax = axes[1, 2]
cf = ax.contourf(X, Y, hazard, levels=levels_hazard, cmap=hazard_cmap, extend='max')
ax.set_title('Flood Hazard', fontsize=12, fontweight='bold')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_aspect('equal')
plt.colorbar(cf, ax=ax, label='h×V (m^2/s)')

plt.suptitle('HydroSIS-2D Full Workflow: Dam Break Simulation', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()

output_file = f"{output_base}/visualization/comprehensive_comparison.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"  已保存: {output_file}")
plt.close()

# ============================================================================
# 最终总结
# ============================================================================
print("\n" + "="*80)
print("完整工作流测试总结")
print("="*80)

print(f"\n[OK] Phase 1: Mesh Generation")
print(f"   - Grid Size: {mesh.ncells:,} cells")
print(f"   - Quality: {metrics.uniformity_score:.3f}")

print(f"\n[OK] Phase 2: Terrain Generation")
print(f"   - Elevation Range: [{terrain.min():.2f}, {terrain.max():.2f}] m")

print(f"\n[OK] Phase 3: Initial Conditions")
print(f"   - Initial Volume: {initial_volume/1000:.2f} thousand m^3")

print(f"\n[OK] Phase 4: 2D Simulation")
print(f"   - Time Steps: {result['steps']}")
print(f"   - Mass Conservation: {result['mass_error']:+.6f}%")
print(f"   - Efficiency: {result['simulated_time']/computation_time:.2f}x realtime")

print(f"\n[OK] Phase 5: Advanced Visualization")
print(f"   - Water Depth Field")
print(f"   - Velocity Magnitude Field")
print(f"   - Velocity Vector Field")
print(f"   - Froude Number Field")
print(f"   - Flood Hazard Map")
print(f"   - Comprehensive Comparison")

print(f"\n" + "="*80)
print("SUCCESS! Full Workflow Test Completed!")
print(f"All outputs saved to: {output_base}/")
print("="*80)

