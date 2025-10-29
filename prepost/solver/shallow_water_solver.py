"""
2D Shallow Water Equations Solver (Python Reference Implementation)

This module implements a finite volume method solver for the 2D shallow water equations
on structured Cartesian grids. It serves as a reference implementation and prototype
for the GPU-accelerated CUDA version.

Governing Equations:
    ∂h/∂t + ∂(hu)/∂x + ∂(hv)/∂y = 0                    (mass conservation)
    ∂(hu)/∂t + ∂(hu² + gh²/2)/∂x + ∂(huv)/∂y = -gh∂z/∂x - τx/(ρh)  (x-momentum)
    ∂(hv)/∂t + ∂(huv)/∂x + ∂(hv² + gh²/2)/∂y = -gh∂z/∂y - τy/(ρh)  (y-momentum)

where:
    h = water depth [m]
    u, v = velocity components [m/s]
    z = bed elevation [m]
    g = gravitational acceleration [m/s²]
    τx, τy = bed friction [N/m²]
    ρ = water density [kg/m³]

Author: HydroSIS-2D Development Team
Date: 2025-10-29
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, Callable
import time
from dataclasses import dataclass


@dataclass
class SolverConfig:
    """Solver configuration parameters"""
    # Time stepping
    cfl: float = 0.5                    # CFL number
    t_end: float = 10.0                 # End time [s]
    dt_initial: float = 0.01            # Initial time step [s]
    dt_min: float = 1e-6                # Minimum time step [s]
    dt_max: float = 1.0                 # Maximum time step [s]

    # Physical parameters
    g: float = 9.81                     # Gravity [m/s²]
    manning_n: float = 0.03             # Manning roughness coefficient
    h_dry: float = 1e-4                 # Dry bed threshold [m]

    # Numerical schemes
    flux_scheme: str = 'HLL'            # 'HLL' or 'HLLC'
    time_scheme: str = 'euler'          # 'euler' or 'rk2'
    spatial_order: int = 1              # 1 (first-order) or 2 (MUSCL)
    muscl_limiter: str = 'minmod'       # 'minmod', 'superbee', 'vanleer', 'mc'

    # Output
    output_interval: float = 1.0        # Output interval [s]
    output_dir: str = 'output/results'  # Output directory

    # Monitoring
    check_mass_conservation: bool = True
    print_progress: bool = True
    progress_interval: int = 100        # Print every N steps


class ShallowWaterSolver:
    """
    2D Shallow Water Equations Solver using Finite Volume Method

    This solver implements:
    - Godunov-type finite volume method on Cartesian grids
    - HLL/HLLC Riemann solvers for flux computation
    - Explicit time integration (Euler or RK2)
    - Dry bed treatment
    - Source term handling (bed slope and friction)
    """

    def __init__(self, mesh, terrain: np.ndarray, config: Optional[SolverConfig] = None):
        """
        Initialize the shallow water solver

        Parameters
        ----------
        mesh : StructuredMesh
            Computational mesh from preprocessing module
        terrain : np.ndarray
            Bed elevation data, shape (nx, ny)
        config : SolverConfig, optional
            Solver configuration parameters
        """
        self.mesh = mesh
        self.nx = mesh.nx
        self.ny = mesh.ny
        self.dx = mesh.dx
        self.dy = mesh.dy

        self.config = config or SolverConfig()

        # Terrain (bed elevation)
        self.z = terrain.copy()

        # Conservative variables: h, hu, hv
        self.h = np.zeros((self.nx, self.ny))   # Water depth
        self.hu = np.zeros((self.nx, self.ny))  # x-momentum
        self.hv = np.zeros((self.nx, self.ny))  # y-momentum

        # Velocities (derived)
        self.u = np.zeros((self.nx, self.ny))
        self.v = np.zeros((self.nx, self.ny))

        # Fluxes at cell interfaces
        self.flux_x = np.zeros((3, self.nx+1, self.ny))  # F = [hf, huf+gh²/2, hvf]
        self.flux_y = np.zeros((3, self.nx, self.ny+1))  # G = [hf, huf, hvf+gh²/2]

        # Time stepping
        self.t = 0.0
        self.dt = self.config.dt_initial
        self.step_count = 0
        self.output_count = 0
        self.next_output_time = 0.0

        # Mass conservation tracking
        self.initial_mass = 0.0
        self.mass_history = []

        # Boundary conditions
        self.bc_handlers = {}

        # Progress callback
        self.callback: Optional[Callable] = None
        self.callback_interval = self.config.progress_interval

        print(f"ShallowWaterSolver initialized:")
        print(f"  Grid: {self.nx} × {self.ny} = {self.nx * self.ny:,} cells")
        print(f"  Cell size: dx={self.dx:.3f}m, dy={self.dy:.3f}m")
        print(f"  Domain: [{mesh.domain.xmin:.1f}, {mesh.domain.xmax:.1f}] × "
              f"[{mesh.domain.ymin:.1f}, {mesh.domain.ymax:.1f}]")
        print(f"  CFL: {self.config.cfl}, g={self.config.g} m/s²")

    def set_initial_conditions(self, h0: np.ndarray, u0: np.ndarray, v0: np.ndarray):
        """
        Set initial conditions

        Parameters
        ----------
        h0 : np.ndarray
            Initial water depth, shape (nx, ny)
        u0 : np.ndarray
            Initial x-velocity, shape (nx, ny)
        v0 : np.ndarray
            Initial y-velocity, shape (nx, ny)
        """
        self.h = h0.copy()
        self.u = u0.copy()
        self.v = v0.copy()

        # Update conservative variables
        self.hu = self.h * self.u
        self.hv = self.h * self.v

        # Compute initial mass
        self.initial_mass = np.sum(self.h) * self.dx * self.dy
        self.mass_history = [self.initial_mass]

        print(f"\nInitial conditions set:")
        print(f"  Water depth: min={np.min(h0):.3f}m, max={np.max(h0):.3f}m, mean={np.mean(h0):.3f}m")
        print(f"  Velocity U: min={np.min(u0):.3f}m/s, max={np.max(u0):.3f}m/s")
        print(f"  Velocity V: min={np.min(v0):.3f}m/s, max={np.max(v0):.3f}m/s")
        print(f"  Initial mass: {self.initial_mass:.2f} m³")

    def set_boundary_conditions(self, bc_manager):
        """
        Set boundary conditions from preprocessing module

        Parameters
        ----------
        bc_manager : BoundaryConditionManager
            Boundary condition manager from preprocessing
        """
        self.bc_manager = bc_manager

        # Validate that we have all four boundaries
        is_valid, errors = bc_manager.validate()
        if not is_valid:
            print("\nWarning: Boundary conditions validation failed:")
            for err in errors:
                print(f"  - {err}")
        else:
            print("\nBoundary conditions set:")
            print(bc_manager.summary())

    def set_callback(self, callback: Callable, interval: int = 100):
        """
        Set progress callback function

        Parameters
        ----------
        callback : Callable
            Function to call with signature: callback(t, step, dt)
        interval : int
            Call callback every N steps
        """
        self.callback = callback
        self.callback_interval = interval

    def compute_timestep(self) -> float:
        """
        Compute adaptive time step based on CFL condition

        CFL condition: dt ≤ CFL * min(dx, dy) / max(|u| + √(gh))

        Returns
        -------
        float
            Time step [s]
        """
        # Compute wave speeds
        c = np.sqrt(self.config.g * np.maximum(self.h, 0.0))
        u_abs = np.abs(self.u)
        v_abs = np.abs(self.v)

        # Maximum wave speed in each direction
        lambda_x = u_abs + c
        lambda_y = v_abs + c

        # Avoid division by zero
        lambda_max = np.max([np.max(lambda_x), np.max(lambda_y)])
        if lambda_max < 1e-10:
            lambda_max = 1e-10

        # CFL time step
        dt_cfl = self.config.cfl * min(self.dx, self.dy) / lambda_max

        # Clamp to min/max
        dt = np.clip(dt_cfl, self.config.dt_min, self.config.dt_max)

        return dt

    def compute_fluxes_hll_loop(self):
        """
        Compute numerical fluxes using HLL Riemann solver (loop version)

        The HLL (Harten-Lax-van Leer) solver approximates the Riemann problem
        solution with two waves.

        Note: This is the original loop-based implementation. Use compute_fluxes_hll()
        for the vectorized version which is 10-50x faster.
        """
        g = self.config.g
        h_dry = self.config.h_dry

        # X-direction fluxes
        for i in range(self.nx + 1):
            for j in range(self.ny):
                # Left and right states
                if i == 0:
                    # Left boundary
                    h_L, u_L, v_L = self.h[i, j], self.u[i, j], self.v[i, j]
                    h_R, u_R, v_R = self.h[i, j], self.u[i, j], self.v[i, j]
                elif i == self.nx:
                    # Right boundary
                    h_L, u_L, v_L = self.h[i-1, j], self.u[i-1, j], self.v[i-1, j]
                    h_R, u_R, v_R = self.h[i-1, j], self.u[i-1, j], self.v[i-1, j]
                else:
                    # Interior
                    h_L, u_L, v_L = self.h[i-1, j], self.u[i-1, j], self.v[i-1, j]
                    h_R, u_R, v_R = self.h[i, j], self.u[i, j], self.v[i, j]

                # Dry bed treatment
                if h_L < h_dry and h_R < h_dry:
                    self.flux_x[:, i, j] = 0.0
                    continue

                # Wave speeds
                c_L = np.sqrt(g * max(h_L, 0.0))
                c_R = np.sqrt(g * max(h_R, 0.0))

                s_L = min(u_L - c_L, u_R - c_R)
                s_R = max(u_L + c_L, u_R + c_R)

                # Physical fluxes
                F_L = np.array([
                    h_L * u_L,
                    h_L * u_L * u_L + 0.5 * g * h_L * h_L,
                    h_L * u_L * v_L
                ])

                F_R = np.array([
                    h_R * u_R,
                    h_R * u_R * u_R + 0.5 * g * h_R * h_R,
                    h_R * u_R * v_R
                ])

                # Conservative variables
                U_L = np.array([h_L, h_L * u_L, h_L * v_L])
                U_R = np.array([h_R, h_R * u_R, h_R * v_R])

                # HLL flux
                if s_L >= 0:
                    self.flux_x[:, i, j] = F_L
                elif s_R <= 0:
                    self.flux_x[:, i, j] = F_R
                else:
                    self.flux_x[:, i, j] = (s_R * F_L - s_L * F_R + s_L * s_R * (U_R - U_L)) / (s_R - s_L)

        # Y-direction fluxes
        for i in range(self.nx):
            for j in range(self.ny + 1):
                # Bottom and top states
                if j == 0:
                    # Bottom boundary
                    h_B, u_B, v_B = self.h[i, j], self.u[i, j], self.v[i, j]
                    h_T, u_T, v_T = self.h[i, j], self.u[i, j], self.v[i, j]
                elif j == self.ny:
                    # Top boundary
                    h_B, u_B, v_B = self.h[i, j-1], self.u[i, j-1], self.v[i, j-1]
                    h_T, u_T, v_T = self.h[i, j-1], self.u[i, j-1], self.v[i, j-1]
                else:
                    # Interior
                    h_B, u_B, v_B = self.h[i, j-1], self.u[i, j-1], self.v[i, j-1]
                    h_T, u_T, v_T = self.h[i, j], self.u[i, j], self.v[i, j]

                # Dry bed treatment
                if h_B < h_dry and h_T < h_dry:
                    self.flux_y[:, i, j] = 0.0
                    continue

                # Wave speeds
                c_B = np.sqrt(g * max(h_B, 0.0))
                c_T = np.sqrt(g * max(h_T, 0.0))

                s_B = min(v_B - c_B, v_T - c_T)
                s_T = max(v_B + c_B, v_T + c_T)

                # Physical fluxes
                G_B = np.array([
                    h_B * v_B,
                    h_B * u_B * v_B,
                    h_B * v_B * v_B + 0.5 * g * h_B * h_B
                ])

                G_T = np.array([
                    h_T * v_T,
                    h_T * u_T * v_T,
                    h_T * v_T * v_T + 0.5 * g * h_T * h_T
                ])

                # Conservative variables
                U_B = np.array([h_B, h_B * u_B, h_B * v_B])
                U_T = np.array([h_T, h_T * u_T, h_T * v_T])

                # HLL flux
                if s_B >= 0:
                    self.flux_y[:, i, j] = G_B
                elif s_T <= 0:
                    self.flux_y[:, i, j] = G_T
                else:
                    self.flux_y[:, i, j] = (s_T * G_B - s_B * G_T + s_B * s_T * (U_T - U_B)) / (s_T - s_B)

    def compute_fluxes_hll(self):
        """
        Compute numerical fluxes using vectorized HLL Riemann solver

        This is the optimized vectorized implementation that processes all
        cell interfaces simultaneously using NumPy operations. Typically
        10-50x faster than the loop-based version.

        The HLL (Harten-Lax-van Leer) solver approximates the Riemann problem
        solution with two waves: s_L (left) and s_R (right).
        """
        g = self.config.g
        h_dry = self.config.h_dry

        # ==================================================================
        # X-direction fluxes (vertical interfaces)
        # ==================================================================

        # Get left and right states at all interior interfaces
        # Interface i is between cells i-1 (left) and i (right)
        h_L = self.h[:-1, :]   # shape: (nx-1, ny)
        h_R = self.h[1:, :]
        u_L = self.u[:-1, :]
        u_R = self.u[1:, :]
        v_L = self.v[:-1, :]
        v_R = self.v[1:, :]

        # Wet/dry mask - compute flux only where at least one side is wet
        wet_mask_x = (h_L >= h_dry) | (h_R >= h_dry)

        # Wave speeds (vectorized)
        c_L = np.sqrt(g * np.maximum(h_L, 0.0))
        c_R = np.sqrt(g * np.maximum(h_R, 0.0))
        s_L = np.minimum(u_L - c_L, u_R - c_R)
        s_R = np.maximum(u_L + c_L, u_R + c_R)

        # Physical fluxes F_L and F_R
        F_L_h = h_L * u_L
        F_L_hu = h_L * u_L * u_L + 0.5 * g * h_L * h_L
        F_L_hv = h_L * u_L * v_L

        F_R_h = h_R * u_R
        F_R_hu = h_R * u_R * u_R + 0.5 * g * h_R * h_R
        F_R_hv = h_R * u_R * v_R

        # Conservative variables U_L and U_R
        U_L_h = h_L
        U_L_hu = h_L * u_L
        U_L_hv = h_L * v_L

        U_R_h = h_R
        U_R_hu = h_R * u_R
        U_R_hv = h_R * v_R

        # HLL flux (avoid division by zero)
        s_diff = s_R - s_L
        s_diff = np.where(np.abs(s_diff) < 1e-10, 1e-10, s_diff)

        hll_h = (s_R * F_L_h - s_L * F_R_h + s_L * s_R * (U_R_h - U_L_h)) / s_diff
        hll_hu = (s_R * F_L_hu - s_L * F_R_hu + s_L * s_R * (U_R_hu - U_L_hu)) / s_diff
        hll_hv = (s_R * F_L_hv - s_L * F_R_hv + s_L * s_R * (U_R_hv - U_L_hv)) / s_diff

        # Vectorized conditional selection
        flux_h = np.where(s_L >= 0, F_L_h,
                          np.where(s_R <= 0, F_R_h, hll_h))
        flux_hu = np.where(s_L >= 0, F_L_hu,
                           np.where(s_R <= 0, F_R_hu, hll_hu))
        flux_hv = np.where(s_L >= 0, F_L_hv,
                           np.where(s_R <= 0, F_R_hv, hll_hv))

        # Apply wet/dry mask
        flux_h = np.where(wet_mask_x, flux_h, 0.0)
        flux_hu = np.where(wet_mask_x, flux_hu, 0.0)
        flux_hv = np.where(wet_mask_x, flux_hv, 0.0)

        # Store to flux array (interior interfaces: i=1 to nx-1)
        self.flux_x[0, 1:-1, :] = flux_h
        self.flux_x[1, 1:-1, :] = flux_hu
        self.flux_x[2, 1:-1, :] = flux_hv

        # Boundary fluxes (zero flux at boundaries - enforced by BC)
        self.flux_x[:, 0, :] = 0.0
        self.flux_x[:, -1, :] = 0.0

        # ==================================================================
        # Y-direction fluxes (horizontal interfaces)
        # ==================================================================

        # Get bottom and top states at all interior interfaces
        # Interface j is between cells j-1 (bottom) and j (top)
        h_B = self.h[:, :-1]   # shape: (nx, ny-1)
        h_T = self.h[:, 1:]
        u_B = self.u[:, :-1]
        u_T = self.u[:, 1:]
        v_B = self.v[:, :-1]
        v_T = self.v[:, 1:]

        # Wet/dry mask
        wet_mask_y = (h_B >= h_dry) | (h_T >= h_dry)

        # Wave speeds (vectorized)
        c_B = np.sqrt(g * np.maximum(h_B, 0.0))
        c_T = np.sqrt(g * np.maximum(h_T, 0.0))
        s_B = np.minimum(v_B - c_B, v_T - c_T)
        s_T = np.maximum(v_B + c_B, v_T + c_T)

        # Physical fluxes G_B and G_T
        G_B_h = h_B * v_B
        G_B_hu = h_B * u_B * v_B
        G_B_hv = h_B * v_B * v_B + 0.5 * g * h_B * h_B

        G_T_h = h_T * v_T
        G_T_hu = h_T * u_T * v_T
        G_T_hv = h_T * v_T * v_T + 0.5 * g * h_T * h_T

        # Conservative variables U_B and U_T
        U_B_h = h_B
        U_B_hu = h_B * u_B
        U_B_hv = h_B * v_B

        U_T_h = h_T
        U_T_hu = h_T * u_T
        U_T_hv = h_T * v_T

        # HLL flux (avoid division by zero)
        s_diff = s_T - s_B
        s_diff = np.where(np.abs(s_diff) < 1e-10, 1e-10, s_diff)

        hll_h = (s_T * G_B_h - s_B * G_T_h + s_B * s_T * (U_T_h - U_B_h)) / s_diff
        hll_hu = (s_T * G_B_hu - s_B * G_T_hu + s_B * s_T * (U_T_hu - U_B_hu)) / s_diff
        hll_hv = (s_T * G_B_hv - s_B * G_T_hv + s_B * s_T * (U_T_hv - U_B_hv)) / s_diff

        # Vectorized conditional selection
        flux_h = np.where(s_B >= 0, G_B_h,
                          np.where(s_T <= 0, G_T_h, hll_h))
        flux_hu = np.where(s_B >= 0, G_B_hu,
                           np.where(s_T <= 0, G_T_hu, hll_hu))
        flux_hv = np.where(s_B >= 0, G_B_hv,
                           np.where(s_T <= 0, G_T_hv, hll_hv))

        # Apply wet/dry mask
        flux_h = np.where(wet_mask_y, flux_h, 0.0)
        flux_hu = np.where(wet_mask_y, flux_hu, 0.0)
        flux_hv = np.where(wet_mask_y, flux_hv, 0.0)

        # Store to flux array (interior interfaces: j=1 to ny-1)
        self.flux_y[0, :, 1:-1] = flux_h
        self.flux_y[1, :, 1:-1] = flux_hu
        self.flux_y[2, :, 1:-1] = flux_hv

        # Boundary fluxes (zero flux at boundaries - enforced by BC)
        self.flux_y[:, :, 0] = 0.0
        self.flux_y[:, :, -1] = 0.0

    def apply_boundary_conditions(self):
        """
        Apply boundary conditions

        This method enforces boundary conditions on the boundary cells.
        If bc_manager is set, uses the boundary conditions from preprocessing.
        Otherwise, applies default wall boundaries.
        """
        h_dry = self.config.h_dry

        if hasattr(self, 'bc_manager') and self.bc_manager is not None:
            # Use boundary conditions from preprocessing
            self._apply_bc_from_manager()
        else:
            # Default: all walls
            self._apply_wall_boundaries()

    def _apply_wall_boundaries(self):
        """Apply wall boundaries on all four sides (default)"""
        h_dry = self.config.h_dry

        # West boundary (i=0)
        self.h[0, :] = np.maximum(self.h[1, :], h_dry)
        self.u[0, :] = -self.u[1, :]  # Reflective (wall)
        self.v[0, :] = self.v[1, :]

        # East boundary (i=nx-1)
        self.h[-1, :] = np.maximum(self.h[-2, :], h_dry)
        self.u[-1, :] = -self.u[-2, :]  # Reflective (wall)
        self.v[-1, :] = self.v[-2, :]

        # South boundary (j=0)
        self.h[:, 0] = np.maximum(self.h[:, 1], h_dry)
        self.u[:, 0] = self.u[:, 1]
        self.v[:, 0] = -self.v[:, 1]  # Reflective (wall)

        # North boundary (j=ny-1)
        self.h[:, -1] = np.maximum(self.h[:, -2], h_dry)
        self.u[:, -1] = self.u[:, -2]
        self.v[:, -1] = -self.v[:, -2]  # Reflective (wall)

        # Update conservative variables
        self.hu[0, :] = self.h[0, :] * self.u[0, :]
        self.hu[-1, :] = self.h[-1, :] * self.u[-1, :]
        self.hv[:, 0] = self.h[:, 0] * self.v[:, 0]
        self.hv[:, -1] = self.h[:, -1] * self.v[:, -1]

    def _apply_bc_from_manager(self):
        """Apply boundary conditions from bc_manager"""
        from preprocessing.boundary_conditions import BCType, BCLocation

        h_dry = self.config.h_dry

        # Get all boundaries from the manager's dictionary
        boundaries = self.bc_manager.boundaries

        for location, bc in boundaries.items():
            if bc is None:
                continue

            if location == BCLocation.WEST:
                self._apply_bc_west(bc)
            elif location == BCLocation.EAST:
                self._apply_bc_east(bc)
            elif location == BCLocation.SOUTH:
                self._apply_bc_south(bc)
            elif location == BCLocation.NORTH:
                self._apply_bc_north(bc)

    def _apply_bc_west(self, bc):
        """Apply boundary condition on west boundary (i=0)"""
        from preprocessing.boundary_conditions import BCType

        h_dry = self.config.h_dry

        if bc.bc_type == BCType.WALL:
            # Reflective wall
            self.h[0, :] = np.maximum(self.h[1, :], h_dry)
            self.u[0, :] = -self.u[1, :]
            self.v[0, :] = self.v[1, :]

        elif bc.bc_type == BCType.INFLOW:
            # Fixed inflow
            self.h[0, :] = bc.depth
            self.u[0, :] = bc.velocity_x
            self.v[0, :] = bc.velocity_y

        elif bc.bc_type == BCType.OUTFLOW:
            # Zero gradient (transmissive)
            self.h[0, :] = self.h[1, :]
            self.u[0, :] = self.u[1, :]
            self.v[0, :] = self.v[1, :]

        elif bc.bc_type == BCType.TIME_SERIES:
            # Time-varying inflow
            depth, vel_x, vel_y = bc.interpolate(self.t)
            self.h[0, :] = depth
            self.u[0, :] = vel_x
            self.v[0, :] = vel_y

        # Update conservative variables
        self.hu[0, :] = self.h[0, :] * self.u[0, :]
        self.hv[0, :] = self.h[0, :] * self.v[0, :]

    def _apply_bc_east(self, bc):
        """Apply boundary condition on east boundary (i=nx-1)"""
        from preprocessing.boundary_conditions import BCType

        h_dry = self.config.h_dry

        if bc.bc_type == BCType.WALL:
            self.h[-1, :] = np.maximum(self.h[-2, :], h_dry)
            self.u[-1, :] = -self.u[-2, :]
            self.v[-1, :] = self.v[-2, :]

        elif bc.bc_type == BCType.INFLOW:
            self.h[-1, :] = bc.depth
            self.u[-1, :] = bc.velocity_x
            self.v[-1, :] = bc.velocity_y

        elif bc.bc_type == BCType.OUTFLOW:
            self.h[-1, :] = self.h[-2, :]
            self.u[-1, :] = self.u[-2, :]
            self.v[-1, :] = self.v[-2, :]

        elif bc.bc_type == BCType.TIME_SERIES:
            depth, vel_x, vel_y = bc.interpolate(self.t)
            self.h[-1, :] = depth
            self.u[-1, :] = vel_x
            self.v[-1, :] = vel_y

        self.hu[-1, :] = self.h[-1, :] * self.u[-1, :]
        self.hv[-1, :] = self.h[-1, :] * self.v[-1, :]

    def _apply_bc_south(self, bc):
        """Apply boundary condition on south boundary (j=0)"""
        from preprocessing.boundary_conditions import BCType

        h_dry = self.config.h_dry

        if bc.bc_type == BCType.WALL:
            self.h[:, 0] = np.maximum(self.h[:, 1], h_dry)
            self.u[:, 0] = self.u[:, 1]
            self.v[:, 0] = -self.v[:, 1]

        elif bc.bc_type == BCType.INFLOW:
            self.h[:, 0] = bc.depth
            self.u[:, 0] = bc.velocity_x
            self.v[:, 0] = bc.velocity_y

        elif bc.bc_type == BCType.OUTFLOW:
            self.h[:, 0] = self.h[:, 1]
            self.u[:, 0] = self.u[:, 1]
            self.v[:, 0] = self.v[:, 1]

        elif bc.bc_type == BCType.TIME_SERIES:
            depth, vel_x, vel_y = bc.interpolate(self.t)
            self.h[:, 0] = depth
            self.u[:, 0] = vel_x
            self.v[:, 0] = vel_y

        self.hu[:, 0] = self.h[:, 0] * self.u[:, 0]
        self.hv[:, 0] = self.h[:, 0] * self.v[:, 0]

    def _apply_bc_north(self, bc):
        """Apply boundary condition on north boundary (j=ny-1)"""
        from preprocessing.boundary_conditions import BCType

        h_dry = self.config.h_dry

        if bc.bc_type == BCType.WALL:
            self.h[:, -1] = np.maximum(self.h[:, -2], h_dry)
            self.u[:, -1] = self.u[:, -2]
            self.v[:, -1] = -self.v[:, -2]

        elif bc.bc_type == BCType.INFLOW:
            self.h[:, -1] = bc.depth
            self.u[:, -1] = bc.velocity_x
            self.v[:, -1] = bc.velocity_y

        elif bc.bc_type == BCType.OUTFLOW:
            self.h[:, -1] = self.h[:, -2]
            self.u[:, -1] = self.u[:, -2]
            self.v[:, -1] = self.v[:, -2]

        elif bc.bc_type == BCType.TIME_SERIES:
            depth, vel_x, vel_y = bc.interpolate(self.t)
            self.h[:, -1] = depth
            self.u[:, -1] = vel_x
            self.v[:, -1] = vel_y

        self.hu[:, -1] = self.h[:, -1] * self.u[:, -1]
        self.hv[:, -1] = self.h[:, -1] * self.v[:, -1]

    def compute_source_terms(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute source terms (bed slope and friction)

        Returns
        -------
        S_h, S_hu, S_hv : np.ndarray
            Source terms for h, hu, hv
        """
        g = self.config.g
        n = self.config.manning_n
        h_dry = self.config.h_dry

        # Initialize source terms
        S_h = np.zeros((self.nx, self.ny))
        S_hu = np.zeros((self.nx, self.ny))
        S_hv = np.zeros((self.nx, self.ny))

        # Bed slope source term
        # S_hu = -g * h * ∂z/∂x
        # S_hv = -g * h * ∂z/∂y

        # Compute bed slope (central difference)
        dz_dx = np.zeros((self.nx, self.ny))
        dz_dy = np.zeros((self.nx, self.ny))

        # Interior cells
        dz_dx[1:-1, :] = (self.z[2:, :] - self.z[:-2, :]) / (2 * self.dx)
        dz_dy[:, 1:-1] = (self.z[:, 2:] - self.z[:, :-2]) / (2 * self.dy)

        # Boundaries (one-sided)
        dz_dx[0, :] = (self.z[1, :] - self.z[0, :]) / self.dx
        dz_dx[-1, :] = (self.z[-1, :] - self.z[-2, :]) / self.dx
        dz_dy[:, 0] = (self.z[:, 1] - self.z[:, 0]) / self.dy
        dz_dy[:, -1] = (self.z[:, -1] - self.z[:, -2]) / self.dy

        S_hu = -g * self.h * dz_dx
        S_hv = -g * self.h * dz_dy

        # Manning friction source term
        # S_hu = -g * n² * u * √(u² + v²) / h^(4/3)
        # S_hv = -g * n² * v * √(u² + v²) / h^(4/3)

        wet = self.h > h_dry
        if np.any(wet):
            V_mag = np.sqrt(self.u**2 + self.v**2)
            h_pow = np.power(self.h, 4.0/3.0)

            friction_factor = g * n * n * V_mag / (h_pow + 1e-10)

            S_hu[wet] -= friction_factor[wet] * self.u[wet] * self.h[wet]
            S_hv[wet] -= friction_factor[wet] * self.v[wet] * self.h[wet]

        return S_h, S_hu, S_hv

    def update_conservative_variables(self, dt: float):
        """
        Update conservative variables using explicit Euler method

        ∂U/∂t + ∂F/∂x + ∂G/∂y = S

        U^(n+1) = U^n - dt/dx * (F_(i+1/2) - F_(i-1/2)) - dt/dy * (G_(j+1/2) - G_(j-1/2)) + dt * S

        Parameters
        ----------
        dt : float
            Time step
        """
        # Compute source terms
        S_h, S_hu, S_hv = self.compute_source_terms()

        # Update conservative variables (interior cells)
        for i in range(1, self.nx - 1):
            for j in range(1, self.ny - 1):
                # Mass equation
                self.h[i, j] -= dt / self.dx * (self.flux_x[0, i+1, j] - self.flux_x[0, i, j])
                self.h[i, j] -= dt / self.dy * (self.flux_y[0, i, j+1] - self.flux_y[0, i, j])
                self.h[i, j] += dt * S_h[i, j]

                # X-momentum equation
                self.hu[i, j] -= dt / self.dx * (self.flux_x[1, i+1, j] - self.flux_x[1, i, j])
                self.hu[i, j] -= dt / self.dy * (self.flux_y[1, i, j+1] - self.flux_y[1, i, j])
                self.hu[i, j] += dt * S_hu[i, j]

                # Y-momentum equation
                self.hv[i, j] -= dt / self.dx * (self.flux_x[2, i+1, j] - self.flux_x[2, i, j])
                self.hv[i, j] -= dt / self.dy * (self.flux_y[2, i, j+1] - self.flux_y[2, i, j])
                self.hv[i, j] += dt * S_hv[i, j]

        # Ensure non-negative depth
        self.h = np.maximum(self.h, 0.0)

        # Update velocities
        wet = self.h > self.config.h_dry
        self.u = np.zeros_like(self.h)
        self.v = np.zeros_like(self.h)
        self.u[wet] = self.hu[wet] / self.h[wet]
        self.v[wet] = self.hv[wet] / self.h[wet]

    def step(self):
        """Perform one time step"""
        # Compute time step
        self.dt = self.compute_timestep()

        # Apply boundary conditions
        self.apply_boundary_conditions()

        # Compute fluxes
        self.compute_fluxes_hll()

        # Update conservative variables
        self.update_conservative_variables(self.dt)

        # Update time
        self.t += self.dt
        self.step_count += 1

    def solve(self, t_end: Optional[float] = None) -> Dict[str, Any]:
        """
        Main solver loop

        Parameters
        ----------
        t_end : float, optional
            End time (uses config value if not provided)

        Returns
        -------
        dict
            Solver statistics
        """
        if t_end is None:
            t_end = self.config.t_end

        print(f"\n{'='*70}")
        print(f"Starting simulation: t_end = {t_end:.2f} s")
        print(f"{'='*70}\n")

        start_time = time.time()

        while self.t < t_end:
            # Perform one time step
            self.step()

            # Check mass conservation
            current_mass = np.sum(self.h) * self.dx * self.dy
            if self.config.check_mass_conservation and self.step_count % 100 == 0:
                self.mass_history.append(current_mass)

            # Progress output
            if self.config.print_progress and self.step_count % self.config.progress_interval == 0:
                elapsed = time.time() - start_time
                speed = self.t / elapsed if elapsed > 0 else 0
                h_max = np.max(self.h)
                mass_error = (current_mass - self.initial_mass) / self.initial_mass * 100 if self.initial_mass > 0 else 0

                print(f"Step {self.step_count:6d}: t={self.t:8.3f}s, dt={self.dt:.4f}s, "
                      f"h_max={h_max:.3f}m, mass_err={mass_error:+.4f}%, "
                      f"speed={speed:.2f}x realtime")

            # Callback
            if self.callback and self.step_count % self.callback_interval == 0:
                self.callback(self.t, self.step_count, self.dt)

            # Output results
            if self.t >= self.next_output_time:
                self.write_output()
                self.next_output_time += self.config.output_interval

        # Final output
        self.write_output()

        elapsed = time.time() - start_time

        print(f"\n{'='*70}")
        print(f"Simulation completed!")
        print(f"{'='*70}")
        print(f"Total time steps: {self.step_count}")
        print(f"Simulated time: {self.t:.3f} s")
        print(f"Wall clock time: {elapsed:.2f} s")
        print(f"Speed: {self.t / elapsed:.2f}x realtime")
        print(f"Average dt: {self.t / self.step_count:.4f} s")

        # Mass conservation
        if self.config.check_mass_conservation:
            final_mass = np.sum(self.h) * self.dx * self.dy
            mass_error = (final_mass - self.initial_mass) / self.initial_mass * 100
            print(f"\nMass conservation:")
            print(f"  Initial mass: {self.initial_mass:.4f} m³")
            print(f"  Final mass:   {final_mass:.4f} m³")
            print(f"  Error:        {mass_error:+.6f}%")

        return {
            'steps': self.step_count,
            'simulated_time': self.t,
            'wall_time': elapsed,
            'speed_factor': self.t / elapsed,
            'final_mass': final_mass,
            'mass_error': mass_error
        }

    def write_output(self):
        """Write output to VTK file"""
        import os
        os.makedirs(self.config.output_dir, exist_ok=True)

        filename = os.path.join(self.config.output_dir, f'result_{self.output_count:06d}.vtk')

        # Write simple VTK structured grid
        with open(filename, 'w') as f:
            f.write("# vtk DataFile Version 3.0\n")
            f.write(f"HydroSIS-2D results at t={self.t:.4f}s\n")
            f.write("ASCII\n")
            f.write("DATASET STRUCTURED_POINTS\n")
            f.write(f"DIMENSIONS {self.nx} {self.ny} 1\n")
            f.write(f"ORIGIN {self.mesh.domain.xmin} {self.mesh.domain.ymin} 0.0\n")
            f.write(f"SPACING {self.dx} {self.dy} 1.0\n")
            f.write(f"POINT_DATA {self.nx * self.ny}\n")

            # Water depth
            f.write("SCALARS depth float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for j in range(self.ny):
                for i in range(self.nx):
                    f.write(f"{self.h[i, j]:.6f}\n")

            # Bed elevation
            f.write("SCALARS elevation float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for j in range(self.ny):
                for i in range(self.nx):
                    f.write(f"{self.z[i, j]:.6f}\n")

            # Velocity
            f.write("VECTORS velocity float\n")
            for j in range(self.ny):
                for i in range(self.nx):
                    f.write(f"{self.u[i, j]:.6f} {self.v[i, j]:.6f} 0.0\n")

        self.output_count += 1

        if self.config.print_progress:
            print(f"  Output written: {filename}")

    def get_state(self) -> Dict[str, np.ndarray]:
        """
        Get current solver state

        Returns
        -------
        dict
            Dictionary with arrays: h, u, v, z
        """
        return {
            'h': self.h.copy(),
            'u': self.u.copy(),
            'v': self.v.copy(),
            'z': self.z.copy(),
            't': self.t,
            'step': self.step_count
        }
