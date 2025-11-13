# -*- coding: utf-8 -*-
"""
Quick feasibility test for Numba JIT compilation

Tests whether Numba can provide additional speedup over
already-vectorized NumPy code.
"""

import numpy as np
import time

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Numba not available, skipping tests")


def numpy_flux_computation(h_L, h_R, u_L, u_R, g=9.81, h_dry=1e-4):
    """Pure NumPy vectorized flux computation"""
    # Wave speeds
    c_L = np.sqrt(g * np.maximum(h_L, 0.0))
    c_R = np.sqrt(g * np.maximum(h_R, 0.0))
    s_L = np.minimum(u_L - c_L, u_R - c_R)
    s_R = np.maximum(u_L + c_L, u_R + c_R)

    # Physical fluxes
    F_L_h = h_L * u_L
    F_L_hu = h_L * u_L * u_L + 0.5 * g * h_L * h_L

    F_R_h = h_R * u_R
    F_R_hu = h_R * u_R * u_R + 0.5 * g * h_R * h_R

    # HLL flux
    s_diff = s_R - s_L
    s_diff = np.where(np.abs(s_diff) < 1e-10, 1e-10, s_diff)

    hll_h = (s_R * F_L_h - s_L * F_R_h + s_L * s_R * (h_R - h_L)) / s_diff
    hll_hu = (s_R * F_L_hu - s_L * F_R_hu + s_L * s_R * (h_R * u_R - h_L * u_L)) / s_diff

    # Conditional selection
    flux_h = np.where(s_L >= 0, F_L_h,
                      np.where(s_R <= 0, F_R_h, hll_h))
    flux_hu = np.where(s_L >= 0, F_L_hu,
                       np.where(s_R <= 0, F_R_hu, hll_hu))

    # Wet/dry mask
    wet_mask = (h_L >= h_dry) | (h_R >= h_dry)
    flux_h = np.where(wet_mask, flux_h, 0.0)
    flux_hu = np.where(wet_mask, flux_hu, 0.0)

    return flux_h, flux_hu


if NUMBA_AVAILABLE:
    @njit
    def numba_flux_computation(h_L, h_R, u_L, u_R, g=9.81, h_dry=1e-4):
        """Numba JIT compiled flux computation"""
        nx, ny = h_L.shape
        flux_h = np.zeros((nx, ny))
        flux_hu = np.zeros((nx, ny))

        for i in range(nx):
            for j in range(ny):
                h_l = h_L[i, j]
                h_r = h_R[i, j]
                u_l = u_L[i, j]
                u_r = u_R[i, j]

                # Check wet/dry
                if h_l < h_dry and h_r < h_dry:
                    continue

                # Wave speeds
                c_l = np.sqrt(g * max(h_l, 0.0))
                c_r = np.sqrt(g * max(h_r, 0.0))
                s_l = min(u_l - c_l, u_r - c_r)
                s_r = max(u_l + c_l, u_r + c_r)

                # Physical fluxes
                F_L_h = h_l * u_l
                F_L_hu = h_l * u_l * u_l + 0.5 * g * h_l * h_l

                F_R_h = h_r * u_r
                F_R_hu = h_r * u_r * u_r + 0.5 * g * h_r * h_r

                # HLL flux
                s_diff = s_r - s_l
                if abs(s_diff) < 1e-10:
                    s_diff = 1e-10

                hll_h = (s_r * F_L_h - s_l * F_R_h + s_l * s_r * (h_r - h_l)) / s_diff
                hll_hu = (s_r * F_L_hu - s_l * F_R_hu + s_l * s_r * (h_r * u_r - h_l * u_l)) / s_diff

                # Conditional selection
                if s_l >= 0:
                    flux_h[i, j] = F_L_h
                    flux_hu[i, j] = F_L_hu
                elif s_r <= 0:
                    flux_h[i, j] = F_R_h
                    flux_hu[i, j] = F_R_hu
                else:
                    flux_h[i, j] = hll_h
                    flux_hu[i, j] = hll_hu

        return flux_h, flux_hu


def benchmark_methods(nx=100, ny=100, iterations=50):
    """Compare NumPy vectorized vs Numba JIT"""
    print(f"\n{'='*70}")
    print(f"Numba Feasibility Test - Grid {nx}x{ny}")
    print(f"{'='*70}")

    # Create test data
    h_L = np.random.uniform(1.0, 10.0, (nx, ny))
    h_R = np.random.uniform(1.0, 10.0, (nx, ny))
    u_L = np.random.uniform(-2.0, 2.0, (nx, ny))
    u_R = np.random.uniform(-2.0, 2.0, (nx, ny))

    # Warm-up
    _ = numpy_flux_computation(h_L, h_R, u_L, u_R)
    if NUMBA_AVAILABLE:
        _ = numba_flux_computation(h_L, h_R, u_L, u_R)

    # Benchmark NumPy
    start = time.time()
    for _ in range(iterations):
        flux_h_np, flux_hu_np = numpy_flux_computation(h_L, h_R, u_L, u_R)
    time_numpy = time.time() - start

    print(f"\nNumPy vectorized:")
    print(f"  Total time: {time_numpy:.3f} s")
    print(f"  Per call: {time_numpy/iterations*1000:.3f} ms")

    if NUMBA_AVAILABLE:
        # Benchmark Numba
        start = time.time()
        for _ in range(iterations):
            flux_h_nb, flux_hu_nb = numba_flux_computation(h_L, h_R, u_L, u_R)
        time_numba = time.time() - start

        print(f"\nNumba JIT:")
        print(f"  Total time: {time_numba:.3f} s")
        print(f"  Per call: {time_numba/iterations*1000:.3f} ms")

        speedup = time_numpy / time_numba
        print(f"\nSpeedup: {speedup:.2f}x")

        # Verify correctness
        flux_h_np, flux_hu_np = numpy_flux_computation(h_L, h_R, u_L, u_R)
        flux_h_nb, flux_hu_nb = numba_flux_computation(h_L, h_R, u_L, u_R)

        max_diff_h = np.max(np.abs(flux_h_np - flux_h_nb))
        max_diff_hu = np.max(np.abs(flux_hu_np - flux_hu_nb))

        print(f"\nNumerical accuracy:")
        print(f"  Max diff in flux_h: {max_diff_h:.2e}")
        print(f"  Max diff in flux_hu: {max_diff_hu:.2e}")

        if speedup < 1.5:
            print(f"\n[WARN]️  Warning: Numba speedup is only {speedup:.2f}x")
            print(f"   Not worth the added complexity over vectorized NumPy.")
            print(f"   Recommendation: Focus on algorithm improvements (MUSCL)")
        else:
            print(f"\n✅ Numba provides {speedup:.2f}x speedup - worth considering!")
    else:
        print("\n[WARN]️  Numba not available")

    print(f"{'='*70}\n")


if __name__ == '__main__':
    if NUMBA_AVAILABLE:
        # Test on different grid sizes
        for nx in [50, 100, 200]:
            benchmark_methods(nx, nx, iterations=50)
    else:
        print("Numba not installed. Install with: pip install numba")
