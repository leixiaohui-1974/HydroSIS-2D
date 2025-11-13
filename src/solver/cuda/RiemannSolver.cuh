/**
 * @file RiemannSolver.cuh
 * @brief CUDA Riemann solvers for shallow water equations
 * @author HydroSIS-2D Team
 * @date 2025-11-13
 */

#ifndef HYDROSIS2D_RIEMANN_SOLVER_CUH
#define HYDROSIS2D_RIEMANN_SOLVER_CUH

#include <cuda_runtime.h>
#include <cmath>

namespace hydrosis2d {
namespace cuda {

/**
 * @brief HLL (Harten-Lax-van Leer) Riemann solver
 *
 * Computes numerical flux at cell interface for 1D Riemann problem.
 *
 * @param hL Left water depth (m)
 * @param uL Left velocity (m/s)
 * @param hR Right water depth (m)
 * @param uR Right velocity (m/s)
 * @param g Gravity (m/s²)
 * @param[out] F_mass Mass flux
 * @param[out] F_momentum Momentum flux
 */
__device__ inline void hll_solver(
    double hL, double uL,
    double hR, double uR,
    double g,
    double& F_mass,
    double& F_momentum
) {
    // Compute wave speeds
    double cL = sqrt(g * hL);  // Left wave speed
    double cR = sqrt(g * hR);  // Right wave speed

    // Estimate wave speeds (simple approach)
    double sL = fmin(uL - cL, uR - cR);
    double sR = fmax(uL + cL, uR + cR);

    // Conservative variables
    double qL_mass = hL;
    double qL_mom = hL * uL;
    double qR_mass = hR;
    double qR_mom = hR * uR;

    // Physical fluxes
    double fL_mass = hL * uL;
    double fL_mom = hL * uL * uL + 0.5 * g * hL * hL;
    double fR_mass = hR * uR;
    double fR_mom = hR * uR * uR + 0.5 * g * hR * hR;

    // HLL flux
    if (sL >= 0.0) {
        // Supersonic flow to the right
        F_mass = fL_mass;
        F_momentum = fL_mom;
    } else if (sR <= 0.0) {
        // Supersonic flow to the left
        F_mass = fR_mass;
        F_momentum = fR_mom;
    } else {
        // Subsonic flow
        F_mass = (sR * fL_mass - sL * fR_mass + sL * sR * (qR_mass - qL_mass)) / (sR - sL);
        F_momentum = (sR * fL_mom - sL * fR_mom + sL * sR * (qR_mom - qL_mom)) / (sR - sL);
    }

    // Dry state handling
    if (hL < 1e-10 && hR < 1e-10) {
        F_mass = 0.0;
        F_momentum = 0.0;
    }
}

/**
 * @brief HLLC (HLL with Contact) Riemann solver
 *
 * More accurate than HLL, resolves contact discontinuity.
 *
 * @param hL Left water depth (m)
 * @param uL Left velocity (m/s)
 * @param hR Right water depth (m)
 * @param uR Right velocity (m/s)
 * @param g Gravity (m/s²)
 * @param[out] F_mass Mass flux
 * @param[out] F_momentum Momentum flux
 */
__device__ inline void hllc_solver(
    double hL, double uL,
    double hR, double uR,
    double g,
    double& F_mass,
    double& F_momentum
) {
    // Dry state handling
    const double dry_tol = 1e-10;
    if (hL < dry_tol && hR < dry_tol) {
        F_mass = 0.0;
        F_momentum = 0.0;
        return;
    }

    // One-sided dry state
    if (hL < dry_tol) {
        hL = 0.0;
        uL = 0.0;
    }
    if (hR < dry_tol) {
        hR = 0.0;
        uR = 0.0;
    }

    // Wave speeds
    double cL = sqrt(g * hL);
    double cR = sqrt(g * hR);

    // Estimate wave speeds (Roe average)
    double h_avg = 0.5 * (hL + hR);
    double u_avg = (uL * sqrt(hL) + uR * sqrt(hR)) / (sqrt(hL) + sqrt(hR) + 1e-12);
    double c_avg = sqrt(g * h_avg);

    double sL = fmin(uL - cL, u_avg - c_avg);
    double sR = fmax(uR + cR, u_avg + c_avg);

    // Conservative variables
    double qL_mass = hL;
    double qL_mom = hL * uL;
    double qR_mass = hR;
    double qR_mom = hR * uR;

    // Physical fluxes
    double fL_mass = hL * uL;
    double fL_mom = hL * uL * uL + 0.5 * g * hL * hL;
    double fR_mass = hR * uR;
    double fR_mom = hR * uR * uR + 0.5 * g * hR * hR;

    // Contact wave speed
    double sM = (sR * qR_mom - sL * qL_mom + fL_mom - fR_mom) /
                (sR * qR_mass - sL * qL_mass + fL_mass - fR_mass + 1e-12);

    // HLLC flux
    if (sL >= 0.0) {
        F_mass = fL_mass;
        F_momentum = fL_mom;
    } else if (sM >= 0.0) {
        // Star region (left)
        double qL_star_mass = qL_mass * (sL - uL) / (sL - sM);
        double qL_star_mom = (qL_mom * (sL - uL) + (fL_mom - sL * qL_mom)) / (sL - sM);

        F_mass = fL_mass + sL * (qL_star_mass - qL_mass);
        F_momentum = fL_mom + sL * (qL_star_mom - qL_mom);
    } else if (sR >= 0.0) {
        // Star region (right)
        double qR_star_mass = qR_mass * (sR - uR) / (sR - sM);
        double qR_star_mom = (qR_mom * (sR - uR) + (fR_mom - sR * qR_mom)) / (sR - sM);

        F_mass = fR_mass + sR * (qR_star_mass - qR_mass);
        F_momentum = fR_mom + sR * (qR_star_mom - qR_mom);
    } else {
        F_mass = fR_mass;
        F_momentum = fR_mom;
    }
}

/**
 * @brief Rotate velocity vector to interface normal direction
 *
 * @param u X-velocity
 * @param v Y-velocity
 * @param nx Normal x-component
 * @param ny Normal y-component
 * @param[out] un Normal velocity
 * @param[out] ut Tangential velocity
 */
__device__ inline void rotateVelocity(
    double u, double v,
    double nx, double ny,
    double& un, double& ut
) {
    un = u * nx + v * ny;     // Normal component
    ut = -u * ny + v * nx;    // Tangential component
}

/**
 * @brief Rotate velocity back to Cartesian coordinates
 */
__device__ inline void rotateVelocityBack(
    double un, double ut,
    double nx, double ny,
    double& u, double& v
) {
    u = un * nx - ut * ny;
    v = un * ny + ut * nx;
}

/**
 * @brief 2D HLLC solver (rotates to interface normal)
 *
 * @param hL Left depth
 * @param uL Left x-velocity
 * @param vL Left y-velocity
 * @param hR Right depth
 * @param uR Right x-velocity
 * @param vR Right y-velocity
 * @param nx Normal x-component
 * @param ny Normal y-component
 * @param g Gravity
 * @param[out] F_mass Mass flux
 * @param[out] F_mom_x X-momentum flux
 * @param[out] F_mom_y Y-momentum flux
 */
__device__ inline void hllc_solver_2d(
    double hL, double uL, double vL,
    double hR, double uR, double vR,
    double nx, double ny,
    double g,
    double& F_mass,
    double& F_mom_x,
    double& F_mom_y
) {
    // Rotate velocities to normal direction
    double unL, utL, unR, utR;
    rotateVelocity(uL, vL, nx, ny, unL, utL);
    rotateVelocity(uR, vR, nx, ny, unR, utR);

    // Solve 1D Riemann problem in normal direction
    double F_mass_n, F_mom_n;
    hllc_solver(hL, unL, hR, unR, g, F_mass_n, F_mom_n);

    // Mass flux (no rotation needed)
    F_mass = F_mass_n;

    // Tangential momentum flux (advection)
    double F_mom_t = 0.5 * (utL + utR) * F_mass_n;

    // Rotate momentum fluxes back to Cartesian
    double ux, uy;
    rotateVelocityBack(F_mom_n, F_mom_t, nx, ny, ux, uy);

    F_mom_x = ux;
    F_mom_y = uy;
}

} // namespace cuda
} // namespace hydrosis2d

#endif // HYDROSIS2D_RIEMANN_SOLVER_CUH
