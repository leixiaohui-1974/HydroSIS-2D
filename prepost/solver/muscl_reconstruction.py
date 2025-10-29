"""
MUSCL reconstruction for second-order spatial accuracy

MUSCL (Monotonic Upstream-centered Scheme for Conservation Laws)
reconstruction increases spatial accuracy from first-order (piecewise
constant) to second-order (piecewise linear) by reconstructing interface
values using cell-centered gradients.

Key components:
1. Gradient calculation
2. Slope limiters (minmod, superbee, van Leer, MC)
3. Interface value reconstruction

References:
- van Leer, B. (1979). Towards the ultimate conservative difference scheme V
- Toro, E.F. (2009). Riemann Solvers and Numerical Methods for Fluid Dynamics
"""

import numpy as np


def minmod(a, b):
    """
    Minmod slope limiter

    The most conservative limiter. Chooses the smallest slope in magnitude
    with the same sign, or zero if signs differ.

    Args:
        a, b: Slope values (can be arrays)

    Returns:
        Limited slope
    """
    return np.where(a * b > 0,
                    np.where(np.abs(a) < np.abs(b), a, b),
                    0.0)


def superbee(a, b):
    """
    Superbee slope limiter

    More aggressive limiter that preserves steep gradients better than minmod.

    Args:
        a, b: Slope values (can be arrays)

    Returns:
        Limited slope
    """
    # Superbee: max(0, min(2*a, b), min(a, 2*b)) when a*b > 0
    term1 = np.minimum(2.0 * np.abs(a), np.abs(b))
    term2 = np.minimum(np.abs(a), 2.0 * np.abs(b))
    max_term = np.maximum(term1, term2)

    # Apply sign and zero if different signs
    result = np.where(a * b > 0,
                     np.sign(a) * max_term,
                     0.0)
    return result


def van_leer(a, b):
    """
    van Leer slope limiter

    Smooth limiter with good balance between accuracy and stability.

    Args:
        a, b: Slope values (can be arrays)

    Returns:
        Limited slope
    """
    # van Leer: (a*b + |a*b|) / (a + b) when a+b != 0, else 0
    num = a * b + np.abs(a * b)
    denom = a + b
    # Avoid division by zero - use np.where to prevent warning
    result = np.where((np.abs(denom) > 1e-10) & (a * b > 0),
                     num / np.where(np.abs(denom) > 1e-10, denom, 1.0),
                     0.0)
    return result


def mc_limiter(a, b, c):
    """
    Monotonized Central (MC) slope limiter

    Three-point limiter that uses central differencing with constraints.

    Args:
        a: Backward difference
        b: Forward difference
        c: Central difference (typically (a+b)/2)

    Returns:
        Limited slope
    """
    # MC: min(2*|a|, 2*|b|, |c|) * sign(c) if a*b > 0, else 0
    term = np.minimum(2.0 * np.abs(a),
                     np.minimum(2.0 * np.abs(b), np.abs(c)))
    result = np.where((a * b > 0) & (a * c > 0) & (b * c > 0),
                     np.sign(c) * term,
                     0.0)
    return result


def compute_gradients(u, dx, limiter='minmod'):
    """
    Compute limited gradients for MUSCL reconstruction

    Args:
        u: Cell-centered values, shape (nx, ny)
        dx: Cell size
        limiter: Slope limiter ('minmod', 'superbee', 'vanleer', 'mc')

    Returns:
        grad_x, grad_y: Limited gradients in x and y directions
    """
    nx, ny = u.shape
    grad_x = np.zeros((nx, ny))
    grad_y = np.zeros((nx, ny))

    # X-direction gradients (interior cells)
    for i in range(1, nx - 1):
        # Backward and forward differences
        du_backward = (u[i, :] - u[i-1, :]) / dx
        du_forward = (u[i+1, :] - u[i, :]) / dx

        if limiter == 'minmod':
            grad_x[i, :] = minmod(du_backward, du_forward)
        elif limiter == 'superbee':
            grad_x[i, :] = superbee(du_backward, du_forward)
        elif limiter == 'vanleer':
            grad_x[i, :] = van_leer(du_backward, du_forward)
        elif limiter == 'mc':
            du_central = (u[i+1, :] - u[i-1, :]) / (2 * dx)
            grad_x[i, :] = mc_limiter(du_backward, du_forward, du_central)
        else:
            raise ValueError(f"Unknown limiter: {limiter}")

    # Y-direction gradients (interior cells)
    for j in range(1, ny - 1):
        # Backward and forward differences
        du_backward = (u[:, j] - u[:, j-1]) / dx
        du_forward = (u[:, j+1] - u[:, j]) / dx

        if limiter == 'minmod':
            grad_y[:, j] = minmod(du_backward, du_forward)
        elif limiter == 'superbee':
            grad_y[:, j] = superbee(du_backward, du_forward)
        elif limiter == 'vanleer':
            grad_y[:, j] = van_leer(du_backward, du_forward)
        elif limiter == 'mc':
            du_central = (u[:, j+1] - u[:, j-1]) / (2 * dx)
            grad_y[:, j] = mc_limiter(du_backward, du_forward, du_central)
        else:
            raise ValueError(f"Unknown limiter: {limiter}")

    return grad_x, grad_y


def muscl_reconstruct_x(u, dx, limiter='minmod'):
    """
    MUSCL reconstruction for x-direction interfaces

    Reconstructs left and right states at vertical interfaces (between cells i and i+1).

    Args:
        u: Cell-centered values, shape (nx, ny)
        dx: Cell size
        limiter: Slope limiter to use

    Returns:
        u_L, u_R: Left and right interface values, shape (nx-1, ny)
    """
    nx, ny = u.shape

    # Compute gradients
    grad_x, _ = compute_gradients(u, dx, limiter)

    # Reconstruct at interfaces
    # Interface i is between cells i-1 (left) and i (right)
    u_L = np.zeros((nx - 1, ny))
    u_R = np.zeros((nx - 1, ny))

    for i in range(nx - 1):
        # Left state: extrapolate from cell i to interface i+1/2
        u_L[i, :] = u[i, :] + 0.5 * dx * grad_x[i, :]

        # Right state: extrapolate from cell i+1 to interface i+1/2
        u_R[i, :] = u[i+1, :] - 0.5 * dx * grad_x[i+1, :]

    return u_L, u_R


def muscl_reconstruct_y(u, dy, limiter='minmod'):
    """
    MUSCL reconstruction for y-direction interfaces

    Reconstructs bottom and top states at horizontal interfaces (between cells j and j+1).

    Args:
        u: Cell-centered values, shape (nx, ny)
        dy: Cell size
        limiter: Slope limiter to use

    Returns:
        u_B, u_T: Bottom and top interface values, shape (nx, ny-1)
    """
    nx, ny = u.shape

    # Compute gradients
    _, grad_y = compute_gradients(u, dy, limiter)

    # Reconstruct at interfaces
    # Interface j is between cells j-1 (bottom) and j (top)
    u_B = np.zeros((nx, ny - 1))
    u_T = np.zeros((nx, ny - 1))

    for j in range(ny - 1):
        # Bottom state: extrapolate from cell j to interface j+1/2
        u_B[:, j] = u[:, j] + 0.5 * dy * grad_y[:, j]

        # Top state: extrapolate from cell j+1 to interface j+1/2
        u_T[:, j] = u[:, j+1] - 0.5 * dy * grad_y[:, j+1]

    return u_B, u_T


def test_limiters():
    """Test slope limiters with various inputs"""
    print("\n" + "="*60)
    print("Testing Slope Limiters")
    print("="*60)

    # Test cases: (a, b, expected behavior)
    test_cases = [
        (1.0, 2.0, "Same sign, a < b"),
        (2.0, 1.0, "Same sign, a > b"),
        (-1.0, -2.0, "Both negative"),
        (1.0, -1.0, "Different signs"),
        (0.0, 1.0, "One zero"),
    ]

    for a, b, desc in test_cases:
        mm = minmod(np.array([a]), np.array([b]))[0]
        sb = superbee(np.array([a]), np.array([b]))[0]
        vl = van_leer(np.array([a]), np.array([b]))[0]

        print(f"\n{desc}: a={a:5.1f}, b={b:5.1f}")
        print(f"  minmod:   {mm:6.2f}")
        print(f"  superbee: {sb:6.2f}")
        print(f"  van Leer: {vl:6.2f}")

    print("\n" + "="*60)


def test_reconstruction():
    """Test MUSCL reconstruction with a simple case"""
    print("\n" + "="*60)
    print("Testing MUSCL Reconstruction")
    print("="*60)

    # Create test data: smooth function
    nx, ny = 10, 5
    dx = 1.0
    x = np.arange(nx) * dx

    # Smooth function: u = sin(x)
    u = np.zeros((nx, ny))
    for j in range(ny):
        u[:, j] = np.sin(2 * np.pi * x / (nx * dx))

    print(f"\nGrid: {nx}×{ny}, dx={dx}")
    print(f"Test function: u = sin(2πx/L)")

    # Test different limiters
    for limiter in ['minmod', 'superbee', 'vanleer']:
        u_L, u_R = muscl_reconstruct_x(u, dx, limiter)

        # u_L[i] is reconstructed from cell i, compare with u[i]
        # u_L shape: (nx-1, ny), u shape: (nx, ny)
        avg_diff = np.mean(np.abs(u_L - u[:-1, :]))

        print(f"\n{limiter} limiter:")
        print(f"  u_L range: [{np.min(u_L):.3f}, {np.max(u_L):.3f}]")
        print(f"  u_R range: [{np.min(u_R):.3f}, {np.max(u_R):.3f}]")
        print(f"  Avg |u_L - u_left_cell|: {avg_diff:.4f}")

    print("\n" + "="*60)


if __name__ == '__main__':
    test_limiters()
    test_reconstruction()
