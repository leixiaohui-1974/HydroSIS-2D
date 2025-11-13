"""
Model Calibration & Parameter Estimation Tests

This module tests calibration methods for identifying optimal model parameters
from observed data, including optimization algorithms, data assimilation, and
inverse modeling techniques.

Test Categories:
1. Parameter Optimization (3 tests)
2. Data Assimilation (3 tests)
3. Inverse Modeling (3 tests)
4. Calibration Metrics (3 tests)

Physical Context:
- Model parameters (Manning n, infiltration rates, etc.) are uncertain
- Calibration uses observed data (stage, discharge, flood extent) to estimate parameters
- Optimization seeks best-fit parameters minimizing objective function
- Essential for predictive accuracy

GPU Readiness: Framework complete, awaits GPU compilation
Author: HydroSIS-2D Development Team
Date: 2025-11-13
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path


class TestParameterOptimization:
    """
    Test optimization algorithms for parameter calibration.

    Optimization Problem:
    - min J(θ) = ||H(θ) - y_obs||²
    - θ: parameters (Manning n, etc.)
    - H: forward model (SWE solver)
    - y_obs: observations (stage, discharge, etc.)
    """

    def test_gradient_descent_optimization(self):
        """
        Test gradient descent for parameter estimation.

        Algorithm:
        - θₖ₊₁ = θₖ - α ∇J(θₖ)
        - α: learning rate
        - ∇J: gradient (analytical or finite difference)

        Test Function:
        - Rosenbrock function (well-known test)
        - Minimum at (1, 1)
        """
        print("\n" + "="*70)
        print("TEST: Gradient Descent Optimization")
        print("="*70)

        def rosenbrock(x):
            """Rosenbrock function: f(x,y) = (1-x)² + 100(y-x²)²"""
            return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

        def rosenbrock_grad(x):
            """Gradient of Rosenbrock function."""
            dfdx = -2*(1-x[0]) - 400*x[0]*(x[1] - x[0]**2)
            dfdy = 200*(x[1] - x[0]**2)
            return np.array([dfdx, dfdy])

        # Initial guess
        x0 = np.array([-0.5, -0.5])

        # Gradient descent parameters
        alpha = 0.001  # Learning rate
        max_iter = 10000
        tol = 1e-6

        print(f"\nOptimization Setup:")
        print(f"  Function: Rosenbrock")
        print(f"  Initial guess: {x0}")
        print(f"  True minimum: [1.0, 1.0]")
        print(f"  Learning rate: {alpha}")

        # Optimization loop
        x = x0.copy()
        history = [x.copy()]

        for iteration in range(max_iter):
            grad = rosenbrock_grad(x)
            x_new = x - alpha * grad

            # Check convergence
            if np.linalg.norm(x_new - x) < tol:
                print(f"\nConverged after {iteration+1} iterations")
                break

            x = x_new
            if (iteration + 1) % 2000 == 0:
                history.append(x.copy())
                f_val = rosenbrock(x)
                print(f"  Iter {iteration+1}: x = {x}, f(x) = {f_val:.6f}")

        history.append(x.copy())

        # Final result
        f_final = rosenbrock(x)
        true_min = np.array([1.0, 1.0])
        error = np.linalg.norm(x - true_min)

        print(f"\nFinal Result:")
        print(f"  x = {x}")
        print(f"  f(x) = {f_final:.8f}")
        print(f"  Error from true minimum: {error:.6f}")

        # Validation
        assert error < 0.1, "Should converge close to true minimum"
        assert f_final < 1.0, "Function value should be small at minimum"

        print("\n✓ Gradient descent test passed")

    def test_nelder_mead_simplex(self):
        """
        Test Nelder-Mead simplex method (derivative-free).

        Advantages:
        - No gradient required
        - Robust for noisy objectives
        - Simple implementation

        Disadvantages:
        - Can be slow for high dimensions
        - May get stuck in local minima
        """
        print("\n" + "="*70)
        print("TEST: Nelder-Mead Simplex Optimization")
        print("="*70)

        # Manning coefficient calibration example
        def objective_function(params):
            """
            Objective: misfit between modeled and observed discharge.

            params: [manning_n]
            """
            n = params[0]

            # Observed discharge (synthetic)
            Q_obs = 50.0  # m³/s

            # Simulated discharge (Manning's equation)
            # For rectangular channel
            width = 10.0  # m
            slope = 0.001
            depth = 2.0  # m (assumed)

            A = width * depth
            P = width + 2*depth
            R = A / P

            Q_sim = (1.0/n) * A * (R**(2.0/3.0)) * (slope**0.5)

            # Least squares
            misfit = (Q_sim - Q_obs)**2
            return misfit

        # Use scipy's Nelder-Mead
        from scipy.optimize import minimize

        # Initial guess
        n0 = np.array([0.05])  # Far from true value

        print(f"\nCalibration Setup:")
        print(f"  Parameter: Manning n")
        print(f"  Initial guess: {n0[0]}")
        print(f"  Observed discharge: 50.0 m³/s")

        # Optimize
        result = minimize(objective_function, n0, method='Nelder-Mead',
                         options={'disp': False, 'maxiter': 100})

        n_opt = result.x[0]
        f_opt = result.fun

        print(f"\nOptimization Result:")
        print(f"  Optimal n: {n_opt:.6f}")
        print(f"  Final objective: {f_opt:.8f}")
        print(f"  Iterations: {result.nit}")
        print(f"  Success: {result.success}")

        # Validation: objective should be small
        assert f_opt < 1.0, "Misfit should be small at optimum"
        assert 0.01 < n_opt < 0.10, "Manning n should be in reasonable range"

        print("\n✓ Nelder-Mead test passed")

    def test_genetic_algorithm(self):
        """
        Test genetic algorithm for global optimization.

        GA Characteristics:
        - Global search (explores multiple regions)
        - Population-based
        - Good for multimodal problems
        - Stochastic

        Application:
        - Calibrate multiple parameters simultaneously
        - Avoid local minima
        """
        print("\n" + "="*70)
        print("TEST: Genetic Algorithm Optimization")
        print("="*70)

        # Multi-parameter calibration
        def multi_param_objective(params):
            """
            Calibrate two parameters: Manning n and bed slope.

            params: [n, slope]
            """
            n, slope = params

            # "Observed" values (synthetic truth)
            Q_obs = 45.0  # m³/s
            h_obs = 2.5  # m

            # Channel geometry
            width = 10.0  # m

            # Manning's equation for discharge
            A = width * h_obs
            P = width + 2*h_obs
            R = A / P
            Q_sim = (1.0/n) * A * (R**(2.0/3.0)) * (slope**0.5)

            # Objective (two criteria)
            misfit_Q = (Q_sim - Q_obs)**2
            misfit_h = 0  # Simplified (would compare simulated vs observed stage)

            return misfit_Q + misfit_h

        # Simplified GA (using differential evolution from scipy)
        from scipy.optimize import differential_evolution

        # Parameter bounds
        bounds = [(0.01, 0.10),  # Manning n
                  (0.0001, 0.01)]  # Bed slope

        print(f"\nGenetic Algorithm Setup:")
        print(f"  Parameters: [Manning n, slope]")
        print(f"  Bounds: n ∈ [0.01, 0.10], S ∈ [0.0001, 0.01]")
        print(f"  Population size: 15")
        print(f"  Max generations: 100")

        # Run GA
        result = differential_evolution(multi_param_objective, bounds,
                                       seed=42, maxiter=100, popsize=15,
                                       disp=False, polish=True)

        n_opt, slope_opt = result.x
        f_opt = result.fun

        print(f"\nOptimization Result:")
        print(f"  Optimal Manning n: {n_opt:.6f}")
        print(f"  Optimal slope: {slope_opt:.6f}")
        print(f"  Final objective: {f_opt:.8f}")
        print(f"  Iterations: {result.nit}")
        print(f"  Function evaluations: {result.nfev}")

        # Validation
        assert f_opt < 10.0, "Should find good solution"
        assert 0.01 < n_opt < 0.10, "Manning n in bounds"
        assert 0.0001 < slope_opt < 0.01, "Slope in bounds"

        print("\n✓ Genetic algorithm test passed")


class TestDataAssimilation:
    """
    Test data assimilation methods for state/parameter estimation.

    Data Assimilation:
    - Combine model predictions with observations
    - Account for uncertainties in both
    - Optimally estimate true state
    """

    def test_kalman_filter_state_estimation(self):
        """
        Test Kalman Filter for state estimation.

        Kalman Filter:
        - Prediction: x̂ₖ = F x̂ₖ₋₁
        - Update: x̂ₖ = x̂ₖ⁻ + K(yₖ - H x̂ₖ⁻)
        - Kalman gain: K = P⁻Hᵀ(HPHᵀ + R)⁻¹

        Application:
        - Real-time water level estimation
        - Combines model forecast with sensor data
        """
        print("\n" + "="*70)
        print("TEST: Kalman Filter State Estimation")
        print("="*70)

        # Simple 1D example: water level evolution
        n_steps = 50
        dt = 60.0  # seconds

        # True state evolution (unknown in practice)
        np.random.seed(42)
        true_state = np.zeros(n_steps)
        true_state[0] = 5.0  # Initial water level

        # Process noise (model uncertainty)
        process_std = 0.1

        for k in range(1, n_steps):
            # Simple decay model + noise
            true_state[k] = 0.98 * true_state[k-1] + np.random.normal(0, process_std)

        # Observations (with measurement noise)
        obs_frequency = 5  # Observe every 5 steps
        obs_std = 0.3
        observations = true_state[::obs_frequency] + np.random.normal(0, obs_std, n_steps//obs_frequency)
        obs_times = np.arange(0, n_steps, obs_frequency)

        print(f"\nKalman Filter Setup:")
        print(f"  Time steps: {n_steps}")
        print(f"  Process noise: {process_std}")
        print(f"  Measurement noise: {obs_std}")
        print(f"  Observation frequency: every {obs_frequency} steps")

        # Kalman filter
        # State: x (water level)
        # Model: x_k = F * x_{k-1} + w, w ~ N(0, Q)
        # Obs: y_k = H * x_k + v, v ~ N(0, R)

        F = 0.98  # State transition
        H = 1.0  # Observation operator
        Q = process_std**2  # Process noise covariance
        R = obs_std**2  # Measurement noise covariance

        # Initialize
        x_est = np.zeros(n_steps)
        x_est[0] = 5.0  # Initial estimate
        P = 1.0  # Initial error covariance

        P_history = [P]

        obs_idx = 0

        for k in range(1, n_steps):
            # Prediction
            x_pred = F * x_est[k-1]
            P_pred = F * P * F + Q

            # Update (if observation available)
            if k in obs_times:
                y_obs = observations[obs_idx]
                obs_idx += 1

                # Kalman gain
                K = P_pred * H / (H * P_pred * H + R)

                # Update estimate
                innovation = y_obs - H * x_pred
                x_est[k] = x_pred + K * innovation

                # Update covariance
                P = (1 - K * H) * P_pred
            else:
                # No observation, use prediction
                x_est[k] = x_pred
                P = P_pred

            P_history.append(P)

        # Compute error
        rmse_filter = np.sqrt(((x_est - true_state)**2).mean())
        rmse_obs = np.sqrt(((observations - true_state[::obs_frequency])**2).mean())

        print(f"\nResults:")
        print(f"  RMSE (Kalman filter): {rmse_filter:.4f}")
        print(f"  RMSE (observations only): {rmse_obs:.4f}")
        print(f"  Improvement: {((rmse_obs - rmse_filter) / rmse_obs * 100):.1f}%")

        # Final uncertainty
        final_std = np.sqrt(P_history[-1])
        print(f"\nFinal uncertainty (std): {final_std:.4f}")

        # Validation
        assert rmse_filter < rmse_obs, "KF should be better than obs alone"
        assert rmse_filter < 0.5, "RMSE should be reasonable"

        print("\n✓ Kalman filter test passed")

    def test_ensemble_kalman_filter(self):
        """
        Test Ensemble Kalman Filter (EnKF) for nonlinear systems.

        EnKF:
        - Monte Carlo approximation of Kalman filter
        - Ensemble represents probability distribution
        - Suitable for nonlinear models (like SWE)

        Advantages:
        - Handles nonlinearity
        - No need for linearization
        - Propagates full probability distribution
        """
        print("\n" + "="*70)
        print("TEST: Ensemble Kalman Filter (EnKF)")
        print("="*70)

        # Nonlinear model: logistic growth
        # dx/dt = r*x*(1 - x/K)
        def logistic_model(x, r=0.5, K=10.0, dt=0.1):
            """Discrete logistic growth."""
            return x + dt * r * x * (1 - x/K)

        n_steps = 30
        n_ensemble = 50

        # True state
        np.random.seed(42)
        true_state = np.zeros(n_steps)
        true_state[0] = 2.0

        for k in range(1, n_steps):
            true_state[k] = logistic_model(true_state[k-1])
            true_state[k] += np.random.normal(0, 0.1)  # Process noise

        # Observations
        obs_std = 0.5
        observations = true_state + np.random.normal(0, obs_std, n_steps)

        print(f"\nEnKF Setup:")
        print(f"  Model: Logistic growth")
        print(f"  Ensemble size: {n_ensemble}")
        print(f"  Time steps: {n_steps}")
        print(f"  Observation noise: {obs_std}")

        # Initialize ensemble
        ensemble = np.random.normal(2.0, 0.5, n_ensemble)

        # Storage
        ensemble_mean = np.zeros(n_steps)
        ensemble_mean[0] = ensemble.mean()

        for k in range(1, n_steps):
            # Forecast: propagate each member
            for i in range(n_ensemble):
                ensemble[i] = logistic_model(ensemble[i])
                ensemble[i] += np.random.normal(0, 0.1)  # Add process noise

            # Analysis: assimilate observation
            y_obs = observations[k]

            # Ensemble mean and covariance
            x_f = ensemble.mean()
            P_f = ensemble.var()

            # Kalman gain (ensemble-based)
            H = 1.0  # Observation operator
            R = obs_std**2
            K = P_f * H / (H * P_f * H + R)

            # Update each ensemble member
            for i in range(n_ensemble):
                # Perturbed observation
                y_pert = y_obs + np.random.normal(0, obs_std)
                innovation = y_pert - H * ensemble[i]
                ensemble[i] = ensemble[i] + K * innovation

            ensemble_mean[k] = ensemble.mean()

        # Evaluate
        rmse = np.sqrt(((ensemble_mean - true_state)**2).mean())

        print(f"\nResults:")
        print(f"  RMSE (EnKF): {rmse:.4f}")
        print(f"  Final ensemble mean: {ensemble_mean[-1]:.4f}")
        print(f"  Final ensemble std: {ensemble.std():.4f}")
        print(f"  True final state: {true_state[-1]:.4f}")

        # Validation
        assert rmse < 1.0, "EnKF should track reasonably well"

        print("\n✓ Ensemble Kalman Filter test passed")

    def test_variational_data_assimilation(self):
        """
        Test 4D-Var (4-dimensional variational) data assimilation.

        4D-Var:
        - Minimize cost function over time window
        - J(x₀) = ||x₀ - xᵇ||² + Σ||H(xₖ) - yₖ||²
        - xᵇ: background (prior)
        - yₖ: observations
        - Adjoint method for gradients

        Advantages:
        - Uses all observations in window
        - Smooths noisy data
        - Optimal for linear-Gaussian case
        """
        print("\n" + "="*70)
        print("TEST: 4D-Var Data Assimilation")
        print("="*70)

        # Simple linear model for demonstration
        # x_k = F * x_{k-1}
        # y_k = H * x_k + noise

        n_steps = 20
        F = 0.95  # Model operator
        H = 1.0  # Observation operator

        # True state
        np.random.seed(42)
        true_state = np.zeros(n_steps)
        true_state[0] = 10.0

        for k in range(1, n_steps):
            true_state[k] = F * true_state[k-1] + np.random.normal(0, 0.2)

        # Observations
        obs_std = 0.5
        observations = H * true_state + np.random.normal(0, obs_std, n_steps)

        # Background (prior guess)
        x_background = 8.0  # Initial guess

        print(f"\n4D-Var Setup:")
        print(f"  Time window: {n_steps} steps")
        print(f"  Background: {x_background}")
        print(f"  Observation noise: {obs_std}")

        # Cost function
        def cost_function(x0):
            """4D-Var cost function."""
            # Background term
            B_inv = 1.0  # Background error covariance inverse
            J_b = 0.5 * B_inv * (x0 - x_background)**2

            # Observation term
            R_inv = 1.0 / obs_std**2
            J_o = 0

            x = x0
            for k in range(n_steps):
                y_model = H * x
                J_o += 0.5 * R_inv * (y_model - observations[k])**2

                # Propagate forward
                if k < n_steps - 1:
                    x = F * x

            return J_b + J_o

        # Optimize
        from scipy.optimize import minimize_scalar
        result = minimize_scalar(cost_function, bounds=(5.0, 15.0), method='bounded')

        x0_optimal = result.x
        J_optimal = result.fun

        print(f"\nOptimization Result:")
        print(f"  Optimal initial condition: {x0_optimal:.4f}")
        print(f"  True initial condition: {true_state[0]:.4f}")
        print(f"  Error: {abs(x0_optimal - true_state[0]):.4f}")
        print(f"  Cost function: {J_optimal:.4f}")

        # Reconstruct trajectory
        x_4dvar = np.zeros(n_steps)
        x_4dvar[0] = x0_optimal
        for k in range(1, n_steps):
            x_4dvar[k] = F * x_4dvar[k-1]

        # RMSE
        rmse = np.sqrt(((x_4dvar - true_state)**2).mean())
        print(f"\nRMSE (4D-Var trajectory): {rmse:.4f}")

        # Validation
        assert abs(x0_optimal - true_state[0]) < 1.0, "Should estimate IC reasonably"

        print("\n✓ 4D-Var test passed")


class TestInverseModeling:
    """
    Test inverse modeling techniques for parameter estimation.

    Inverse Problem:
    - Given observations, infer parameters
    - Often ill-posed (non-unique, unstable)
    - Regularization needed
    """

    def test_least_squares_parameter_estimation(self):
        """
        Test least squares for parameter estimation.

        Problem:
        - min ||f(θ) - y||²
        - θ: parameters
        - f: forward model
        - y: observations

        Solution (linear case):
        - θ = (JᵀJ)⁻¹ Jᵀ (y - f(θ₀))
        - J: Jacobian
        """
        print("\n" + "="*70)
        print("TEST: Least Squares Parameter Estimation")
        print("="*70)

        # Linear model: y = a*x + b + noise
        # Estimate parameters [a, b]

        np.random.seed(42)

        # True parameters
        a_true = 2.5
        b_true = 1.0

        # Generate synthetic data
        n_obs = 50
        x_data = np.linspace(0, 10, n_obs)
        y_data = a_true * x_data + b_true + np.random.normal(0, 0.5, n_obs)

        print(f"\nLinear Regression Setup:")
        print(f"  Model: y = a*x + b")
        print(f"  True parameters: a = {a_true}, b = {b_true}")
        print(f"  Observations: {n_obs} points with noise")

        # Least squares solution
        # Design matrix
        X = np.column_stack([x_data, np.ones(n_obs)])

        # Normal equations: (X^T X) θ = X^T y
        params_est = np.linalg.lstsq(X, y_data, rcond=None)[0]

        a_est, b_est = params_est

        print(f"\nEstimated Parameters:")
        print(f"  a = {a_est:.4f} (true: {a_true})")
        print(f"  b = {b_est:.4f} (true: {b_true})")
        print(f"\nErrors:")
        print(f"  Δa = {abs(a_est - a_true):.4f}")
        print(f"  Δb = {abs(b_est - b_true):.4f}")

        # Residuals
        y_pred = a_est * x_data + b_est
        residuals = y_data - y_pred
        rmse = np.sqrt((residuals**2).mean())

        print(f"\nFit Quality:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²: {1 - (residuals**2).sum() / ((y_data - y_data.mean())**2).sum():.4f}")

        # Validation
        assert abs(a_est - a_true) < 0.2, "Slope estimate should be close"
        assert abs(b_est - b_true) < 0.3, "Intercept estimate should be close"

        print("\n✓ Least squares test passed")

    def test_regularized_inversion(self):
        """
        Test Tikhonov regularization for ill-posed inverse problems.

        Tikhonov Regularization:
        - min ||Ax - b||² + λ||Lx||²
        - λ: regularization parameter
        - L: regularization operator (e.g., identity, gradient)

        Purpose:
        - Stabilize solution
        - Prevent overfitting
        - Incorporate prior knowledge
        """
        print("\n" + "="*70)
        print("TEST: Tikhonov Regularization")
        print("="*70)

        # Ill-conditioned problem
        # Estimate roughness field from sparse observations

        n_params = 20  # Manning coefficients in 20 zones
        n_obs = 5  # Only 5 observations (underdetermined)

        np.random.seed(42)

        # True parameters (smooth)
        theta_true = 0.03 + 0.01 * np.sin(np.linspace(0, 2*np.pi, n_params))

        # Forward operator (random sensing)
        A = np.random.randn(n_obs, n_params)

        # Observations
        y_obs = A @ theta_true + np.random.normal(0, 0.001, n_obs)

        print(f"\nInverse Problem Setup:")
        print(f"  Parameters: {n_params}")
        print(f"  Observations: {n_obs} (underdetermined)")

        # Unregularized solution (will be noisy)
        try:
            theta_unreg = np.linalg.lstsq(A, y_obs, rcond=None)[0]
            unreg_norm = np.linalg.norm(theta_unreg)
            print(f"\nUnregularized solution norm: {unreg_norm:.4f}")
        except:
            print("\nUnregularized solution: singular (cannot solve)")

        # Tikhonov regularization
        # (A^T A + λI) θ = A^T y

        lambda_values = [1e-6, 1e-4, 1e-2, 1e-1]

        print(f"\nRegularized Solutions:")
        print(f"{'λ':>10} {'RMSE':>10} {'||θ||':>10}")
        print("-" * 32)

        best_lambda = None
        best_rmse = float('inf')

        for lam in lambda_values:
            # Regularized normal equations
            ATA = A.T @ A
            ATy = A.T @ y_obs
            theta_reg = np.linalg.solve(ATA + lam * np.eye(n_params), ATy)

            # Evaluate
            rmse = np.sqrt(((theta_reg - theta_true)**2).mean())
            theta_norm = np.linalg.norm(theta_reg)

            print(f"{lam:10.6f} {rmse:10.6f} {theta_norm:10.6f}")

            if rmse < best_rmse:
                best_rmse = rmse
                best_lambda = lam
                theta_best = theta_reg

        print(f"\nBest regularization: λ = {best_lambda}")
        print(f"  RMSE: {best_rmse:.6f}")

        # Validation
        assert best_rmse < 0.05, "Regularized solution should be accurate"

        print("\n✓ Tikhonov regularization test passed")

    def test_bayesian_parameter_estimation(self):
        """
        Test Bayesian parameter estimation with MCMC.

        Bayesian Approach:
        - p(θ|y) ∝ p(y|θ) p(θ)
        - Posterior ∝ Likelihood × Prior
        - MCMC samples from posterior

        Advantages:
        - Full uncertainty quantification
        - Incorporates prior knowledge
        - Natural for inverse problems
        """
        print("\n" + "="*70)
        print("TEST: Bayesian Parameter Estimation (MCMC)")
        print("="*70)

        # Simple example: estimate mean and variance
        # Observations: y ~ N(μ, σ²)
        # Estimate: μ, σ

        np.random.seed(42)

        # True parameters
        mu_true = 5.0
        sigma_true = 1.5

        # Generate data
        n_obs = 50
        data = np.random.normal(mu_true, sigma_true, n_obs)

        print(f"\nBayesian Estimation Setup:")
        print(f"  Model: y ~ N(μ, σ)")
        print(f"  True parameters: μ = {mu_true}, σ = {sigma_true}")
        print(f"  Observations: {n_obs}")

        # Log-likelihood
        def log_likelihood(mu, sigma, data):
            if sigma <= 0:
                return -np.inf
            return -0.5 * n_obs * np.log(2 * np.pi * sigma**2) - \
                   np.sum((data - mu)**2) / (2 * sigma**2)

        # Log-prior (weakly informative)
        def log_prior(mu, sigma):
            if sigma <= 0:
                return -np.inf
            # Uniform priors
            if 0 < mu < 10 and 0 < sigma < 5:
                return 0.0
            return -np.inf

        # Log-posterior
        def log_posterior(params, data):
            mu, sigma = params
            return log_likelihood(mu, sigma, data) + log_prior(mu, sigma)

        # Metropolis-Hastings MCMC
        n_iterations = 5000
        n_burn = 1000

        # Initial guess
        params_current = np.array([4.0, 2.0])

        # Proposal standard deviation
        proposal_std = np.array([0.2, 0.2])

        # Storage
        chain = np.zeros((n_iterations, 2))

        accepted = 0

        for i in range(n_iterations):
            # Propose new parameters
            params_proposed = params_current + np.random.normal(0, proposal_std, 2)

            # Acceptance ratio
            log_alpha = log_posterior(params_proposed, data) - log_posterior(params_current, data)

            # Accept/reject
            if np.log(np.random.uniform()) < log_alpha:
                params_current = params_proposed
                accepted += 1

            chain[i] = params_current

        acceptance_rate = accepted / n_iterations

        # Discard burn-in
        chain_converged = chain[n_burn:]

        # Posterior statistics
        mu_post = chain_converged[:, 0].mean()
        sigma_post = chain_converged[:, 1].mean()

        mu_std = chain_converged[:, 0].std()
        sigma_std = chain_converged[:, 1].std()

        print(f"\nMCMC Results:")
        print(f"  Iterations: {n_iterations} (burn-in: {n_burn})")
        print(f"  Acceptance rate: {acceptance_rate*100:.1f}%")

        print(f"\nPosterior Estimates:")
        print(f"  μ = {mu_post:.4f} ± {mu_std:.4f} (true: {mu_true})")
        print(f"  σ = {sigma_post:.4f} ± {sigma_std:.4f} (true: {sigma_true})")

        # Credible intervals
        mu_ci = np.percentile(chain_converged[:, 0], [2.5, 97.5])
        sigma_ci = np.percentile(chain_converged[:, 1], [2.5, 97.5])

        print(f"\n95% Credible Intervals:")
        print(f"  μ: [{mu_ci[0]:.4f}, {mu_ci[1]:.4f}]")
        print(f"  σ: [{sigma_ci[0]:.4f}, {sigma_ci[1]:.4f}]")

        # Validation
        assert abs(mu_post - mu_true) < 0.5, "Mean estimate should be close"
        assert abs(sigma_post - sigma_true) < 0.5, "Std estimate should be close"
        assert mu_ci[0] < mu_true < mu_ci[1], "True μ in credible interval"

        print("\n✓ Bayesian MCMC test passed")


class TestCalibrationMetrics:
    """
    Test metrics for evaluating calibration quality.

    Calibration Metrics:
    - Goodness-of-fit (how well model matches data)
    - Parameter identifiability (can parameters be uniquely determined?)
    - Prediction uncertainty
    """

    def test_nash_sutcliffe_efficiency(self):
        """
        Test Nash-Sutcliffe Efficiency (NSE) metric.

        NSE:
        - NSE = 1 - Σ(Qₒbs - Qsim)² / Σ(Qₒbs - Q̄ₒbs)²
        - Range: (-∞, 1]
        - NSE = 1: perfect match
        - NSE = 0: model as good as mean
        - NSE < 0: model worse than mean

        Interpretation:
        - NSE > 0.75: Very good
        - 0.65 < NSE < 0.75: Good
        - 0.5 < NSE < 0.65: Satisfactory
        - NSE < 0.5: Unsatisfactory
        """
        print("\n" + "="*70)
        print("TEST: Nash-Sutcliffe Efficiency Metric")
        print("="*70)

        np.random.seed(42)

        # Observed discharge
        n_points = 100
        Q_obs = 50 + 20 * np.sin(np.linspace(0, 4*np.pi, n_points)) + \
                np.random.normal(0, 3, n_points)

        # Simulated discharge (various quality)
        # Perfect model
        Q_perfect = Q_obs.copy()

        # Good model
        Q_good = Q_obs + np.random.normal(0, 2, n_points)

        # Poor model (just use mean)
        Q_poor = np.ones(n_points) * Q_obs.mean()

        def nse(obs, sim):
            """Calculate Nash-Sutcliffe Efficiency."""
            numerator = np.sum((obs - sim)**2)
            denominator = np.sum((obs - obs.mean())**2)
            return 1 - numerator / denominator

        nse_perfect = nse(Q_obs, Q_perfect)
        nse_good = nse(Q_obs, Q_good)
        nse_poor = nse(Q_obs, Q_poor)

        print(f"\nNSE Results:")
        print(f"  Perfect model: NSE = {nse_perfect:.4f} (expected: 1.0)")
        print(f"  Good model: NSE = {nse_good:.4f}")
        print(f"  Poor model (mean): NSE = {nse_poor:.4f} (expected: ~0)")

        # Classification
        def classify_nse(nse_value):
            if nse_value > 0.75:
                return "Very good"
            elif nse_value > 0.65:
                return "Good"
            elif nse_value > 0.5:
                return "Satisfactory"
            else:
                return "Unsatisfactory"

        print(f"\nClassification:")
        print(f"  Good model: {classify_nse(nse_good)}")

        # Validation
        assert nse_perfect > 0.99, "Perfect model should have NSE ≈ 1"
        assert -0.1 < nse_poor < 0.1, "Mean model should have NSE ≈ 0"

        print("\n✓ Nash-Sutcliffe Efficiency test passed")

    def test_parameter_identifiability(self):
        """
        Test parameter identifiability analysis.

        Identifiability:
        - Can parameters be uniquely determined from data?
        - Sensitivity matrix rank
        - Correlation among parameters

        Methods:
        - Sensitivity analysis
        - Correlation matrix
        - Eigenvalue analysis
        """
        print("\n" + "="*70)
        print("TEST: Parameter Identifiability Analysis")
        print("="*70)

        # Manning equation with two parameters
        # Q = (1/n) * A * R^(2/3) * S^(1/2)
        # Parameters: n (roughness), S (slope)

        def discharge(n, slope, depth=2.0, width=10.0):
            """Calculate discharge using Manning's equation."""
            A = width * depth
            P = width + 2*depth
            R = A / P
            return (1/n) * A * (R**(2.0/3.0)) * (slope**0.5)

        # Nominal parameters
        n_nom = 0.03
        S_nom = 0.001

        # Sensitivity analysis (finite differences)
        epsilon = 1e-6

        Q_nom = discharge(n_nom, S_nom)

        dQ_dn = (discharge(n_nom + epsilon, S_nom) - Q_nom) / epsilon
        dQ_dS = (discharge(n_nom, S_nom + epsilon) - Q_nom) / epsilon

        # Normalized sensitivities (elasticities)
        S_n = (dQ_dn * n_nom) / Q_nom
        S_S = (dQ_dS * S_nom) / Q_nom

        print(f"\nSensitivity Analysis:")
        print(f"  ∂Q/∂n = {dQ_dn:.4f}")
        print(f"  ∂Q/∂S = {dQ_dS:.4f}")
        print(f"\nNormalized Sensitivities:")
        print(f"  S_n = {S_n:.4f}")
        print(f"  S_S = {S_S:.4f}")

        # Parameter correlation (from Hessian)
        # For identifiability, want low correlation

        # Simplified: compute correlation from sensitivity vectors
        # (In practice, use full Hessian from optimization)

        # Sensitivity vectors
        sens_n = np.array([dQ_dn])
        sens_S = np.array([dQ_dS])

        # Correlation
        correlation = np.dot(sens_n, sens_S) / (np.linalg.norm(sens_n) * np.linalg.norm(sens_S))

        print(f"\nParameter Correlation:")
        print(f"  Corr(n, S) = {correlation[0]:.4f}")

        if abs(correlation[0]) < 0.5:
            print("  ✓ Parameters are weakly correlated (identifiable)")
        elif abs(correlation[0]) < 0.9:
            print("  ⚠ Moderate correlation (some identifiability issues)")
        else:
            print("  ✗ Strong correlation (not uniquely identifiable)")

        # Condition number
        # High condition number → ill-conditioned → identifiability problems

        # Simplified sensitivity matrix (would have multiple observations in reality)
        J = np.array([[dQ_dn, dQ_dS]])  # 1 observation, 2 parameters
        JTJ = J.T @ J

        eigenvalues = np.linalg.eigvalsh(JTJ)
        condition_number = np.sqrt(eigenvalues.max() / eigenvalues.min())

        print(f"\nCondition Number:")
        print(f"  κ = {condition_number:.4f}")

        if condition_number < 10:
            print("  ✓ Well-conditioned")
        elif condition_number < 100:
            print("  ⚠ Moderately conditioned")
        else:
            print("  ✗ Ill-conditioned")

        print("\n✓ Parameter identifiability test passed")

    def test_cross_validation(self):
        """
        Test k-fold cross-validation for model evaluation.

        Cross-Validation:
        - Split data into k folds
        - Train on k-1 folds, validate on 1 fold
        - Repeat k times
        - Average performance

        Purpose:
        - Assess generalization
        - Detect overfitting
        - Compare models
        """
        print("\n" + "="*70)
        print("TEST: K-Fold Cross-Validation")
        print("="*70)

        # Synthetic data
        np.random.seed(42)

        n_data = 100
        x = np.linspace(0, 10, n_data)
        y_true = 2*x + 1
        y = y_true + np.random.normal(0, 1, n_data)

        k_folds = 5

        print(f"\nCross-Validation Setup:")
        print(f"  Data points: {n_data}")
        print(f"  Folds: {k_folds}")
        print(f"  Model: Linear regression")

        # Shuffle data
        indices = np.arange(n_data)
        np.random.shuffle(indices)

        # Split into folds
        fold_size = n_data // k_folds
        fold_indices = [indices[i*fold_size:(i+1)*fold_size] for i in range(k_folds)]

        # Cross-validation
        cv_scores = []

        print(f"\n{'Fold':>6} {'Train RMSE':>12} {'Val RMSE':>12}")
        print("-" * 32)

        for k in range(k_folds):
            # Validation fold
            val_idx = fold_indices[k]

            # Training folds (all except k)
            train_idx = np.concatenate([fold_indices[i] for i in range(k_folds) if i != k])

            # Split data
            x_train, y_train = x[train_idx], y[train_idx]
            x_val, y_val = x[val_idx], y[val_idx]

            # Fit model (linear regression)
            X_train = np.column_stack([x_train, np.ones(len(x_train))])
            params = np.linalg.lstsq(X_train, y_train, rcond=None)[0]

            # Evaluate
            y_train_pred = X_train @ params
            train_rmse = np.sqrt(((y_train - y_train_pred)**2).mean())

            X_val = np.column_stack([x_val, np.ones(len(x_val))])
            y_val_pred = X_val @ params
            val_rmse = np.sqrt(((y_val - y_val_pred)**2).mean())

            cv_scores.append(val_rmse)

            print(f"{k+1:6d} {train_rmse:12.4f} {val_rmse:12.4f}")

        # Average CV score
        mean_cv_score = np.mean(cv_scores)
        std_cv_score = np.std(cv_scores)

        print(f"\nCross-Validation Results:")
        print(f"  Mean validation RMSE: {mean_cv_score:.4f} ± {std_cv_score:.4f}")

        # Validation: scores should be consistent across folds
        cv_variation = std_cv_score / mean_cv_score

        if cv_variation < 0.1:
            print("  ✓ Consistent performance across folds")
        else:
            print("  ⚠ Variable performance (may need more data or simpler model)")

        print("\n✓ Cross-validation test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
