#!/usr/bin/env python3
"""
Explore KKLT solution branches using adaptive step size.

Strategy: Start with 1 step (Newton), if that fails try 2, 4, 8, 16, etc.
For each step size, try 5 random initializations.

This finds the minimum number of steps needed for convergence while
still exploring multiple branches.
"""

import sys
import time
from pathlib import Path
import numpy as np

# Paths
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_BASE = ROOT_DIR / "resources/small_cc_2107.09064_source/anc/paper_data"
CYTOOLS_2021 = ROOT_DIR / "vendor/cytools_mcallister_2107"

sys.path.insert(0, str(CYTOOLS_2021))

from cytools import Polytope
from compute_kahler_param import SparseKappa, _path_follow
from compute_t_uncorrected import _compute_V_from_kappa


# Constants
LAMBDA_OBS = 2.888e-122


def load_example(example_name: str) -> dict:
    """Load all data for an example."""
    data_dir = DATA_BASE / example_name

    points = np.array([[int(x) for x in line.split(',')]
                       for line in (data_dir / "points.dat").read_text().strip().split('\n')])
    heights = np.array([float(x) for x in (data_dir / "corrected_heights.dat").read_text().strip().split(',')])
    basis = [int(x) for x in (data_dir / "basis.dat").read_text().strip().split(',')]
    kklt_basis = [int(x) for x in (data_dir / "kklt_basis.dat").read_text().strip().split(',')]
    c_i = np.array([int(x) for x in (data_dir / "target_volumes.dat").read_text().strip().split(',')])

    g_s = float((data_dir / "g_s.dat").read_text().strip())
    W0 = float((data_dir / "W_0.dat").read_text().strip())
    c_tau = float((data_dir / "c_tau.dat").read_text().strip())
    cy_vol_expected = float((data_dir / "cy_vol.dat").read_text().strip())
    t_expected = np.array([float(x) for x in (data_dir / "kahler_param.dat").read_text().strip().split(',')])

    poly = Polytope(points)
    tri = poly.triangulate(heights=heights)
    cy = tri.get_cy()
    cy.set_divisor_basis(basis)

    kappa_basis = cy.intersection_numbers(in_basis=True)
    kappa_all = cy.intersection_numbers(in_basis=False)
    kappa = SparseKappa(kappa_basis, kappa_all, basis, kklt_basis)

    return {
        "kappa": kappa,
        "c_i": c_i,
        "g_s": g_s,
        "W0": W0,
        "c_tau": c_tau,
        "cy_vol_expected": cy_vol_expected,
        "t_expected": t_expected,
        "h11": cy.h11(),
        "h21": cy.h21(),
    }


def solve_with_n_steps(kappa: SparseKappa, tau_target: np.ndarray,
                       t_init: np.ndarray, n_steps: int,
                       tol: float = 1e-6, newton_tol: float = 1e-8,
                       max_newton: int = 5) -> tuple:
    """
    Solve τ(t) = τ_target using n_steps of predictor-corrector.

    Each step has:
    - Predictor: Euler step toward τ_step
    - Corrector: Newton iterations to converge exactly to τ_step

    n_steps=1 is essentially Newton (one big jump).
    n_steps=100 is conservative predictor-corrector.

    Returns: (t, V, success, final_residual)
    """
    t = t_init.copy()
    tau_init = kappa.compute_tau_kklt(t)
    tau_target_norm = np.linalg.norm(tau_target)

    for m in range(n_steps):
        alpha = (m + 1) / n_steps
        tau_step = (1 - alpha) * tau_init + alpha * tau_target
        tau_step_norm = np.linalg.norm(tau_step)

        # Predictor: one Euler step
        tau_current = kappa.compute_tau_kklt(t)
        delta_tau = tau_step - tau_current
        J = kappa.compute_jacobian_kklt(t)

        try:
            epsilon, _, _, _ = np.linalg.lstsq(J, delta_tau, rcond=1e-10)
        except np.linalg.LinAlgError:
            return None, None, False, np.inf

        t_pred = t + epsilon

        # Corrector: Newton iterations to converge exactly to τ_step
        t_corr = t_pred.copy()
        for _ in range(max_newton):
            tau_corr = kappa.compute_tau_kklt(t_corr)
            residual = tau_corr - tau_step
            if np.linalg.norm(residual) / tau_step_norm < newton_tol:
                break
            J_corr = kappa.compute_jacobian_kklt(t_corr)
            try:
                delta, _, _, _ = np.linalg.lstsq(J_corr, -residual, rcond=1e-10)
            except np.linalg.LinAlgError:
                break

            # Simple damping
            step_norm = np.linalg.norm(delta)
            if step_norm > 1.0:
                delta = delta / step_norm
            t_corr = t_corr + delta

        t = t_corr

        # Check for divergence
        if np.any(np.isnan(t)) or np.any(np.abs(t) > 1e6):
            return None, None, False, np.inf

    # Check final convergence AND V > 0 only at the end
    tau_final = kappa.compute_tau_kklt(t)
    residual = np.linalg.norm(tau_final - tau_target) / tau_target_norm
    V = _compute_V_from_kappa(kappa, t)

    return t, V, residual < tol and V > 0, residual


def solve_adaptive(kappa: SparseKappa, tau_target: np.ndarray,
                   t_init: np.ndarray, max_steps: int = 256,
                   tol: float = 1e-6) -> tuple:
    """
    Solve using adaptive step size: try 1, 2, 4, 8, ... steps until success.

    Returns: (t, V, success, n_steps_used, final_residual)
    """
    n_steps = 1
    while n_steps <= max_steps:
        t, V, success, residual = solve_with_n_steps(kappa, tau_target, t_init, n_steps, tol)
        if success and V is not None and V > 0:
            return t, V, True, n_steps, residual
        n_steps *= 2

    return None, None, False, max_steps, np.inf


def generate_random_init(h11: int, init_type: int, t_expected: np.ndarray = None) -> np.ndarray:
    """Generate a random initialization based on type."""
    if init_type == 0:
        # Uniform scaled
        scale = 10 ** np.random.uniform(-1, 2)
        return np.ones(h11) * scale
    elif init_type == 1:
        # Random positive
        scale = 10 ** np.random.uniform(-1, 2)
        return np.abs(np.random.randn(h11)) * scale + 0.1
    elif init_type == 2:
        # Sparse (few large values)
        t = np.ones(h11) * 0.1
        n_large = max(1, np.random.randint(1, h11 // 5 + 1))
        large_idx = np.random.choice(h11, n_large, replace=False)
        t[large_idx] = np.random.uniform(1, 20, n_large)
        return t
    elif init_type == 3 and t_expected is not None:
        # Perturbed from expected
        t = t_expected * (1 + 0.5 * np.random.randn(h11))
        return np.abs(t) + 0.1
    else:
        # Log-uniform components
        return 10 ** np.random.uniform(-1, 1.5, h11)


def run_full_pipeline(kappa: SparseKappa, t_uncorrected: np.ndarray,
                      c_i: np.ndarray, c_tau: float, g_s: float, W0: float,
                      h11: int, h21: int, e_K0: float = 0.0912) -> dict:
    """Run full pipeline from t_uncorrected to V₀."""
    tau_target = c_i.astype(float) / c_tau
    t_corrected, tau_achieved, conv = _path_follow(kappa, tau_target, t_uncorrected, n_steps=200)

    if not conv or t_corrected is None:
        return {"success": False, "error": "Phase 2 failed"}

    V_raw = _compute_V_from_kappa(kappa, t_corrected)
    chi = 2 * (h11 - h21)
    BBHL = 1.2020569 * chi / (4 * (2 * np.pi)**3)
    V_string = V_raw - BBHL

    if V_string <= 0:
        return {"success": False, "error": "V_string <= 0"}

    V0 = -3 * e_K0 * (g_s**7 / (4 * V_string)**2) * W0**2

    return {
        "success": True,
        "t_corrected": t_corrected,
        "V_string": V_string,
        "V0": V0,
        "V0_over_lambda": abs(V0 / LAMBDA_OBS),
    }


def explore_branches(example_name: str = "5-81-3213",
                     max_time_seconds: float = 60,
                     n_tries_per_step: int = 5,
                     verbose: bool = True):
    """
    Explore branches using adaptive step size.

    For each step size (1, 2, 4, 8, ...), try n_tries_per_step random inits.
    """
    print("=" * 70)
    print(f"ADAPTIVE STEP SIZE EXPLORATION - {example_name}")
    print("=" * 70)
    print()

    data = load_example(example_name)
    kappa = data["kappa"]
    c_i = data["c_i"]
    h11 = data["h11"]
    tau_target = c_i.astype(float)

    print(f"h11 = {h11}, h21 = {data['h21']}")
    print(f"McAllister V_string = {data['cy_vol_expected']:.2f}")
    print(f"Strategy: Try n_steps = 1, 2, 4, 8, ... with {n_tries_per_step} random inits each")
    print()

    # Track results
    solutions = []  # (V_unc, t_unc, t_init, n_steps, pipeline_result, timing)
    step_stats = {}  # n_steps -> (attempts, successes)

    start_time = time.time()
    np.random.seed(int(time.time()) % 10000)

    n_attempts = 0
    n_converged = 0

    print("Step size exploration:")
    print("-" * 70)

    # Test each step size
    for n_steps in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        if time.time() - start_time > max_time_seconds:
            break

        step_attempts = 0
        step_successes = 0
        step_start = time.time()

        for try_idx in range(n_tries_per_step):
            if time.time() - start_time > max_time_seconds:
                break

            n_attempts += 1
            init_type = try_idx % 5
            t_init = generate_random_init(h11, init_type, data["t_expected"])
            t_init_saved = t_init.copy()

            # Scale init to match τ magnitude
            tau_init = kappa.compute_tau_kklt(t_init)
            mean_tau_init = np.mean(np.abs(tau_init))
            mean_tau_target = np.mean(np.abs(tau_target))
            if mean_tau_init > 0 and mean_tau_target > 0:
                scale = np.sqrt(mean_tau_target / mean_tau_init)
                if 0.01 < scale < 100:
                    t_init = t_init * scale

            # Try with exactly n_steps
            t_unc, V_unc, success, residual = solve_with_n_steps(
                kappa, tau_target, t_init, n_steps
            )

            step_attempts += 1
            if success and V_unc is not None and V_unc > 0:
                step_successes += 1
                n_converged += 1

                # Check if new branch
                is_new = True
                for existing_V, _, _, _, _, _ in solutions:
                    if abs(V_unc - existing_V) / max(abs(V_unc), abs(existing_V)) < 0.05:
                        is_new = False
                        break

                if is_new:
                    # Run full pipeline
                    result = run_full_pipeline(
                        kappa, t_unc, c_i,
                        data["c_tau"], data["g_s"], data["W0"],
                        data["h11"], data["h21"]
                    )

                    if result["success"]:
                        elapsed = time.time() - start_time
                        timing = {"found_at": elapsed, "n_steps": n_steps}
                        solutions.append((V_unc, t_unc.copy(), t_init_saved, n_steps, result, timing))

                        if verbose:
                            print(f"  [{elapsed:5.1f}s] n_steps={n_steps:3d}: Branch {len(solutions):2d} "
                                  f"V_unc={V_unc:8.1f} → V_string={result['V_string']:8.2f}")

        step_time = time.time() - step_start
        step_stats[n_steps] = (step_attempts, step_successes, step_time)

        if verbose and step_attempts > 0:
            rate = step_successes / step_attempts * 100
            print(f"  n_steps={n_steps:3d}: {step_successes}/{step_attempts} succeeded ({rate:.0f}%) in {step_time:.2f}s")

    # Summary
    print()
    print("=" * 70)
    print("STEP SIZE ANALYSIS")
    print("=" * 70)
    print(f"{'n_steps':>8} {'attempts':>10} {'successes':>10} {'rate':>8} {'time':>8}")
    print("-" * 50)
    for n_steps in sorted(step_stats.keys()):
        attempts, successes, step_time = step_stats[n_steps]
        rate = successes / attempts * 100 if attempts > 0 else 0
        print(f"{n_steps:8d} {attempts:10d} {successes:10d} {rate:7.1f}% {step_time:7.2f}s")

    print()
    print("=" * 70)
    print("BRANCHES FOUND")
    print("=" * 70)
    print(f"Total attempts: {n_attempts}")
    print(f"Converged: {n_converged}")
    print(f"Unique branches: {len(solutions)}")
    print()

    if not solutions:
        print("No valid branches found!")
        return

    solutions.sort(key=lambda x: x[4]["V_string"])

    print(f"{'#':>3} {'n_steps':>8} {'V_unc':>10} {'V_string':>10} {'V₀':>12}")
    print("-" * 50)
    for i, (V_unc, t_unc, t_init, n_steps, result, timing) in enumerate(solutions):
        print(f"{i+1:3d} {n_steps:8d} {V_unc:10.1f} {result['V_string']:10.2f} {result['V0']:12.2e}")

    # Find optimal step size
    print()
    min_working_steps = min(s[3] for s in solutions) if solutions else None
    if min_working_steps:
        print(f"Minimum working n_steps: {min_working_steps}")

    return solutions, step_stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Explore KKLT branches with adaptive step size")
    parser.add_argument("--example", default="5-81-3213", help="Example name")
    parser.add_argument("--time", type=float, default=60, help="Max search time in seconds")
    parser.add_argument("--tries", type=int, default=5, help="Random inits per step size")

    args = parser.parse_args()

    explore_branches(args.example, max_time_seconds=args.time, n_tries_per_step=args.tries)
