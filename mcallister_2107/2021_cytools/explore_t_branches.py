#!/usr/bin/env python3
"""
Explore multiple solution branches for KKLT moduli stabilization.

Runs many random initializations, finds all valid t_uncorrected solutions,
then runs the FULL pipeline (Phase 1 → Phase 2 → V_string → V₀) for each.

Goal: Find branches with V₀ closest to observed Λ ≈ 10⁻¹²²
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
from compute_t_uncorrected import _compute_V_from_kappa, _predictor_corrector


# Constants
LAMBDA_OBS = 2.888e-122  # Observed cosmological constant in Planck units


def load_example(example_name: str) -> dict:
    """Load all data for an example."""
    data_dir = DATA_BASE / example_name

    points = np.array([[int(x) for x in line.split(',')]
                       for line in (data_dir / "points.dat").read_text().strip().split('\n')])
    heights = np.array([float(x) for x in (data_dir / "corrected_heights.dat").read_text().strip().split(',')])
    basis = [int(x) for x in (data_dir / "basis.dat").read_text().strip().split(',')]
    kklt_basis = [int(x) for x in (data_dir / "kklt_basis.dat").read_text().strip().split(',')]
    c_i = np.array([int(x) for x in (data_dir / "target_volumes.dat").read_text().strip().split(',')])

    # McAllister's values
    g_s = float((data_dir / "g_s.dat").read_text().strip())
    W0 = float((data_dir / "W_0.dat").read_text().strip())
    c_tau = float((data_dir / "c_tau.dat").read_text().strip())
    cy_vol_expected = float((data_dir / "cy_vol.dat").read_text().strip())
    t_expected = np.array([float(x) for x in (data_dir / "kahler_param.dat").read_text().strip().split(',')])

    # Build CY
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


def run_full_pipeline(kappa: SparseKappa, t_uncorrected: np.ndarray,
                      c_i: np.ndarray, c_tau: float, g_s: float, W0: float,
                      h11: int, h21: int, e_K0: float = 0.0912) -> dict:
    """
    Run full pipeline from t_uncorrected to V₀.

    Phase 2: Path-follow to target τ
    Phase 3: Compute V_string with BBHL
    Phase 4: Compute V₀
    """
    # Phase 2: target τ = c_i / c_tau (ignoring χ/24 for speed)
    tau_target = c_i.astype(float) / c_tau

    t_corrected, tau_achieved, conv = _path_follow(kappa, tau_target, t_uncorrected, n_steps=200)

    if not conv or t_corrected is None:
        return {"success": False, "error": "Phase 2 failed"}

    # Phase 3: V_string with BBHL
    V_raw = _compute_V_from_kappa(kappa, t_corrected)
    chi = 2 * (h11 - h21)
    BBHL = 1.2020569 * chi / (4 * (2 * np.pi)**3)
    V_string = V_raw - BBHL

    if V_string <= 0:
        return {"success": False, "error": "V_string <= 0"}

    # Phase 4: V₀ = -3 × e^K₀ × (g_s⁷/(4V)²) × W₀²
    V0 = -3 * e_K0 * (g_s**7 / (4 * V_string)**2) * W0**2

    return {
        "success": True,
        "t_corrected": t_corrected,
        "V_string": V_string,
        "V0": V0,
        "V0_over_lambda": abs(V0 / LAMBDA_OBS),
    }


def find_t_uncorrected(kappa: SparseKappa, c_i: np.ndarray,
                       t_init: np.ndarray, n_steps: int = 100,
                       scale_init: bool = True) -> tuple:
    """
    Try to find t_uncorrected from a given initialization.

    The starting point t_init determines which BRANCH of solutions we find.
    See BRANCH_DISCOVERY_NOTES.md for details on multiple solution branches.

    Args:
        kappa: SparseKappa with intersection numbers
        c_i: Target divisor volumes (dual Coxeter numbers)
        t_init: Starting point in Kähler cone. Determines which branch.
        n_steps: Number of predictor-corrector steps
        scale_init: If True, scale t_init to match τ magnitude (for random search).
                    If False, use t_init exactly (for fixed/deterministic init).

    Returns (t, V, success)
    """
    tau_target = c_i.astype(float)
    t_start = t_init.copy()

    if scale_init:
        # Scale initialization to match τ magnitude
        tau_init = kappa.compute_tau_kklt(t_start)
        if np.mean(tau_init) > 0:
            scale = np.sqrt(np.mean(tau_target) / np.mean(tau_init))
            if 0.01 < scale < 100:
                t_start = t_start * scale

    # Use predictor-corrector
    t_result, tau_result, conv = _predictor_corrector(kappa, tau_target, t_start, n_steps=n_steps)

    if not conv or t_result is None:
        return None, None, False

    V = _compute_V_from_kappa(kappa, t_result)
    return t_result, V, True


def explore_branches(example_name: str = "5-81-3213",
                     max_time_seconds: float = 300,
                     verbose: bool = True):
    """
    Explore branches by running many initializations.

    Collects unique solutions and runs full pipeline on each.
    """
    print("=" * 70)
    print(f"BRANCH EXPLORATION - {example_name}")
    print("=" * 70)
    print()

    # Load data
    data = load_example(example_name)
    kappa = data["kappa"]
    c_i = data["c_i"]
    h11 = data["h11"]

    print(f"h11 = {h11}, h21 = {data['h21']}")
    print(f"McAllister V_string = {data['cy_vol_expected']:.2f}")
    print(f"Target: Λ_obs = {LAMBDA_OBS:.3e}")
    print()

    # Track unique solutions (by V value, with tolerance)
    # Each entry: (V_uncorrected, t_uncorrected, t_init_used, pipeline_result)
    solutions = []
    n_attempts = 0
    n_converged = 0
    n_V_positive = 0

    start_time = time.time()
    np.random.seed(int(time.time()) % 10000)  # Different seed each run

    print("Searching for branches...")
    print("-" * 70)

    while time.time() - start_time < max_time_seconds:
        n_attempts += 1

        # Generate random initialization
        init_type = n_attempts % 5
        if init_type == 0:
            # Uniform scaled
            scale = 10 ** np.random.uniform(-1, 2)
            t_init = np.ones(h11) * scale
        elif init_type == 1:
            # Random positive
            scale = 10 ** np.random.uniform(-1, 2)
            t_init = np.abs(np.random.randn(h11)) * scale + 0.1
        elif init_type == 2:
            # Sparse (few large values)
            t_init = np.ones(h11) * 0.1
            n_large = np.random.randint(1, h11 // 5)
            large_idx = np.random.choice(h11, n_large, replace=False)
            t_init[large_idx] = np.random.uniform(1, 20, n_large)
        elif init_type == 3:
            # Perturbed from McAllister's expected
            t_init = data["t_expected"] * (1 + 0.5 * np.random.randn(h11))
            t_init = np.abs(t_init) + 0.1
        else:
            # Log-uniform components
            t_init = 10 ** np.random.uniform(-1, 1.5, h11)

        # Save t_init before it gets scaled
        t_init_saved = t_init.copy()

        # Try to find t_uncorrected
        t_unc, V_unc, success = find_t_uncorrected(kappa, c_i, t_init, n_steps=150)

        if not success:
            continue

        n_converged += 1

        if V_unc <= 0:
            continue

        n_V_positive += 1

        # Check if this is a new branch (different V)
        is_new = True
        for existing_V, _, _, _, _ in solutions:
            if abs(V_unc - existing_V) / max(abs(V_unc), abs(existing_V)) < 0.05:
                is_new = False
                break

        if not is_new:
            continue

        # Run full pipeline
        result = run_full_pipeline(
            kappa, t_unc, c_i,
            data["c_tau"], data["g_s"], data["W0"],
            data["h11"], data["h21"]
        )

        if result["success"]:
            elapsed = time.time() - start_time
            # Calculate time since last branch
            if solutions:
                last_time = solutions[-1][4]["found_at"]
                time_since_last = elapsed - last_time
            else:
                time_since_last = elapsed

            timing_info = {
                "found_at": elapsed,
                "time_since_last": time_since_last,
                "attempts_so_far": n_attempts,
            }
            solutions.append((V_unc, t_unc.copy(), t_init_saved, result, timing_info))

            if verbose:
                print(f"[{elapsed:5.1f}s] Branch {len(solutions):2d}: "
                      f"V_unc={V_unc:8.1f} → V_string={result['V_string']:8.2f} → "
                      f"V₀={result['V0']:.2e} (Δt={time_since_last:.1f}s, attempt #{n_attempts})")

        # Progress update
        if n_attempts % 100 == 0 and verbose:
            elapsed = time.time() - start_time
            print(f"  ... {n_attempts} attempts, {n_converged} converged, "
                  f"{n_V_positive} V>0, {len(solutions)} unique branches [{elapsed:.0f}s]")

    # Summary
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Total attempts: {n_attempts}")
    print(f"Converged: {n_converged}")
    print(f"V > 0: {n_V_positive}")
    print(f"Unique branches: {len(solutions)}")
    print()

    if not solutions:
        print("No valid branches found!")
        return

    # Sort by V_string (index 3 is the result dict)
    solutions.sort(key=lambda x: x[3]["V_string"])

    print("All branches found (sorted by V_string):")
    print("-" * 90)
    print(f"{'#':>3} {'V_unc':>10} {'V_string':>10} {'V₀':>12} {'V₀/Λ':>12} {'found_at':>10} {'Δt':>8}")
    print("-" * 90)

    for i, (V_unc, t_unc, t_init, result, timing) in enumerate(solutions):
        print(f"{i+1:3d} {V_unc:10.1f} {result['V_string']:10.2f} "
              f"{result['V0']:12.2e} {result['V0_over_lambda']:12.2e} "
              f"{timing['found_at']:10.1f}s {timing['time_since_last']:7.1f}s")

    print("-" * 90)
    print()

    # Timing summary
    total_time = time.time() - start_time
    print(f"Timing summary:")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Total attempts: {n_attempts}")
    print(f"  Attempts/sec: {n_attempts/total_time:.1f}")
    print(f"  Avg time between branches: {total_time/len(solutions):.1f}s")
    print()

    # Find closest to McAllister
    mcallister_V = data["cy_vol_expected"]
    closest_to_mcallister = min(solutions, key=lambda x: abs(x[3]["V_string"] - mcallister_V))
    print(f"Closest to McAllister (V_string={mcallister_V:.2f}):")
    print(f"  V_string = {closest_to_mcallister[3]['V_string']:.2f}")
    print(f"  V₀ = {closest_to_mcallister[3]['V0']:.2e}")
    print(f"  Found at: {closest_to_mcallister[4]['found_at']:.1f}s (attempt #{closest_to_mcallister[4]['attempts_so_far']})")
    print(f"  t_init (first 10): {closest_to_mcallister[2][:10].round(4).tolist()}")
    print()

    # Find closest to Λ_obs (smallest |V₀|)
    closest_to_lambda = min(solutions, key=lambda x: abs(x[3]["V0"]))
    print(f"Closest to Λ_obs (smallest |V₀|):")
    print(f"  V_string = {closest_to_lambda[3]['V_string']:.2f}")
    print(f"  V₀ = {closest_to_lambda[3]['V0']:.2e}")
    print(f"  |V₀/Λ_obs| = {closest_to_lambda[3]['V0_over_lambda']:.2e}")
    print(f"  Found at: {closest_to_lambda[4]['found_at']:.1f}s (attempt #{closest_to_lambda[4]['attempts_so_far']})")
    print(f"  t_init (first 10): {closest_to_lambda[2][:10].round(4).tolist()}")
    print()

    # Find largest V_string (smallest |V₀| since V₀ ∝ 1/V²)
    largest_V = max(solutions, key=lambda x: x[3]["V_string"])
    print(f"Largest V_string (→ smallest |V₀|):")
    print(f"  V_string = {largest_V[3]['V_string']:.2f}")
    print(f"  V₀ = {largest_V[3]['V0']:.2e}")

    # Print all t_inits for reproducibility
    print()
    print("=" * 70)
    print("T_INIT VALUES FOR EACH BRANCH (for deterministic validation)")
    print("=" * 70)
    for i, (V_unc, t_unc, t_init, result, timing) in enumerate(solutions):
        # Save as compact representation
        print(f"\nBranch {i+1} (V_string={result['V_string']:.2f}, found at {timing['found_at']:.1f}s):")
        print(f"  t_init = np.array({t_init.tolist()})")

    return solutions


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Explore KKLT solution branches")
    parser.add_argument("--example", default="5-81-3213", help="Example name")
    parser.add_argument("--time", type=float, default=300, help="Max search time in seconds")

    args = parser.parse_args()

    explore_branches(args.example, max_time_seconds=args.time)
