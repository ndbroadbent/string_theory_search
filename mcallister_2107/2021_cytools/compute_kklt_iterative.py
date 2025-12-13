#!/usr/bin/env python3
"""
Step 16: KKLT Moduli Stabilization - Solve for Kähler moduli t.

From arXiv:2107.09064 Section 5.2, equation 5.13:

    (1/2) κ_ijk t^j t^k = c_i/c_τ + χ(D_i)/24 - GV_correction(t)

PURE FUNCTION INTERFACE:
    solve_kklt(kappa_sparse, c_i, g_s, W0, chi_D, gv_invariants, h11, h21) -> dict

    Inputs (all computed by upstream pipeline steps):
        kappa_sparse: Sparse intersection numbers (array [[i,j,k,val],...] in CYTools 2021)
        c_i: Dual Coxeter numbers array (Step 3 - orientifold)
        g_s: String coupling (Step 13 - racetrack)
        W0: Flux superpotential magnitude (Step 14 - racetrack)
        chi_D: Divisor Euler characteristics array (Step 15 - compute_chi_divisor)
        gv_invariants: Dict {(q1,q2,...): N_q} (Step 8 - compute_gv_invariants)
        h11, h21: Hodge numbers (Step 5)

    Output:
        t: Kähler moduli solution (h11 values)
        V_string: String frame volume (with BBHL correction)
        tau_achieved: Achieved divisor volumes
        converged: Whether solver converged

NO DATA FILES ARE LOADED IN THE PURE FUNCTION.
Validation harness at bottom loads McAllister data for testing only.

ALGORITHM:
1. Compute c_τ = 2π / (g_s × ln(1/W₀))
2. Compute zeroth-order target: τ_target = c_i/c_τ + χ(D_i)/24
3. Initialize t from uniform starting point
4. Iterate: solve for t, update GV correction, repeat until convergence
5. Compute V_string = (1/6)κt³ - BBHL

NOTES:
- Extended Kähler cone: Solution t may have negative values (~19/214 for McAllister)
- Multiple solutions exist; starting point determines which one is found
- Uses sparse κ representation for efficiency (6400 vs 10M entries for h11=214)

Validation: Tests against all McAllister examples.
"""

import sys
from pathlib import Path
import time

# Paths
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent.parent
DATA_BASE = ROOT_DIR / "resources/small_cc_2107.09064_source/anc/paper_data"

# Use CYTools 2021 for consistency
CYTOOLS_2021 = ROOT_DIR / "vendor/cytools_mcallister_2107"
sys.path.insert(0, str(CYTOOLS_2021))

import numpy as np
from scipy.special import zeta, spence
from cytools import Polytope

# Import from sibling modules
from compute_target_tau import compute_c_tau
from compute_chi_divisor import compute_chi_divisor

# McAllister examples (name, h11_primal, h21_primal)
MCALLISTER_EXAMPLES = [
    ("4-214-647", 214, 4),
    ("5-113-4627-main", 113, 5),
    ("5-113-4627-alternative", 113, 5),
    ("5-81-3213", 81, 5),
    # ("7-51-13590", 51, 7),  # primal is non-favorable in CYTools 2021
]


# =============================================================================
# SPARSE TENSOR CLASS
# =============================================================================


class SparseIntersectionTensor:
    """
    Sparse representation of intersection tensor κ_ijk.

    For h11=214, dense tensor = 10M entries (80MB).
    Sparse stores only ~6400 non-zero entries (<1KB).

    Handles both CYTools 2021 (array format) and latest (dict format).
    """

    def __init__(self, cy_or_kappa, h11: int = None):
        """
        Initialize from CY object or raw kappa data.

        Args:
            cy_or_kappa: Either CYTools CalabiYau object or sparse kappa data
            h11: Required if passing raw kappa data
        """
        if hasattr(cy_or_kappa, 'h11'):
            # CYTools CY object
            self.h11 = cy_or_kappa.h11()
            kappa_sparse = cy_or_kappa.intersection_numbers(in_basis=True)
        else:
            # Raw kappa data
            kappa_sparse = cy_or_kappa
            if h11 is None:
                raise ValueError("h11 required when passing raw kappa data")
            self.h11 = h11

        # Store as list of (i, j, k, val) with canonical (sorted) indices
        self.entries = []
        seen = set()

        # Handle both dict (latest CYTools) and array (2021) formats
        if hasattr(kappa_sparse, 'items'):
            # Dict format: {(i,j,k): val, ...}
            for (i, j, k), val in kappa_sparse.items():
                if val == 0:
                    continue
                key = tuple(sorted([i, j, k]))
                if key not in seen:
                    seen.add(key)
                    self.entries.append((i, j, k, float(val)))
        else:
            # Array format: [[i, j, k, val], ...]
            for row in kappa_sparse:
                i, j, k, val = int(row[0]), int(row[1]), int(row[2]), row[3]
                if val == 0:
                    continue
                key = tuple(sorted([i, j, k]))
                if key not in seen:
                    seen.add(key)
                    self.entries.append((i, j, k, float(val)))

        self.n_entries = len(self.entries)

    def compute_tau(self, t: np.ndarray) -> np.ndarray:
        """
        Compute τ_m = (1/2) Σ_{j,k} κ_mjk t^j t^k using sparse operations.

        For canonical entry (i,j,k) with i≤j≤k, we add contributions to τ_i, τ_j, τ_k.
        """
        tau = np.zeros(self.h11)

        for i, j, k, val in self.entries:
            if i == j == k:
                # κ_iii: τ_i += κ_iii t_i²
                tau[i] += val * t[i] * t[i]
            elif i == j:
                # κ_iik (i<k): τ_i += 2κ t_i t_k, τ_k += κ t_i²
                tau[i] += 2 * val * t[i] * t[k]
                tau[k] += val * t[i] * t[i]
            elif j == k:
                # κ_ijj (i<j): τ_i += κ t_j², τ_j += 2κ t_i t_j
                tau[i] += val * t[j] * t[j]
                tau[j] += 2 * val * t[i] * t[j]
            elif i == k:
                # κ_iji - shouldn't happen with sorted storage, but handle it
                tau[i] += 2 * val * t[i] * t[j]
                tau[j] += val * t[i] * t[i]
            else:
                # κ_ijk (i<j<k): τ_i += 2κ t_j t_k, τ_j += 2κ t_i t_k, τ_k += 2κ t_i t_j
                tau[i] += 2 * val * t[j] * t[k]
                tau[j] += 2 * val * t[i] * t[k]
                tau[k] += 2 * val * t[i] * t[j]

        return 0.5 * tau

    def compute_V(self, t: np.ndarray) -> float:
        """
        Compute V = (1/6) κ_ijk t^i t^j t^k using sparse operations.
        """
        V = 0.0

        for i, j, k, val in self.entries:
            if i == j == k:
                mult = 1
            elif i == j or j == k or i == k:
                mult = 3
            else:
                mult = 6

            V += val * t[i] * t[j] * t[k] * mult

        return V / 6.0

    def compute_jacobian(self, t: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian J_mk = ∂τ_m/∂t^k = κ_mpk t^p (sum over p).

        For canonical entry (i,j,k), κ_ijk contributes to multiple J elements.
        """
        J = np.zeros((self.h11, self.h11))

        for i, j, k, val in self.entries:
            if i == j == k:
                J[i, i] += val * t[i]
            elif i == j:
                J[i, i] += val * t[k]
                J[i, k] += val * t[i]
                J[k, i] += val * t[i]
            elif j == k:
                J[i, j] += val * t[j]
                J[j, i] += val * t[j]
                J[j, j] += val * t[i]
            else:
                J[i, j] += val * t[k]
                J[i, k] += val * t[j]
                J[j, i] += val * t[k]
                J[j, k] += val * t[i]
                J[k, i] += val * t[j]
                J[k, j] += val * t[i]

        return J


# =============================================================================
# PURE COMPUTATION FUNCTIONS
# =============================================================================


def li2(z: float) -> float:
    """
    Compute polylogarithm Li₂(z) = Σ_{n=1}^∞ z^n / n².

    Uses scipy.special.spence: Li₂(z) = spence(1-z) for real z ≤ 1.
    """
    if abs(z) < 1e-100:
        return 0.0
    if z <= 1:
        return float(spence(1 - z))
    # For z > 1, Li₂(z) gives complex values. For KKLT we expect z < 1.
    return float(spence(1 - z))


def compute_gv_correction(gv_invariants: dict, t: np.ndarray) -> np.ndarray:
    """
    Compute GV correction to target τ from worldsheet instantons.

    From eq 5.13:
        GV_correction(t)_i = (1/(2π)²) Σ_q q_i N_q Li₂(e^{-2πq·t})

    Args:
        gv_invariants: Dict {(q1,q2,...): N_q} of curve classes to GV invariants
        t: Current Kähler moduli

    Returns:
        GV correction for each divisor τ_i
    """
    h11 = len(t)
    correction = np.zeros(h11)

    if not gv_invariants:
        return correction

    prefactor = 1.0 / (4 * np.pi * np.pi)  # (1/(2π)²)

    for q_tuple, N_q in gv_invariants.items():
        q = np.array(q_tuple, dtype=np.float64)

        # Compute q·t
        q_dot_t = np.dot(q, t)

        # Compute Li₂(e^{-2π q·t})
        arg = np.exp(-2 * np.pi * q_dot_t)

        # Skip negligible contributions (e^{-2π q·t} < 10^{-100} for q·t > ~37)
        if abs(arg) < 1e-100:
            continue

        li2_val = li2(arg)

        # Add contribution: q_i × N_q × Li₂(...)
        for i in range(h11):
            correction[i] += q[i] * N_q * li2_val

    return prefactor * correction


def iterative_solve(kappa: SparseIntersectionTensor, target_tau: np.ndarray,
                    n_steps: int = 500, t_init: np.ndarray = None,
                    tol: float = 1e-3, verbose: bool = False) -> tuple:
    """
    McAllister's iterative algorithm (Section 5.2, eqs 5.8-5.11).

    Solve: τ_i = (1/2) κ_ijk t^j t^k = target_tau_i

    Uses damped Newton interpolating from τ_init to target_tau.

    Args:
        kappa: Sparse intersection tensor
        target_tau: Target divisor volumes
        n_steps: Number of interpolation steps
        t_init: Explicit starting point
        tol: Convergence tolerance (RMS error)
        verbose: Print progress

    Returns: (t_solution, tau_achieved, converged)
    """
    h11 = len(target_tau)

    # Initialize with uniform t if not provided
    if t_init is None:
        t_init = np.ones(h11)
        # Scale to target τ magnitude
        tau_init = kappa.compute_tau(t_init)
        if np.mean(tau_init) > 0:
            scale = np.sqrt(np.mean(target_tau) / np.mean(tau_init))
            t_init = t_init * scale

    t = t_init.copy()
    tau_init = kappa.compute_tau(t)

    if verbose:
        print(f"  Initial τ mean: {np.mean(tau_init):.2f}, target mean: {np.mean(target_tau):.2f}")

    converged = False

    for m in range(n_steps):
        alpha = (m + 1) / n_steps
        tau_target_step = (1 - alpha) * tau_init + alpha * target_tau

        tau_current = kappa.compute_tau(t)
        delta_tau = tau_target_step - tau_current

        # Solve linear system with regularization for ill-conditioned Jacobian
        J = kappa.compute_jacobian(t)

        # Use lstsq for robustness against singular/ill-conditioned J
        epsilon, _, _, _ = np.linalg.lstsq(J, delta_tau, rcond=1e-10)

        # Damped Newton update
        step_size = 1.0
        t_new = t + step_size * epsilon

        # Backtrack if τ becomes invalid (NaN or error increases)
        tau_new = kappa.compute_tau(t_new)
        error_new = np.sqrt(np.mean((tau_new - tau_target_step)**2))

        for _ in range(20):
            if not np.any(np.isnan(tau_new)) and error_new < np.sqrt(np.mean((tau_current - tau_target_step)**2)) * 2:
                break
            step_size *= 0.5
            t_new = t + step_size * epsilon
            tau_new = kappa.compute_tau(t_new)
            error_new = np.sqrt(np.mean((tau_new - tau_target_step)**2))
        else:
            # If still failing, use smaller step
            t_new = t + 0.01 * epsilon

        t = t_new

        # Check convergence
        tau_achieved = kappa.compute_tau(t)
        error = np.sqrt(np.mean((tau_achieved - target_tau)**2))

        if error < tol:
            converged = True
            if verbose:
                print(f"  Converged at step {m+1}/{n_steps}, RMS error = {error:.2e}")
            break

        if verbose and (m + 1) % max(1, n_steps // 5) == 0:
            print(f"  Step {m+1}/{n_steps}: RMS error = {error:.6f}")

    tau_final = kappa.compute_tau(t)
    return t, tau_final, converged


def solve_kklt(kappa_sparse, c_i: np.ndarray, g_s: float, W0: float,
               chi_D: np.ndarray, gv_invariants: dict, h11: int, h21: int,
               max_gv_iterations: int = 5, verbose: bool = False) -> dict:
    """
    PURE FUNCTION: Solve KKLT moduli stabilization.

    From eq 5.13:
        (1/2) κ_ijk t^j t^k = c_i/c_τ + χ(D_i)/24 - GV_correction(t)

    Args:
        kappa_sparse: Sparse intersection numbers (array or dict format)
        c_i: Dual Coxeter numbers (1 for D3, 6 for O7)
        g_s: String coupling
        W0: Flux superpotential magnitude
        chi_D: Divisor Euler characteristics
        gv_invariants: Dict {(q1,q2,...): N_q}
        h11, h21: Hodge numbers
        max_gv_iterations: Maximum GV update iterations
        verbose: Print progress

    Returns:
        Dict with t, V_string, tau_achieved, converged
    """
    # Build sparse tensor
    kappa = SparseIntersectionTensor(kappa_sparse, h11=h11)

    # Step 1: Compute c_τ
    c_tau = compute_c_tau(g_s, W0)
    if verbose:
        print(f"  c_τ = {c_tau:.6f}")

    # Zeroth-order target (no GV correction)
    tau_target_zeroth = c_i / c_tau + chi_D / 24.0
    if verbose:
        print(f"  τ_target (zeroth) range: [{tau_target_zeroth.min():.2f}, {tau_target_zeroth.max():.2f}]")

    # Initialize t
    t_current = np.ones(h11)
    tau_init = kappa.compute_tau(t_current)
    if np.mean(tau_init) > 0:
        scale = np.sqrt(np.mean(tau_target_zeroth) / np.mean(tau_init))
        t_current = t_current * scale

    # GV iteration loop
    gv_correction = np.zeros(h11)
    t_prev = None
    converged = False

    for gv_iter in range(max_gv_iterations):
        # Compute current target
        tau_target = tau_target_zeroth - gv_correction

        if verbose:
            print(f"\n  GV iteration {gv_iter + 1}/{max_gv_iterations}")

        # Solve for t
        t_current, tau_achieved, step_converged = iterative_solve(
            kappa, tau_target,
            n_steps=500,
            t_init=t_current,
            tol=1e-3,
            verbose=False
        )

        if not step_converged:
            if verbose:
                print(f"    Solver failed to converge")
            break

        # Check GV iteration convergence
        if t_prev is not None:
            t_change = np.linalg.norm(t_current - t_prev) / np.linalg.norm(t_current)
            if verbose:
                print(f"    t change: {t_change:.2e}")
            if t_change < 1e-4:
                converged = True
                if verbose:
                    print(f"    GV iteration converged")
                break
        t_prev = t_current.copy()

        # Update GV correction
        gv_correction = compute_gv_correction(gv_invariants, t_current)

    # Final results
    V_classical = kappa.compute_V(t_current)

    # BBHL correction
    chi = 2 * (h11 - h21)
    BBHL = zeta(3) * chi / (4 * (2 * np.pi) ** 3)
    V_string = V_classical - BBHL

    if verbose:
        print(f"\n  V_classical = {V_classical:.2f}")
        print(f"  BBHL = {BBHL:.6f} (χ={chi})")
        print(f"  V_string = {V_string:.2f}")

    return {
        "t": t_current,
        "tau_achieved": kappa.compute_tau(t_current),
        "tau_target": tau_target_zeroth - gv_correction,
        "V_classical": V_classical,
        "V_string": V_string,
        "BBHL": BBHL,
        "c_tau": c_tau,
        "chi_D": chi_D,
        "gv_correction": gv_correction,
        "converged": converged,
    }


# =============================================================================
# DATA LOADING FOR TESTS
# =============================================================================


def load_primal_points(example_name: str) -> np.ndarray:
    """Load primal polytope points (points.dat)."""
    data_dir = DATA_BASE / example_name
    lines = (data_dir / "points.dat").read_text().strip().split('\n')
    return np.array([[int(x) for x in line.split(',')] for line in lines])


def load_heights(example_name: str, corrected: bool = True) -> np.ndarray:
    """Load triangulation heights."""
    data_dir = DATA_BASE / example_name
    filename = "corrected_heights.dat" if corrected else "heights.dat"
    text = (data_dir / filename).read_text().strip()
    return np.array([float(x) for x in text.split(',')])


def load_basis(example_name: str) -> list:
    """Load divisor basis indices."""
    data_dir = DATA_BASE / example_name
    text = (data_dir / "basis.dat").read_text().strip()
    return [int(x) for x in text.split(',')]


def load_kklt_basis(example_name: str) -> np.ndarray:
    """Load KKLT basis indices from kklt_basis.dat."""
    data_dir = DATA_BASE / example_name
    basis_path = data_dir / "kklt_basis.dat"
    if not basis_path.exists():
        return None
    text = basis_path.read_text().strip()
    return np.array([int(x) for x in text.split(",")])


def load_c_i(example_name: str) -> np.ndarray:
    """Load c_i values from target_volumes.dat."""
    data_dir = DATA_BASE / example_name
    text = (data_dir / "target_volumes.dat").read_text().strip()
    return np.array([int(x) for x in text.split(",")])


def load_g_s(example_name: str) -> float:
    """Load string coupling from g_s.dat."""
    data_dir = DATA_BASE / example_name
    return float((data_dir / "g_s.dat").read_text().strip())


def load_W0(example_name: str) -> float:
    """Load flux superpotential from W_0.dat."""
    data_dir = DATA_BASE / example_name
    return float((data_dir / "W_0.dat").read_text().strip())


def load_cy_vol(example_name: str) -> float:
    """Load expected V_string from cy_vol.dat."""
    data_dir = DATA_BASE / example_name
    return float((data_dir / "cy_vol.dat").read_text().strip())


def load_kahler_params(example_name: str, corrected: bool = True) -> np.ndarray:
    """Load McAllister's solved Kähler parameters."""
    data_dir = DATA_BASE / example_name
    filename = "corrected_kahler_param.dat" if corrected else "kahler_param.dat"
    text = (data_dir / filename).read_text().strip()
    return np.array([float(x) for x in text.split(',')])


# =============================================================================
# VALIDATION TESTS
# =============================================================================


def test_sparse_tensor(example_name: str, verbose: bool = True) -> dict:
    """
    Validate sparse tensor implementation against dense computation.

    Uses McAllister's pre-solved t to verify V and τ computations.
    """
    if verbose:
        print("=" * 70)
        print(f"SPARSE TENSOR VALIDATION - {example_name}")
        print("=" * 70)

    # Load polytope
    points = load_primal_points(example_name)
    poly = Polytope(points)

    # Check if favorable
    try:
        is_fav = poly.is_favorable(lattice="N")
    except TypeError:
        is_fav = poly.is_favorable()

    if not is_fav:
        if verbose:
            print("  SKIP: Polytope is non-favorable")
        return {"example_name": example_name, "passed": True, "skipped": True}

    # Build CY with McAllister's triangulation
    heights = load_heights(example_name, corrected=True)
    tri = poly.triangulate(heights=heights)
    cy = tri.get_cy()

    # Set McAllister's basis
    basis = load_basis(example_name)
    cy.set_divisor_basis(basis)
    h11, h21 = cy.h11(), cy.h21()

    if verbose:
        print(f"  h11={h11}, h21={h21}")

    # Build sparse and dense tensors
    kappa_sparse_obj = SparseIntersectionTensor(cy)

    kappa_raw = cy.intersection_numbers(in_basis=True)
    kappa_dense = np.zeros((h11, h11, h11))
    for row in kappa_raw:
        i, j, k, val = int(row[0]), int(row[1]), int(row[2]), row[3]
        for perm in [(i, j, k), (i, k, j), (j, i, k), (j, k, i), (k, i, j), (k, j, i)]:
            kappa_dense[perm] = val

    if verbose:
        print(f"  Sparse κ: {kappa_sparse_obj.n_entries} non-zero entries")

    # Load McAllister's corrected t
    t = load_kahler_params(example_name, corrected=True)

    # Compare V
    V_sparse = kappa_sparse_obj.compute_V(t)
    V_dense = np.einsum('ijk,i,j,k->', kappa_dense, t, t, t) / 6.0

    # Compare τ
    tau_sparse = kappa_sparse_obj.compute_tau(t)
    tau_dense = 0.5 * np.einsum('ijk,j,k->i', kappa_dense, t, t)
    tau_diff = np.max(np.abs(tau_sparse - tau_dense))

    # Compare Jacobian
    J_sparse = kappa_sparse_obj.compute_jacobian(t)
    J_dense = np.einsum('ijk,j->ik', kappa_dense, t)
    J_diff = np.max(np.abs(J_sparse - J_dense))

    if verbose:
        print(f"\n  V comparison:")
        print(f"    V (sparse) = {V_sparse:.10f}")
        print(f"    V (dense)  = {V_dense:.10f}")
        print(f"    Difference = {abs(V_sparse - V_dense):.2e}")
        print(f"  τ max difference = {tau_diff:.2e}")
        print(f"  J max difference = {J_diff:.2e}")

    passed = (abs(V_sparse - V_dense) < 1e-6 and tau_diff < 1e-6 and J_diff < 1e-6)
    status = "PASS" if passed else "FAIL"

    if verbose:
        print(f"\n{status}: {example_name}")

    return {
        "example_name": example_name,
        "passed": passed,
        "V_diff": abs(V_sparse - V_dense),
        "tau_diff": tau_diff,
        "J_diff": J_diff,
    }


def test_example(example_name: str, expected_h11: int, verbose: bool = True) -> dict:
    """
    Test V_string computation using McAllister's pre-solved t.

    This validates that our formula V_string = (1/6)κt³ - BBHL gives
    the correct result when applied to McAllister's solution.

    NOTE: The KKLT solver (finding t from scratch) is a hard numerical
    problem requiring proper initialization. This test focuses on validating
    the V_string computation itself.
    """
    if verbose:
        print("=" * 70)
        print(f"TEST - {example_name} (primal h11={expected_h11})")
        print("=" * 70)

    # Load polytope
    points = load_primal_points(example_name)
    poly = Polytope(points)

    # Check if favorable
    try:
        is_fav = poly.is_favorable(lattice="N")
    except TypeError:
        is_fav = poly.is_favorable()

    if not is_fav:
        if verbose:
            print("  SKIP: Polytope is non-favorable")
        return {"example_name": example_name, "passed": True, "skipped": True}

    # Build CY with McAllister's triangulation
    heights = load_heights(example_name, corrected=True)
    tri = poly.triangulate(heights=heights)
    cy = tri.get_cy()

    # Set McAllister's basis
    basis = load_basis(example_name)
    cy.set_divisor_basis(basis)
    h11, h21 = cy.h11(), cy.h21()

    if verbose:
        print(f"  h11={h11}, h21={h21}")

    # Build sparse tensor
    kappa = SparseIntersectionTensor(cy)

    if verbose:
        print(f"  Sparse κ: {kappa.n_entries} non-zero entries")

    # Load McAllister's pre-solved t
    t = load_kahler_params(example_name, corrected=True)

    if verbose:
        print(f"  Loaded McAllister's t: {len(t)} values")
        print(f"  t range: [{t.min():.4f}, {t.max():.4f}]")

    # Compute V_string using our sparse tensor
    V_classical = kappa.compute_V(t)
    chi = 2 * (h11 - h21)
    BBHL = zeta(3) * chi / (4 * (2 * np.pi) ** 3)
    V_computed = V_classical - BBHL

    # Load expected
    V_expected = load_cy_vol(example_name)

    if verbose:
        print(f"\n  V_classical = {V_classical:.4f}")
        print(f"  BBHL = {BBHL:.6f} (χ={chi})")
        print(f"  V_string (computed) = {V_computed:.4f}")
        print(f"  V_string (expected) = {V_expected:.4f}")

    # Compute error
    rel_error = abs(V_computed - V_expected) / V_expected

    if verbose:
        print(f"  Relative error = {100*rel_error:.6f}%")

    # Pass if within 0.001% (should be essentially exact)
    passed = rel_error < 0.00001

    if verbose:
        status = "PASS" if passed else "FAIL"
        print(f"\n{status}: {example_name} (error = {100*rel_error:.6f}%)")

    return {
        "example_name": example_name,
        "passed": passed,
        "V_computed": V_computed,
        "V_expected": V_expected,
        "rel_error": rel_error,
    }


def main():
    """Test sparse tensor and V_string computation against all McAllister examples."""
    print("=" * 70)
    print("KKLT COMPONENTS - MCALLISTER EXAMPLES (CYTools 2021)")
    print("Tests: Sparse κ tensor, V_string = (1/6)κt³ - BBHL")
    print("=" * 70)
    print("\nNOTE: V_string validated using McAllister's pre-solved t")
    print("      7-51-13590 excluded (primal non-favorable)")

    # First validate sparse tensor
    print("\n" + "=" * 70)
    print("PART 1: SPARSE TENSOR VALIDATION")
    print("=" * 70)

    sparse_results = []
    for name, h11, h21 in MCALLISTER_EXAMPLES:
        result = test_sparse_tensor(name, verbose=True)
        sparse_results.append(result)
        print()

    # Then test V_string computation
    print("\n" + "=" * 70)
    print("PART 2: V_STRING COMPUTATION VALIDATION")
    print("=" * 70)

    solver_results = []
    for name, h11, h21 in MCALLISTER_EXAMPLES:
        result = test_example(name, h11, verbose=True)
        solver_results.append(result)
        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nSparse tensor:")
    for r in sparse_results:
        if r.get("skipped"):
            print(f"  SKIP: {r['example_name']:30s} (non-favorable)")
        else:
            status = "PASS" if r["passed"] else "FAIL"
            print(f"  {status}: {r['example_name']:30s}")

    print("\nV_string computation:")
    all_passed = True
    for r in solver_results:
        if r.get("skipped"):
            print(f"  SKIP: {r['example_name']:30s} (non-favorable)")
        else:
            status = "PASS" if r["passed"] else "FAIL"
            print(f"  {status}: {r['example_name']:30s} error={100*r['rel_error']:.6f}%")
            all_passed = all_passed and r["passed"]

    print()
    if all_passed:
        print(f"All {len(solver_results)} examples PASSED")
        print("V_string = (1/6)κt³ - BBHL formula validated")
    else:
        n_passed = sum(1 for r in solver_results if r["passed"])
        print(f"{n_passed}/{len(solver_results)} examples passed")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
