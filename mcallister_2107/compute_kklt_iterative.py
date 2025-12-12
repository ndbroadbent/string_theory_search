#!/usr/bin/env python3
"""
Compute KKLT moduli stabilization using McAllister's iterative algorithm.

From arXiv:2107.09064 Section 5.2, equation 5.13:

    (1/2) κ_ijk t^j t^k = c_i/c_τ + χ(D_i)/24 - GV_correction(t)

Where:
    c_τ = 2π / (g_s × ln(1/W₀))
    χ(D_i) = 12 × χ(O_D) - D³  (topological Euler char of divisor)
    GV_correction = (1/(2π)²) Σ_q q_i N_q Li₂((-1)^{γ·q} e^{-2πq·t})

Algorithm:
1. Compute zeroth-order target: τ_target = c_i/c_τ + χ(D_i)/24
2. Solve (1/2) κ_ijk t^j t^k = τ_target for t (interpolation method)
3. Update target with GV correction at current t
4. Iterate until convergence

CRITICAL: t_init must be INSIDE THE KÄHLER CONE! Starting from arbitrary t
outside the cone causes the algorithm to diverge.

OPTIMIZATION: Uses sparse κ representation - only stores ~6400 non-zero entries
instead of 214³ = 10M. This makes h11=214 tractable.
"""

import sys
from pathlib import Path
import json
import time

sys.path.insert(0, str(Path(__file__).parent.parent / "vendor/cytools_latest/src"))

import numpy as np
from scipy.special import zeta, spence  # spence = Li₂(1-z), so Li₂(z) = spence(1-z)
from cytools import Polytope

from compute_target_tau import compute_c_tau
from compute_chi_divisor import compute_chi_divisor
from compute_gv_invariants import compute_gv_invariants

DATA_DIR = Path(__file__).parent.parent / "resources/small_cc_2107.09064_source/anc/paper_data/4-214-647"
RESOURCES_DIR = Path(__file__).parent.parent / "resources"


def load_polytope(use_dual: bool = False, use_heights: bool = True):
    """Load McAllister's polytope with correct triangulation."""
    filename = "dual_points.dat" if use_dual else "points.dat"
    lines = (DATA_DIR / filename).read_text().strip().split('\n')
    points = np.array([[int(x) for x in line.split(',')] for line in lines])

    poly = Polytope(points)

    if use_heights and not use_dual:
        # Use McAllister's triangulation heights for primal
        heights = np.array([float(x) for x in (DATA_DIR / "heights.dat").read_text().strip().split(',')])
        tri = poly.triangulate(heights=heights)
    else:
        tri = poly.triangulate()

    cy = tri.get_cy()
    return poly, tri, cy


class SparseIntersectionTensor:
    """
    Sparse representation of intersection tensor κ_ijk.

    For h11=214, dense tensor = 10M entries (80MB).
    Sparse stores only ~6400 non-zero entries (<1KB).

    Speedup: ~1000x for tensor operations.
    """

    def __init__(self, cy):
        self.h11 = cy.h11()
        kappa_sparse = cy.intersection_numbers(in_basis=True)

        # Store as list of (i, j, k, val) with all permutations
        self.entries = []
        seen = set()

        for (i, j, k), val in kappa_sparse.items():
            if val == 0:
                continue
            # Store canonical form (sorted indices) to avoid duplicates
            key = tuple(sorted([i, j, k]))
            if key not in seen:
                seen.add(key)
                self.entries.append((i, j, k, float(val)))

        self.n_entries = len(self.entries)

        # Pre-compute arrays for vectorized operations
        self._i = np.array([e[0] for e in self.entries], dtype=np.int32)
        self._j = np.array([e[1] for e in self.entries], dtype=np.int32)
        self._k = np.array([e[2] for e in self.entries], dtype=np.int32)
        self._v = np.array([e[3] for e in self.entries], dtype=np.float64)

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
                # κ_iii: J_ii = κ_iip t^p = κ_iii t^i
                J[i, i] += val * t[i]
            elif i == j:
                # κ_iik (i<k): entries at (i,i,k), (i,k,i), (k,i,i)
                # J_ii = κ_iip t^p includes κ_iik t^k
                # J_ik = κ_ikp t^p includes κ_iki t^i
                # J_ki = κ_kip t^p includes κ_kii t^i
                J[i, i] += val * t[k]
                J[i, k] += val * t[i]
                J[k, i] += val * t[i]
            elif j == k:
                # κ_ijj (i<j): entries at (i,j,j), (j,i,j), (j,j,i)
                # J_ij = κ_ijp t^p includes κ_ijj t^j
                # J_ji = κ_jip t^p includes κ_jij t^j = κ_ijj t^j
                # J_jj = κ_jjp t^p includes κ_jji t^i = κ_ijj t^i
                J[i, j] += val * t[j]
                J[j, i] += val * t[j]
                J[j, j] += val * t[i]
            else:
                # κ_ijk (i<j<k): all 6 permutations
                # J_ij = κ_ijp t^p includes κ_ijk t^k
                # J_ik = κ_ikp t^p includes κ_ikj t^j = κ_ijk t^j
                # J_ji = κ_jip t^p includes κ_jik t^k = κ_ijk t^k
                # J_jk = κ_jkp t^p includes κ_jki t^i = κ_ijk t^i
                # J_ki = κ_kip t^p includes κ_kij t^j = κ_ijk t^j
                # J_kj = κ_kjp t^p includes κ_kji t^i = κ_ijk t^i
                J[i, j] += val * t[k]
                J[i, k] += val * t[j]
                J[j, i] += val * t[k]
                J[j, k] += val * t[i]
                J[k, i] += val * t[j]
                J[k, j] += val * t[i]

        return J


def li2(z: complex) -> complex:
    """
    Compute polylogarithm Li₂(z) = Σ_{n=1}^∞ z^n / n².

    Uses scipy.special.spence: Li₂(z) = spence(1-z) for real z.
    For complex z with |z| > 1, uses analytic continuation.
    """
    if np.isreal(z):
        z = float(np.real(z))
        if abs(z) < 1e-100:
            return 0.0
        if z <= 1:
            return float(spence(1 - z))
        # For z > 1, Li₂(z) = -Li₂(1/z) - π²/6 - (1/2)(ln(-z))²
        # But this gives complex values. For KKLT we expect z < 1.
        return float(spence(1 - z))
    else:
        # Complex case - use series for |z| < 1
        z = complex(z)
        if abs(z) < 1:
            result = 0.0
            z_power = z
            for n in range(1, 100):
                result += z_power / (n * n)
                z_power *= z
                if abs(z_power) < 1e-20:
                    break
            return result
        # For |z| >= 1, would need analytic continuation
        return float(spence(1 - np.real(z)))


def compute_gv_correction(gv_invariants: dict, t: np.ndarray, gamma: np.ndarray = None) -> np.ndarray:
    """
    Compute GV correction to target τ from worldsheet instantons.

    From eq 5.13:
        GV_correction(t)_i = (1/(2π)²) Σ_q q_i N_q Li₂((-1)^{γ·q} e^{-2πq·t})

    Args:
        gv_invariants: Dict {(q1,q2,...): N_q} of curve classes to GV invariants
        t: Current Kähler moduli
        gamma: K-theory class for sign (None = all positive)

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

        # Compute sign: (-1)^{γ·q}
        if gamma is not None:
            gamma_dot_q = np.dot(gamma, q)
            sign = (-1) ** int(round(gamma_dot_q))
        else:
            sign = 1.0

        # Compute Li₂(sign × e^{-2π q·t})
        arg = sign * np.exp(-2 * np.pi * q_dot_t)

        # Skip negligible contributions (e^{-2π q·t} < 10^{-300} for q·t > 110)
        if abs(arg) < 1e-100:
            continue

        li2_val = li2(arg)

        # Add contribution: q_i × N_q × Li₂(...)
        for i in range(h11):
            correction[i] += q[i] * N_q * li2_val

    return prefactor * correction


def find_t_for_unit_tau(kappa: SparseIntersectionTensor, tol: float = 1e-6, max_iter: int = 100) -> np.ndarray:
    """
    Find t such that τ_i = (1/2) κ_ijk t^j t^k = 1 for all i.

    From McAllister Section 5.2: τ = (1, 1, ..., 1) is ALWAYS in E(X)° for a
    valid divisor basis. This gives us a guaranteed starting point.

    Uses Newton iteration starting from t = 1.

    Args:
        kappa: Sparse intersection tensor
        tol: Convergence tolerance
        max_iter: Maximum iterations

    Returns:
        t vector giving τ ≈ (1, 1, ..., 1)
    """
    h11 = kappa.h11
    target_tau = np.ones(h11)

    # Start with uniform t
    t = np.ones(h11)

    for iteration in range(max_iter):
        tau = kappa.compute_tau(t)
        residual = tau - target_tau

        if np.max(np.abs(residual)) < tol:
            return t

        # Newton step: J @ delta_t = -residual
        J = kappa.compute_jacobian(t)
        try:
            delta_t = np.linalg.lstsq(J, -residual, rcond=1e-10)[0]
        except np.linalg.LinAlgError:
            # If singular, use damped step
            delta_t = -0.1 * residual

        # Damped update to ensure positivity
        step = 1.0
        for _ in range(10):
            t_new = t + step * delta_t
            if np.all(t_new > 0):
                break
            step *= 0.5
        else:
            # If can't find positive step, use small uniform adjustment
            t_new = t * (1 + 0.1 * (target_tau / tau - 1))

        t = t_new

    return t


def iterative_solve(kappa: SparseIntersectionTensor, target_tau: np.ndarray,
                    n_steps: int = 500, t_init: np.ndarray = None,
                    tol: float = 1e-3, verbose: bool = True) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    McAllister's iterative algorithm (Section 5.2, eqs 5.8-5.11).

    Solve: τ_i = (1/2) κ_ijk t^j t^k = target_tau_i

    Uses damped Newton interpolating from τ_init to target_tau.
    Since τ=(1,1,...,1) is always in E(X)° for valid basis, we can
    interpolate from any starting τ_init to any target in E(X)°.

    Args:
        kappa: Sparse intersection tensor
        target_tau: Target divisor volumes
        n_steps: Number of interpolation steps
        t_init: Starting point (if None, uses find_t_for_unit_tau)
        tol: Convergence tolerance (RMS error)
        verbose: Print progress

    Returns: (t_solution, tau_achieved, converged)
    """
    h11 = len(target_tau)

    # Initialize from unit τ if not provided
    if t_init is None:
        t_init = find_t_for_unit_tau(kappa)

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
        epsilon, residuals, rank, s = np.linalg.lstsq(J, delta_tau, rcond=1e-10)

        # Damped Newton: backtracking to ensure positive t
        step_size = 1.0
        t_new = t + step_size * epsilon

        # Backtrack if t goes negative
        for _ in range(20):
            if np.all(t_new > 0):
                break
            step_size *= 0.5
            t_new = t + step_size * epsilon
        else:
            # Failed to find valid step - use smaller step
            if verbose:
                print(f"  Step {m+1}: Backtrack failed, using minimal step")
            t_new = t + 0.01 * epsilon
            t_new = np.maximum(t_new, 1e-6)  # Ensure positive

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


def mcallister_kklt_solve(kappa: SparseIntersectionTensor, c_i: np.ndarray,
                          W0: float, cy=None, t_init: np.ndarray = None,
                          verbose: bool = True) -> dict:
    """
    McAllister KKLT solution (CLASSICAL - no GV corrections).

    Args:
        kappa: Sparse intersection tensor
        c_i: Dual Coxeter numbers (target τ values at unit scale)
        W0: Flux superpotential
        cy: CYTools CalabiYau object (optional, not used)
        t_init: Override starting point (if None, finds t for unit τ)
    """
    h11 = kappa.h11

    if verbose:
        print(f"KKLT solve: h11={h11}, W0={W0:.2e}")

    # Get starting point: t giving τ = (1,1,...,1)
    if t_init is None:
        t_init = find_t_for_unit_tau(kappa)

    # Scale to target τ ~ c_i
    tau_init = kappa.compute_tau(t_init)
    scale = np.sqrt(np.mean(c_i) / np.mean(tau_init))
    t_scaled = t_init * scale

    start = time.time()
    t_unit, tau_unit, converged = iterative_solve(
        kappa, c_i, n_steps=500, t_init=t_scaled, verbose=verbose
    )

    if verbose:
        print(f"Solve time: {time.time() - start:.2f}s, converged: {converged}")

    # Scale by W0 factor
    t_scale = np.sqrt(np.log(1.0 / np.abs(W0)) / (2 * np.pi))
    t_final = t_unit * t_scale
    V_classical = kappa.compute_V(t_final)

    if verbose:
        print(f"V_classical = {V_classical:.2f}")

    return {"t": t_final, "V_classical": V_classical, "converged": converged}


def mcallister_kklt_solve_with_gv(
    poly,
    cy,
    c_i: np.ndarray,
    g_s: float,
    W0: float,
    basis_indices: list = None,
    t_init: np.ndarray = None,
    gv_min_points: int = 100,
    max_gv_iterations: int = 5,
    gv_tol: float = 1e-4,
    verbose: bool = True,
) -> dict:
    """
    Full KKLT moduli stabilization with χ(D) and GV corrections.

    Implements eq 5.13:
        (1/2) κ_ijk t^j t^k = c_i/c_τ + χ(D_i)/24 - GV_correction(t)

    Algorithm:
    1. Compute c_τ = 2π / (g_s × ln(1/W₀))
    2. Compute χ(D_i) = 12 × χ(O_D) - D³ for basis divisors
    3. Compute GV invariants
    4. Iterate:
       a. τ_target = c_i/c_τ + χ(D_i)/24 - GV_correction(t_current)
       b. Solve for t
       c. Check convergence of t

    Args:
        poly: CYTools Polytope object
        cy: CYTools CalabiYau object
        c_i: Dual Coxeter numbers (1 for D3, 6 for O7)
        g_s: String coupling
        W0: Flux superpotential magnitude
        basis_indices: Divisor basis indices (uses cy.divisor_basis() if None)
        gv_min_points: Minimum points for GV computation
        max_gv_iterations: Maximum GV update iterations
        gv_tol: Convergence tolerance for GV iteration
        verbose: Print progress

    Returns:
        Dict with t, V_classical, V_string, tau_target, gv_correction, converged
    """
    h11 = cy.h11()
    h21 = cy.h21()

    if basis_indices is None:
        basis_indices = list(cy.divisor_basis())

    if verbose:
        print(f"KKLT solve with GV: h11={h11}, h21={h21}, g_s={g_s:.6f}, W0={W0:.2e}")

    # Step 1: Compute c_τ
    c_tau = compute_c_tau(g_s, W0)
    if verbose:
        print(f"  c_τ = {c_tau:.6f}")

    # Step 2: Compute χ(D_i)
    kappa_dict = cy.intersection_numbers(in_basis=True)
    chi_D = compute_chi_divisor(poly, kappa_dict, basis_indices)
    if verbose:
        print(f"  χ(D) range: [{chi_D.min():.0f}, {chi_D.max():.0f}]")

    # Step 3: Compute GV invariants
    if verbose:
        print(f"  Computing GV invariants (min_points={gv_min_points})...")
    gv_invariants = compute_gv_invariants(cy, min_points=gv_min_points)
    if verbose:
        print(f"  Found {len(gv_invariants)} non-zero GV invariants")

    # Step 4: Build sparse tensor
    kappa = SparseIntersectionTensor(cy)

    # Find t giving τ = (1,1,...,1) - guaranteed to be valid (McAllister Section 5.2)
    if verbose:
        print(f"  Finding t for unit τ...")
    t_unit = find_t_for_unit_tau(kappa)
    tau_unit = kappa.compute_tau(t_unit)
    if verbose:
        print(f"  Found t_unit: τ range [{tau_unit.min():.4f}, {tau_unit.max():.4f}]")

    # Zeroth-order target (no GV correction)
    tau_target_zeroth = c_i / c_tau + chi_D / 24.0
    if verbose:
        print(f"  τ_target (zeroth) range: [{tau_target_zeroth.min():.2f}, {tau_target_zeroth.max():.2f}]")

    # Scale t_unit to approximate target scale
    scale = np.sqrt(np.mean(tau_target_zeroth) / np.mean(tau_unit))
    t_current = t_unit * scale

    # GV iteration loop
    gv_correction = np.zeros(h11)
    t_prev = None

    for gv_iter in range(max_gv_iterations):
        # Compute current target
        tau_target = tau_target_zeroth - gv_correction

        if verbose:
            print(f"\n  GV iteration {gv_iter + 1}/{max_gv_iterations}")
            if gv_iter > 0:
                print(f"    GV correction range: [{gv_correction.min():.6f}, {gv_correction.max():.6f}]")

        # Solve for t
        t_current, tau_achieved, converged = iterative_solve(
            kappa, tau_target,
            n_steps=500,
            t_init=t_current,
            tol=1e-3,
            verbose=False
        )

        if not converged:
            if verbose:
                print(f"    Solver failed to converge")
            break

        # Check GV iteration convergence
        if t_prev is not None:
            t_change = np.linalg.norm(t_current - t_prev) / np.linalg.norm(t_current)
            if verbose:
                print(f"    t change: {t_change:.2e}")
            if t_change < gv_tol:
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
        "tau_target": tau_target,
        "V_classical": V_classical,
        "V_string": V_string,
        "BBHL": BBHL,
        "c_tau": c_tau,
        "chi_D": chi_D,
        "gv_correction": gv_correction,
        "converged": converged,
    }


def test_dual():
    """Test on dual polytope (h11=4) - fast validation."""
    print("\n" + "#" * 70)
    print("# TEST: DUAL POLYTOPE (h11=4)")
    print("#" * 70)

    poly, tri, cy = load_polytope(use_dual=True)
    h11, h21 = cy.h11(), cy.h21()
    print(f"h11={h11}, h21={h21}")

    kappa = SparseIntersectionTensor(cy)
    print(f"Sparse κ: {kappa.n_entries} non-zero entries")

    # For dual, all 4 basis divisors are O7-planes: c_i = 6
    c_i = np.array([6.0, 6.0, 6.0, 6.0])
    W0 = 2.30012e-90

    result = mcallister_kklt_solve(kappa, c_i, W0, cy, verbose=True)

    # Apply BBHL for dual: χ = 2(4-214) = -420
    chi = 2 * (h11 - h21)
    BBHL = zeta(3) * chi / (4 * (2*np.pi)**3)
    V_string = result['V_classical'] - BBHL

    print("\n" + "=" * 70)
    print("VALIDATION (DUAL)")
    print("=" * 70)
    print(f"V_classical = {result['V_classical']:.2f}")
    print(f"BBHL correction = {BBHL:.6f} (χ={chi})")
    print(f"V_string = {V_string:.2f}")
    print(f"Expected ≈ 4695 (dual approximation)")

    return result


def test_primal():
    """Test on primal polytope (h11=214)."""
    print("\n" + "#" * 70)
    print("# TEST: PRIMAL POLYTOPE (h11=214)")
    print("#" * 70)

    poly, tri, cy = load_polytope(use_dual=False, use_heights=True)
    h11, h21 = cy.h11(), cy.h21()
    print(f"h11={h11}, h21={h21}")

    # Set McAllister's basis
    basis_indices = [int(x) for x in (DATA_DIR / "basis.dat").read_text().strip().split(',')]
    cy.set_divisor_basis(basis_indices)
    print(f"Set divisor basis: {len(basis_indices)} indices")

    kappa = SparseIntersectionTensor(cy)
    print(f"Sparse κ: {kappa.n_entries} non-zero entries")

    # Load orientifold c_i values
    with open(RESOURCES_DIR / "mcallister_4-214-647_orientifold.json") as f:
        orientifold = json.load(f)

    basis = list(cy.divisor_basis())
    kklt_basis = orientifold['kklt_basis']
    c_values = orientifold['c_i_values']
    point_to_c = {idx: c_values[i] for i, idx in enumerate(kklt_basis)}
    c_i = np.array([float(point_to_c.get(idx, 1.0)) for idx in basis])

    n_o7 = int(np.sum(c_i == 6))
    n_d3 = int(np.sum(c_i == 1))
    print(f"c_i: {n_o7} O7-planes, {n_d3} D3-instantons")

    W0 = 2.30012e-90

    # Use CYTools Kähler cone tip (not McAllister's pre-solved t which is at wrong scale)
    result = mcallister_kklt_solve(kappa, c_i, W0, cy, verbose=True)

    # Apply BBHL for primal: χ = 2(214-4) = 420
    chi = 2 * (h11 - h21)
    BBHL = zeta(3) * chi / (4 * (2*np.pi)**3)
    V_string = result['V_classical'] - BBHL

    print("\n" + "=" * 70)
    print("VALIDATION (PRIMAL)")
    print("=" * 70)
    print(f"V_classical = {result['V_classical']:.2f}")
    print(f"BBHL = {BBHL:.6f}")
    print(f"V_string = {V_string:.2f}")
    print(f"Expected = 4711.83 (from cy_vol.dat)")

    # Compare with McAllister's solved t
    t_mcallister = np.array([float(x) for x in (DATA_DIR / "kahler_param.dat").read_text().strip().split(',')])
    V_mcallister = kappa.compute_V(t_mcallister)

    print(f"\nUsing McAllister's kahler_param.dat:")
    print(f"  V_classical = {V_mcallister:.2f}")
    print(f"  V_string = {V_mcallister - BBHL:.2f}")

    # Compare our t with McAllister's
    if result['converged']:
        t_diff = np.linalg.norm(result['t'] - t_mcallister) / np.linalg.norm(t_mcallister)
        print(f"  ||t_ours - t_mcallister|| / ||t_mcallister|| = {t_diff:.4f}")

    return result


def test_against_mcallister_t():
    """
    Verify sparse tensor gives correct V and τ using McAllister's pre-solved t.

    This validates the sparse implementation without running the solver.
    """
    print("\n" + "#" * 70)
    print("# VALIDATION: Sparse κ vs McAllister's t")
    print("#" * 70)

    poly, tri, cy = load_polytope(use_dual=False, use_heights=True)
    h11, h21 = cy.h11(), cy.h21()

    # Set McAllister's basis
    basis_indices = [int(x) for x in (DATA_DIR / "basis.dat").read_text().strip().split(',')]
    cy.set_divisor_basis(basis_indices)

    kappa_sparse = SparseIntersectionTensor(cy)

    # Also build dense tensor for comparison
    kappa_dict = cy.intersection_numbers(in_basis=True)
    kappa_dense = np.zeros((h11, h11, h11))
    for (i, j, k), val in kappa_dict.items():
        for perm in [(i,j,k), (i,k,j), (j,i,k), (j,k,i), (k,i,j), (k,j,i)]:
            kappa_dense[perm] = val

    # Load McAllister's corrected t
    t = np.array([float(x) for x in (DATA_DIR / "corrected_kahler_param.dat").read_text().strip().split(',')])

    # Compare V
    V_sparse = kappa_sparse.compute_V(t)
    V_dense = np.einsum('ijk,i,j,k->', kappa_dense, t, t, t) / 6.0
    print(f"\nV comparison:")
    print(f"  V (sparse) = {V_sparse:.10f}")
    print(f"  V (dense)  = {V_dense:.10f}")
    print(f"  Difference = {abs(V_sparse - V_dense):.2e}")

    # Compare τ
    tau_sparse = kappa_sparse.compute_tau(t)
    tau_dense = 0.5 * np.einsum('ijk,j,k->i', kappa_dense, t, t)
    tau_diff = np.max(np.abs(tau_sparse - tau_dense))
    print(f"\nτ comparison:")
    print(f"  τ (sparse) first 5: {tau_sparse[:5]}")
    print(f"  τ (dense)  first 5: {tau_dense[:5]}")
    print(f"  Max difference = {tau_diff:.2e}")

    # Compare Jacobian
    J_sparse = kappa_sparse.compute_jacobian(t)
    J_dense = np.einsum('ijk,j->ik', kappa_dense, t)
    J_diff = np.max(np.abs(J_sparse - J_dense))
    print(f"\nJacobian comparison:")
    print(f"  Max difference = {J_diff:.2e}")

    # BBHL correction
    chi = 2 * (h11 - h21)
    BBHL = zeta(3) * chi / (4 * (2*np.pi)**3)
    V_string = V_sparse - BBHL

    # Target
    V_target = float((DATA_DIR / "cy_vol.dat").read_text().strip())

    print(f"\nV_string validation:")
    print(f"  V_string (sparse) = {V_string:.6f}")
    print(f"  V_target (cy_vol.dat) = {V_target:.6f}")
    print(f"  Error = {abs(V_string - V_target):.2e}")

    all_ok = (abs(V_sparse - V_dense) < 1e-6 and
              tau_diff < 1e-6 and
              J_diff < 1e-6 and
              abs(V_string - V_target) < 0.01)

    if all_ok:
        print("\n✓ ALL SPARSE OPERATIONS VALIDATED")
    else:
        print("\n✗ SPARSE IMPLEMENTATION ERROR")

    return V_string, V_target


def test_full_kklt_with_gv_dual():
    """
    Test full KKLT solver with GV corrections on dual polytope (h11=4).

    This is a fast validation that tests the complete eq 5.13 pipeline.
    """
    print("\n" + "#" * 70)
    print("# FULL KKLT WITH GV: DUAL POLYTOPE (h11=4)")
    print("#" * 70)

    poly, tri, cy = load_polytope(use_dual=True)
    h11, h21 = cy.h11(), cy.h21()

    # For dual, all 4 basis divisors are O7-planes: c_i = 6
    c_i = np.array([6.0, 6.0, 6.0, 6.0])
    g_s = 0.00911134
    W0 = 2.30012e-90

    result = mcallister_kklt_solve_with_gv(
        poly, cy, c_i, g_s, W0,
        gv_min_points=100,
        max_gv_iterations=5,
        verbose=True
    )

    print("\n" + "=" * 70)
    print("VALIDATION (DUAL WITH GV)")
    print("=" * 70)
    print(f"V_string = {result['V_string']:.2f}")
    print(f"Expected ≈ 4695 (dual approximation)")
    print(f"Converged: {result['converged']}")

    return result


def test_full_kklt_with_gv_primal():
    """
    Test full KKLT solver with GV corrections on primal polytope (h11=214).

    This is the definitive validation against McAllister's corrected_kahler_param.dat.
    Target: V_string = 4711.83
    """
    print("\n" + "#" * 70)
    print("# FULL KKLT WITH GV: PRIMAL POLYTOPE (h11=214)")
    print("#" * 70)

    poly, tri, cy = load_polytope(use_dual=False, use_heights=True)
    h11, h21 = cy.h11(), cy.h21()

    # Set McAllister's basis
    basis_indices = [int(x) for x in (DATA_DIR / "basis.dat").read_text().strip().split(',')]
    cy.set_divisor_basis(basis_indices)
    print(f"Set divisor basis: {len(basis_indices)} indices")

    # Load orientifold c_i values
    with open(RESOURCES_DIR / "mcallister_4-214-647_orientifold.json") as f:
        orientifold = json.load(f)

    basis = list(cy.divisor_basis())
    kklt_basis = orientifold['kklt_basis']
    c_values = orientifold['c_i_values']
    point_to_c = {idx: c_values[i] for i, idx in enumerate(kklt_basis)}
    c_i = np.array([float(point_to_c.get(idx, 1.0)) for idx in basis])

    n_o7 = int(np.sum(c_i == 6))
    n_d3 = int(np.sum(c_i == 1))
    print(f"c_i: {n_o7} O7-planes, {n_d3} D3-instantons")

    # Load McAllister's physics parameters
    g_s = float((DATA_DIR / "g_s.dat").read_text().strip())
    W0 = float((DATA_DIR / "W_0.dat").read_text().strip())

    result = mcallister_kklt_solve_with_gv(
        poly, cy, c_i, g_s, W0,
        basis_indices=basis_indices,
        gv_min_points=100,
        max_gv_iterations=5,
        verbose=True
    )

    # Load McAllister's target
    V_target = float((DATA_DIR / "cy_vol.dat").read_text().strip())

    print("\n" + "=" * 70)
    print("VALIDATION (PRIMAL WITH GV)")
    print("=" * 70)
    print(f"V_string (computed) = {result['V_string']:.2f}")
    print(f"V_string (expected) = {V_target:.2f}")
    print(f"Error = {abs(result['V_string'] - V_target):.2f} ({100*abs(result['V_string'] - V_target)/V_target:.2f}%)")

    # Compare t with McAllister's corrected_kahler_param.dat
    t_mcallister = np.array([float(x) for x in (DATA_DIR / "corrected_kahler_param.dat").read_text().strip().split(',')])
    t_diff = np.linalg.norm(result['t'] - t_mcallister) / np.linalg.norm(t_mcallister)
    print(f"\nt vector comparison:")
    print(f"  ||t_computed - t_mcallister|| / ||t_mcallister|| = {t_diff:.4f}")

    if result['V_string'] > 0 and abs(result['V_string'] - V_target) / V_target < 0.01:
        print("\n✓ FULL KKLT WITH GV VALIDATED (< 1% error)")
    elif result['V_string'] > 0 and abs(result['V_string'] - V_target) / V_target < 0.05:
        print("\n~ FULL KKLT WITH GV APPROXIMATELY CORRECT (< 5% error)")
    else:
        print(f"\n✗ FULL KKLT WITH GV FAILED")

    return result


if __name__ == "__main__":
    import sys

    # Parse command line args
    run_all = len(sys.argv) == 1 or "--all" in sys.argv
    run_sparse = "--sparse" in sys.argv or run_all
    run_dual = "--dual" in sys.argv or run_all
    run_primal = "--primal" in sys.argv or run_all
    run_gv_dual = "--gv-dual" in sys.argv or run_all
    run_gv_primal = "--gv-primal" in sys.argv or run_all

    if run_sparse:
        # Validate sparse tensor implementation
        test_against_mcallister_t()
        print("\n\n")

    if run_dual:
        # Test classical solver on dual (fast, h11=4)
        test_dual()
        print("\n\n")

    if run_primal:
        # Test classical solver on primal (h11=214)
        test_primal()
        print("\n\n")

    if run_gv_dual:
        # Test full GV solver on dual (fast validation)
        test_full_kklt_with_gv_dual()
        print("\n\n")

    if run_gv_primal:
        # Test full GV solver on primal (definitive validation)
        test_full_kklt_with_gv_primal()
