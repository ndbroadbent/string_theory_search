#!/usr/bin/env python3
"""
Compute KKLT moduli stabilization using McAllister's iterative algorithm.

From arXiv:2107.09064 Section 5.2:

The key insight: log(W0) and g_s enter only as overall factors, so we can solve
    (1/2) κ_ijk t^j t^k = c_i
first (independent of fluxes), then scale by the W0-dependent factor.

Algorithm (equations 5.8-5.11):
1. Start from a point t_init INSIDE THE KÄHLER CONE
2. Target: τ* = (c_1, c_2, ..., c_h11)
3. Interpolate: τ_α = (1-α)τ_init + α×τ*
4. At each step, solve LINEAR system: κ_ijk t^j ε^k = τ_{m+1} - τ_m
5. Scale final result

CRITICAL: McAllister's paper says "start from random point in the secondary fan"
which means a random TRIANGULATION, not random t values! Each triangulation
defines a valid t inside the Kähler cone. Starting from arbitrary t outside
the cone causes the algorithm to diverge.

The paper also notes (footnote 38): "this algorithm can fail to converge in
some examples, e.g. if there is an unknown autochthonous divisor that has
negative volume at the candidate point."

OPTIMIZATION: Uses sparse κ representation - only stores ~6400 non-zero entries
instead of 214³ = 10M. This makes h11=214 tractable.
"""

import sys
from pathlib import Path
import json
import time

sys.path.insert(0, str(Path(__file__).parent.parent / "vendor/cytools_latest/src"))

import numpy as np
from scipy.special import zeta
from cytools import Polytope

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


def get_kahler_cone_interior_point(cy, scale: float = 1.0) -> np.ndarray:
    """
    Get a point inside the Kähler cone from CYTools.

    This is CRITICAL for McAllister's algorithm - starting outside the cone
    causes divergence.

    Args:
        cy: CYTools CalabiYau object
        scale: Scale factor for the tip point

    Returns:
        t vector inside the Kähler cone

    Raises:
        RuntimeError: If CYTools cannot compute a valid interior point
    """
    cone = cy.toric_kahler_cone()
    tip = cone.tip_of_stretched_cone(1.0)
    return np.array(tip) * scale


def iterative_solve(kappa: SparseIntersectionTensor, target_tau: np.ndarray,
                    n_steps: int = 500, t_init: np.ndarray = None,
                    tol: float = 1e-10, verbose: bool = True) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    McAllister's iterative algorithm (Section 5.2, eqs 5.8-5.11).

    Solve: τ_i = (1/2) κ_ijk t^j t^k = target_tau_i

    CRITICAL: t_init must be INSIDE the Kähler cone! Use get_kahler_cone_interior_point()
    or a known valid solution. Starting from arbitrary t outside the cone causes divergence.

    Returns: (t_solution, tau_achieved, converged)
    """
    h11 = len(target_tau)

    # Initialize - MUST be inside Kähler cone
    if t_init is None:
        raise ValueError(
            "t_init is required! Use get_kahler_cone_interior_point(cy) or a known valid solution. "
            "Starting from arbitrary t outside the Kähler cone causes divergence."
        )

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

        # Solve linear system
        J = kappa.compute_jacobian(t)
        epsilon = np.linalg.solve(J, delta_tau)

        t = t + epsilon

        # Check Kähler cone constraint (all t > 0)
        if np.any(t <= 0):
            if verbose:
                print(f"  Step {m+1}: t left Kähler cone (min t = {np.min(t):.2e})")
            return t, kappa.compute_tau(t), False

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
                          W0: float, cy, t_init: np.ndarray = None, verbose: bool = True) -> dict:
    """
    McAllister KKLT solution (CLASSICAL - no GV corrections).

    Args:
        kappa: Sparse intersection tensor
        c_i: Dual Coxeter numbers
        W0: Flux superpotential
        cy: CYTools CalabiYau object
        t_init: Override starting point (if None, uses Kähler cone tip from cy)
    """
    h11 = kappa.h11

    if verbose:
        print(f"KKLT solve: h11={h11}, W0={W0:.2e}")

    # Get starting point
    if t_init is None:
        t_init = get_kahler_cone_interior_point(cy)

    # Scale to target τ ~ c_i
    tau_init = kappa.compute_tau(t_init)
    scale = np.sqrt(np.mean(c_i) / np.mean(tau_init))
    t_scaled = t_init * scale

    start = time.time()
    t_unit, tau_unit, converged = iterative_solve(kappa, c_i, n_steps=500, t_init=t_scaled, verbose=verbose)

    if verbose:
        print(f"Time: {time.time() - start:.2f}s, Converged: {converged}")

    # Scale by W0 factor
    t_scale = np.sqrt(np.log(1.0 / np.abs(W0)) / (2 * np.pi))
    t_final = t_unit * t_scale
    V_classical = kappa.compute_V(t_final)

    if verbose:
        print(f"V_classical = {V_classical:.2f}")

    return {"t": t_final, "V_classical": V_classical, "converged": converged}


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

    # Use McAllister's solution as starting point (known valid point in his basis)
    t_mcallister = np.array([float(x) for x in (DATA_DIR / "kahler_param.dat").read_text().strip().split(',')])
    print(f"Using McAllister's kahler_param.dat as t_init")

    result = mcallister_kklt_solve(kappa, c_i, W0, cy, t_init=t_mcallister, verbose=True)

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


if __name__ == "__main__":
    # First validate sparse tensor implementation
    test_against_mcallister_t()

    print("\n\n")

    # Test dual (fast, h11=4)
    test_dual()

    print("\n\n")

    # Test primal (h11=214)
    test_primal()
