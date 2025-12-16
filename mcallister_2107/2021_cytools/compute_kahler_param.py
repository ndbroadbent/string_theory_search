#!/usr/bin/env python3
"""
Compute Kähler moduli t from KKLT stabilization.

ALGORITHM: McAllister's path-following (Section 5.2)
- Interpolate from τ_init to τ_target in N steps
- At each step, solve LINEAR system: κᵢⱼₖ tʲ εᵏ = Δτᵢ
- This is O(N × h11³), NOT nonlinear optimization

CRITICAL: basis.dat vs kklt_basis.dat differ by 4 divisors.
- For basis divisors: τ_i = (1/2) κ_ijk t^j t^k (in_basis=True)
- For non-basis KKLT: τ_a = (1/2) κ_{a,basis[j],basis[k]} t^j t^k (in_basis=False)

Validation: Tests against corrected_kahler_param.dat for all 5 McAllister examples.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

# Paths
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent.parent
DATA_BASE = ROOT_DIR / "resources/small_cc_2107.09064_source/anc/paper_data"
CYTOOLS_2021 = ROOT_DIR / "vendor/cytools_mcallister_2107"
CYTOOLS_LATEST = ROOT_DIR / "vendor/cytools_latest/src"

sys.path.insert(0, str(CYTOOLS_2021))

from cytools import Polytope

# Import from sibling modules
from compute_target_tau import compute_c_tau
from compute_rigidity_combinatorial import compute_rigidity

MCALLISTER_EXAMPLES = [
    ("4-214-647", 214, 4, True),
    ("5-113-4627-main", 113, 5, True),
    ("5-113-4627-alternative", 113, 5, True),
    ("5-81-3213", 81, 5, True),
    ("7-51-13590", 51, 7, False),
]


# =============================================================================
# SPARSE KAPPA TENSOR
# =============================================================================


class SparseKappa:
    """
    Sparse representation of intersection tensor κ.

    Stores κ as a list of (i, j, k, val) tuples and builds sparse
    matrices for efficient τ and Jacobian computation.
    """

    def __init__(self, kappa_basis, kappa_all, basis: list, kklt_basis: list):
        """
        Args:
            kappa_basis: Intersection numbers in basis coords (in_basis=True)
            kappa_all: Intersection numbers in all-divisor coords (in_basis=False)
            basis: List of point indices for divisor basis
            kklt_basis: List of point indices for KKLT divisors
        """
        self.h11 = len(basis)
        self.n_kklt = len(kklt_basis)
        self.basis = basis
        self.kklt_basis = kklt_basis
        self.pt_to_basis_idx = {pt: i for i, pt in enumerate(basis)}

        # Build basis kappa entries (handle both array and dict formats)
        self.kappa_basis_entries = []
        if hasattr(kappa_basis, 'items'):
            # Dict format (latest CYTools)
            for (i, j, k), val in kappa_basis.items():
                if val != 0:
                    self.kappa_basis_entries.append((i, j, k, val))
        else:
            # Array format (CYTools 2021)
            for row in kappa_basis:
                i, j, k = int(row[0]), int(row[1]), int(row[2])
                val = row[3]
                if val != 0:
                    self.kappa_basis_entries.append((i, j, k, val))

        # Build all-divisor kappa dict for non-basis KKLT divisors
        self.kappa_all_dict = {}
        if hasattr(kappa_all, 'items'):
            # Dict format (latest CYTools)
            for (i, j, k), val in kappa_all.items():
                if val != 0:
                    for perm in [(i,j,k), (i,k,j), (j,i,k), (j,k,i), (k,i,j), (k,j,i)]:
                        self.kappa_all_dict[perm] = val
        else:
            # Array format (CYTools 2021)
            for row in kappa_all:
                i, j, k = int(row[0]), int(row[1]), int(row[2])
                val = row[3]
                if val != 0:
                    for perm in [(i,j,k), (i,k,j), (j,i,k), (j,k,i), (k,i,j), (k,j,i)]:
                        self.kappa_all_dict[perm] = val

        # Precompute mapping from kklt index to basis index (or -1 if non-basis)
        self.kklt_to_basis = []
        for pt in kklt_basis:
            self.kklt_to_basis.append(self.pt_to_basis_idx.get(pt, -1))

    def compute_tau_kklt(self, t: np.ndarray) -> np.ndarray:
        """
        Compute τ for all KKLT divisors.

        Formula: τ_i = (1/2) Σ_{j,k} κ_{ijk} t^j t^k

        For basis divisors, build full tensor and use einsum for correctness.
        For non-basis divisors, use kappa_all_dict with mixed indices.
        """
        tau = np.zeros(self.n_kklt)

        # Build full tensor for basis (required for correct symmetrization)
        kappa_tensor = np.zeros((self.h11, self.h11, self.h11))
        for i, j, k, val in self.kappa_basis_entries:
            # Fill ALL permutations (symmetric tensor)
            for perm in [(i,j,k), (i,k,j), (j,i,k), (j,k,i), (k,i,j), (k,j,i)]:
                kappa_tensor[perm] = val

        # τ_i = (1/2) κ_ijk t^j t^k
        tau_basis = 0.5 * np.einsum('ijk,j,k->i', kappa_tensor, t, t)

        # Map basis τ to kklt τ
        for a in range(self.n_kklt):
            basis_idx = self.kklt_to_basis[a]
            if basis_idx >= 0:
                tau[a] = tau_basis[basis_idx]
            else:
                # Non-basis divisor - use kappa_all_dict (already symmetrized)
                pt = self.kklt_basis[a]
                tau_a = 0.0
                for j, bj in enumerate(self.basis):
                    for k, bk in enumerate(self.basis):
                        key = (pt, bj, bk)
                        if key in self.kappa_all_dict:
                            tau_a += self.kappa_all_dict[key] * t[j] * t[k]
                tau[a] = 0.5 * tau_a

        return tau

    def compute_jacobian_kklt(self, t: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian J[a,k] = ∂τ_a/∂t^k for KKLT divisors.

        From τ_i = (1/2) κ_{ijk} t^j t^k:
        J[i,k] = ∂τ_i/∂t^k = κ_{ijk} t^j  (factor of 2 from symmetry of j,k)
        """
        J = np.zeros((self.n_kklt, self.h11))

        # Build full tensor (reuse same symmetrization as compute_tau_kklt)
        kappa_tensor = np.zeros((self.h11, self.h11, self.h11))
        for i, j, k, val in self.kappa_basis_entries:
            for perm in [(i,j,k), (i,k,j), (j,i,k), (j,k,i), (k,i,j), (k,j,i)]:
                kappa_tensor[perm] = val

        # J[i,k] = κ_{ijk} t^j
        J_basis = np.einsum('ijk,j->ik', kappa_tensor, t)

        # Map to KKLT indices
        for a in range(self.n_kklt):
            basis_idx = self.kklt_to_basis[a]
            if basis_idx >= 0:
                J[a, :] = J_basis[basis_idx, :]
            else:
                # Non-basis divisor - use kappa_all_dict
                pt = self.kklt_basis[a]
                for j, bj in enumerate(self.basis):
                    for k, bk in enumerate(self.basis):
                        key = (pt, bj, bk)
                        if key in self.kappa_all_dict:
                            # J[a,k] = Σ_j κ_{a,j,k} t^j
                            J[a, k] += self.kappa_all_dict[key] * t[j]

        return J


# =============================================================================
# CHI COMPUTATION
# =============================================================================


def compute_chi_for_kklt_divisors(poly, kappa_all, kklt_basis: list) -> np.ndarray:
    """Compute χ(D_i) = 12 × χ(O_D) - D³ for KKLT divisors."""
    kappa_dict = {}
    if hasattr(kappa_all, 'items'):
        # Dict format (latest CYTools)
        for (i, j, k), val in kappa_all.items():
            kappa_dict[(i, j, k)] = val
    else:
        # Array format (CYTools 2021)
        for row in kappa_all:
            i, j, k = int(row[0]), int(row[1]), int(row[2])
            kappa_dict[(i, j, k)] = row[3]

    rigidity = compute_rigidity(poly)
    chi = np.zeros(len(kklt_basis))

    for idx, pt in enumerate(kklt_basis):
        r = rigidity.get(pt, {})
        g = r.get("n_interior", 0)
        pt_type = r.get("type", "unknown")

        if pt_type == "vertex":
            chi_O = 1 + g
        elif "1-face" in pt_type:
            chi_O = 1 - g
        else:
            chi_O = 1 + g

        D_cubed = kappa_dict.get((pt, pt, pt), 0)
        chi[idx] = 12 * chi_O - D_cubed

    return chi


# =============================================================================
# NEWTON-RAPHSON SOLVER WITH LINE SEARCH
# =============================================================================


def solve_kklt_newton(kappa: SparseKappa, tau_target: np.ndarray,
                      max_iters: int = 500, tol: float = 1e-6) -> tuple:
    """
    Solve τ(t) = τ_target using Newton-Raphson with backtracking line search.

    This is more robust than path-following when the initialization is far
    from the solution (e.g., when t has many negative components).

    Returns:
        (t, tau_achieved, converged)
    """
    h11 = kappa.h11

    # Try multiple starting points
    best_error = float('inf')
    best_t = None

    for init_val in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        t = np.ones(h11) * init_val

        for _ in range(max_iters):
            tau_current = kappa.compute_tau_kklt(t)
            delta_tau = tau_target - tau_current

            error = np.sqrt(np.mean(delta_tau**2))
            if error < tol:
                if error < best_error:
                    best_error = error
                    best_t = t.copy()
                break

            # Compute Jacobian and use truncated SVD for numerical stability
            J = kappa.compute_jacobian_kklt(t)
            U, s, Vt = np.linalg.svd(J, full_matrices=False)

            # Truncated SVD inverse (regularization for near-singular Jacobian)
            s_inv = np.where(s > 1e-10 * s.max(), 1.0 / s, 0)
            epsilon = Vt.T @ (s_inv * (U.T @ delta_tau))

            # Backtracking line search
            step_size = 1.0
            for _ in range(10):
                t_new = t + step_size * epsilon
                tau_new = kappa.compute_tau_kklt(t_new)
                new_error = np.sqrt(np.mean((tau_new - tau_target)**2))
                if new_error < error:
                    t = t_new
                    break
                step_size *= 0.5
            else:
                break  # Line search failed, try next init

        # Check if this init gave a better result
        tau_final = kappa.compute_tau_kklt(t)
        error = np.sqrt(np.mean((tau_final - tau_target)**2))
        if error < best_error:
            best_error = error
            best_t = t.copy()

    if best_t is None:
        # Fallback to uniform init result
        best_t = np.ones(h11)
        best_error = float('inf')

    tau_achieved = kappa.compute_tau_kklt(best_t)
    converged = best_error < 0.001 * np.mean(tau_target)

    return best_t, tau_achieved, converged


def _path_follow(kappa: SparseKappa, tau_target: np.ndarray,
                 t_init: np.ndarray, n_steps: int = 200) -> tuple:
    """Internal: Path-follow from t_init to tau_target."""
    t = t_init.copy()
    tau_init = kappa.compute_tau_kklt(t)

    for m in range(n_steps):
        alpha = (m + 1) / n_steps
        tau_step = (1 - alpha) * tau_init + alpha * tau_target

        tau_current = kappa.compute_tau_kklt(t)
        delta_tau = tau_step - tau_current

        J = kappa.compute_jacobian_kklt(t)
        epsilon, _, _, _ = np.linalg.lstsq(J, delta_tau, rcond=1e-8)
        t = t + epsilon

        if np.abs(t).max() > 1e6:
            return None, None, False  # Diverged

    tau_achieved = kappa.compute_tau_kklt(t)
    error = np.sqrt(np.mean((tau_achieved - tau_target)**2))
    rel_error = error / np.mean(tau_target)

    return t, tau_achieved, rel_error < 0.001


def solve_kklt_path_following(kappa: SparseKappa, tau_target: np.ndarray,
                               c_i: np.ndarray = None, n_steps: int = 200) -> tuple:
    """
    Two-phase path-following algorithm:

    Phase 1: Solve uniform t → τ = c_i (uncorrected KKLT, correct branch)
    Phase 2: Path-follow from that t to target τ (with corrections)

    The key insight: McAllister's "uncorrected" t satisfies τ = c_i exactly.
    This is the correct branch that gives positive V. Starting from any other
    initialization risks landing on a different branch with negative V.

    Falls back to Newton-Raphson if both phases fail.

    Args:
        kappa: SparseKappa object with intersection numbers
        tau_target: Target τ values (c_i/c_τ + χ/24)
        c_i: Dual Coxeter numbers (1 for D3, 6 for O7-planes)
        n_steps: Number of path-following steps per phase

    Returns:
        (t, tau_achieved, converged)
    """
    h11 = kappa.h11
    n_kklt = kappa.n_kklt

    # Phase 1: Find t where τ = c_i (uncorrected KKLT = correct branch)
    if c_i is not None:
        # Use c_i directly as the phase 1 target
        tau_phase1 = c_i.astype(float)
    else:
        # Fallback: estimate c_i from tau_target pattern
        # c_i values are 1 or 6, so tau_target ∝ c_i
        # Use scaled version that preserves the ratio pattern
        tau_phase1 = tau_target * (np.mean([1, 6]) / np.mean(tau_target))

    # Start from uniform t, scaled to match phase 1 target magnitude
    t_uniform = np.ones(h11) * 0.5
    tau_at_uniform = kappa.compute_tau_kklt(t_uniform)

    if np.mean(tau_at_uniform) > 0:
        scale = np.sqrt(np.mean(tau_phase1) / np.mean(tau_at_uniform))
        t_uniform = t_uniform * scale

    # Phase 1: uniform → τ = c_i (correct branch)
    t_phase1, _, conv1 = _path_follow(kappa, tau_phase1, t_uniform, n_steps)

    if conv1 and t_phase1 is not None:
        # Phase 2: τ = c_i → target τ (with corrections)
        t_final, tau_final, conv2 = _path_follow(kappa, tau_target, t_phase1, n_steps)
        if conv2 and t_final is not None:
            return t_final, tau_final, True

    # Fallback: Try Newton-Raphson
    return solve_kklt_newton(kappa, tau_target)


# =============================================================================
# PURE COMPUTATION FUNCTION
# =============================================================================


def compute_kahler_param(kappa_basis, kappa_all, c_i: np.ndarray, g_s: float,
                         W0: float, chi_kklt: np.ndarray, basis: list,
                         kklt_basis: list) -> dict:
    """
    Compute Kähler moduli t from KKLT stabilization.

    Uses path-following with LINEAR solves (NOT scipy optimization).
    """
    kappa = SparseKappa(kappa_basis, kappa_all, basis, kklt_basis)

    # Target τ (zeroth order: c_i/c_τ + χ/24)
    c_tau = compute_c_tau(g_s, W0)
    tau_target = c_i / c_tau + chi_kklt / 24.0

    # Solve using path-following
    t, tau_achieved, converged = solve_kklt_path_following(kappa, tau_target)

    return {
        "t": t,
        "tau_kklt_achieved": tau_achieved,
        "tau_kklt_target": tau_target,
        "c_tau": c_tau,
        "converged": converged,
    }


# =============================================================================
# DATA LOADING
# =============================================================================


def load_primal_points(example_name: str) -> np.ndarray:
    lines = (DATA_BASE / example_name / "points.dat").read_text().strip().split('\n')
    return np.array([[int(x) for x in line.split(',')] for line in lines])


def load_heights(example_name: str) -> np.ndarray:
    text = (DATA_BASE / example_name / "corrected_heights.dat").read_text().strip()
    return np.array([float(x) for x in text.split(',')])


def load_basis(example_name: str) -> list:
    text = (DATA_BASE / example_name / "basis.dat").read_text().strip()
    return [int(x) for x in text.split(',')]


def load_kklt_basis(example_name: str) -> list:
    text = (DATA_BASE / example_name / "kklt_basis.dat").read_text().strip()
    return [int(x) for x in text.split(',')]


def load_c_i(example_name: str) -> np.ndarray:
    text = (DATA_BASE / example_name / "target_volumes.dat").read_text().strip()
    return np.array([int(x) for x in text.split(',')])


def load_g_s(example_name: str) -> float:
    return float((DATA_BASE / example_name / "g_s.dat").read_text().strip())


def load_W0(example_name: str) -> float:
    return float((DATA_BASE / example_name / "W_0.dat").read_text().strip())


def load_kahler_params_expected(example_name: str) -> np.ndarray:
    text = (DATA_BASE / example_name / "corrected_kahler_param.dat").read_text().strip()
    return np.array([float(x) for x in text.split(',')])


def load_target_volumes_expected(example_name: str) -> np.ndarray:
    text = (DATA_BASE / example_name / "corrected_target_volumes.dat").read_text().strip()
    return np.array([float(x) for x in text.split(',')])


def get_cy(points: np.ndarray, heights: np.ndarray, basis: list, is_favorable: bool):
    """Build CY, handling favorable vs non-favorable."""
    if is_favorable:
        poly = Polytope(points)
        tri = poly.triangulate(heights=heights)
        cy = tri.get_cy()
        cy.set_divisor_basis(basis)
        return cy, poly
    else:
        # Use latest CYTools for non-favorable
        mods = [k for k in list(sys.modules.keys()) if 'cytools' in k]
        for m in mods:
            del sys.modules[m]

        sys.path.insert(0, str(CYTOOLS_LATEST))
        from cytools import Polytope as PL
        from cytools import config
        config.enable_experimental_features()

        poly = PL(points)
        tri = poly.triangulate(heights=heights)
        cy = tri.get_cy()
        cy.set_divisor_basis(basis)

        # Restore 2021
        mods = [k for k in list(sys.modules.keys()) if 'cytools' in k]
        for m in mods:
            del sys.modules[m]
        sys.path.remove(str(CYTOOLS_LATEST))
        sys.path.insert(0, str(CYTOOLS_2021))

        return cy, poly


# =============================================================================
# VALIDATION TEST
# =============================================================================


def test_example(example_name: str, h11_expected: int, h21_expected: int,
                 is_favorable: bool, verbose: bool = True) -> dict:
    """Test Kähler param computation against McAllister's data."""
    if verbose:
        print("=" * 70)
        print(f"TEST - {example_name} (h11={h11_expected})")
        print("=" * 70)

    # Load data
    points = load_primal_points(example_name)
    heights = load_heights(example_name)
    basis = load_basis(example_name)
    kklt_basis = load_kklt_basis(example_name)
    c_i = load_c_i(example_name)
    g_s = load_g_s(example_name)
    W0 = load_W0(example_name)
    t_expected = load_kahler_params_expected(example_name)
    tau_kklt_expected = load_target_volumes_expected(example_name)

    if verbose:
        print(f"\n  Loading CY and computing intersection numbers...")

    # Build CY
    cy, poly = get_cy(points, heights, basis, is_favorable)

    # Get intersection numbers
    kappa_basis = cy.intersection_numbers(in_basis=True)
    kappa_all = cy.intersection_numbers(in_basis=False)

    # Build sparse kappa
    kappa = SparseKappa(kappa_basis, kappa_all, basis, kklt_basis)

    if verbose:
        print(f"  Inputs:")
        print(f"    g_s = {g_s:.6f}, W₀ = {W0:.2e}")
        print(f"    c_i: {sum(c_i == 6)} O7s, {sum(c_i == 1)} D3s")
        print(f"    basis: {len(basis)}, kklt_basis: {len(kklt_basis)}")
        shared = len(set(basis) & set(kklt_basis))
        print(f"    Shared: {shared}")

    # ===========================================================================
    # TEST 1: Solve using McAllister's τ_expected (verify solver)
    # ===========================================================================
    if verbose:
        print(f"\n  [Test 1] Solving with McAllister's τ as target...")

    t_from_tau, tau_achieved, converged = solve_kklt_path_following(
        kappa, tau_kklt_expected, n_steps=200
    )

    # Compare achieved τ with expected
    rms_tau = np.sqrt(np.mean((tau_achieved - tau_kklt_expected)**2))
    rel_rms_tau = rms_tau / np.mean(tau_kklt_expected)

    # Compare t with expected
    corr_t = np.corrcoef(t_from_tau, t_expected)[0, 1]

    if verbose:
        print(f"    Converged: {converged}")
        print(f"    τ rel_rms: {rel_rms_tau:.6f}")
        print(f"    t correlation: {corr_t:.6f}")
        print(f"    t range: [{t_from_tau.min():.4f}, {t_from_tau.max():.4f}]")
        print(f"    t expected: [{t_expected.min():.4f}, {t_expected.max():.4f}]")

    # ===========================================================================
    # TEST 2: Solve using our formula (c_i/c_τ + χ/24)
    # ===========================================================================
    if verbose:
        print(f"\n  [Test 2] Solving with our zeroth-order formula...")

    chi_kklt = compute_chi_for_kklt_divisors(poly, kappa_all, kklt_basis)
    c_tau = compute_c_tau(g_s, W0)
    tau_zeroth = c_i / c_tau + chi_kklt / 24.0

    t_from_zeroth, tau_achieved_zeroth, converged_zeroth = solve_kklt_path_following(
        kappa, tau_zeroth, n_steps=200
    )

    # Compare with McAllister's τ
    rms_tau_zeroth = np.sqrt(np.mean((tau_achieved_zeroth - tau_kklt_expected)**2))
    rel_rms_tau_zeroth = rms_tau_zeroth / np.mean(tau_kklt_expected)

    if verbose:
        print(f"    χ(D) range: [{chi_kklt.min():.1f}, {chi_kklt.max():.1f}]")
        print(f"    c_τ = {c_tau:.4f}")
        print(f"    Converged: {converged_zeroth}")
        print(f"    τ_zeroth rel_rms vs McAllister: {rel_rms_tau_zeroth:.4f}")

    # Pass if Test 1 works (solver is correct)
    passed = converged and rel_rms_tau < 0.001

    if verbose:
        print(f"\n{'PASS' if passed else 'FAIL'}: {example_name}")
        print(f"  (Test 1 τ error: {rel_rms_tau:.6f}, Test 2 τ error vs McAllister: {rel_rms_tau_zeroth:.4f})")

    return {
        "example_name": example_name,
        "passed": passed,
        "rel_rms_tau": rel_rms_tau,
        "correlation_t": corr_t,
        "converged": converged,
        "rel_rms_tau_zeroth": rel_rms_tau_zeroth,
    }


def main():
    """Test all 5 McAllister examples."""
    print("=" * 70)
    print("COMPUTE KÄHLER PARAMS - PATH-FOLLOWING ALGORITHM")
    print("=" * 70)

    results = []
    for name, h11, h21, is_favorable in MCALLISTER_EXAMPLES:
        result = test_example(name, h11, h21, is_favorable, verbose=True)
        results.append(result)
        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_passed = True
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  {status}: {r['example_name']:30s} "
              f"τ_err={r['rel_rms_tau']:.6f} t_corr={r['correlation_t']:.4f}")
        all_passed = all_passed and r["passed"]

    print()
    if all_passed:
        print("All 5 examples PASSED")
    else:
        print("Some examples FAILED")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
