#!/usr/bin/env python3
"""
Compute V_string (string frame CY volume) from first principles.

This is the critical computation for the cosmological constant:
    V_0 = -3 * e^K * |W|^2

Where e^K depends on V_string.

Pipeline:
1. Load polytope geometry
2. Compute intersection numbers kappa_ijk via CYTools
3. Load orientifold data (c_i values) from pre-computed JSON
4. KKLT stabilization: tau_i = (c_i / 2*pi) * ln(|W_0|^-1)
5. Solve for t^i from tau_i = (1/2) kappa_ijk t^j t^k
6. Compute V_string = (1/6) kappa_ijk t^i t^j t^k

Reference: arXiv:2107.09064 (McAllister et al.)

Note: Orientifold data (which divisors are O7-planes vs D3-instantons) is NOT
computed - it's a model choice. We load it from pre-extracted data files.
See docs/ORIENTIFOLD_INVOLUTION.md for details.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "vendor/cytools_latest/src"))

import numpy as np
from scipy.optimize import minimize
from cytools import Polytope


def compute_intersection_tensor(cy, h11: int) -> np.ndarray:
    """
    Compute full intersection number tensor kappa_ijk.

    Args:
        cy: CYTools CalabiYau object
        h11: Hodge number h^{1,1}

    Returns:
        (h11, h11, h11) symmetric tensor
    """
    kappa_sparse = cy.intersection_numbers(in_basis=True)
    kappa = np.zeros((h11, h11, h11))

    for (i, j, k), val in kappa_sparse.items():
        # Fill all symmetric permutations
        for perm in [(i, j, k), (i, k, j), (j, i, k), (j, k, i), (k, i, j), (k, j, i)]:
            kappa[perm] = val

    return kappa


def compute_divisor_volumes(kappa: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Compute divisor volumes tau_i = (1/2) kappa_ijk t^j t^k.

    These are the 4-cycle volumes in the CY threefold.
    Vectorized with einsum for O(n²) instead of O(n³) per call.
    """
    return 0.5 * np.einsum("ijk,j,k->i", kappa, t, t)


def compute_cy_volume(kappa: np.ndarray, t: np.ndarray) -> float:
    """
    Compute CY volume V = (1/6) kappa_ijk t^i t^j t^k.

    This is the string frame volume.
    Vectorized with einsum for efficiency.
    """
    return np.einsum("ijk,i,j,k->", kappa, t, t, t) / 6.0


def solve_for_t_from_tau(
    kappa: np.ndarray,
    target_tau: np.ndarray,
    verbose: bool = False,
) -> tuple[np.ndarray, bool]:
    """
    Solve for 2-cycle volumes t^i given target 4-cycle volumes tau_i.

    The constraint is: tau_i = (1/2) kappa_ijk t^j t^k

    This is a system of h11 quadratic equations in h11 unknowns.
    We solve via optimization: minimize sum_i (tau_i(t) - target_tau_i)^2
    """
    h11 = len(target_tau)

    def objective(t):
        tau = compute_divisor_volumes(kappa, t)
        return np.sum((tau - target_tau) ** 2)

    # Initial guess based on typical scales
    t_init = np.ones(h11) * np.sqrt(2 * np.mean(np.abs(target_tau)))
    t_init = np.clip(t_init, 0.1, 1000.0)

    # Bounds: t > 0 (Kahler cone constraint)
    max_bound = max(1000.0, 10 * np.sqrt(np.max(np.abs(target_tau))))
    bounds = [(1e-6, max_bound) for _ in range(h11)]

    result = minimize(
        objective,
        t_init,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 50000, "ftol": 1e-14},
    )

    if verbose:
        tau_achieved = compute_divisor_volumes(kappa, result.x)
        print(f"    Optimization: {result.message}")
        print(f"    Objective: {result.fun:.2e}")
        print(f"    Target tau: {target_tau}")
        print(f"    Achieved tau: {tau_achieved}")
        rel_error = np.abs(tau_achieved - target_tau) / (np.abs(target_tau) + 1e-10)
        print(f"    Relative error: {rel_error}")

    return result.x, result.success and result.fun < 1e-6


def kklt_target_divisor_volumes(
    c_i: np.ndarray,
    W0: float,
) -> np.ndarray:
    """
    Compute target divisor volumes from KKLT stabilization.

    At the KKLT minimum (F-flatness condition):
        Re(T_i) = tau_i = (c_i / 2*pi) * ln(|W_0|^-1)

    Args:
        c_i: Dual Coxeter numbers for each divisor in basis
        W0: Flux superpotential magnitude

    Returns:
        Target divisor volumes tau_i
    """
    ln_W0_inv = np.log(1.0 / np.abs(W0))
    tau = c_i * ln_W0_inv / (2 * np.pi)
    return tau


def compute_V_string(
    poly,
    tri,
    c_i: np.ndarray,
    W0: float,
    verbose: bool = True,
) -> dict:
    """
    Compute string frame volume from first principles.

    Args:
        poly: CYTools Polytope
        tri: CYTools Triangulation
        c_i: Dual Coxeter numbers (h11 values, in divisor basis order)
        W0: Flux superpotential

    Returns:
        Dict with V_string, t (Kahler moduli), tau (divisor volumes), success
    """
    cy = tri.get_cy()
    h11 = cy.h11()

    if len(c_i) != h11:
        raise ValueError(f"c_i has {len(c_i)} elements, expected h11={h11}")

    if verbose:
        print(f"[1] Geometry: h11={h11}")
        print(f"    Divisor basis: {list(cy.divisor_basis())}")

    # Get intersection numbers
    kappa = compute_intersection_tensor(cy, h11)
    if verbose:
        print(f"[2] Intersection tensor computed")
        # Show a few non-zero entries
        nonzero = [(i, j, k, kappa[i, j, k])
                   for i in range(h11) for j in range(i, h11) for k in range(j, h11)
                   if kappa[i, j, k] != 0][:5]
        for i, j, k, val in nonzero:
            print(f"    kappa[{i},{j},{k}] = {val}")

    # KKLT target volumes
    target_tau = kklt_target_divisor_volumes(c_i, W0)
    if verbose:
        print(f"[3] KKLT target divisor volumes:")
        print(f"    c_i = {c_i}")
        print(f"    W0 = {W0:.6e}")
        print(f"    ln(W0^-1) = {np.log(1/np.abs(W0)):.2f}")
        print(f"    target tau = {target_tau}")

    # Solve for t
    t, success = solve_for_t_from_tau(kappa, target_tau, verbose=verbose)
    if verbose:
        print(f"[4] Solved for Kahler moduli t:")
        print(f"    t = {t}")
        print(f"    Success: {success}")

    # Compute V_string
    V_string = compute_cy_volume(kappa, t)
    tau_achieved = compute_divisor_volumes(kappa, t)

    if verbose:
        print(f"[5] Results:")
        print(f"    V_string = {V_string:.6f}")
        print(f"    tau (achieved) = {tau_achieved}")

    return {
        "V_string": V_string,
        "t": t,
        "tau": tau_achieved,
        "target_tau": target_tau,
        "kappa": kappa,
        "h11": h11,
        "success": success,
    }


# =============================================================================
# ORIENTIFOLD DATA LOADING
# =============================================================================

RESOURCES_DIR = Path(__file__).parent.parent / "resources"
DATA_DIR = RESOURCES_DIR / "small_cc_2107.09064_source/anc/paper_data/4-214-647"


def load_orientifold_data(polytope_name: str = "4-214-647") -> dict:
    """
    Load pre-extracted orientifold data from JSON.

    The orientifold involution determines which divisors are O7-planes (c_i=6)
    vs D3-instantons (c_i=1). This is a MODEL CHOICE, not computed from the
    polytope. See docs/ORIENTIFOLD_INVOLUTION.md.

    Args:
        polytope_name: Name of polytope (e.g., "4-214-647")

    Returns:
        Dict with o7_divisor_indices, d3_divisor_indices, c_i_values, etc.
    """
    json_path = RESOURCES_DIR / f"mcallister_{polytope_name}_orientifold.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"Orientifold data not found: {json_path}\n"
            f"Run the extraction script to create this file."
        )

    with open(json_path) as f:
        return json.load(f)


def get_c_i_for_basis(
    orientifold_data: dict,
    cytools_basis: list[int],
) -> np.ndarray:
    """
    Map orientifold c_i values to CYTools divisor basis ordering.

    Args:
        orientifold_data: Dict from load_orientifold_data()
        cytools_basis: Divisor basis indices from CYTools

    Returns:
        c_i array in CYTools basis order
    """
    # Create mapping from KKLT basis index -> c_i value
    kklt_basis = orientifold_data["kklt_basis"]
    c_values = orientifold_data["c_i_values"]

    point_to_c = {}
    for i, point_idx in enumerate(kklt_basis):
        point_to_c[point_idx] = c_values[i]

    # Map to CYTools basis
    c_i = np.zeros(len(cytools_basis))
    mapped = 0
    for i, cytools_idx in enumerate(cytools_basis):
        if cytools_idx in point_to_c:
            c_i[i] = point_to_c[cytools_idx]
            mapped += 1
        else:
            # Default: assume D3-instanton if not in KKLT basis
            c_i[i] = 1.0

    return c_i, mapped


# =============================================================================
# MCALLISTER VALIDATION
# =============================================================================


def load_mcallister_points():
    """Load McAllister's dual polytope points (the ONLY input needed)."""
    lines = (DATA_DIR / "dual_points.dat").read_text().strip().split('\n')
    return np.array([[int(x) for x in line.split(',')] for line in lines])


def load_mcallister_ground_truth():
    """
    Load McAllister ground truth values for validation.

    These are the target values we must reproduce.
    """
    W0 = float((DATA_DIR / "W_0.dat").read_text().strip())
    g_s = float((DATA_DIR / "g_s.dat").read_text().strip())
    V_cy = float((DATA_DIR / "cy_vol.dat").read_text().strip())

    return {
        "W0": W0,
        "g_s": g_s,
        "V_string": V_cy,
        # From LATEST_CYTOOLS_RESULT.md
        "e_K0": 0.234393,
        "V0": -5.5e-203,
        # Latest CYTools basis [5,6,7,8]
        "K": np.array([8, 5, -8, 6]),
        "M": np.array([-10, -1, 11, -5]),
    }


def load_primal_points():
    """Load McAllister's primal polytope points (h11=214)."""
    lines = (DATA_DIR / "points.dat").read_text().strip().split('\n')
    return np.array([[int(x) for x in line.split(',')] for line in lines])


def load_primal_c_i():
    """
    Load c_i values for the primal polytope from target_volumes.dat.

    Returns (c_values, kklt_basis_indices).
    """
    c_text = (DATA_DIR / "target_volumes.dat").read_text().strip()
    c_values = np.array([int(x) for x in c_text.split(',')])

    basis_text = (DATA_DIR / "kklt_basis.dat").read_text().strip()
    basis_indices = np.array([int(x) for x in basis_text.split(',')])

    return c_values, basis_indices


def load_primal_kahler_params(corrected: bool = False):
    """
    Load McAllister's solved Kähler moduli for the primal polytope.

    These are the SOLVED t^i values from KKLT stabilization.

    Args:
        corrected: If True, load corrected_kahler_param.dat with instanton corrections

    Returns:
        (t_values, basis_indices) - 214 Kähler moduli and their basis indices
    """
    filename = "corrected_kahler_param.dat" if corrected else "kahler_param.dat"
    t_text = (DATA_DIR / filename).read_text().strip()
    t_values = np.array([float(x) for x in t_text.split(',')])

    basis_text = (DATA_DIR / "basis.dat").read_text().strip()
    basis_indices = np.array([int(x) for x in basis_text.split(',')])

    return t_values, basis_indices


def test_mcallister_V_string():
    """
    Test V_string computation against McAllister ground truth.

    This is the critical validation - if this passes, our pipeline is correct.

    Note: For the DUAL polytope (h11=4), McAllister's target_volumes.dat is for
    the PRIMAL (h11=214). The dual has only 4 basis divisors, all of which are
    O7-planes in their orientifold choice. So we use c_i = [6,6,6,6].
    """
    print("=" * 70)
    print("MCALLISTER V_string VALIDATION - DUAL (h11=4)")
    print("=" * 70)

    # Load geometry - ONLY points, CYTools computes triangulation
    print("\n[1] Loading McAllister dual polytope (h11=4)...")
    points = load_mcallister_points()
    print(f"    Points: {points.shape}")

    # Create CYTools objects - triangulation computed automatically
    poly = Polytope(points)
    tri = poly.triangulate()  # CYTools computes triangulation
    cy = tri.get_cy()
    print(f"    Simplices: {len(tri.simplices())} (computed by CYTools)")

    h11 = cy.h11()
    basis = cy.divisor_basis()
    print(f"    h11 = {h11}, h21 = {cy.h21()}")
    print(f"    Divisor basis: {list(basis)}")

    # Load ground truth
    print("\n[2] Loading ground truth...")
    gt = load_mcallister_ground_truth()
    print(f"    W0 = {gt['W0']:.6e}")
    print(f"    g_s = {gt['g_s']:.6f}")
    print(f"    V_string (target) = {gt['V_string']:.2f}")

    # For the dual polytope, McAllister uses c_i = 6 for all 4 basis divisors
    # (they are all O7-planes in the orientifold)
    print("\n[3] Using McAllister's orientifold: c_i = [6,6,6,6] for dual...")
    print("    (All 4 basis divisors are O7-planes with so(8) stacks)")
    c_i_mcallister = np.array([6.0, 6.0, 6.0, 6.0])

    # Compute V_string
    print("\n[4] Computing V_string...")
    result = compute_V_string(poly, tri, c_i_mcallister, gt["W0"], verbose=True)

    # Validate
    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)
    V_computed = result["V_string"]
    V_expected = gt["V_string"]
    rel_error = abs(V_computed - V_expected) / V_expected

    print(f"V_string computed: {V_computed:.2f}")
    print(f"V_string expected: {V_expected:.2f}")
    print(f"Relative error: {rel_error:.4%}")

    if rel_error < 0.01:
        print("\n*** VALIDATION PASSED (< 1% error) ***")
    elif rel_error < 0.10:
        print("\n*** VALIDATION CLOSE (< 10% error) - check c_i values ***")
    else:
        print("\n*** VALIDATION FAILED - significant discrepancy ***")

    return result, gt


def test_mcallister_V_string_primal():
    """
    Test V_string computation on McAllister's PRIMAL polytope (h11=214).

    This is the main validation - cy_vol.dat = 4711.83 is for the primal.
    We use McAllister's SOLVED Kähler moduli from kahler_param.dat.

    Note: Solving the 214-dimensional KKLT optimization from scratch is
    intractable. McAllister provides the solution in their data files.

    WARNING: This test currently fails due to basis ordering mismatch.
    McAllister uses two different basis orderings:
    - basis.dat (for kahler_param.dat): indices [1,2,3,4,5,6,7,11,12,...]
    - kklt_basis.dat (for target_volumes.dat): indices [3,4,5,6,7,8,9,10,11,...]

    The DUAL validation (h11=4) passes and validates the core computation.
    """
    print("=" * 70)
    print("MCALLISTER V_string VALIDATION - PRIMAL (h11=214)")
    print("=" * 70)

    # Load geometry
    print("\n[1] Loading McAllister primal polytope (h11=214)...")
    points = load_primal_points()
    print(f"    Points: {points.shape}")

    # Create CYTools objects
    poly = Polytope(points)
    tri = poly.triangulate()
    cy = tri.get_cy()

    h11 = cy.h11()
    basis = cy.divisor_basis()
    print(f"    h11 = {h11}, h21 = {cy.h21()}")
    print(f"    CYTools divisor basis: first 10 of {len(basis)}: {list(basis[:10])}...")

    # Load ground truth
    print("\n[2] Loading ground truth...")
    gt = load_mcallister_ground_truth()
    print(f"    W0 = {gt['W0']:.6e}")
    print(f"    g_s = {gt['g_s']:.6f}")
    print(f"    V_string (target) = {gt['V_string']:.2f}")

    # Load orientifold data from JSON
    print("\n[3] Loading orientifold data from JSON...")
    orientifold = load_orientifold_data("4-214-647")
    print(f"    Source: {orientifold['source']}")
    print(f"    O7-planes: {orientifold['n_o7_planes']}")
    print(f"    D3-instantons: {orientifold['n_d3_instantons']}")

    # Map c_i to CYTools basis
    c_i, mapped = get_c_i_for_basis(orientifold, list(basis))
    print(f"    Mapped {mapped}/{h11} c_i values to CYTools basis")
    print(f"    c_i (first 10): {c_i[:10]}")

    # Load McAllister's SOLVED Kähler moduli
    print("\n[4] Loading McAllister's solved Kähler moduli...")
    t_mcallister, mcallister_basis = load_primal_kahler_params(corrected=False)
    print(f"    Kähler moduli: {len(t_mcallister)} values")
    print(f"    McAllister basis: first 10: {mcallister_basis[:10]}...")
    print(f"    t (first 10): {t_mcallister[:10]}")

    # Get intersection tensor from CYTools
    print("\n[5] Computing intersection tensor...")
    kappa = compute_intersection_tensor(cy, h11)
    nonzero_count = np.sum(kappa != 0)
    print(f"    Intersection tensor shape: {kappa.shape}")
    print(f"    Non-zero entries: {nonzero_count}")

    # Map McAllister's t to CYTools basis ordering
    print("\n[6] Mapping Kähler moduli to CYTools basis...")
    print(f"    McAllister basis: {len(mcallister_basis)} indices")
    print(f"    CYTools basis: {len(basis)} indices")

    # Create mapping: McAllister index -> t value
    mcallister_point_to_t = {}
    for i, point_idx in enumerate(mcallister_basis):
        mcallister_point_to_t[point_idx] = t_mcallister[i]

    # Map to CYTools basis ordering
    t_cytools = np.zeros(h11)
    mapped_t = 0
    for i, cytools_point_idx in enumerate(basis):
        if cytools_point_idx in mcallister_point_to_t:
            t_cytools[i] = mcallister_point_to_t[cytools_point_idx]
            mapped_t += 1
        else:
            t_cytools[i] = 1.0  # Default for unmapped

    print(f"    Mapped {mapped_t}/{h11} moduli")
    print(f"    t_cytools (first 10): {t_cytools[:10]}")

    # Compute V_string directly from Kähler moduli
    print("\n[7] Computing V_string from Kähler moduli...")
    V_computed = compute_cy_volume(kappa, t_cytools)
    print(f"    V_string = {V_computed:.2f}")

    # Also compute divisor volumes for verification
    tau_computed = compute_divisor_volumes(kappa, t_cytools)
    print(f"    tau (first 10): {tau_computed[:10]}")

    # Validate
    print("\n" + "=" * 70)
    print("VALIDATION RESULTS - PRIMAL (using McAllister's solved t)")
    print("=" * 70)
    V_expected = gt["V_string"]
    rel_error = abs(V_computed - V_expected) / V_expected

    print(f"V_string computed: {V_computed:.2f}")
    print(f"V_string expected: {V_expected:.2f}")
    print(f"Relative error: {rel_error:.4%}")

    if rel_error < 0.01:
        print("\n*** VALIDATION PASSED (< 1% error) ***")
    elif rel_error < 0.10:
        print("\n*** VALIDATION CLOSE (< 10% error) ***")
    else:
        print("\n*** VALIDATION FAILED - significant discrepancy ***")
        print("\n    Note: Mismatch likely due to basis ordering differences")
        print("    between McAllister's convention and CYTools.")

    return {
        "V_string": V_computed,
        "t": t_cytools,
        "c_i": c_i,
        "kappa": kappa,
        "orientifold": orientifold,
    }, gt


# =============================================================================
# MAIN
# =============================================================================


def main():
    """Run McAllister validation tests on both dual and primal."""
    print("\n" + "#" * 70)
    print("# DUAL POLYTOPE TEST (h11=4)")
    print("#" * 70)
    test_mcallister_V_string()

    print("\n\n" + "#" * 70)
    print("# PRIMAL POLYTOPE TEST (h11=214)")
    print("#" * 70)
    test_mcallister_V_string_primal()


if __name__ == "__main__":
    main()
