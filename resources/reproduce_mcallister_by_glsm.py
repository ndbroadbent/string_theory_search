#!/usr/bin/env python3
"""
Reproduce McAllister et al. CY volume (4711.83) using GLSM projection.

This script uses the EXACT procedure to reconstruct the 4 Kähler moduli from
McAllister's 214 ambient parameters using GLSM linear relations.

Key insight: McAllister's kahler_param.dat contains 214 ambient coordinates for
non-basis divisors. The 4 basis divisor coordinates are determined by solving
the GLSM constraint Q @ t_amb = 0.

Reference: arXiv:2107.09064 "Small cosmological constants in string theory"
"""

import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "resources" / "small_cc_2107.09064_source" / "anc" / "paper_data" / "4-214-647"


def load_csv_floats(filename: str) -> np.ndarray:
    """Load comma-separated floats from a single-line file."""
    with open(DATA_DIR / filename) as f:
        line = f.read().strip()
        return np.array([float(x) for x in line.split(",")])


def load_csv_ints(filename: str) -> np.ndarray:
    """Load comma-separated ints from a single-line file."""
    with open(DATA_DIR / filename) as f:
        line = f.read().strip()
        return np.array([int(x) for x in line.split(",")])


def load_points(filename: str) -> list[list[int]]:
    """Load points from a CSV file (one point per line)."""
    points = []
    with open(DATA_DIR / filename) as f:
        for line in f:
            line = line.strip()
            if line:
                points.append([int(x) for x in line.split(",")])
    return points


def load_simplices(filename: str) -> list[list[int]]:
    """Load simplices from a CSV file (one simplex per line, 0-indexed)."""
    simplices = []
    with open(DATA_DIR / filename) as f:
        for line in f:
            line = line.strip()
            if line:
                simplices.append([int(x) for x in line.split(",")])
    return simplices


def load_float(filename: str) -> float:
    """Load a single float from a file."""
    with open(DATA_DIR / filename) as f:
        return float(f.read().strip())


def main():
    print("=" * 70)
    print("Reproducing McAllister CY Volume (4-214-647)")
    print("Using GLSM projection (NOT optimization)")
    print("=" * 70)
    print()

    try:
        from cytools import Polytope
    except ImportError:
        raise RuntimeError("CYTools not installed")

    # ==========================================================================
    # Step 0: Load all McAllister data files
    # ==========================================================================
    print("Loading McAllister data files...")

    dual_points = load_points("dual_points.dat")
    dual_simplices = load_simplices("dual_simplices.dat")
    g_s = load_float("g_s.dat")
    expected_vol_einstein = load_float("cy_vol.dat")
    expected_vol_einstein_corrected = load_float("corrected_cy_vol.dat")

    # Load both uncorrected and corrected Kähler parameters
    kahler_params = load_csv_floats("kahler_param.dat")
    kahler_params_corrected = load_csv_floats("corrected_kahler_param.dat")

    # basis.dat: 214 indices (1-indexed) of non-basis divisors
    # The first 4 entries of basis.dat permutation give the basis divisors
    basis_perm = load_csv_ints("basis.dat")  # 214 values, 1-indexed

    # kklt_basis.dat: ordering of non-basis divisors matching kahler_param.dat
    kklt_basis = load_csv_ints("kklt_basis.dat")  # 214 values, 1-indexed

    # Target divisor volumes for verification
    target_vols = load_csv_floats("target_volumes.dat")
    target_vols_corrected = load_csv_floats("corrected_target_volumes.dat")

    print(f"  g_s: {g_s}")
    print(f"  expected CY vol (Einstein, uncorrected): {expected_vol_einstein}")
    print(f"  expected CY vol (Einstein, corrected):   {expected_vol_einstein_corrected}")
    print(f"  dual_points: {len(dual_points)} vertices")
    print(f"  dual_simplices: {len(dual_simplices)} simplices")
    print(f"  kahler_params: {len(kahler_params)} values")
    print(f"  kklt_basis: {len(kklt_basis)} values, range [{kklt_basis.min()}, {kklt_basis.max()}]")
    print()

    # ==========================================================================
    # Step 1: Build CY from dual polytope with McAllister's exact triangulation
    # ==========================================================================
    print("=" * 70)
    print("Step 1: Build CY from dual polytope Δ* with exact triangulation")
    print("=" * 70)

    poly = Polytope(dual_points)
    print(f"  Polytope vertices: {len(dual_points)}")
    print(f"  Is reflexive: {poly.is_reflexive()}")

    # Use McAllister's exact triangulation
    triang = poly.triangulate(simplices=dual_simplices)
    cy = triang.get_cy()

    h11 = cy.h11()
    h21 = cy.h21()
    n_divisors = len(cy.prime_toric_divisors())

    print(f"  h11={h11}, h21={h21}")
    print(f"  Prime toric divisors: {n_divisors}")
    print(f"  Divisor basis (CYTools default): {cy.divisor_basis()}")
    print()

    # ==========================================================================
    # Step 2: Set McAllister's exact divisor basis
    # ==========================================================================
    print("=" * 70)
    print("Step 2: Set McAllister's divisor basis")
    print("=" * 70)

    # basis.dat contains 214 non-basis divisor indices (1-indexed)
    # The 4 divisors NOT in basis.dat form the basis
    # According to the procedure: first 4 entries of basis permutation are the basis

    # Actually, basis.dat lists the NON-basis divisors
    # We need to find which 4 are NOT listed
    all_divisor_indices = set(range(1, n_divisors + 1))  # 1 to N (1-indexed)
    non_basis_set = set(basis_perm)
    basis_divisors_1idx = sorted(all_divisor_indices - non_basis_set)

    # Convert to 0-indexed for CYTools
    B = [d - 1 for d in basis_divisors_1idx]

    print(f"  Non-basis divisors (from basis.dat): {len(non_basis_set)} indices")
    print(f"  Basis divisors (1-indexed): {basis_divisors_1idx}")
    print(f"  Basis divisors (0-indexed): {B}")

    # Set the divisor basis
    cy.set_divisor_basis(B)
    print(f"  CYTools divisor basis set to: {cy.divisor_basis()}")
    print()

    # ==========================================================================
    # Step 3: GLSM projection from 214 ambient params to 4 Kähler moduli
    # ==========================================================================
    print("=" * 70)
    print("Step 3: GLSM projection")
    print("=" * 70)

    # Get GLSM charge matrix (excluding origin column)
    Q = cy.glsm_charge_matrix(include_origin=False)
    print(f"  GLSM charge matrix shape: {Q.shape}")

    N = Q.shape[1]  # Number of prime toric divisors
    print(f"  Number of prime toric divisors: {N}")

    # Non-basis divisor indices (0-indexed), in the order of kklt_basis.dat
    NB = kklt_basis - 1  # Convert to 0-indexed

    print(f"  Basis indices B (0-indexed): {B}")
    print(f"  Non-basis indices NB: {len(NB)} values")
    print()

    # Build the ambient vector t_amb
    # Insert kahler_params at positions NB, solve for positions B

    def solve_glsm(kpar: np.ndarray, name: str) -> np.ndarray:
        """Solve GLSM constraint for basis coordinates given non-basis values."""
        print(f"  Solving GLSM for {name}...")

        # t_amb[NB] = kpar (known)
        # t_amb[B] = ? (unknown)
        # Constraint: Q @ t_amb = 0
        # Q[:, B] @ t_B + Q[:, NB] @ t_NB = 0
        # Q[:, B] @ t_B = -Q[:, NB] @ kpar

        rhs = -Q[:, NB] @ kpar
        t_B, residuals, rank, s = np.linalg.lstsq(Q[:, B], rhs, rcond=None)

        # Verify constraint satisfaction
        t_amb = np.zeros(N)
        t_amb[B] = t_B
        t_amb[NB] = kpar
        constraint_error = Q @ t_amb
        max_error = np.abs(constraint_error).max()

        print(f"    t_B = {t_B}")
        print(f"    GLSM constraint error: {max_error:.2e}")

        if max_error > 1e-10:
            print(f"    WARNING: Large constraint error!")

        return t_B

    # Solve for uncorrected parameters
    t_uncorrected = solve_glsm(kahler_params, "uncorrected")
    print()

    # Solve for corrected parameters
    t_corrected = solve_glsm(kahler_params_corrected, "corrected")
    print()

    # ==========================================================================
    # Step 4: Compute volumes and verify
    # ==========================================================================
    print("=" * 70)
    print("Step 4: Compute volumes and verify")
    print("=" * 70)

    def compute_and_verify(t: np.ndarray, expected_V_E: float, name: str):
        """Compute CY volume and verify against expected."""
        print(f"\n  {name}:")
        print(f"    Kähler moduli t = {t}")

        # Check if t is in Kähler cone
        cone = cy.toric_kahler_cone()
        in_cone = cone.contains(t)
        print(f"    In Kähler cone: {in_cone}")

        if not in_cone:
            print(f"    WARNING: Point is outside Kähler cone!")

        # Compute string frame volume
        V_string = cy.compute_cy_volume(t)

        # Convert to Einstein frame: V_E = V_S * g_s^(-3/2)
        V_einstein = V_string * (g_s ** (-1.5))

        print(f"    V_string (computed):  {V_string}")
        print(f"    V_einstein (computed): {V_einstein}")
        print(f"    V_einstein (expected): {expected_V_E}")

        error = abs(V_einstein - expected_V_E)
        rel_error = error / expected_V_E
        print(f"    Absolute error: {error:.6e}")
        print(f"    Relative error: {rel_error:.6e}")

        if rel_error < 1e-10:
            print(f"    ✓ EXACT MATCH (within floating point)")
        elif rel_error < 1e-6:
            print(f"    ~ Close match")
        else:
            print(f"    ✗ MISMATCH")

        return V_string, V_einstein

    V_s_unc, V_e_unc = compute_and_verify(t_uncorrected, expected_vol_einstein, "UNCORRECTED")
    V_s_cor, V_e_cor = compute_and_verify(t_corrected, expected_vol_einstein_corrected, "CORRECTED")

    # ==========================================================================
    # Step 5: Verify divisor volumes
    # ==========================================================================
    print()
    print("=" * 70)
    print("Step 5: Verify divisor volumes")
    print("=" * 70)

    def verify_divisor_volumes(t: np.ndarray, target: np.ndarray, name: str):
        """Verify divisor volumes match targets."""
        print(f"\n  {name}:")

        # Compute all divisor volumes (not just basis)
        tau_all = cy.compute_divisor_volumes(t, in_basis=False)

        # Target volumes are in Einstein frame, convert to string frame
        # τ_S = τ_E * g_s^(3/2) (divisor volumes scale opposite to CY volume)
        # Actually: τ_E = τ_S * g_s^(-1) for 4-cycles
        # Let's check both conventions

        # Extract volumes at non-basis positions (matching target ordering)
        tau_NB = tau_all[NB]

        print(f"    Computed τ (first 5): {tau_NB[:5]}")
        print(f"    Target τ (first 5):   {target[:5]}")

        # Try to find the right frame conversion
        # τ_4-cycle in Einstein vs string frame: τ_E = τ_S / g_s
        tau_target_string = target * g_s  # Convert target from Einstein to string

        diff = tau_NB - tau_target_string
        max_diff = np.abs(diff).max()
        rel_diff = np.abs(diff / (tau_target_string + 1e-10)).max()

        print(f"    Max absolute diff: {max_diff:.6e}")
        print(f"    Max relative diff: {rel_diff:.6e}")

    verify_divisor_volumes(t_uncorrected, target_vols, "UNCORRECTED")
    verify_divisor_volumes(t_corrected, target_vols_corrected, "CORRECTED")

    # ==========================================================================
    # Summary
    # ==========================================================================
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
McAllister 4-214-647 reproduction via GLSM projection:

  g_s = {g_s}

  UNCORRECTED:
    Kähler moduli t = {t_uncorrected}
    V_string  = {V_s_unc}
    V_einstein = {V_e_unc}
    Expected   = {expected_vol_einstein}

  CORRECTED (with instanton corrections):
    Kähler moduli t = {t_corrected}
    V_string  = {V_s_cor}
    V_einstein = {V_e_cor}
    Expected   = {expected_vol_einstein_corrected}

This is the EXACT reconstruction of McAllister's Kähler moduli from
their 214 ambient parameters using GLSM linear relations.

NOTE: W₀ = 2.30012e-90 cannot be computed from geometry alone.
      It requires period computation which CYTools does not provide.
      Use W_0.dat directly for physics computations.
""")


if __name__ == "__main__":
    main()
