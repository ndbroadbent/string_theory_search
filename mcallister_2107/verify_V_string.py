#!/usr/bin/env python3
"""
Verify V_string = 4711 using McAllister's pre-computed Kähler moduli.

McAllister SEARCHED and FOUND the solution. We VERIFY the pipeline.

Data files:
- kahler_param.dat: 214 solved t^i values
- basis.dat: 214 divisor indices (which divisors these t^i correspond to)
- cy_vol.dat: 4711.83 (the target V_string)

The challenge: map McAllister's basis to CYTools' basis to compute
V_string = (1/6) κ_ijk t^i t^j t^k
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "vendor/cytools_latest/src"))

import numpy as np
from cytools import Polytope

DATA_DIR = Path(__file__).parent.parent / "resources/small_cc_2107.09064_source/anc/paper_data/4-214-647"


def load_mcallister_data():
    """Load all McAllister data files."""
    # Polytope points
    lines = (DATA_DIR / "points.dat").read_text().strip().split('\n')
    points = np.array([[int(x) for x in line.split(',')] for line in lines])

    # Solved Kähler moduli
    t_text = (DATA_DIR / "kahler_param.dat").read_text().strip()
    t_values = np.array([float(x) for x in t_text.split(',')])

    # Basis indices (which divisors the t values correspond to)
    basis_text = (DATA_DIR / "basis.dat").read_text().strip()
    basis_indices = np.array([int(x) for x in basis_text.split(',')])

    # Ground truth volume
    V_target = float((DATA_DIR / "cy_vol.dat").read_text().strip())

    return {
        "points": points,
        "t": t_values,
        "basis": basis_indices,
        "V_target": V_target,
    }


def get_cytools_basis(cy):
    """Get CYTools' divisor basis."""
    return np.array(list(cy.divisor_basis()))


def compute_V_direct(kappa, t):
    """Compute V = (1/6) κ_ijk t^i t^j t^k."""
    return np.einsum('ijk,i,j,k->', kappa, t, t, t) / 6.0


def verify_basis_overlap():
    """Check overlap between McAllister basis and CYTools basis."""
    print("=" * 70)
    print("STEP 1: Analyze Basis Overlap")
    print("=" * 70)

    data = load_mcallister_data()

    poly = Polytope(data["points"])
    tri = poly.triangulate()
    cy = tri.get_cy()

    mcallister_basis = set(data["basis"])
    cytools_basis = set(get_cytools_basis(cy))

    common = mcallister_basis & cytools_basis
    only_mcallister = mcallister_basis - cytools_basis
    only_cytools = cytools_basis - mcallister_basis

    print(f"McAllister basis: {len(mcallister_basis)} divisors")
    print(f"CYTools basis: {len(cytools_basis)} divisors")
    print(f"Common: {len(common)} divisors")
    print(f"Only in McAllister: {only_mcallister}")
    print(f"Only in CYTools: {only_cytools}")

    return {
        "common": common,
        "only_mcallister": only_mcallister,
        "only_cytools": only_cytools,
        "data": data,
        "cy": cy,
    }


def verify_with_cytools_kappa():
    """
    Attempt 1: Use CYTools intersection numbers with mapped t values.
    """
    print("\n" + "=" * 70)
    print("STEP 2: Verify with CYTools κ (mapped basis)")
    print("=" * 70)

    data = load_mcallister_data()

    poly = Polytope(data["points"])
    tri = poly.triangulate()
    cy = tri.get_cy()

    h11 = cy.h11()
    cytools_basis = get_cytools_basis(cy)
    mcallister_basis = data["basis"]
    mcallister_t = data["t"]

    print(f"h11 = {h11}")
    print(f"CYTools basis (first 10): {list(cytools_basis[:10])}")
    print(f"McAllister basis (first 10): {list(mcallister_basis[:10])}")

    # Create mapping: divisor index -> t value
    div_to_t = {int(div): t for div, t in zip(mcallister_basis, mcallister_t)}

    # Map to CYTools basis ordering
    t_mapped = np.zeros(h11)
    mapped_count = 0
    for i, div in enumerate(cytools_basis):
        if div in div_to_t:
            t_mapped[i] = div_to_t[div]
            mapped_count += 1
        else:
            t_mapped[i] = 1.0  # Default for unmapped

    print(f"Mapped {mapped_count}/{h11} divisors")
    print(f"t_mapped (first 10): {t_mapped[:10]}")

    # Get CYTools intersection numbers
    kappa_sparse = cy.intersection_numbers(in_basis=True)
    kappa = np.zeros((h11, h11, h11))
    for (i, j, k), val in kappa_sparse.items():
        for perm in [(i,j,k), (i,k,j), (j,i,k), (j,k,i), (k,i,j), (k,j,i)]:
            kappa[perm] = val

    # Compute V_string
    V_computed = compute_V_direct(kappa, t_mapped)
    V_target = data["V_target"]

    print(f"\nV_string computed: {V_computed:.2f}")
    print(f"V_string target: {V_target:.2f}")
    print(f"Relative error: {abs(V_computed - V_target) / V_target * 100:.2f}%")

    return V_computed, V_target


def verify_with_direct_kappa():
    """
    Attempt 2: Compute intersection numbers directly for McAllister's basis.

    Use CYTools' intersection_numbers() but with explicit divisor indices.
    """
    print("\n" + "=" * 70)
    print("STEP 3: Verify with Direct κ Computation")
    print("=" * 70)

    data = load_mcallister_data()

    poly = Polytope(data["points"])
    tri = poly.triangulate()
    cy = tri.get_cy()

    mcallister_basis = list(data["basis"])
    mcallister_t = data["t"]
    h11 = len(mcallister_basis)

    print(f"Computing κ for McAllister's {h11} basis divisors...")

    # Get ALL intersection numbers (not in any particular basis)
    # Then extract the ones for McAllister's divisors
    try:
        # Try to get intersection numbers for specific divisors
        kappa_all = cy.intersection_numbers(in_basis=False)

        # Build κ tensor for McAllister's basis
        kappa = np.zeros((h11, h11, h11))

        for (d1, d2, d3), val in kappa_all.items():
            # Check if all three divisors are in McAllister's basis
            if d1 in mcallister_basis and d2 in mcallister_basis and d3 in mcallister_basis:
                i = mcallister_basis.index(d1)
                j = mcallister_basis.index(d2)
                k = mcallister_basis.index(d3)
                for perm in [(i,j,k), (i,k,j), (j,i,k), (j,k,i), (k,i,j), (k,j,i)]:
                    kappa[perm] = val

        nonzero = np.sum(kappa != 0)
        print(f"Non-zero κ entries: {nonzero}")

        # Compute V_string
        V_computed = compute_V_direct(kappa, mcallister_t)
        V_target = data["V_target"]

        print(f"\nV_string computed: {V_computed:.2f}")
        print(f"V_string target: {V_target:.2f}")
        print(f"Relative error: {abs(V_computed - V_target) / V_target * 100:.2f}%")

        return V_computed, V_target

    except Exception as e:
        print(f"Error: {e}")
        return None, data["V_target"]


def verify_with_set_basis():
    """
    Attempt 3: Tell CYTools to use McAllister's basis explicitly.
    """
    print("\n" + "=" * 70)
    print("STEP 4: Verify with CYTools set_divisor_basis()")
    print("=" * 70)

    data = load_mcallister_data()

    poly = Polytope(data["points"])
    tri = poly.triangulate()
    cy = tri.get_cy()

    mcallister_basis = list(data["basis"])
    mcallister_t = data["t"]
    h11 = cy.h11()

    print(f"h11 = {h11}")
    print(f"McAllister basis has {len(mcallister_basis)} divisors")

    # Try to set the divisor basis to McAllister's
    try:
        cy.set_divisor_basis(mcallister_basis)
        new_basis = list(cy.divisor_basis())
        print(f"Set basis to McAllister's: {new_basis[:10]}...")

        # Get intersection numbers in new basis
        kappa_sparse = cy.intersection_numbers(in_basis=True)
        kappa = np.zeros((h11, h11, h11))
        for (i, j, k), val in kappa_sparse.items():
            for perm in [(i,j,k), (i,k,j), (j,i,k), (j,k,i), (k,i,j), (k,j,i)]:
                kappa[perm] = val

        nonzero = np.sum(kappa != 0)
        print(f"Non-zero κ entries: {nonzero}")

        # Compute V_string
        V_computed = compute_V_direct(kappa, mcallister_t)
        V_target = data["V_target"]

        print(f"\nV_string computed: {V_computed:.2f}")
        print(f"V_string target: {V_target:.2f}")
        print(f"Relative error: {abs(V_computed - V_target) / V_target * 100:.2f}%")

        if abs(V_computed - V_target) / V_target < 0.01:
            print("\n*** SUCCESS: V_string matches within 1%! ***")

        return V_computed, V_target

    except Exception as e:
        print(f"Error setting basis: {e}")
        import traceback
        traceback.print_exc()
        return None, data["V_target"]


def main():
    print("#" * 70)
    print("# VERIFY V_string = 4711 using McAllister's kahler_param.dat")
    print("#" * 70)

    # Step 1: Analyze basis overlap
    verify_basis_overlap()

    # Step 2: Try with CYTools basis (mapped)
    verify_with_cytools_kappa()

    # Step 3: Try computing κ directly for McAllister's basis
    verify_with_direct_kappa()

    # Step 4: Try setting CYTools to use McAllister's basis
    verify_with_set_basis()


if __name__ == "__main__":
    main()
