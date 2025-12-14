#!/usr/bin/env python3
"""
Compute basis transformation T from GLSM linear relations.

Given old_basis and new_basis indices, compute T such that:
    D_old[i] = sum_j T[i,j] * D_new[j]

The GLSM linear relations tell us how divisors are related.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "vendor/cytools_latest/src"))
from cytools import Polytope


def compute_T_from_glsm(cy, old_basis: list, new_basis: list) -> np.ndarray:
    """
    Compute transformation matrix T from GLSM linear relations.

    Args:
        cy: CalabiYau object
        old_basis: list of divisor indices in old basis
        new_basis: list of divisor indices in new basis

    Returns:
        T: transformation matrix where D_old = T @ D_new
    """
    h = len(old_basis)
    assert len(new_basis) == h

    # Get GLSM linear relations: L @ D = 0
    L = cy.triangulation().polytope().glsm_linear_relations()
    print(f"GLSM linear relations shape: {L.shape}")
    print(f"L =\n{L}\n")

    T = np.zeros((h, h))

    for i, old_idx in enumerate(old_basis):
        if old_idx in new_basis:
            # Old basis divisor is also in new basis - just identity
            j = new_basis.index(old_idx)
            T[i, j] = 1
            print(f"D_{old_idx} = D_{old_idx} (in both bases)")
        else:
            # Need to express D_old_idx in terms of new_basis using GLSM relations
            # Find a relation involving D_old_idx
            for row_idx, row in enumerate(L):
                coeff = row[old_idx]
                if abs(coeff) > 1e-10:
                    # Found a relation: coeff * D_old_idx + sum(row[j] * D_j) = 0
                    # => D_old_idx = -sum(row[j] * D_j) / coeff (for j != old_idx)

                    print(f"Using GLSM row {row_idx}: {row}")

                    expr = []
                    for j, new_idx in enumerate(new_basis):
                        if new_idx != old_idx:
                            c = -row[new_idx] / coeff
                            T[i, j] = c
                            if abs(c) > 1e-10:
                                expr.append(f"{c:+.0f}*D_{new_idx}")

                    print(f"D_{old_idx} = {' '.join(expr) if expr else '0'}")
                    break

    return T


def transform_fluxes(K_old, M_old, T):
    """
    Transform fluxes from old basis to new basis.

    K is covariant: K_new = T^{-1} @ K_old
    M is contravariant: M_new = T^T @ M_old
    """
    T_inv = np.linalg.inv(T)
    K_new = T_inv @ K_old
    M_new = T.T @ M_old
    return np.round(K_new).astype(int), np.round(M_new).astype(int)


def load_mcallister_example(name: str):
    """Load a McAllister example by folder name."""
    DATA_DIR = Path(__file__).parent.parent / f"resources/small_cc_2107.09064_source/anc/paper_data/{name}"

    # Load dual polytope
    lines = (DATA_DIR / "dual_points.dat").read_text().strip().split("\n")
    dual_points = np.array([[int(x) for x in line.split(",")] for line in lines])

    # Load fluxes
    K = np.array([int(x) for x in (DATA_DIR / "K_vec.dat").read_text().strip().split(",")])
    M = np.array([int(x) for x in (DATA_DIR / "M_vec.dat").read_text().strip().split(",")])

    # Load expected values
    g_s = float((DATA_DIR / "g_s.dat").read_text().strip())
    W0 = float((DATA_DIR / "W_0.dat").read_text().strip())
    V_string = float((DATA_DIR / "cy_vol.dat").read_text().strip())

    return {
        "name": name,
        "dual_points": dual_points,
        "K": K,
        "M": M,
        "g_s": g_s,
        "W0": W0,
        "V_string": V_string,
    }


def test_example(example: dict, verbose: bool = True):
    """Test basis transformation for one example."""
    name = example["name"]
    dual_points = example["dual_points"]
    K_old = example["K"]
    M_old = example["M"]

    poly = Polytope(dual_points)
    tri = poly.triangulate()
    cy = tri.get_cy()

    if verbose:
        print(f"\n{'='*60}")
        print(f"Example: {name}")
        print(f"{'='*60}")
        print(f"h11={cy.h11()}, h21={cy.h21()}")

    # Get old basis from CYTools 2021 (need to load with that version)
    # For now, we use set_divisor_basis to match McAllister's
    # First, get the new (latest) basis
    new_basis = list(cy.divisor_basis())

    # Load McAllister's basis from their data
    # The basis.dat file contains h11 indices - but for the DUAL that's h11_dual = h21_primal
    # Actually, let's use CYTools 2021 to get the old basis
    sys.path.insert(0, str(Path(__file__).parent.parent / "vendor/cytools_mcallister_2107"))

    # Clear cached imports
    mods_to_remove = [k for k in list(sys.modules.keys()) if 'cytools' in k]
    for mod in mods_to_remove:
        del sys.modules[mod]

    from cytools import Polytope as Polytope2021

    poly_2021 = Polytope2021(dual_points)
    tri_2021 = poly_2021.triangulate()
    cy_2021 = tri_2021.get_cy()
    old_basis = list(cy_2021.divisor_basis())

    # Restore latest cytools
    mods_to_remove = [k for k in list(sys.modules.keys()) if 'cytools' in k]
    for mod in mods_to_remove:
        del sys.modules[mod]
    sys.path.pop(0)

    if verbose:
        print(f"Old basis (2021): {old_basis}")
        print(f"New basis (latest): {new_basis}")
        print(f"K_old = {K_old}")
        print(f"M_old = {M_old}")

    # Reload latest cytools
    from cytools import Polytope as PolytopeLatest
    poly_new = PolytopeLatest(dual_points)
    tri_new = poly_new.triangulate()
    cy_new = tri_new.get_cy()

    # Compute T
    if old_basis == new_basis:
        if verbose:
            print("Bases are identical - no transformation needed")
        return True

    T = compute_T_from_glsm(cy_new, old_basis, new_basis)

    if verbose:
        print(f"\nT =\n{T.astype(int)}")
        print(f"det(T) = {np.linalg.det(T):.0f}")

    # Transform fluxes
    K_new, M_new = transform_fluxes(K_old, M_old, T)

    if verbose:
        print(f"\nK_new = {K_new}")
        print(f"M_new = {M_new}")

    # Verify by computing e^{K0} in both bases
    def compute_eK0(cy, K, M):
        kappa_dict = cy.intersection_numbers(in_basis=True)
        h = cy.h11()
        kappa = np.zeros((h, h, h))
        for (i, j, k), val in kappa_dict.items():
            for perm in [(i,j,k), (i,k,j), (j,i,k), (j,k,i), (k,i,j), (k,j,i)]:
                kappa[perm] = val
        N = np.einsum('abc,c->ab', kappa, M)
        if abs(np.linalg.det(N)) < 1e-10:
            return None
        p = np.linalg.solve(N, K)
        kappa_p3 = np.einsum('abc,a,b,c->', kappa, p, p, p)
        return 1.0 / ((4.0/3.0) * kappa_p3)

    # Set old basis and compute
    cy_new.set_divisor_basis(old_basis)
    eK0_old = compute_eK0(cy_new, K_old, M_old)

    # Set new basis and compute
    cy_new.set_divisor_basis(new_basis)
    eK0_new = compute_eK0(cy_new, K_new, M_new)

    if verbose:
        print(f"\ne^{{K0}} (old basis): {eK0_old:.6f}")
        print(f"e^{{K0}} (new basis): {eK0_new:.6f}")

    if eK0_old is not None and eK0_new is not None and np.isclose(eK0_old, eK0_new):
        if verbose:
            print("✓ e^{K0} MATCHES - transformation correct!")
        return True
    else:
        if verbose:
            print("✗ e^{K0} MISMATCH")
        return False


def main():
    print("=" * 60)
    print("Testing basis transformation for all 5 McAllister examples")
    print("=" * 60)

    examples = [
        "4-214-647",
        "5-113-4627-main",
        "5-113-4627-alternative",
        "5-81-3213",
        "7-51-13590",
    ]

    results = []
    for name in examples:
        example = load_mcallister_example(name)
        success = test_example(example, verbose=True)
        results.append((name, success))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, success in results:
        status = "✓" if success else "✗"
        print(f"  {status} {name}")

    all_passed = all(s for _, s in results)
    print(f"\n{'All tests passed!' if all_passed else 'Some tests failed.'}")


if __name__ == "__main__":
    main()
