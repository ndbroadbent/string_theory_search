#!/usr/bin/env python3
"""Compare GV invariants: McAllister vs CYTools/cygv."""

import numpy as np
from pathlib import Path


def main():
    DATA_DIR = Path("resources/small_cc_2107.09064_source/anc/paper_data/4-214-647")

    # Load McAllister's data
    dual_curves = []
    with open(DATA_DIR / "dual_curves.dat") as f:
        for line in f:
            row = [int(x) for x in line.strip().split(",")]
            dual_curves.append(row)
    dual_curves = np.array(dual_curves)

    gv_mca = []
    with open(DATA_DIR / "dual_curves_gv.dat") as f:
        content = f.read()
        gv_mca = [int(float(x)) for x in content.strip().split(",")]
    gv_mca = np.array(gv_mca)

    print(f"McAllister GV invariants: {len(gv_mca)}")
    print(f"Curve class shape: {dual_curves.shape}")
    print()

    # The first few McAllister GV invariants
    print("First 20 McAllister GV invariants:")
    for i in range(20):
        curve = dual_curves[i]
        print(f"  {tuple(curve)}: {gv_mca[i]}")

    print()

    # Now compare with our computed ones
    from cytools import Polytope

    dual_pts = np.loadtxt(DATA_DIR / "dual_points.dat", delimiter=',').astype(int)
    p_dual = Polytope(dual_pts)
    tri = p_dual.triangulate()
    cy = tri.get_cy()

    print(f"CYTools CY: h11={cy.h11()}, h21={cy.h21()}")

    gvs_computed = cy.compute_gvs(min_points=100, format='dok')
    print(f"Computed GV invariants: {len(gvs_computed)}")
    print()

    # The computed GVs have 4 components (h11=4)
    # McAllister's have 9 components - need to understand the basis

    print("Computed GV invariants:")
    for k, v in list(gvs_computed.items())[:20]:
        print(f"  {k}: {v}")

    print()
    print("=" * 50)
    print("Matching check:")
    print("=" * 50)

    # Try to find matches by looking at the GLSM structure
    Q = cy.glsm_charge_matrix(include_origin=False)
    print(f"GLSM charge matrix shape: {Q.shape}")
    print(f"GLSM charges:")
    print(Q)

    print()
    print("Key observation:")
    print("  McAllister's curves have 9 components (ambient space)")
    print("  Our computed curves have 4 components (h11 basis)")
    print("  The GLSM charge matrix should relate them")
    print()

    # McAllister's curve (-6, 2, 3, -1, 1, 1, 0, 0, 0) has GV = 252
    # Our curve (1, 0, 0, 0) has GV = 252
    # These should be the same curve class!

    # Let's verify: if we multiply GLSM charges by the curve class...
    # Q is 4x9, so Q @ curve gives a 4-vector in H^{1,1} basis

    print("Testing basis relationship:")
    mca_curve_252 = np.array([-6, 2, 3, -1, 1, 1, 0, 0, 0])
    print(f"  McAllister curve for N=252: {mca_curve_252}")

    # Q @ curve would give... something
    # Actually, curves are in the dual basis to divisors
    # Let me check the relationship differently

    # The curve basis should be related to the Mori cone
    try:
        curve_basis = cy.curve_basis(include_origin=False, as_matrix=True)
        print(f"  Curve basis shape: {curve_basis.shape}")
        print(f"  Curve basis:\n{curve_basis}")
    except Exception as e:
        print(f"  Could not get curve basis: {e}")


if __name__ == "__main__":
    main()
