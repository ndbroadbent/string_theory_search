#!/usr/bin/env python3
"""
Compute Gopakumar-Vafa invariants for a Calabi-Yau threefold.

GV invariants N_q count BPS states and appear in:
1. Worldsheet instanton corrections to the prepotential
2. The KKLT target τ formula (eq 5.13)

Uses CYTools' compute_gvs() which computes genus-zero GV invariants
via mirror symmetry.

Reference: arXiv:2107.09064 section 5.3
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "vendor/cytools_latest/src"))

import numpy as np
from cytools import Polytope


def compute_gv_invariants(cy, min_points: int = 100) -> dict:
    """
    Compute genus-zero Gopakumar-Vafa invariants.

    Args:
        cy: CYTools CalabiYau object
        min_points: Minimum number of lattice points to compute (controls degree)

    Returns:
        Dictionary mapping curve class tuples to GV invariants {(q1,q2,...): N_q}
    """
    gv_obj = cy.compute_gvs(min_points=min_points)
    gv_invariants = {}
    for q, N_q in gv_obj.dok.items():
        if N_q != 0:
            gv_invariants[tuple(q)] = int(N_q)
    return gv_invariants


# =============================================================================
# VALIDATION
# =============================================================================

def main():
    """Validate against McAllister 4-214-647 GV data."""
    DATA_DIR = Path(__file__).parent.parent / "resources/small_cc_2107.09064_source/anc/paper_data/4-214-647"

    print("=" * 70)
    print("GV Invariants: Compute and Validate")
    print("=" * 70)

    # Load McAllister's GV data (for dual polytope, h11=4)
    dual_curves = []
    with open(DATA_DIR / "dual_curves.dat") as f:
        for line in f:
            row = [int(x) for x in line.strip().split(",")]
            dual_curves.append(row)
    dual_curves = np.array(dual_curves)

    gv_mcallister = []
    with open(DATA_DIR / "dual_curves_gv.dat") as f:
        content = f.read()
        gv_mcallister = [int(float(x)) for x in content.strip().split(",")]

    print(f"\nMcAllister GV data: {len(gv_mcallister)} curve classes")

    # Load dual polytope and compute GV
    dual_pts = np.loadtxt(DATA_DIR / "dual_points.dat", delimiter=',').astype(int)
    poly = Polytope(dual_pts)
    tri = poly.triangulate()
    cy = tri.get_cy()

    print(f"CY: h11={cy.h11()}, h21={cy.h21()}")

    gv_computed = compute_gv_invariants(cy, min_points=100)
    print(f"Computed: {len(gv_computed)} non-zero GV invariants")

    # McAllister uses 9-component ambient space curves, we use h11=4 basis
    # Build mapping from our basis to McAllister's values
    # Known matches from compare_gv.py:
    #   Our (1,0,0,0) = McAllister (-6,2,3,-1,1,1,0,0,0) = 252
    #   Our (2,0,0,0) = McAllister (-12,4,6,-2,2,2,0,0,0) = -9252
    #   Our (3,0,0,0) = McAllister (-18,6,9,-3,3,3,0,0,0) = 848628

    print("\nValidation (known curve classes):")
    expected = {
        (1, 0, 0, 0): 252,
        (2, 0, 0, 0): -9252,
        (3, 0, 0, 0): 848628,
        (0, 0, 0, 1): 420,
    }

    all_match = True
    for q, N_expected in expected.items():
        N_computed = gv_computed.get(q, None)
        match = N_computed == N_expected
        status = "✓" if match else "✗"
        print(f"  {q}: computed={N_computed}, expected={N_expected} {status}")
        if not match:
            all_match = False

    if all_match:
        print(f"\n✓ GV INVARIANTS VALIDATED")
    else:
        print(f"\n✗ GV INVARIANTS MISMATCH")
        raise AssertionError("GV invariants do not match")


if __name__ == "__main__":
    main()
