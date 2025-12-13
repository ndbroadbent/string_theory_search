#!/usr/bin/env python3
"""
Verify the e^K₀ formula from eq 6.12.

CORRECT FORMULA (from LaTeX source AdS4_v3.tex line 1171):
  e^{K₀} := (4/3 × κ̃_abc p^a p^b p^c)^{-1}
          = 1 / (4/3 × κ_p3)
          = (3/4) / κ_p3

The ^{-1} applies to the ENTIRE expression including the 4/3 factor!

McAllister's paper gives EXPLICIT numerical values we can check:
- 5-113-4627-main: e^K₀ = 1170672/12843563 = 0.09117
- 7-51-13590: e^K₀ = 5488000/20186543 = 0.27188

Using (3/4) / κ_p3 should match these EXACTLY.
"""

import sys
from pathlib import Path
import numpy as np
from fractions import Fraction

SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
CYTOOLS_2021 = ROOT_DIR / "vendor/cytools_mcallister_2107"
DATA_BASE = ROOT_DIR / "resources/small_cc_2107.09064_source/anc/paper_data"


def load_example_data(example_name: str) -> dict:
    """Load K, M vectors."""
    data_dir = DATA_BASE / example_name
    K = np.array([int(x) for x in (data_dir / "K_vec.dat").read_text().strip().split(",")])
    M = np.array([int(x) for x in (data_dir / "M_vec.dat").read_text().strip().split(",")])
    return {"K": K, "M": M}


def load_dual_points(example_name: str) -> np.ndarray:
    lines = (DATA_BASE / example_name / "dual_points.dat").read_text().strip().split('\n')
    return np.array([[int(x) for x in line.split(',')] for line in lines])


def load_simplices(example_name: str) -> list:
    lines = (DATA_BASE / example_name / "dual_simplices.dat").read_text().strip().split('\n')
    return [[int(x) for x in line.split(',')] for line in lines]


def get_cytools_2021_kappa(dual_pts, simplices, K, M):
    """Get κ tensor and compute κ_p3 using McAllister's CYTools 2021."""
    # Clear cytools modules
    mods = [k for k in list(sys.modules.keys()) if 'cytools' in k]
    for m in mods:
        del sys.modules[m]

    sys.path.insert(0, str(CYTOOLS_2021))
    from cytools import Polytope

    poly = Polytope(dual_pts)
    tri = poly.triangulate(simplices=simplices, check_input_simplices=False)
    cy = tri.get_cy()

    # Get kappa tensor
    kappa_sparse = cy.intersection_numbers(in_basis=True)
    h11 = cy.h11()
    kappa = np.zeros((h11, h11, h11))
    for row in kappa_sparse:
        i, j, k = int(row[0]), int(row[1]), int(row[2])
        val = row[3]
        for perm in [(i, j, k), (i, k, j), (j, i, k), (j, k, i), (k, i, j), (k, j, i)]:
            kappa[perm] = val

    divisor_basis = list(cy.divisor_basis())

    # Compute p from N @ p = K where N_ab = κ_abc M^c
    N = np.einsum('abc,c->ab', kappa, M)
    p = np.linalg.solve(N, K)

    # Compute κ_p3 = κ_abc p^a p^b p^c
    kappa_p3 = np.einsum('abc,a,b,c->', kappa, p, p, p)

    sys.path.remove(str(CYTOOLS_2021))

    return kappa, p, kappa_p3, divisor_basis


def main():
    print("=" * 70)
    print("VERIFYING e^K₀ = (3/4) / κ̃_p3 (eq 6.12)")
    print("=" * 70)

    # Known values from the paper (EXPLICIT fractions)
    known_eK0 = {
        "5-113-4627-main": Fraction(1170672, 12843563),  # eq 6.12
        "7-51-13590": Fraction(5488000, 20186543),  # around eq 6.29-6.30
    }

    examples = [
        "5-113-4627-main",
        "7-51-13590",
        "4-214-647",
        "5-113-4627-alternative",
        "5-81-3213",
    ]

    for example in examples:
        print(f"\n{'='*60}")
        print(f"Example: {example}")
        print(f"{'='*60}")

        dual_pts = load_dual_points(example)
        simplices = load_simplices(example)
        data = load_example_data(example)

        kappa, p, kappa_p3, basis = get_cytools_2021_kappa(
            dual_pts, simplices, data['K'], data['M']
        )

        print(f"Divisor basis: {basis}")
        print(f"K = {data['K']}")
        print(f"M = {data['M']}")
        print(f"p = {p}")

        # Compute e^K₀ using CORRECT formula: e^K₀ = (4/3 × κ_p3)^{-1} = (3/4) / κ_p3
        eK0_computed = (3/4) / kappa_p3

        print(f"\nκ_p3 = κ_abc p^a p^b p^c = {kappa_p3:.6f}")
        print(f"e^K₀ = (3/4) / κ_p3 = {eK0_computed:.6f}")

        if example in known_eK0:
            eK0_paper = float(known_eK0[example])
            print(f"\nPaper's e^K₀ = {known_eK0[example]} = {eK0_paper:.6f}")
            ratio = eK0_computed / eK0_paper
            print(f"Ratio computed/paper = {ratio:.6f}")

            # What κ_p3 would be needed to get paper's value?
            kappa_p3_needed = (3/4) / eK0_paper
            print(f"\nκ_p3 needed for paper's e^K₀: {kappa_p3_needed:.6f}")
            print(f"κ_p3 computed: {kappa_p3:.6f}")
            print(f"Ratio needed/computed: {kappa_p3_needed/kappa_p3:.6f}")

        # Show some kappa values
        print(f"\nSample κ values:")
        h11 = len(p)
        for i in range(h11):
            for j in range(i, h11):
                for k in range(j, h11):
                    val = kappa[i, j, k]
                    if val != 0:
                        print(f"  κ_{{{i}{j}{k}}} = {int(val)}")


if __name__ == "__main__":
    main()
