#!/usr/bin/env python3
"""
Verify the V₀ formula across all McAllister examples.

V₀ = -3 × e^{K₀} × (g_s⁷ / (4V[0])²) × W₀²  (eq. 6.24)

Goal: Find what e^{K₀} McAllister actually uses and compare with our κ_p3 computation.
"""

import sys
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
CYTOOLS_2021 = ROOT_DIR / "vendor/cytools_mcallister_2107"
DATA_BASE = ROOT_DIR / "resources/small_cc_2107.09064_source/anc/paper_data"


def load_data(example_name: str) -> dict:
    """Load all McAllister data for an example."""
    data_dir = DATA_BASE / example_name

    K = np.array([int(x) for x in (data_dir / "K_vec.dat").read_text().strip().split(",")])
    M = np.array([int(x) for x in (data_dir / "M_vec.dat").read_text().strip().split(",")])
    g_s = float((data_dir / "g_s.dat").read_text().strip())
    W0 = float((data_dir / "W_0.dat").read_text().strip())
    V_string = float((data_dir / "cy_vol.dat").read_text().strip())

    # Load c_tau if available
    c_tau_file = data_dir / "c_tau.dat"
    c_tau = float(c_tau_file.read_text().strip()) if c_tau_file.exists() else None

    return {
        "K": K, "M": M, "g_s": g_s, "W0": W0, "V_string": V_string, "c_tau": c_tau
    }


def load_dual_points(example_name: str) -> np.ndarray:
    lines = (DATA_BASE / example_name / "dual_points.dat").read_text().strip().split('\n')
    return np.array([[int(x) for x in line.split(',')] for line in lines])


def load_simplices(example_name: str) -> list:
    lines = (DATA_BASE / example_name / "dual_simplices.dat").read_text().strip().split('\n')
    return [[int(x) for x in line.split(',')] for line in lines]


def compute_kappa_p3(dual_pts, simplices, K, M):
    """Compute κ_abc p^a p^b p^c using CYTools 2021."""
    mods = [k for k in list(sys.modules.keys()) if 'cytools' in k]
    for m in mods:
        del sys.modules[m]

    sys.path.insert(0, str(CYTOOLS_2021))
    from cytools import Polytope

    poly = Polytope(dual_pts)
    tri = poly.triangulate(simplices=simplices, check_input_simplices=False)
    cy = tri.get_cy()

    kappa_sparse = cy.intersection_numbers(in_basis=True)
    h11 = cy.h11()
    kappa = np.zeros((h11, h11, h11))
    for row in kappa_sparse:
        i, j, k = int(row[0]), int(row[1]), int(row[2])
        val = row[3]
        for perm in [(i, j, k), (i, k, j), (j, i, k), (j, k, i), (k, i, j), (k, j, i)]:
            kappa[perm] = val

    N = np.einsum('abc,c->ab', kappa, M)
    p = np.linalg.solve(N, K)

    kappa_p3 = np.einsum('abc,a,b,c->', kappa, p, p, p)

    sys.path.remove(str(CYTOOLS_2021))

    return kappa_p3, p


# From the paper:
# h21=5, h11=113 (first): e^{K₀} = 1170672/12843563 (eq 6.12)
# h21=7, h11=51: e^{K₀} = 5488000/20186543 (mentioned in eq 6.29-6.30 area)

KNOWN_EK0 = {
    "5-113-4627-main": 1170672 / 12843563,
    "7-51-13590": 5488000 / 20186543,
}

# V₀ values from the paper
KNOWN_V0 = {
    "5-113-4627-main": -1.68e-144,  # eq 6.24
    "5-113-4627-alternative": -3.31e-214,  # eq 6.33
    "7-51-13590": -8.4e-216,  # need to find
    "5-81-3213": None,  # need to find
    "4-214-647": -5.5e-203,  # eq 6.63
}


def main():
    examples = [
        "5-113-4627-main",
        "5-113-4627-alternative",
        "7-51-13590",
        "5-81-3213",
        "4-214-647",
    ]

    print("=" * 80)
    print("VERIFYING V₀ FORMULA: V₀ = -3 × e^{K₀} × (g_s⁷ / (4V)²) × W₀²")
    print("=" * 80)

    for example in examples:
        print(f"\n{'='*60}")
        print(f"Example: {example}")
        print(f"{'='*60}")

        try:
            data = load_data(example)
            dual_pts = load_dual_points(example)
            simplices = load_simplices(example)

            print(f"  g_s = {data['g_s']:.6f}")
            print(f"  W₀ = {data['W0']:.2e}")
            print(f"  V_string = {data['V_string']:.2f}")
            if data['c_tau']:
                print(f"  c_τ = {data['c_tau']:.5f}")

            # Compute κ_p3
            kappa_p3, p = compute_kappa_p3(dual_pts, simplices, data['K'], data['M'])
            eK0_computed = (4/3) / kappa_p3

            print(f"\n  Computed κ_p3 = {kappa_p3:.6f}")
            print(f"  Computed e^{{K₀}} = (4/3)/κ_p3 = {eK0_computed:.6f}")

            # Check against known e^{K₀} if available
            if example in KNOWN_EK0:
                eK0_known = KNOWN_EK0[example]
                print(f"  Paper's e^{{K₀}} = {eK0_known:.6f}")
                print(f"  Ratio paper/computed = {eK0_known/eK0_computed:.4f}")

            # Compute V₀ using our e^{K₀}
            prefactor = data['g_s']**7 / (4 * data['V_string'])**2
            V0_computed = -3 * eK0_computed * prefactor * data['W0']**2

            print(f"\n  Prefactor g_s⁷/(4V)² = {prefactor:.6e}")
            print(f"  W₀² = {data['W0']**2:.2e}")
            print(f"  V₀ (computed e^{{K₀}}) = {V0_computed:.2e}")

            # Back-calculate e^{K₀} from known V₀ if available
            if example in KNOWN_V0 and KNOWN_V0[example] is not None:
                V0_target = KNOWN_V0[example]
                eK0_backcalc = V0_target / (-3 * prefactor * data['W0']**2)

                print(f"\n  Paper's V₀ = {V0_target:.2e}")
                print(f"  Back-calc e^{{K₀}} = {eK0_backcalc:.6f}")
                print(f"  Ratio back-calc/computed = {eK0_backcalc/eK0_computed:.4f}")

                # Check if there's a consistent factor
                ratio = eK0_backcalc / eK0_computed
                print(f"\n  Analysis:")
                print(f"    κ_p3 ratio (needed/computed) = {(4/3)/eK0_backcalc / kappa_p3:.4f}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
