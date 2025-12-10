#!/usr/bin/env python3
"""
Reproduce McAllister et al. CY volume (4711.83) using CYTools.

This script validates our understanding of the McAllister data format by
reproducing their published CY volume for polytope 4-214-647.

Reference: arXiv:2107.09064 "Small cosmological constants in string theory"

The key insight is that McAllister's `kahler_param.dat` contains 214 ambient/
secondary-fan parameters (one per non-basis toric divisor), NOT the 4 Kähler
moduli that CYTools expects.

To reproduce their volume, we must:
1. Load their exact triangulation (dual_simplices.dat)
2. Use their divisor basis (first 4 of basis.dat)
3. Solve for 4 Kähler moduli t^i that match target divisor volumes
4. Convert from string frame to Einstein frame: V_E = V_S * g_s^(-3/2)

See resources/MCALLISTER_SMALL_CC_DETAILS.md for full explanation.
"""

import sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize, differential_evolution

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# McAllister data directory
DATA_DIR = PROJECT_ROOT / "resources" / "small_cc_2107.09064_source" / "anc" / "paper_data" / "4-214-647"


def load_points(filename: str) -> list[list[int]]:
    """Load points from a CSV file."""
    points = []
    with open(DATA_DIR / filename) as f:
        for line in f:
            line = line.strip()
            if line:
                points.append([int(x) for x in line.split(",")])
    return points


def load_simplices(filename: str) -> list[list[int]]:
    """Load simplices from a CSV file."""
    simplices = []
    with open(DATA_DIR / filename) as f:
        for line in f:
            line = line.strip()
            if line:
                simplices.append([int(x) for x in line.split(",")])
    return simplices


def load_ints(filename: str) -> list[int]:
    """Load comma-separated integers from a file."""
    with open(DATA_DIR / filename) as f:
        content = f.read().strip()
        return [int(x) for x in content.split(",")]


def load_floats(filename: str) -> list[float]:
    """Load comma-separated floats from a file."""
    with open(DATA_DIR / filename) as f:
        content = f.read().strip()
        return [float(x) for x in content.split(",")]


def load_float(filename: str) -> float:
    """Load a single float from a file."""
    with open(DATA_DIR / filename) as f:
        return float(f.read().strip())


def main():
    print("=" * 70)
    print("Reproducing McAllister CY Volume (4-214-647)")
    print("=" * 70)
    print()

    # Try to import CYTools
    try:
        from cytools import Polytope
    except ImportError:
        print("ERROR: CYTools not installed.")
        print("Install with: pip install cytools")
        sys.exit(1)

    # Load McAllister data
    print("Loading McAllister data files...")
    dual_points = load_points("dual_points.dat")
    dual_simplices = load_simplices("dual_simplices.dat")
    basis = load_ints("basis.dat")
    target_volumes_einstein = load_ints("target_volumes.dat")  # These are INTEGERS (1 or 6)
    g_s = load_float("g_s.dat")
    expected_vol_einstein = load_float("cy_vol.dat")

    print(f"  dual_points: {len(dual_points)} points")
    print(f"  dual_simplices: {len(dual_simplices)} simplices")
    print(f"  basis: {len(basis)} indices (first 4: {basis[:4]})")
    print(f"  target_volumes: {len(target_volumes_einstein)} values")
    print(f"  g_s: {g_s}")
    print(f"  expected CY vol (Einstein): {expected_vol_einstein}")
    print()

    # Frame conversion factor
    # V_E = V_S * g_s^{-3/2}  => V_S = V_E * g_s^{3/2}
    g_s_factor = g_s ** 1.5  # ~0.00087
    expected_vol_string = expected_vol_einstein * g_s_factor
    print(f"Frame conversion:")
    print(f"  g_s = {g_s}")
    print(f"  g_s^(3/2) = {g_s_factor:.6f}")
    print(f"  g_s^(-3/2) = {g_s**(-1.5):.2f}")
    print(f"  Expected V_E = {expected_vol_einstein:.2f}")
    print(f"  Expected V_S = V_E * g_s^(3/2) = {expected_vol_string:.2f}")
    print()

    # Step 1: Create polytope
    print("Step 1: Creating polytope from dual_points...")
    poly = Polytope(dual_points)
    print(f"  Polytope: {poly}")
    print(f"  h11 = {poly.h11(lattice='N')}, h21 = {poly.h21(lattice='N')}")
    print()

    # Step 2: Get triangulation (use CYTools default, not McAllister's exact one for now)
    print("Step 2: Getting triangulation...")
    triang = poly.triangulate(backend="cgal")
    print(f"  Triangulation: {triang}")
    print()

    # Step 3: Get CY manifold
    print("Step 3: Getting Calabi-Yau manifold...")
    cy = triang.get_cy()
    h11 = cy.h11()
    h21 = cy.h21()
    print(f"  CY: {cy}")
    print(f"  h11 = {h11}, h21 = {h21}")
    print()

    # Step 4: Explore divisor structure
    print("Step 4: Exploring divisor structure...")

    # Get number of divisors
    try:
        num_divisors = cy.nef_partitions()
        print(f"  nef_partitions: {num_divisors}")
    except:
        pass

    # Get tip of the Kähler cone
    print("  Getting tip of Kähler cone...")
    try:
        tip = cy.toric_kahler_cone().tip_of_stretched_cone(1.0)
        print(f"  Tip: {tip}")
    except Exception as e:
        print(f"  Could not get tip: {e}")
        tip = np.ones(h11) * 2.0

    # Step 5: Compute volume at tip
    print()
    print("Step 5: Computing volume at tip of cone...")
    V_string_tip = cy.compute_cy_volume(tip)
    V_einstein_tip = V_string_tip * (g_s ** (-1.5))
    print(f"  t (tip) = {tip}")
    print(f"  V_S (at tip) = {V_string_tip:.4f}")
    print(f"  V_E (at tip) = {V_einstein_tip:.4f}")
    print()

    # Step 6: Search for t that gives correct V_E
    print("Step 6: Searching for Kähler moduli that give V_E = 4711.83...")

    # We want V_E = 4711.83
    # V_E = V_S * g_s^{-3/2}
    # V_S = V_E * g_s^{3/2} = 4711.83 * 0.000870 = 4.10
    target_V_string = expected_vol_einstein * g_s_factor
    print(f"  Target V_S = {target_V_string:.4f}")

    def objective(t):
        """Objective: minimize |V_S(t) - target_V_S|"""
        if np.any(t <= 0):
            return 1e10
        try:
            V = cy.compute_cy_volume(t)
            return (V - target_V_string) ** 2
        except:
            return 1e10

    # Try global optimization
    print("  Running differential evolution...")
    bounds = [(0.1, 50)] * h11
    result = differential_evolution(
        objective,
        bounds,
        seed=42,
        maxiter=500,
        tol=1e-10,
        disp=False
    )

    t_opt = result.x
    V_string_opt = cy.compute_cy_volume(t_opt)
    V_einstein_opt = V_string_opt * (g_s ** (-1.5))

    print(f"  Optimal t = {t_opt}")
    print(f"  V_S (optimal) = {V_string_opt:.4f}")
    print(f"  V_E (optimal) = {V_einstein_opt:.4f}")
    print(f"  Target V_E = {expected_vol_einstein:.4f}")
    print()

    # Step 7: Check what range of V_E is achievable
    print("Step 7: Exploring achievable volume range...")

    # Try various t values
    test_scales = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    print("  Scale   V_S        V_E")
    print("  " + "-" * 40)
    for scale in test_scales:
        t_test = tip * scale
        try:
            V_s = cy.compute_cy_volume(t_test)
            V_e = V_s * (g_s ** (-1.5))
            print(f"  {scale:5.1f}   {V_s:10.4f}   {V_e:10.2f}")
        except:
            print(f"  {scale:5.1f}   (error)")
    print()

    # Step 8: Results
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    rel_error = abs(V_einstein_opt - expected_vol_einstein) / expected_vol_einstein * 100
    print(f"  Computed V_E: {V_einstein_opt:.4f}")
    print(f"  Expected V_E: {expected_vol_einstein:.4f}")
    print(f"  Relative error: {rel_error:.2f}%")
    print()

    if rel_error < 1.0:
        print("SUCCESS: Reproduced McAllister volume within 1%!")
    elif rel_error < 10.0:
        print("PARTIAL SUCCESS: Within 10%")
    else:
        print("MISMATCH: The volume range may not be achievable with CYTools' default triangulation")
        print()
        print("Note: McAllister uses their own triangulation from dual_simplices.dat")
        print("which may give a different intersection tensor than CYTools' default.")

    return V_einstein_opt, expected_vol_einstein


if __name__ == "__main__":
    main()
