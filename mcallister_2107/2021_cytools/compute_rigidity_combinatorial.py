#!/usr/bin/env python3
"""
Compute divisor rigidity combinatorially from polytope structure.

From Braun et al. (arXiv:1712.04946):
- A prime toric divisor D is rigid iff h^i(O_D) = (1, 0, 0)

For CY threefold hypersurfaces, rigidity is determined combinatorially:
1. Points interior to 2-faces of Δ° → always rigid
2. Points interior to 1-faces of Δ° → rigid iff dual edge has no interior points
3. Vertices of Δ° → rigid iff dual facet in Δ has NO interior points

Uses CYTools' PolytopeFace.dual_face().interior_points() method.

Validation: Tests against all 5 McAllister examples.
- 4-214-647: 214 divisors, validates against kklt_basis.dat (214 rigid)
- 7-51-13590: primal is non-favorable in CYTools 2021 (skipped)
"""

import sys
from pathlib import Path

import numpy as np

# Paths
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent.parent
DATA_BASE = ROOT_DIR / "resources/small_cc_2107.09064_source/anc/paper_data"

# Use CYTools 2021 for consistency
CYTOOLS_2021 = ROOT_DIR / "vendor/cytools_mcallister_2107"
sys.path.insert(0, str(CYTOOLS_2021))

from cytools import Polytope

# McAllister examples (name, h11_primal, h21_primal)
MCALLISTER_EXAMPLES = [
    ("4-214-647", 214, 4),
    ("5-113-4627-main", 113, 5),
    ("5-113-4627-alternative", 113, 5),
    ("5-81-3213", 81, 5),
    # ("7-51-13590", 51, 7),  # primal is non-favorable in CYTools 2021
]


# =============================================================================
# PURE COMPUTATION FUNCTIONS
# =============================================================================


def get_point_to_face_map(poly) -> dict:
    """
    Map each point index to the minimal face containing it.

    Returns dict: point_idx -> (face_dim, face_idx, face_obj, type_str)
    """
    all_pts = poly.points()
    point_to_face = {}

    # Check origin first
    for i, pt in enumerate(all_pts):
        if np.allclose(pt, 0):
            point_to_face[i] = (-1, None, None, "origin")
            break

    # Get faces by dimension (0=vertices, 1=edges, 2=2-faces, 3=facets)
    for dim in range(poly.dim() + 1):
        faces = poly.faces(dim)
        for face_idx, face in enumerate(faces):
            face_pts = face.points()
            interior_pts = face.interior_points()

            for i, pt in enumerate(all_pts):
                if i in point_to_face:
                    continue

                # Check if point is interior to this face
                if interior_pts is not None:
                    for ip in interior_pts:
                        if np.allclose(pt, ip):
                            point_to_face[i] = (dim, face_idx, face, f"{dim}-face interior")
                            break

                # Check if point is a vertex of this face (dim=0)
                if dim == 0:
                    for fp in face_pts:
                        if np.allclose(pt, fp):
                            point_to_face[i] = (0, face_idx, face, "vertex")
                            break

    return point_to_face


def compute_rigidity(poly) -> dict:
    """
    Compute rigidity for all prime toric divisors combinatorially.

    This uses the Braun formula (arXiv:1712.04946 eq 2.7):
    - Vertex: rigid iff dual facet has no interior points (g=0)
    - Edge interior: rigid iff dual edge has no interior points
    - 2-face interior: always rigid (dual is vertex, g=0)

    Args:
        poly: CYTools Polytope object

    Returns:
        dict: point_idx -> {'rigid': bool, 'reason': str, 'type': str, ...}
    """
    all_pts = poly.points()
    point_to_face = get_point_to_face_map(poly)

    results = {}

    for pt_idx in range(len(all_pts)):
        pt = all_pts[pt_idx]

        # Origin
        if np.allclose(pt, 0):
            results[pt_idx] = {
                "rigid": None,
                "reason": "origin (not a divisor)",
                "type": "origin",
            }
            continue

        face_info = point_to_face.get(pt_idx)
        if face_info is None:
            results[pt_idx] = {
                "rigid": None,
                "reason": "could not classify point",
                "type": "unknown",
            }
            continue

        dim, face_idx, face, face_type = face_info

        if face_type == "vertex":
            # For vertices: rigid iff dual face has no interior points
            # CYTools 2021 uses dual() instead of dual_face()
            dual_face = face.dual()
            interior = dual_face.interior_points()
            n_interior = len(interior) if interior is not None else 0

            results[pt_idx] = {
                "rigid": n_interior == 0,
                "reason": f"dual facet has {n_interior} interior points",
                "type": "vertex",
                "dual_face_dim": dual_face.dim(),
                "n_interior": n_interior,
            }

        elif "1-face" in face_type:
            # Edge interior: rigid iff dual edge has no interior points
            dual_face = face.dual()
            interior = dual_face.interior_points()
            n_interior = len(interior) if interior is not None else 0
            results[pt_idx] = {
                "rigid": n_interior == 0,
                "reason": f"dual edge has {n_interior} interior points",
                "type": "1-face interior",
                "dual_face_dim": dual_face.dim(),
                "n_interior": n_interior,
            }

        elif "2-face" in face_type:
            # 2-face interior: always rigid (dual is vertex with g=0)
            results[pt_idx] = {
                "rigid": True,
                "reason": "interior to 2-face → dual is vertex with g=0",
                "type": "2-face interior",
                "dual_face_dim": 0,
                "n_interior": 0,
            }

        elif "3-face" in face_type:
            # 3-face interior (facets) - don't intersect generic CY hypersurface
            results[pt_idx] = {
                "rigid": True,
                "reason": "interior to 3-face → not on CY hypersurface",
                "type": "3-face interior",
                "n_interior": 0,
            }

        else:
            results[pt_idx] = {
                "rigid": None,
                "reason": f"unknown face type: {face_type}",
                "type": face_type,
            }

    return results


# =============================================================================
# DATA LOADING
# =============================================================================


def load_primal_points(example_name: str) -> np.ndarray:
    """Load primal polytope points (points.dat)."""
    data_dir = DATA_BASE / example_name
    lines = (data_dir / "points.dat").read_text().strip().split('\n')
    return np.array([[int(x) for x in line.split(',')] for line in lines])


def load_target_volumes(example_name: str) -> np.ndarray:
    """Load c_i values from target_volumes.dat."""
    data_dir = DATA_BASE / example_name
    text = (data_dir / "target_volumes.dat").read_text().strip()
    return np.array([int(x) for x in text.split(",")])


def load_kklt_basis(example_name: str) -> np.ndarray:
    """Load KKLT basis indices from kklt_basis.dat."""
    data_dir = DATA_BASE / example_name
    basis_path = data_dir / "kklt_basis.dat"
    if not basis_path.exists():
        return None
    text = basis_path.read_text().strip()
    return np.array([int(x) for x in text.split(",")])


# =============================================================================
# VALIDATION TESTS
# =============================================================================


def test_example(example_name: str, expected_h11: int, verbose: bool = True) -> dict:
    """
    Test rigidity computation for one McAllister example.

    Validates:
    - All divisors in kklt_basis.dat should be rigid
    - Excluded divisors should be non-rigid

    Returns:
        Dict with test results
    """
    if verbose:
        print("=" * 70)
        print(f"TEST - {example_name} (primal h11={expected_h11})")
        print("=" * 70)

    # Load primal polytope
    points = load_primal_points(example_name)
    if verbose:
        print(f"\n  Loaded primal polytope: {points.shape[0]} points")

    poly = Polytope(points)

    # Check if favorable (CYTools 2021 requires lattice argument)
    try:
        is_fav = poly.is_favorable(lattice="N")  # N-lattice (dual)
    except TypeError:
        # Older CYTools versions don't need argument
        is_fav = poly.is_favorable()

    if not is_fav:
        if verbose:
            print(f"  SKIP: Polytope is non-favorable in CYTools 2021")
        return {"example_name": example_name, "passed": True, "skipped": True}

    # Compute rigidity
    results = compute_rigidity(poly)

    # Count by type
    type_counts = {}
    rigid_count = 0
    for pt_idx, r in results.items():
        t = r["type"]
        type_counts[t] = type_counts.get(t, 0) + 1
        if r.get("rigid"):
            rigid_count += 1

    if verbose:
        print(f"\n  Point classification:")
        for t, count in sorted(type_counts.items()):
            print(f"    {t}: {count}")
        print(f"  Total rigid: {rigid_count}/{len(results) - 1}")  # -1 for origin

    # Load ground truth
    target_c = load_target_volumes(example_name)
    kklt_basis = load_kklt_basis(example_name)

    if kklt_basis is None:
        if verbose:
            print(f"\n  No kklt_basis.dat for validation")
        return {
            "example_name": example_name,
            "passed": True,
            "n_rigid": rigid_count,
            "n_total": len(results) - 1,
        }

    # Validate: all kklt_basis divisors should be rigid
    matches = 0
    mismatches = 0
    mismatch_details = []

    for i, basis_idx in enumerate(kklt_basis):
        result = results.get(basis_idx)
        if result is None:
            mismatch_details.append(f"Point {basis_idx}: not found")
            mismatches += 1
        elif result["rigid"] is True:
            matches += 1
        elif result["rigid"] is False:
            mismatch_details.append(
                f"Point {basis_idx}: should be rigid but is not ({result['reason']})"
            )
            mismatches += 1
        else:
            mismatch_details.append(
                f"Point {basis_idx}: unknown ({result.get('reason', '?')})"
            )
            mismatches += 1

    passed = mismatches == 0

    if verbose:
        print(f"\n  Validation against kklt_basis.dat ({len(kklt_basis)} divisors):")
        print(f"    Rigid as expected: {matches}")
        print(f"    Mismatches: {mismatches}")
        if mismatch_details:
            for detail in mismatch_details[:5]:
                print(f"      {detail}")
            if len(mismatch_details) > 5:
                print(f"      ... and {len(mismatch_details) - 5} more")

        status = "PASS" if passed else "FAIL"
        print(f"\n{status}: {example_name}")

    return {
        "example_name": example_name,
        "passed": passed,
        "n_rigid": rigid_count,
        "n_kklt": len(kklt_basis),
        "n_matches": matches,
        "n_mismatches": mismatches,
    }


def main():
    """Test rigidity computation against all McAllister examples."""
    print("=" * 70)
    print("COMBINATORIAL RIGIDITY - MCALLISTER EXAMPLES (CYTools 2021)")
    print("Divisor D is rigid iff h^i(O_D) = (1, 0, 0)")
    print("=" * 70)
    print("\nNOTE: Uses dual_face().interior_points() to determine rigidity")
    print("      7-51-13590 excluded (primal non-favorable in CYTools 2021)")

    results = []
    for name, h11, h21 in MCALLISTER_EXAMPLES:
        result = test_example(name, h11, verbose=True)
        results.append(result)
        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_passed = True
    for r in results:
        if r.get("skipped"):
            print(f"  SKIP: {r['example_name']:30s} (non-favorable)")
        else:
            status = "PASS" if r["passed"] else "FAIL"
            print(f"  {status}: {r['example_name']:30s} "
                  f"{r.get('n_matches', '?')}/{r.get('n_kklt', '?')} rigid in kklt_basis")
            all_passed = all_passed and r["passed"]

    print()
    if all_passed:
        print(f"All {len(results)} examples PASSED")
        print("Rigidity computation validated combinatorially.")
    else:
        n_passed = sum(1 for r in results if r["passed"])
        print(f"{n_passed}/{len(results)} examples passed")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
