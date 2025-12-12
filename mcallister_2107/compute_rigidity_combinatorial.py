#!/usr/bin/env python3
"""
Compute divisor rigidity combinatorially from polytope structure.

From McAllister et al. (arXiv:1712.04946):
- A prime toric divisor D is rigid iff h^i(O_D) = (1, 0, 0)

For CY threefold hypersurfaces, rigidity is determined combinatorially:
1. Points interior to 2-faces of Δ° → always rigid
2. Points interior to 1-faces of Δ° → always rigid (in Δ-favorable models)
3. Vertices of Δ° → rigid iff dual facet in Δ has NO interior points

Uses CYTools' PolytopeFace.dual_face().interior_points() method.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "vendor/cytools_latest/src"))

from cytools import Polytope

DATA_DIR = (
    Path(__file__).parent.parent
    / "resources/small_cc_2107.09064_source/anc/paper_data/4-214-647"
)


def load_points(filename: str) -> np.ndarray:
    """Load polytope points from .dat file."""
    lines = (DATA_DIR / filename).read_text().strip().split("\n")
    return np.array([[int(x) for x in line.split(",")] for line in lines])


def load_target_volumes() -> np.ndarray:
    """Load c_i values from target_volumes.dat."""
    text = (DATA_DIR / "target_volumes.dat").read_text().strip()
    return np.array([int(x) for x in text.split(",")])


def load_kklt_basis() -> np.ndarray:
    """Load KKLT basis indices."""
    text = (DATA_DIR / "kklt_basis.dat").read_text().strip()
    return np.array([int(x) for x in text.split(",")])


def get_point_to_face_map(poly) -> dict:
    """
    Map each point index to the minimal face containing it.

    Returns dict: point_idx -> (face_dim, face_idx, face_obj)
    """
    all_pts = poly.points()
    n_pts = len(all_pts)

    point_to_face = {}

    # Check origin first
    for i, pt in enumerate(all_pts):
        if np.allclose(pt, 0):
            point_to_face[i] = (-1, None, None, "origin")
            break

    # Get faces by dimension
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
    Compute rigidity for all prime toric divisors using CYTools.

    Returns dict: point_idx -> {'rigid': bool, 'reason': str, ...}
    """
    all_pts = poly.points()
    n_pts = len(all_pts)
    vertices = poly.vertices()

    # Map vertices to point indices
    vertex_to_point_idx = {}
    for i, v in enumerate(vertices):
        for j, pt in enumerate(all_pts):
            if np.allclose(v, pt):
                vertex_to_point_idx[i] = j
                break

    # Get 0-faces (vertices) for dual face computation
    faces_0d = poly.faces(0)

    # Map point indices to face locations
    point_to_face = get_point_to_face_map(poly)

    results = {}

    for pt_idx in range(n_pts):
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
            dual_face = face.dual_face()
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
            # Points interior to 1-faces: always rigid (Δ-favorable)
            results[pt_idx] = {
                "rigid": True,
                "reason": "interior to 1-face → always rigid",
                "type": "1-face interior",
            }

        elif "2-face" in face_type:
            # Points interior to 2-faces: always rigid
            results[pt_idx] = {
                "rigid": True,
                "reason": "interior to 2-face → always rigid",
                "type": "2-face interior",
            }

        elif "3-face" in face_type:
            # Points interior to 3-faces (facets): always rigid
            results[pt_idx] = {
                "rigid": True,
                "reason": "interior to 3-face → always rigid",
                "type": "3-face interior",
            }

        else:
            results[pt_idx] = {
                "rigid": None,
                "reason": f"unknown face type: {face_type}",
                "type": face_type,
            }

    return results


def test_primal_polytope() -> int:
    """
    Test on McAllister's primal polytope (h11=214).

    Returns 0 on success, 1 on failure.
    """
    print("=" * 70)
    print("COMBINATORIAL RIGIDITY - McAllister PRIMAL (h11=214)")
    print("=" * 70)

    points = load_points("points.dat")
    print(f"\n[1] Loaded {len(points)} points")

    poly = Polytope(points)
    print(f"    Polytope dim: {poly.dim()}")
    print(f"    Is reflexive: {poly.is_reflexive()}")
    print(f"    Vertices: {len(poly.vertices())}")

    # Load ground truth
    target_c = load_target_volumes()
    kklt_basis = load_kklt_basis()
    print(f"\n[2] Ground truth from target_volumes.dat:")
    print(f"    {len(target_c)} c_i values in KKLT basis")
    print(f"    c_i=1 (D3-instanton): {np.sum(target_c == 1)}")
    print(f"    c_i=6 (O7-plane): {np.sum(target_c == 6)}")

    print("\n[3] Computing rigidity via CYTools dual_face()...")
    results = compute_rigidity(poly)

    # Count by type
    type_counts = {}
    rigid_count = 0
    for pt_idx, r in results.items():
        t = r["type"]
        type_counts[t] = type_counts.get(t, 0) + 1
        if r["rigid"]:
            rigid_count += 1

    print("\n[4] Point classification:")
    for t, count in sorted(type_counts.items()):
        print(f"    {t}: {count}")
    print(f"\n    Total rigid: {rigid_count}/{len(results) - 1}")  # -1 for origin

    # Validate against ground truth
    print("\n[5] Validation against target_volumes.dat:")

    matches = 0
    mismatches = 0
    mismatch_details = []

    for i, (basis_idx, c_i) in enumerate(zip(kklt_basis, target_c)):
        expected_rigid = c_i > 0  # Both 1 and 6 indicate contributing divisor
        result = results.get(basis_idx)

        if result is None:
            mismatch_details.append(f"Point {basis_idx}: not found in results")
            mismatches += 1
            continue

        computed_rigid = result["rigid"]

        if computed_rigid == expected_rigid:
            matches += 1
        elif computed_rigid is None:
            mismatch_details.append(
                f"Point {basis_idx}: c_i={c_i}, computed=None ({result['reason']})"
            )
            mismatches += 1
        else:
            mismatch_details.append(
                f"Point {basis_idx}: c_i={c_i}, computed={computed_rigid} ({result['reason']})"
            )
            mismatches += 1

    print(f"    Matches: {matches}/{len(target_c)}")
    print(f"    Mismatches: {mismatches}/{len(target_c)}")

    if mismatch_details:
        print("\n    Mismatch details:")
        for detail in mismatch_details[:10]:
            print(f"      {detail}")
        if len(mismatch_details) > 10:
            print(f"      ... and {len(mismatch_details) - 10} more")

    if mismatches == 0:
        print("\n*** VALIDATION PASSED ***")
        return 0
    else:
        print(f"\n*** VALIDATION FAILED: {mismatches} mismatches ***")
        return 1


def test_dual_polytope() -> int:
    """
    Test on McAllister's dual polytope (h11=4).

    Returns 0 on success, 1 on failure.
    """
    print("=" * 70)
    print("COMBINATORIAL RIGIDITY - McAllister DUAL (h11=4)")
    print("=" * 70)

    points = load_points("dual_points.dat")
    print(f"\n[1] Loaded {len(points)} points")

    poly = Polytope(points)
    tri = poly.triangulate()
    cy = tri.get_cy()

    print(f"    h11 = {cy.h11()}, h21 = {cy.h21()}")
    print(f"    Vertices: {len(poly.vertices())}")

    print("\n[2] Computing rigidity via CYTools dual_face()...")
    results = compute_rigidity(poly)

    print("\n[3] Results:")
    rigid_count = 0
    for pt_idx in sorted(results.keys()):
        r = results[pt_idx]
        if r["type"] == "origin":
            continue
        status = "RIGID" if r["rigid"] else ("NOT RIGID" if r["rigid"] is False else "UNKNOWN")
        print(f"    Point {pt_idx}: {r['type']:20s} -> {status:10s} ({r['reason']})")
        if r["rigid"]:
            rigid_count += 1

    print(f"\n    Total rigid: {rigid_count}/{len(results) - 1}")
    return 0


def main() -> int:
    """Run tests. Returns 0 on success, 1 on failure."""
    test_dual_polytope()
    print("\n")
    return test_primal_polytope()


if __name__ == "__main__":
    sys.exit(main())
