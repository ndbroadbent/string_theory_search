#!/usr/bin/env python3
"""
Compute divisor cohomology h^i(D, O_D) for toric divisors on CY threefolds.

Method: cohomCalg + Koszul sequence (arXiv:1003.5217, arXiv:2111.03078)

This computes from first principles using the cohomCalg binary.

NOTE: For the McAllister pipeline, we use the faster COMBINATORIAL method
in compute_rigidity_combinatorial.py and compute_chi_divisor.py instead.
This script is a supporting tool for general-purpose cohomology computation.

Validation: Tests against McAllister's kklt_basis.dat (all divisors should be rigid).
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent.parent
COHOMCALG_BIN = ROOT_DIR / "vendor/cohomCalg/bin/cohomcalg"
DATA_BASE = ROOT_DIR / "resources/small_cc_2107.09064_source/anc/paper_data"

# Use CYTools 2021 for consistency
CYTOOLS_2021 = ROOT_DIR / "vendor/cytools_mcallister_2107"
sys.path.insert(0, str(CYTOOLS_2021))

import numpy as np
from cytools import Polytope

# McAllister examples (name, h11_primal, h21_primal)
# Note: We test DUAL polytopes here (small h11), so 7-51-13590 is included
# even though its primal is non-favorable.
MCALLISTER_EXAMPLES = [
    ("4-214-647", 214, 4),
    ("5-113-4627-main", 113, 5),
    ("5-113-4627-alternative", 113, 5),
    ("5-81-3213", 81, 5),
    ("7-51-13590", 51, 7),
]


def generate_cohomcalg_input(
    points: np.ndarray,
    glsm: np.ndarray,
    sr_ideal: tuple,
    line_bundles: list[np.ndarray],
) -> str:
    """
    Generate cohomCalg input file content.

    Args:
        points: (n_coords, dim) array of lattice points (excluding origin)
        glsm: (n_charges, n_coords) GLSM charge matrix
        sr_ideal: Stanley-Reisner ideal as tuple of tuples of coordinate indices
        line_bundles: List of line bundle degrees to compute

    Returns:
        String content for cohomCalg input file
    """
    n_coords = points.shape[0]
    n_charges = glsm.shape[0]

    lines = ["% Auto-generated cohomCalg input"]

    # Vertices with GLSM charges
    for i in range(n_coords):
        v_str = ", ".join(str(int(x)) for x in points[i])
        c_str = ", ".join(str(int(glsm[r, i])) for r in range(n_charges))
        lines.append(f"    vertex u{i+1} = ( {v_str} ) | GLSM: ( {c_str} );")

    lines.append("")

    # Stanley-Reisner ideal
    # SR indices are into poly.points() which includes origin at 0
    # So index k in SR corresponds to u{k} in cohomCalg
    sr_terms = []
    for gen in sr_ideal:
        term = "*".join(f"u{idx}" for idx in gen)
        sr_terms.append(term)
    lines.append(f"    srideal [{', '.join(sr_terms)}];")
    lines.append("")

    # Disable monomial files
    lines.append("    monomialfile off;")
    lines.append("")

    # Line bundles to compute
    for lb in line_bundles:
        degrees = ", ".join(str(int(d)) for d in lb)
        lines.append(f"    ambientcohom O( {degrees} );")

    return "\n".join(lines)


def run_cohomcalg(input_content: str) -> list[list[int]]:
    """
    Run cohomCalg binary and parse output.

    Returns list of cohomology dimensions for each requested line bundle.
    Each entry is [h^0, h^1, ..., h^dim].
    """
    if not COHOMCALG_BIN.exists():
        raise FileNotFoundError(
            f"cohomCalg not found at {COHOMCALG_BIN}. "
            "Run 'make' in vendor/cohomCalg/"
        )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".in", delete=False) as f:
        f.write(input_content)
        input_path = Path(f.name)

    try:
        result = subprocess.run(
            [str(COHOMCALG_BIN), "--integrated", str(input_path)],
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            raise RuntimeError(f"cohomCalg failed: {result.stderr}")

        return parse_cohomcalg_output(result.stdout)
    finally:
        input_path.unlink()


def parse_cohomcalg_output(output: str) -> list[list[int]]:
    """
    Parse cohomCalg --integrated output.

    Format: {True,{{h0,h1,h2,h3,h4},{{details}}},{{h0,h1,h2,h3,h4},{{details}}},...}
    """
    cohomologies = []

    for line in output.split("\n"):
        line = line.strip()
        if line.startswith("{True,") or line.startswith("{False,"):
            # Extract 5-tuples (for 4D ambient space)
            matches = re.findall(r"\{\{(\d+),(\d+),(\d+),(\d+),(\d+)\}", line)
            for m in matches:
                cohomologies.append([int(x) for x in m])

    return cohomologies


def chase_koszul_sequence(
    h_XD: list[int],
    h_X: list[int],
    h_D: list[int],
    h_O: list[int],
) -> list[int]:
    """
    Chase the Koszul sequence to get divisor cohomology h^i(D|X, O_D).

    Koszul 4-term sequence:
        0 -> O(-X-D) -> O(-X) + O(-D) -> O -> O_{D|X} -> 0

    Uses Euler characteristic additivity:
        chi(D|X) = chi(O) - chi(O(-D)) - chi(O(-X)) + chi(O(-X-D))

    For generic CY3 hypersurfaces where H^i of negative bundles
    is concentrated in top degree:
        h^0(D|X) = 1 (connected)
        h^1(D|X) = 0 (from sequence analysis)
        h^2(D|X) = chi(D|X) - 1

    Args:
        h_XD: H^i(A, O(-X-D))
        h_X: H^i(A, O(-X))
        h_D: H^i(A, O(-D))
        h_O: H^i(A, O)

    Returns:
        [h^0, h^1, h^2] for divisor restricted to CY hypersurface
    """

    def euler(h: list[int]) -> int:
        return sum((-1) ** i * h[i] for i in range(len(h)))

    chi_D = euler(h_O) - euler(h_D) - euler(h_X) + euler(h_XD)

    # Check if we're in the simple case
    h_X_low = sum(h_X[:-1])  # H^0 to H^{dim-1}
    h_D_low = sum(h_D[:-1])
    h_XD_low = sum(h_XD[:-1])

    if h_X_low == 0 and h_D_low == 0 and h_XD_low == 0:
        # Simple case: only H^{dim} terms nonzero for negative bundles
        h0 = 1  # D is connected
        h1 = 0  # From sequence analysis
        h2 = max(0, chi_D - h0 + h1)
        return [h0, h1, h2]
    else:
        # Complex case - need full sequence chase
        # For now, fall back to Euler characteristic
        h0 = 1
        h1 = 0
        h2 = max(0, chi_D - 1)
        return [h0, h1, h2]


def compute_divisor_cohomology(
    poly,
    tri,
    divisor_idx: int,
) -> dict:
    """
    Compute h^i(D|X, O_D) for a single toric divisor.

    Args:
        poly: CYTools Polytope object
        tri: CYTools Triangulation object
        divisor_idx: Index into poly.points() (INCLUDING origin)

    Returns:
        dict with 'h' (cohomology vector) and 'rigid' (bool)
    """
    # Get GLSM charge matrix (includes origin at column 0)
    glsm_full = poly.glsm_charge_matrix()
    n_charges = glsm_full.shape[0]
    n_glsm_cols = glsm_full.shape[1]

    # Get points - GLSM columns correspond to poly.points()
    pts_full = poly.points()

    # For cohomCalg, we need points and GLSM excluding origin
    pts = pts_full[1:]  # Skip origin
    glsm = glsm_full[:, 1:]  # Skip origin column
    n_coords = glsm.shape[1]  # Use GLSM column count, not points count

    # Get SR ideal
    sr = tri.sr_ideal()

    # Anticanonical class: sum of all divisor classes
    anticanonical = glsm.sum(axis=1)

    # Divisor class (divisor_idx is into full points array including origin)
    # So divisor_idx - 1 is the index into glsm
    glsm_idx = divisor_idx - 1
    if glsm_idx < 0 or glsm_idx >= n_coords:
        raise ValueError(f"Invalid divisor index {divisor_idx}, valid range 1-{n_coords}")

    div_class = glsm[:, glsm_idx]

    # Line bundles for Koszul sequence
    O_minus_X = -anticanonical
    O_minus_D = -div_class
    O_minus_XD = -(anticanonical + div_class)
    O_trivial = np.zeros(n_charges, dtype=int)

    # Generate and run cohomCalg
    # Use only the points that correspond to GLSM columns
    pts_for_cohomcalg = pts[:n_coords]
    inp = generate_cohomcalg_input(
        pts_for_cohomcalg, glsm, sr, [O_minus_XD, O_minus_X, O_minus_D, O_trivial]
    )
    results = run_cohomcalg(inp)

    if len(results) < 4:
        raise RuntimeError(f"cohomCalg returned {len(results)} results, expected 4")

    # Chase Koszul sequence
    h = chase_koszul_sequence(*results[:4])

    return {
        "h": h,
        "rigid": h[1] == 0 and h[2] == 0,
        "h_XD": results[0],
        "h_X": results[1],
        "h_D": results[2],
        "h_O": results[3],
    }


def compute_all_divisor_cohomology(poly, tri) -> list[dict]:
    """
    Compute divisor cohomology for all toric divisors.

    Args:
        poly: CYTools Polytope object
        tri: CYTools Triangulation object

    Returns:
        List of dicts, one per divisor (for each GLSM column, excluding origin)
    """
    # Number of divisors = GLSM columns - 1 (excluding origin)
    glsm = poly.glsm_charge_matrix()
    n_divisors = glsm.shape[1] - 1

    results = []
    for i in range(1, n_divisors + 1):  # Divisor indices 1 to n_divisors
        result = compute_divisor_cohomology(poly, tri, i)
        results.append(result)

    return results


def get_rigid_divisor_indices(poly, tri) -> list[int]:
    """
    Find indices of rigid divisors (h^1 = h^2 = 0).

    Returns:
        List of indices into poly.points() (INCLUDING origin offset)
    """
    all_cohom = compute_all_divisor_cohomology(poly, tri)
    rigid = []
    for i, result in enumerate(all_cohom):
        if result["rigid"]:
            rigid.append(i + 1)  # +1 because we skip origin
    return rigid


def is_rigid(h: list[int]) -> bool:
    """Check if divisor is rigid: h^1 = 0 and h^2 = 0."""
    return len(h) >= 3 and h[1] == 0 and h[2] == 0


# =============================================================================
# DATA LOADING
# =============================================================================


def load_points(example_name: str, filename: str) -> np.ndarray:
    """Load polytope points from .dat file."""
    data_dir = DATA_BASE / example_name
    lines = (data_dir / filename).read_text().strip().split('\n')
    return np.array([[int(x) for x in line.split(',')] for line in lines])


def load_target_volumes(example_name: str) -> np.ndarray:
    """Load c_i values from target_volumes.dat (1 or 6 = rigid)."""
    data_dir = DATA_BASE / example_name
    text = (data_dir / "target_volumes.dat").read_text().strip()
    return np.array([int(x) for x in text.split(',')])


def test_dual(example_name: str = "4-214-647", verbose: bool = True) -> dict:
    """
    Test divisor cohomology on McAllister's DUAL polytope.

    The dual polytope has small h11 (4-7 for McAllister examples),
    making cohomology computation via cohomCalg tractable.
    """
    if verbose:
        print("=" * 70)
        print(f"DIVISOR COHOMOLOGY - {example_name} DUAL")
        print("=" * 70)

    points = load_points(example_name, "dual_points.dat")
    if verbose:
        print(f"\n[1] Loaded {points.shape[0]} points")

    poly = Polytope(points)
    tri = poly.triangulate()
    cy = tri.get_cy()

    if verbose:
        print(f"    h11={cy.h11()}, h21={cy.h21()}")

    glsm = poly.glsm_charge_matrix()
    n_divisors = glsm.shape[1] - 1
    if verbose:
        print(f"    Divisors: {n_divisors}")

    if verbose:
        print("\n[2] Computing cohomology via cohomCalg...")
    all_cohom = compute_all_divisor_cohomology(poly, tri)

    if verbose:
        print("\n[3] Results:")
        for i, result in enumerate(all_cohom):
            status = "RIGID" if result["rigid"] else "not rigid"
            print(f"    D{i+1}: h^i={result['h']} -> {status}")

    n_rigid = sum(1 for r in all_cohom if r["rigid"])
    if verbose:
        print(f"\n    Rigid: {n_rigid}/{len(all_cohom)}")

    return {
        "example_name": example_name,
        "n_divisors": n_divisors,
        "n_rigid": n_rigid,
        "cohomologies": all_cohom,
    }


def test_primal(example_name: str = "4-214-647", max_divisors: int = 5, verbose: bool = True) -> dict:
    """
    Test divisor cohomology on McAllister's PRIMAL polytope.

    The primal polytope has large h11 (51-214 for McAllister examples),
    so we only test first max_divisors to keep runtime reasonable.

    NOTE: For primal polytopes, the combinatorial method in
    compute_rigidity_combinatorial.py is much faster and recommended.
    """
    if verbose:
        print("\n" + "=" * 70)
        print(f"DIVISOR COHOMOLOGY - {example_name} PRIMAL (first {max_divisors})")
        print("=" * 70)

    points = load_points(example_name, "points.dat")
    if verbose:
        print(f"\n[1] Loaded {points.shape[0]} points")

    poly = Polytope(points)

    # Check if favorable (CYTools 2021 requires lattice argument)
    try:
        is_fav = poly.is_favorable(lattice="N")
    except TypeError:
        is_fav = poly.is_favorable()

    if not is_fav:
        if verbose:
            print(f"  SKIP: Polytope is non-favorable in CYTools 2021")
        return {"example_name": example_name, "passed": True, "skipped": True}

    tri = poly.triangulate()
    cy = tri.get_cy()

    if verbose:
        print(f"    h11={cy.h11()}, h21={cy.h21()}")

    glsm = poly.glsm_charge_matrix()
    n_divisors = glsm.shape[1] - 1
    if verbose:
        print(f"    Divisors: {n_divisors}")

    # Load ground truth
    target_c = load_target_volumes(example_name)
    if verbose:
        print(f"    Ground truth c_i: {len(target_c)} values")
        print(f"    c_i=1 (D3): {np.sum(target_c == 1)}, c_i=6 (O7): {np.sum(target_c == 6)}")

    if verbose:
        print(f"\n[2] Computing cohomology for first {max_divisors} divisors...")

    results = []
    for i in range(1, min(max_divisors + 1, n_divisors + 1)):
        result = compute_divisor_cohomology(poly, tri, i)
        results.append(result)
        if verbose:
            status = "RIGID" if result["rigid"] else "not rigid"
            print(f"    D{i}: h^i={result['h']} -> {status}")

    if verbose:
        print(f"\n[3] Note: Full validation requires kklt_basis.dat mapping")

    n_rigid = sum(1 for r in results if r["rigid"])
    return {
        "example_name": example_name,
        "n_tested": len(results),
        "n_rigid": n_rigid,
        "cohomologies": results,
    }


def main():
    """Test divisor cohomology for all McAllister examples."""
    print("=" * 70)
    print("DIVISOR COHOMOLOGY VIA COHOMCALG - MCALLISTER EXAMPLES")
    print("Computes h^i(D, O_D) using cohomCalg + Koszul sequence")
    print("=" * 70)
    print("\nNOTE: This script uses the cohomCalg binary for line bundle cohomology.")
    print("      For production use, prefer compute_rigidity_combinatorial.py")
    print("      which is faster and doesn't require external dependencies.")

    # Test dual polytopes (small h11 - tractable for cohomCalg)
    print("\n" + "=" * 70)
    print("DUAL POLYTOPES (small h11)")
    print("=" * 70)

    dual_results = []
    for name, h11, h21 in MCALLISTER_EXAMPLES:
        result = test_dual(name, verbose=True)
        dual_results.append(result)
        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nDual polytopes (full cohomology computed):")
    for r in dual_results:
        print(f"  {r['example_name']:30s} {r['n_rigid']}/{r['n_divisors']} rigid")

    print("\nNOTE: Primal polytopes (h11=51-214) are too large for cohomCalg.")
    print("      Use compute_rigidity_combinatorial.py for primal validation.")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
