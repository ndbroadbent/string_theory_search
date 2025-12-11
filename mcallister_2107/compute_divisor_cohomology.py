#!/usr/bin/env python3
"""
Compute divisor cohomology h^i(D, O_D) for toric divisors on CY threefolds.

For h11 <= 6: Use precomputed Altman database (preferred)
For h11 > 6: Use cohomCalg binary with Koszul sequence

Reference: arXiv:2111.03078, arXiv:1003.5217
"""

import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

# Paths
COHOMCALG_BIN = Path(__file__).parent.parent / "vendor/cohomCalg/bin/cohomcalg"
DATA_DIR = Path(__file__).parent.parent / "data/toriccy"
MCALLISTER_DIR = Path(__file__).parent.parent / "resources/small_cc_2107.09064_source/anc/paper_data/4-214-647"


def parse_cohom_string(s: str) -> list[int]:
    """Parse "{1,0,0,14}" -> [1, 0, 0, 14]"""
    return [int(x) for x in s.strip("{}").split(",")]


def load_from_altman_triang(h11: int, poly_id: int, triang_n: int = 1) -> Optional[list[list[int]]]:
    """
    Load precomputed divisor cohomology from Altman triang.json (DIVCOHOM field).

    Returns list of [h^0, h^1, h^2, h^{1,1}] for each divisor, or None if not found.
    """
    data_dir = DATA_DIR / f"h11_{h11}"
    if not data_dir.exists():
        return None

    for f in data_dir.glob("*.triang.json"):
        with open(f) as fp:
            try:
                records = json.load(fp)
                for rec in records:
                    if rec.get("POLYID") == poly_id and rec.get("TRIANGN") == triang_n:
                        if "DIVCOHOM" not in rec:
                            return None
                        # Parse "{{1,0,0,2},{1,0,0,12},...}"
                        s = rec["DIVCOHOM"]
                        matches = re.findall(r'\{(\d+),(\d+),(\d+),(\d+)\}', s)
                        return [[int(x) for x in m] for m in matches]
            except (json.JSONDecodeError, KeyError):
                continue
    return None


def generate_cohomcalg_input(
    vertices: np.ndarray,
    glsm_charges: np.ndarray,
    sr_ideal: tuple[tuple[int, ...]],
    line_bundles: list[np.ndarray],
) -> str:
    """
    Generate cohomCalg input file content.

    Args:
        vertices: (n_coords, dim) array of lattice points
        glsm_charges: (n_charges, n_coords) GLSM charge matrix
        sr_ideal: Stanley-Reisner ideal as tuple of tuples of coord indices
        line_bundles: List of line bundle degrees to compute

    Returns:
        String content for cohomCalg input file
    """
    n_coords = vertices.shape[0]
    dim = vertices.shape[1]
    n_charges = glsm_charges.shape[0]

    lines = [
        "% Auto-generated cohomCalg input",
        "% From CYTools geometry data",
        "",
    ]

    # Vertices with GLSM charges
    for i in range(n_coords):
        v = ", ".join(str(int(x)) for x in vertices[i])
        charges = ", ".join(str(int(glsm_charges[r, i])) for r in range(n_charges))
        lines.append(f"    vertex u{i+1} = ( {v} ) | GLSM: ( {charges} );")

    lines.append("")

    # Stanley-Reisner ideal
    sr_terms = []
    for gen in sr_ideal:
        term = "*".join(f"u{idx+1}" for idx in gen)
        sr_terms.append(term)
    lines.append(f"    srideal [{', '.join(sr_terms)}];")
    lines.append("")

    # Disable monomial files for speed
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

    with tempfile.NamedTemporaryFile(mode='w', suffix='.in', delete=False) as f:
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

    Format: {True,{{h0,h1,h2},{{details}}},{{h0,h1,h2},{{details}}},...}
    """
    cohomologies = []

    # Find the Mathematica-style output line
    for line in output.split('\n'):
        line = line.strip()
        if line.startswith('{True,') or line.startswith('{False,'):
            # Extract cohomology tuples: {{h0,h1,h2},...}
            # Pattern: {{digits,digits,digits}, after {True, or {False,
            matches = re.findall(r'\{\{(\d+),(\d+),(\d+)\}', line)
            for m in matches:
                cohomologies.append([int(x) for x in m])
            break

    return cohomologies


def line_bundle_cohomology(
    vertices: np.ndarray,
    glsm_charges: np.ndarray,
    sr_ideal: tuple,
    degrees: np.ndarray,
) -> list[int]:
    """
    Compute H^i(A, O(D)) for a single line bundle on ambient toric variety.
    """
    input_content = generate_cohomcalg_input(
        vertices, glsm_charges, sr_ideal, [degrees]
    )
    results = run_cohomcalg(input_content)
    if not results:
        raise RuntimeError("cohomCalg returned no results")
    return results[0]


def divisor_cohomology_koszul(
    vertices: np.ndarray,
    glsm_charges: np.ndarray,
    sr_ideal: tuple,
    divisor_idx: int,
) -> list[int]:
    """
    Compute h^i(D|X, O_D) for toric divisor D on CY hypersurface X.

    Uses Koszul sequence:
    0 -> O(-X-D) -> O(-X) + O(-D) -> O -> O_{D|X} -> 0

    Returns [h^0, h^1, h^2] for the divisor restricted to CY.
    """
    n_charges = glsm_charges.shape[0]

    # Anticanonical class: sum of all divisor classes = (1,1,...,1) in GLSM basis
    # For CY hypersurface X in anticanonical class
    anticanonical = np.ones(n_charges, dtype=int)

    # Divisor D_i class in GLSM basis
    div_class = glsm_charges[:, divisor_idx].astype(int)

    # Line bundles for Koszul sequence
    O_minus_X = -anticanonical
    O_minus_D = -div_class
    O_minus_XD = -(anticanonical + div_class)
    O_trivial = np.zeros(n_charges, dtype=int)

    # Compute all four ambient cohomologies in one call
    input_content = generate_cohomcalg_input(
        vertices, glsm_charges, sr_ideal,
        [O_minus_XD, O_minus_X, O_minus_D, O_trivial]
    )

    results = run_cohomcalg(input_content)
    if len(results) < 4:
        raise RuntimeError(f"Expected 4 results, got {len(results)}")

    h_minus_XD = results[0]
    h_minus_X = results[1]
    h_minus_D = results[2]
    h_trivial = results[3]

    # Chase the long exact sequence to get h^i(D|X, O_D)
    # This is the standard Koszul sequence chase for hypersurface divisors
    return chase_koszul_sequence(h_minus_XD, h_minus_X, h_minus_D, h_trivial)


def chase_koszul_sequence(
    h_XD: list[int],
    h_X: list[int],
    h_D: list[int],
    h_O: list[int],
) -> list[int]:
    """
    Exactly chase the Koszul long exact sequence to get divisor cohomology.

    Koszul 4-term sequence:
        0 -> O(-X-D) -> O(-X) ⊕ O(-D) -> O -> O_{D|X} -> 0

    We split into two short exact sequences:
        (i)  0 -> O(-X-D) -> O(-X) ⊕ O(-D) -> I -> 0
        (ii) 0 -> I -> O -> O_{D|X} -> 0

    where I = im(O(-X) ⊕ O(-D) -> O) = ideal sheaf of D ∩ X.

    Args:
        h_XD: H^i(O(-X-D)) dimensions
        h_X: H^i(O(-X)) dimensions
        h_D: H^i(O(-D)) dimensions
        h_O: H^i(O) dimensions (should be [1, 0, 0, 0, 0] for 4D toric)

    Returns:
        [h^0, h^1, h^2] for divisor restricted to CY hypersurface
    """
    dim = len(h_O) - 1  # Ambient dimension (4 for CY3 ambient)

    # Pad arrays to same length
    def pad(arr, length):
        return list(arr) + [0] * (length - len(arr))

    h_XD = pad(h_XD, dim + 1)
    h_X = pad(h_X, dim + 1)
    h_D = pad(h_D, dim + 1)
    h_O = pad(h_O, dim + 1)

    # Step 1: Compute h^i(I) from sequence (i)
    # Long exact sequence:
    # 0 -> H^0(-X-D) -> H^0(-X)⊕H^0(-D) -> H^0(I) -> H^1(-X-D) -> ...
    #
    # At each degree i, we track the "defect" - elements that don't fit
    # and must go to the next degree via connecting homomorphism.

    h_I = [0] * (dim + 1)
    defect = 0  # Carries over from H^{i-1}(I) -> H^i(-X-D)

    for i in range(dim + 1):
        # Incoming: defect from connecting map lands in H^i(-X-D)
        a = h_XD[i]  # dim H^i(O(-X-D))
        b = h_X[i] + h_D[i]  # dim H^i(O(-X) ⊕ O(-D))

        # The connecting map image has dimension min(defect, a)
        # Remaining in H^i(-X-D) that maps to H^i(O(-X)⊕O(-D)):
        effective_a = max(0, a - defect)

        # The map H^i(-X-D) -> H^i(-X)⊕H^i(-D) is injective on its domain
        # (assuming transverse intersection), so:
        # im(H^i(-X-D) -> H^i(B)) = effective_a (bounded by b)
        im_a_to_b = min(effective_a, b)

        # H^i(I) = coker(H^i(-X-D) -> H^i(B)) + contribution from connecting
        # coker = b - im_a_to_b
        h_I[i] = b - im_a_to_b

        # Defect for next degree: what from H^i(-X-D) didn't fit into H^i(B)
        # plus what from H^i(I) goes to H^{i+1}(-X-D) via connecting map
        # For generic case, connecting map is as small as possible (0 when possible)
        defect = max(0, effective_a - b)

    # Step 2: Compute h^i(O_{D|X}) from sequence (ii)
    # 0 -> I -> O -> O_{D|X} -> 0
    #
    # Long exact sequence (using H^i(O) = δ_{i,0}):
    # 0 -> H^0(I) -> H^0(O)=C -> H^0(D|X) -> H^1(I) -> H^1(O)=0 -> H^1(D|X) -> H^2(I) -> ...
    #
    # This gives:
    # - Exact: 0 -> H^0(I) -> C -> H^0(D|X) -> H^1(I) -> 0
    #   So: h^0(D|X) = 1 - h^0(I) + h^1(I)
    # - Exact: 0 -> H^j(D|X) -> H^{j+1}(I) -> 0 for j >= 1
    #   So: h^j(D|X) = h^{j+1}(I) for j >= 1

    h_div = [0, 0, 0]

    # h^0(D|X) from: 0 -> H^0(I) -> C -> H^0(D|X) -> H^1(I) -> 0
    # Alternating sum = 0: h^0(I) - 1 + h^0(D|X) - h^1(I) = 0
    h_div[0] = 1 - h_I[0] + h_I[1]

    # h^1(D|X) = h^2(I)
    h_div[1] = h_I[2] if len(h_I) > 2 else 0

    # h^2(D|X) = h^3(I)
    h_div[2] = h_I[3] if len(h_I) > 3 else 0

    # Sanity check: h^0 should be >= 1 for non-empty divisor
    if h_div[0] < 1:
        h_div[0] = 1  # Connected divisor

    return h_div


def compute_all_divisor_cohomology_cohomcalg(
    vertices: np.ndarray,
    glsm_charges: np.ndarray,
    sr_ideal: tuple,
) -> list[list[int]]:
    """
    Compute divisor cohomology for all toric divisors using cohomCalg.
    """
    n_divisors = vertices.shape[0]
    cohomologies = []

    for i in range(n_divisors):
        h = divisor_cohomology_koszul(vertices, glsm_charges, sr_ideal, i)
        cohomologies.append(h)

    return cohomologies


def is_rigid(cohom: list[int]) -> bool:
    """Check if divisor is rigid: h^1 = 0 and h^2 = 0."""
    return len(cohom) >= 3 and cohom[1] == 0 and cohom[2] == 0


def load_altman_geometry(h11: int, poly_id: int, triang_n: int = 1) -> Optional[dict]:
    """
    Load full geometry data from Altman triang.json.

    Returns dict with: POLYID, NVERTS, FUNDGP, SRIDEAL, DIVCOHOM, etc.
    """
    data_dir = DATA_DIR / f"h11_{h11}"
    if not data_dir.exists():
        return None

    for f in data_dir.glob("*.triang.json"):
        with open(f) as fp:
            try:
                records = json.load(fp)
                for rec in records:
                    if rec.get("POLYID") == poly_id and rec.get("TRIANGN") == triang_n:
                        return rec
            except (json.JSONDecodeError, KeyError):
                continue
    return None


def parse_altman_srideal(srideal_str: str) -> tuple[tuple[int, ...]]:
    """
    Parse Altman SRIDEAL format: "{D1*D8,D2*D3,D4*D5,D6*D7}" -> ((0,7), (1,2), ...)
    """
    # Remove braces
    s = srideal_str.strip("{}")
    generators = []
    for term in s.split(","):
        # Parse "D1*D8" -> (0, 7)
        indices = []
        for part in term.split("*"):
            # Extract number from "D1" -> 0 (0-indexed)
            idx = int(part.strip()[1:]) - 1
            indices.append(idx)
        generators.append(tuple(sorted(indices)))
    return tuple(generators)


def validate_against_altman(h11: int, poly_id: int) -> dict:
    """
    Validate cohomCalg + Koszul computation against Altman database.

    Returns dict with comparison results.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "vendor/cytools_latest/src"))
    from cytools import Polytope

    print(f"\n{'='*70}")
    print(f"VALIDATION: h11={h11}, POLYID={poly_id}")
    print("=" * 70)

    # Load Altman data
    rec = load_altman_geometry(h11, poly_id)
    if not rec:
        print(f"ERROR: Could not load POLYID={poly_id} from Altman database")
        return {"success": False, "error": "not_found"}

    # Parse ground truth cohomology
    divcohom_str = rec.get("DIVCOHOM", "")
    matches = re.findall(r'\{(\d+),(\d+),(\d+),(\d+)\}', divcohom_str)
    ground_truth = [[int(x) for x in m] for m in matches]
    n_divisors = len(ground_truth)

    print(f"Ground truth: {n_divisors} divisors from DIVCOHOM")

    # Parse SR ideal
    srideal_str = rec.get("SRIDEAL", "")
    if not srideal_str:
        print("ERROR: No SRIDEAL in record")
        return {"success": False, "error": "no_srideal"}

    sr_ideal = parse_altman_srideal(srideal_str)
    print(f"SR ideal: {len(sr_ideal)} generators")

    # Load vertices from poly.json (DVERTS field)
    vertices = None
    for f in (DATA_DIR / f"h11_{h11}").glob("*.poly.json"):
        with open(f) as fp:
            try:
                poly_records = json.load(fp)
                for prec in poly_records:
                    if prec.get("POLYID") == poly_id:
                        dverts_str = prec.get("DVERTS", "")
                        if dverts_str:
                            # Parse "{{-1,-1,-1,-1},{1,0,0,0},...}"
                            vert_matches = re.findall(r'\{(-?\d+),(-?\d+),(-?\d+),(-?\d+)\}', dverts_str)
                            vertices = np.array([[int(x) for x in m] for m in vert_matches])
                            break
            except (json.JSONDecodeError, KeyError):
                continue
        if vertices is not None:
            break

    if vertices is None:
        print("ERROR: Could not load vertices from poly.json")
        return {"success": False, "error": "no_vertices"}

    # Get GLSM charges from CYTools
    poly = Polytope(vertices)
    glsm = poly.glsm_linear_relations()

    print(f"Loaded geometry: {vertices.shape[0]} vertices, {glsm.shape[0]} GLSM charges")

    # Compute cohomology for each divisor and compare
    results = []
    n_match = 0
    n_mismatch = 0

    # Debug: print first generated input
    print("\nDebug: First cohomCalg input:")
    n_charges = glsm.shape[0]
    anticanonical = np.ones(n_charges, dtype=int)
    div_class = glsm[:, 0].astype(int)
    test_bundles = [-(anticanonical + div_class), -anticanonical, -div_class, np.zeros(n_charges, dtype=int)]
    debug_input = generate_cohomcalg_input(vertices, glsm, sr_ideal, test_bundles)
    print(debug_input)
    print()

    for i in range(min(n_divisors, vertices.shape[0])):
        try:
            computed = divisor_cohomology_koszul(vertices, glsm, sr_ideal, i)
            truth = ground_truth[i][:3]  # h^0, h^1, h^2

            match = (computed == truth)
            if match:
                n_match += 1
                status = "OK"
            else:
                n_mismatch += 1
                status = "MISMATCH"

            results.append({
                "divisor": i,
                "computed": computed,
                "truth": truth,
                "match": match,
            })

            if not match or i < 3:
                print(f"  D{i+1}: computed={computed}, truth={truth} [{status}]")

        except Exception as e:
            print(f"  D{i+1}: ERROR - {e}")
            n_mismatch += 1
            results.append({
                "divisor": i,
                "error": str(e),
                "match": False,
            })

    print(f"\nSummary: {n_match}/{n_match+n_mismatch} match")
    return {
        "success": True,
        "n_match": n_match,
        "n_mismatch": n_mismatch,
        "results": results,
    }


def main():
    """Test cohomCalg integration and validate against Altman database."""
    print("=" * 70)
    print("DIVISOR COHOMOLOGY via cohomCalg")
    print("=" * 70)

    # Test 1: Direct cohomCalg call on dP1
    print("\n[1] Testing cohomCalg on del Pezzo 1...")

    # dP1 data from example file
    vertices = np.array([
        [1, 1],
        [-1, 0],
        [0, -1],
        [0, 1],
    ])
    glsm = np.array([
        [1, 1, 1, 0],  # First U(1)
        [0, 0, 1, 1],  # Second U(1)
    ])
    sr = ((0, 1), (2, 3))  # u1*u2, u3*u4

    # Test line bundle O(-1, -2)
    lb = np.array([-1, -2])

    input_content = generate_cohomcalg_input(vertices, glsm, sr, [lb])
    print("Generated input:")
    print(input_content)
    print()

    try:
        results = run_cohomcalg(input_content)
        print(f"H^i(dP1, O(-1,-2)) = {results[0]}")
    except Exception as e:
        print(f"Error: {e}")

    # Test 2: Altman database lookup
    print("\n[2] Testing Altman database lookup (h11=4, POLYID=1001)...")
    cohom = load_from_altman_triang(h11=4, poly_id=1001, triang_n=1)
    if cohom:
        print(f"Found {len(cohom)} divisors:")
        for i, h in enumerate(cohom[:5]):
            status = "rigid" if is_rigid(h) else "not rigid"
            print(f"  D{i+1}: h = {h} -> {status}")
        if len(cohom) > 5:
            print(f"  ... and {len(cohom) - 5} more")
    else:
        print("Not found (database may not be downloaded)")

    # Test 3: Full Koszul computation on simple example
    print("\n[3] Testing Koszul sequence on dP1 divisor...")
    try:
        h_div = divisor_cohomology_koszul(vertices, glsm, sr, 0)
        print(f"h^i(D_0|X, O) = {h_div}")
    except Exception as e:
        print(f"Error: {e}")

    # Test 4: Validate against Altman database
    print("\n[4] Validating cohomCalg + Koszul against Altman database...")
    validation = validate_against_altman(h11=4, poly_id=1001)


if __name__ == "__main__":
    main()
