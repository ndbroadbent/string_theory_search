#!/usr/bin/env python3
"""
Compute dual Coxeter numbers c_i for KKLT moduli stabilization.

The KKLT superpotential is:
    W = W_0 + sum_i A_i exp(-2*pi*T_i / c_i)

Where c_i depends on the source of the non-perturbative effect:
- c_i = 1 for D3-brane instantons on rigid divisors
- c_i = 6 for gaugino condensation on O7-planes with SO(8) gauge group

This computes from first principles using cohomCalg for divisor cohomology.

Reference: arXiv:2107.09064 (McAllister et al.)
"""

import re
from pathlib import Path
from typing import Optional

import numpy as np

# Import divisor cohomology computation
from compute_divisor_cohomology import (
    compute_all_divisor_cohomology,
    get_rigid_divisor_indices,
)

# McAllister data path
MCALLISTER_DIR = (
    Path(__file__).parent.parent
    / "resources/small_cc_2107.09064_source/anc/paper_data/4-214-647"
)


# =============================================================================
# DUAL COXETER NUMBERS
# =============================================================================


DUAL_COXETER_NUMBERS = {
    # ADE Lie algebras (as functions of rank n)
    "su": lambda n: n,  # SU(N): c2 = N
    "so": lambda n: n - 2,  # SO(N): c2 = N - 2
    "sp": lambda n: n + 1,  # Sp(N): c2 = N + 1
    # Exceptional algebras
    "e6": 12,
    "e7": 18,
    "e8": 30,
    "f4": 9,
    "g2": 4,
    # Common in string theory
    "so(8)": 6,  # Most common in McAllister - O7-planes with 4 D7s
    "su(5)": 5,  # GUT group
}


def get_dual_coxeter(gauge_algebra: str) -> int:
    """
    Get dual Coxeter number for a gauge algebra.

    Args:
        gauge_algebra: String like "so(8)", "su(5)", "e8", etc.

    Returns:
        Dual Coxeter number c2
    """
    s = gauge_algebra.lower().strip()

    # Check for exact match
    if s in DUAL_COXETER_NUMBERS:
        val = DUAL_COXETER_NUMBERS[s]
        return val if isinstance(val, int) else val(0)

    # Check for parametric match (su(n), so(n), sp(n))
    match = re.match(r"(su|so|sp)\((\d+)\)", s)
    if match:
        family = match.group(1)
        n = int(match.group(2))
        func = DUAL_COXETER_NUMBERS[family]
        return func(n)

    raise ValueError(f"Unknown gauge algebra: {gauge_algebra}")


# =============================================================================
# ORIENTIFOLD INVOLUTIONS
# =============================================================================


def parse_coordinate_involution(invol_str: str, n_coords: int) -> dict[int, int]:
    """
    Parse an orientifold involution defined by coordinate negation.

    Format: "x1, x3, x5" means negate coordinates 1, 3, 5 (1-indexed)

    The involution acts as x_i -> -x_i for listed coordinates.
    Fixed divisors are those where the involution acts trivially.

    Returns:
        Dict mapping coordinate index to its image (or itself if fixed)
    """
    mapping = {}

    # Initialize identity
    for i in range(1, n_coords + 1):
        mapping[i] = i

    if not invol_str.strip():
        return mapping

    # Parse negated coordinates
    negated = set()
    for part in invol_str.split(","):
        part = part.strip()
        if part.startswith("x"):
            idx = int(part[1:])
            negated.add(idx)

    # For negated coordinates, the corresponding divisor {x_i = 0} is FIXED
    # (since negating x_i doesn't change the locus x_i = 0)
    # This is where O7-planes live

    return mapping, negated


def identify_o7_divisors(
    involution_negated: set[int],
    rigid_divisor_indices: list[int],
    n_coords: int,
) -> list[int]:
    """
    Identify which rigid divisors host O7-planes.

    For an involution that negates coordinates {x_i1, x_i2, ...},
    the O7-planes wrap the fixed divisors {x_ij = 0}.

    Only RIGID fixed divisors contribute to the superpotential.

    Args:
        involution_negated: Set of coordinate indices that are negated
        rigid_divisor_indices: Indices of rigid divisors (from cohomology)
        n_coords: Total number of coordinates

    Returns:
        List of divisor indices that are O7-planes
    """
    o7_divisors = []

    for div_idx in rigid_divisor_indices:
        # div_idx is 1-indexed (coordinate index)
        if div_idx in involution_negated:
            o7_divisors.append(div_idx)

    return o7_divisors


def compute_c_i_values(
    poly,
    tri,
    involution_negated: Optional[set[int]] = None,
) -> dict:
    """
    Compute c_i values for all divisors that contribute to the superpotential.

    Args:
        poly: CYTools Polytope object
        tri: CYTools Triangulation object
        involution_negated: Set of coordinate indices negated by orientifold
                           If None, all rigid divisors get c_i = 1

    Returns:
        Dict with:
            - c_values: list of c_i for each divisor (0 if doesn't contribute)
            - rigid_indices: indices of rigid divisors
            - o7_indices: indices of O7-plane divisors
            - n_total: total number of divisors
    """
    # Get all divisor cohomology
    all_cohom = compute_all_divisor_cohomology(poly, tri)
    n_divisors = len(all_cohom)

    # Identify rigid divisors
    rigid_indices = []
    for i, result in enumerate(all_cohom):
        if result["rigid"]:
            rigid_indices.append(i + 1)  # 1-indexed

    # Identify O7-planes if involution is given
    o7_indices = []
    if involution_negated:
        o7_indices = identify_o7_divisors(
            involution_negated, rigid_indices, n_divisors
        )

    # Assign c_i values
    c_values = []
    for i in range(n_divisors):
        div_idx = i + 1  # 1-indexed
        if div_idx in rigid_indices:
            if div_idx in o7_indices:
                c_values.append(6)  # O7-plane with SO(8)
            else:
                c_values.append(1)  # D3-brane instanton
        else:
            c_values.append(0)  # Doesn't contribute

    return {
        "c_values": c_values,
        "rigid_indices": rigid_indices,
        "o7_indices": o7_indices,
        "n_total": n_divisors,
        "n_rigid": len(rigid_indices),
        "n_o7": len(o7_indices),
    }


# =============================================================================
# MCALLISTER DATA LOADING
# =============================================================================


def load_mcallister_c_i() -> tuple[np.ndarray, np.ndarray]:
    """
    Load c_i values from McAllister's 4-214-647 data.

    These are ground-truth values for validation.

    Returns:
        (c_values, basis_indices) where:
        - c_values: array of 214 dual Coxeter numbers (1 or 6)
        - basis_indices: 1-indexed divisor indices in KKLT basis
    """
    target_path = MCALLISTER_DIR / "target_volumes.dat"
    if not target_path.exists():
        raise FileNotFoundError(f"McAllister data not found: {target_path}")

    with open(target_path) as f:
        content = f.read().strip()
        c_values = np.array([int(x) for x in content.split(",")])

    basis_path = MCALLISTER_DIR / "kklt_basis.dat"
    if basis_path.exists():
        with open(basis_path) as f:
            content = f.read().strip()
            basis_indices = np.array([int(x) for x in content.split(",")])
    else:
        basis_indices = np.arange(1, len(c_values) + 1)

    return c_values, basis_indices


def get_mcallister_c_i_stats() -> dict:
    """Get statistics about McAllister's c_i values."""
    c_values, basis_indices = load_mcallister_c_i()

    return {
        "n_total": len(c_values),
        "n_o7": int(np.sum(c_values == 6)),
        "n_d3": int(np.sum(c_values == 1)),
        "c_values": c_values,
        "basis_indices": basis_indices,
    }


# =============================================================================
# MAIN
# =============================================================================


def main():
    """Test c_i computation."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent / "vendor/cytools_latest/src"))
    from cytools import Polytope

    print("=" * 70)
    print("DUAL COXETER NUMBERS (c_i) for KKLT")
    print("=" * 70)

    # Test 1: Dual Coxeter numbers reference
    print("\n[1] Dual Coxeter numbers reference:")
    algebras = ["so(8)", "su(5)", "e6", "e7", "e8", "so(10)", "sp(2)"]
    for alg in algebras:
        c2 = get_dual_coxeter(alg)
        print(f"  {alg}: c2 = {c2}")

    # Test 2: Compute c_i for h11=4 polytope
    print("\n[2] Computing c_i for h11=4 polytope:")
    vertices = np.array(
        [
            [-1, -1, -1, -1],
            [-1, -1, -1, 0],
            [-1, -1, 0, -1],
            [-1, -1, 0, 0],
            [-1, 0, -1, -1],
            [-1, 0, 0, 1],
            [0, -1, -1, -1],
            [1, 1, 1, 1],
        ]
    )

    poly = Polytope(vertices)
    tri = poly.triangulate()

    # Without involution (all rigid get c_i = 1)
    result = compute_c_i_values(poly, tri)
    print(f"  Total divisors: {result['n_total']}")
    print(f"  Rigid divisors: {result['n_rigid']}")
    print(f"  c_i values: {result['c_values']}")

    # With example involution (negate x1)
    print("\n  With involution negating x1:")
    result_with_invol = compute_c_i_values(poly, tri, involution_negated={1})
    print(f"  O7-planes: {result_with_invol['n_o7']}")
    print(f"  c_i values: {result_with_invol['c_values']}")

    # Test 3: McAllister ground truth
    print("\n[3] McAllister 4-214-647 ground truth:")
    try:
        stats = get_mcallister_c_i_stats()
        print(f"  Total in KKLT basis: {stats['n_total']}")
        print(f"  O7-planes (c_i=6): {stats['n_o7']}")
        print(f"  D3-instantons (c_i=1): {stats['n_d3']}")
        print(f"  First 10 c_i: {stats['c_values'][:10]}")
    except FileNotFoundError as e:
        print(f"  Error: {e}")


if __name__ == "__main__":
    main()
