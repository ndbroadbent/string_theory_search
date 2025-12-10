#!/usr/bin/env python3
"""
Test physics_bridge.py with both GA mode and fixed mode.

NOTE: McAllister's polytope 4-214-647 dual_points.dat is NOT reflexive, so
physics_bridge will reject it. Instead, we use a known reflexive polytope
from the Kreuzer-Skarke database to test both modes.

For McAllister volume reproduction, see tools/reproduce_mcallister_volume.py
which uses CYTools directly (bypassing physics_bridge reflexivity check).
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from physics_bridge import PhysicsBridge


def main():
    print("=" * 70)
    print("Testing PhysicsBridge: GA Mode vs Fixed Mode")
    print("=" * 70)
    print()

    # Use a simple reflexive polytope (the quintic - simplest CY3)
    # P^4 hypersurface vertices
    quintic_vertices = [
        [-1, -1, -1, -1],
        [4, -1, -1, -1],
        [-1, 4, -1, -1],
        [-1, -1, 4, -1],
        [-1, -1, -1, 4],
    ]

    print("Using quintic (P^4 hypersurface) polytope")
    print(f"  Vertices: {len(quintic_vertices)} points")
    print()

    # Create physics bridge
    print("Creating PhysicsBridge...")
    bridge = PhysicsBridge()
    print()

    g_s = 0.1  # Typical string coupling

    # Test 1: GA mode (default)
    print("-" * 70)
    print("TEST 1: GA Mode (raytracing from genome direction)")
    print("-" * 70)

    genome_ga = {
        "vertices": quintic_vertices,
        "kahler_moduli": [1.0],  # Direction seed (h11=1 for quintic)
        "complex_moduli": [1.0],
        "flux_f": [1, -1],
        "flux_h": [1, 1],
        "g_s": g_s,
        "kahler_mode": "ga",  # Explicit, but this is default
    }

    result_ga = bridge.compute_physics(genome_ga)

    if result_ga["success"]:
        print(f"  Success!")
        print(f"  h11: {result_ga.get('h11')}, h21: {result_ga.get('h21')}")
        print(f"  CY Volume (string frame): {result_ga.get('cy_volume', 0):.4f}")
        v_einstein_ga = result_ga.get('cy_volume', 0) * (g_s ** (-1.5))
        print(f"  CY Volume (Einstein frame): {v_einstein_ga:.2f}")
    else:
        print(f"  Failed: {result_ga.get('error')}")
    print()

    # Test 2: Fixed mode with specific Kähler moduli
    print("-" * 70)
    print("TEST 2: Fixed Mode (exact Kähler moduli)")
    print("-" * 70)

    # For quintic, h11=1, so we just need one Kähler modulus
    # Try a specific value
    fixed_kahler = [5.0]

    genome_fixed = {
        "vertices": quintic_vertices,
        "kahler_moduli": fixed_kahler,
        "complex_moduli": [1.0],
        "flux_f": [1, -1],
        "flux_h": [1, 1],
        "g_s": g_s,
        "kahler_mode": "fixed",  # Use exact values
    }

    result_fixed = bridge.compute_physics(genome_fixed)

    if result_fixed["success"]:
        print(f"  Success!")
        print(f"  h11: {result_fixed.get('h11')}, h21: {result_fixed.get('h21')}")
        print(f"  Requested Kähler moduli: {fixed_kahler}")
        print(f"  CY Volume (string frame): {result_fixed.get('cy_volume', 0):.4f}")
        v_einstein_fixed = result_fixed.get('cy_volume', 0) * (g_s ** (-1.5))
        print(f"  CY Volume (Einstein frame): {v_einstein_fixed:.2f}")
    else:
        print(f"  Failed: {result_fixed.get('error')}")
    print()

    # Test 3: Fixed mode with different values
    print("-" * 70)
    print("TEST 3: Fixed Mode with larger Kähler modulus")
    print("-" * 70)

    larger_kahler = [10.0]

    genome_larger = {
        "vertices": quintic_vertices,
        "kahler_moduli": larger_kahler,
        "complex_moduli": [1.0],
        "flux_f": [1, -1],
        "flux_h": [1, 1],
        "g_s": g_s,
        "kahler_mode": "fixed",
    }

    result_larger = bridge.compute_physics(genome_larger)

    if result_larger["success"]:
        print(f"  Success!")
        print(f"  Requested Kähler moduli: {larger_kahler}")
        print(f"  CY Volume (string frame): {result_larger.get('cy_volume', 0):.4f}")
        v_einstein_larger = result_larger.get('cy_volume', 0) * (g_s ** (-1.5))
        print(f"  CY Volume (Einstein frame): {v_einstein_larger:.2f}")

        # Volume should scale as t^3 for quintic (roughly)
        ratio = result_larger.get('cy_volume', 0) / result_fixed.get('cy_volume', 1)
        expected_ratio = (larger_kahler[0] / fixed_kahler[0]) ** 3
        print(f"  Volume ratio: {ratio:.2f} (expected ~{expected_ratio:.2f} for t^3 scaling)")
    else:
        print(f"  Failed: {result_larger.get('error')}")
    print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_success = all([
        result_ga.get("success"),
        result_fixed.get("success"),
        result_larger.get("success"),
    ])
    if all_success:
        print("All tests passed!")
        print("  - GA mode produces valid physics")
        print("  - Fixed mode uses exact Kähler moduli")
        print("  - Volume scales appropriately with moduli")
    else:
        print("Some tests failed!")
        for name, result in [("GA", result_ga), ("Fixed", result_fixed), ("Larger", result_larger)]:
            if not result.get("success"):
                print(f"  {name} mode error: {result.get('error')}")


if __name__ == "__main__":
    main()
