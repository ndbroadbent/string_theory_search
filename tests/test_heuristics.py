#!/usr/bin/env python3
"""
Snapshot tests for polytope heuristics computation.

Run tests:
    pytest tests/test_heuristics.py -v

Update snapshots:
    UPDATE_SNAPSHOTS=1 pytest tests/test_heuristics.py -v
"""

import json
import os
import pytest
import numpy as np
from pathlib import Path
from dataclasses import asdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from compute_all_heuristics import compute_all_heuristics, PolytopeHeuristics


SNAPSHOTS_DIR = Path(__file__).parent / "snapshots"
UPDATE_SNAPSHOTS = os.environ.get("UPDATE_SNAPSHOTS", "").lower() in ("1", "true", "yes")


# Test polytopes
TEST_POLYTOPES = {
    "cross_polytope_4d": {
        "h11": 5,
        "h21": 8,
        "vertex_count": 8,
        "vertices": [
            1, 0, 0, 0,
            -1, 0, 0, 0,
            0, 1, 0, 0,
            0, -1, 0, 0,
            0, 0, 1, 0,
            0, 0, -1, 0,
            0, 0, 0, 1,
            0, 0, 0, -1,
        ],
    },
    "asymmetric_simplex": {
        "h11": 10,
        "h21": 7,
        "vertex_count": 5,
        "vertices": [
            0, 0, 0, 0,
            2, 0, 0, 0,
            0, 3, 0, 0,
            0, 0, 1, 0,
            1, 1, 1, 2,
        ],
    },
}


def load_snapshot(name: str) -> dict | None:
    """Load a snapshot from disk."""
    path = SNAPSHOTS_DIR / f"{name}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def save_snapshot(name: str, data: dict):
    """Save a snapshot to disk."""
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOTS_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    print(f"Updated snapshot: {path}")


def compare_values(actual, expected, path="", tolerance=1e-9):
    """
    Recursively compare values, returning list of differences.
    """
    differences = []

    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            differences.append(f"{path}: expected dict, got {type(actual).__name__}")
        else:
            all_keys = set(expected.keys()) | set(actual.keys())
            for key in all_keys:
                if key not in expected:
                    differences.append(f"{path}.{key}: unexpected key in actual")
                elif key not in actual:
                    differences.append(f"{path}.{key}: missing key in actual")
                else:
                    differences.extend(
                        compare_values(actual[key], expected[key], f"{path}.{key}", tolerance)
                    )

    elif isinstance(expected, list):
        if not isinstance(actual, list):
            differences.append(f"{path}: expected list, got {type(actual).__name__}")
        elif len(actual) != len(expected):
            differences.append(f"{path}: length mismatch ({len(actual)} vs {len(expected)})")
        else:
            for i, (a, e) in enumerate(zip(actual, expected)):
                differences.extend(compare_values(a, e, f"{path}[{i}]", tolerance))

    elif isinstance(expected, float):
        if not isinstance(actual, (int, float)):
            differences.append(f"{path}: expected number, got {type(actual).__name__}")
        elif np.isnan(expected) and np.isnan(actual):
            pass  # Both NaN is ok
        elif np.isnan(expected) or np.isnan(actual):
            differences.append(f"{path}: NaN mismatch ({actual} vs {expected})")
        elif abs(actual - expected) > tolerance and abs(actual - expected) > tolerance * abs(expected):
            differences.append(f"{path}: {actual} != {expected} (diff={actual - expected})")

    elif isinstance(expected, (int, bool)):
        if actual != expected:
            differences.append(f"{path}: {actual} != {expected}")

    elif expected is None:
        if actual is not None:
            differences.append(f"{path}: expected None, got {actual}")

    else:
        if actual != expected:
            differences.append(f"{path}: {actual} != {expected}")

    return differences


class TestHeuristicsSnapshots:
    """Snapshot tests for heuristics computation."""

    @pytest.mark.parametrize("polytope_name", TEST_POLYTOPES.keys())
    def test_heuristics_snapshot(self, polytope_name):
        """Test heuristics against saved snapshot."""
        polytope = TEST_POLYTOPES[polytope_name]
        # Use deterministic IDs based on polytope name
        polytope_ids = {"cross_polytope_4d": 1, "asymmetric_simplex": 2}
        h = compute_all_heuristics(polytope, polytope_id=polytope_ids.get(polytope_name, 0))
        actual = asdict(h)

        # Remove polytope_id from comparison since it's just an identifier
        del actual["polytope_id"]

        # Handle NaN values for JSON serialization
        def sanitize(obj):
            if isinstance(obj, dict):
                return {k: sanitize(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize(v) for v in obj]
            elif isinstance(obj, float) and np.isnan(obj):
                return "NaN"
            elif isinstance(obj, float) and np.isinf(obj):
                return "Inf" if obj > 0 else "-Inf"
            return obj

        def desanitize(obj):
            if isinstance(obj, dict):
                return {k: desanitize(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [desanitize(v) for v in obj]
            elif obj == "NaN":
                return float("nan")
            elif obj == "Inf":
                return float("inf")
            elif obj == "-Inf":
                return float("-inf")
            return obj

        snapshot_name = f"heuristics_{polytope_name}"
        expected = load_snapshot(snapshot_name)

        if expected is None or UPDATE_SNAPSHOTS:
            save_snapshot(snapshot_name, sanitize(actual))
            if expected is None:
                pytest.skip(f"Created new snapshot: {snapshot_name}")
            return

        expected = desanitize(expected)
        differences = compare_values(actual, expected)

        if differences:
            diff_str = "\n".join(differences[:20])  # Limit output
            if len(differences) > 20:
                diff_str += f"\n... and {len(differences) - 20} more differences"
            pytest.fail(f"Snapshot mismatch for {polytope_name}:\n{diff_str}")


class TestHeuristicsBasic:
    """Basic sanity tests that don't depend on snapshots."""

    def test_embedding_dimensions_consistent(self):
        """All polytopes should produce same embedding dimension."""
        dims = set()
        for name, polytope in TEST_POLYTOPES.items():
            h = compute_all_heuristics(polytope, polytope_id=1)
            emb = h.to_embedding()
            dims.add(len(emb))

        assert len(dims) == 1, f"Inconsistent embedding dimensions: {dims}"

    def test_embedding_mostly_finite(self):
        """Embeddings should be mostly finite (allow a few NaN for edge cases)."""
        for name, polytope in TEST_POLYTOPES.items():
            h = compute_all_heuristics(polytope, polytope_id=1)
            emb = h.to_embedding()
            nan_count = sum(1 for x in emb if np.isnan(x))
            inf_count = sum(1 for x in emb if np.isinf(x))

            # Allow up to 5% NaN/Inf
            max_bad = len(emb) * 0.05
            assert nan_count <= max_bad, f"{name}: too many NaN ({nan_count}/{len(emb)})"
            assert inf_count <= max_bad, f"{name}: too many Inf ({inf_count}/{len(emb)})"

    def test_sphericity_bounds(self):
        """Sphericity should be between 0 and 1."""
        for name, polytope in TEST_POLYTOPES.items():
            h = compute_all_heuristics(polytope, polytope_id=1)
            assert 0 <= h.sphericity <= 1, f"{name}: sphericity={h.sphericity}"

    def test_symmetry_bounds(self):
        """Symmetry scores should be between 0 and 1."""
        for name, polytope in TEST_POLYTOPES.items():
            h = compute_all_heuristics(polytope, polytope_id=1)
            for axis in ['x', 'y', 'z', 'w']:
                val = getattr(h, f'symmetry_{axis}')
                assert 0 <= val <= 1, f"{name}: symmetry_{axis}={val}"

    def test_cross_polytope_is_symmetric(self):
        """Cross-polytope should have high symmetry."""
        polytope = TEST_POLYTOPES["cross_polytope_4d"]
        h = compute_all_heuristics(polytope, polytope_id=1)

        # Perfect symmetry on all axes
        assert h.symmetry_x == 1.0
        assert h.symmetry_y == 1.0
        assert h.symmetry_z == 1.0
        assert h.symmetry_w == 1.0

        # Perfect sphericity
        assert h.sphericity == 1.0

    def test_vertex_count_matches(self):
        """Computed vertex count should match input."""
        for name, polytope in TEST_POLYTOPES.items():
            h = compute_all_heuristics(polytope, polytope_id=1)
            assert h.vertex_count == polytope["vertex_count"]

    def test_chirality_values_differ_for_asymmetric(self):
        """Chirality values should differ per axis for asymmetric polytopes."""
        polytope = TEST_POLYTOPES["asymmetric_simplex"]
        h = compute_all_heuristics(polytope, polytope_id=1)

        chiralities = [h.chirality_x, h.chirality_y, h.chirality_z, h.chirality_w]
        unique_values = set(chiralities)

        # Asymmetric polytope should have different chirality on different axes
        assert len(unique_values) > 1, (
            f"Chirality values should differ for asymmetric polytope, "
            f"but got identical values: {chiralities}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
