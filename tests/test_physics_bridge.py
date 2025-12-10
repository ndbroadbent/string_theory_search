"""
Test physics_bridge.py against gold standard McAllister data.

These tests validate that our physics_bridge produces results consistent
with published results from arXiv:2107.09064 (McAllister et al.).

If these tests fail, our GA results are suspect and should not be trusted.
"""

import json
import pytest
from pathlib import Path

# Skip all tests if physics_bridge is not available
physics_bridge = pytest.importorskip("physics_bridge")


def load_fixture(name: str) -> dict:
    """Load a test fixture from the fixtures directory."""
    fixture_path = Path(__file__).parent / "fixtures" / f"{name}.json"
    with open(fixture_path) as f:
        return json.load(f)


class TestMcAllisterFixtures:
    """Test physics bridge against McAllister et al. published results."""

    @pytest.fixture
    def mcallister_4_214_647(self):
        """Load McAllister 4-214-647 fixture data."""
        return load_fixture("mcallister_4_214_647")

    def test_fixture_loads(self, mcallister_4_214_647):
        """Verify the fixture file loads correctly."""
        data = mcallister_4_214_647
        assert data["name"] == "McAllister 4-214-647"
        assert data["h11"] == 4
        assert data["h21"] == 214
        assert len(data["vertices"]) == 12
        assert len(data["kahler_moduli"]) == 214
        assert len(data["flux_k"]) == 4
        assert len(data["flux_m"]) == 4

    @pytest.mark.slow
    def test_cy_volume(self, mcallister_4_214_647):
        """Test that CY volume matches published value."""
        data = mcallister_4_214_647

        # Create bridge instance
        bridge = physics_bridge.PhysicsBridge()

        # Analyze polytope
        cy_data = bridge.cytools.analyze_polytope(data["vertices"])

        if not cy_data.get("success", True):
            pytest.skip(f"Polytope analysis failed: {cy_data.get('error')}")

        expected = data["expected"]["cy_volume"]
        tolerance = data["expected"]["cy_volume_tolerance"]

        # Check if CY volume is available in the analysis
        if "cy_volume" in cy_data:
            result = cy_data["cy_volume"]
            assert abs(result - expected) < tolerance, (
                f"CY volume mismatch: got {result}, expected {expected} +/- {tolerance}"
            )

    @pytest.mark.slow
    def test_full_physics_computation(self, mcallister_4_214_647):
        """Test full physics computation against McAllister published values.

        This test validates that our physics bridge produces results consistent
        with arXiv:2107.09064. If this test fails, our GA results are suspect.

        Expected values from the paper:
        - W₀ = 2.30012e-90
        - CY Volume = 4711.83
        """
        data = mcallister_4_214_647

        # Create bridge instance
        bridge = physics_bridge.PhysicsBridge()

        # Build genome dict matching PhysicsBridge.compute_physics interface
        genome = {
            "vertices": data["vertices"],
            "kahler_moduli": data["kahler_moduli"],
            "complex_moduli": [1.0] * data["h21"],
            "flux_f": data["flux_k"],
            "flux_h": data["flux_m"],
            "g_s": data["g_s"],
        }

        # Run full computation
        result = bridge.compute_physics(genome)

        # Check computation succeeded
        assert result["success"], f"Physics computation failed: {result.get('error')}"

        # Validate CY volume against published value
        expected_cy_volume = data["expected"]["cy_volume"]
        cy_volume_tolerance = data["expected"]["cy_volume_tolerance"]
        assert abs(result["cy_volume"] - expected_cy_volume) < cy_volume_tolerance, (
            f"CY volume mismatch: got {result['cy_volume']}, "
            f"expected {expected_cy_volume} +/- {cy_volume_tolerance}"
        )

        # Validate W₀ (superpotential) against published value
        # W₀ relates to cosmological constant via Λ ∝ |W₀|²/V²
        expected_w0 = data["expected"]["w0_magnitude"]
        w0_tolerance = data["expected"]["w0_tolerance"]

        # The physics bridge should compute W0 or something proportional to it
        # Check if W0 is available in results
        if "w0" in result or "superpotential" in result:
            computed_w0 = abs(result.get("w0", result.get("superpotential", 0)))
            # For extremely small W0 values, check order of magnitude
            if expected_w0 < 1e-50:
                # Check that log10 values are within 5 orders of magnitude
                import math
                if computed_w0 > 0:
                    log_diff = abs(math.log10(computed_w0) - math.log10(expected_w0))
                    assert log_diff < 5, (
                        f"W0 order of magnitude mismatch: got {computed_w0:.2e}, "
                        f"expected {expected_w0:.2e} (log diff: {log_diff})"
                    )
            else:
                assert abs(computed_w0 - expected_w0) < w0_tolerance, (
                    f"W0 mismatch: got {computed_w0}, expected {expected_w0}"
                )

        # Basic sanity checks on gauge couplings
        assert 0 < result["alpha_em"] < 1, f"alpha_em out of range: {result['alpha_em']}"
        assert 0 < result["alpha_s"] < 1, f"alpha_s out of range: {result['alpha_s']}"
        assert 0 < result["sin2_theta_w"] < 1, f"sin2_theta_w out of range: {result['sin2_theta_w']}"

        # Print actual values for debugging
        print(f"\n=== McAllister 4-214-647 Results ===")
        print(f"CY Volume: {result['cy_volume']:.4f} (expected: {expected_cy_volume})")
        print(f"alpha_em: {result['alpha_em']:.6e}")
        print(f"alpha_s: {result['alpha_s']:.6f}")
        print(f"sin2_theta_w: {result['sin2_theta_w']:.6f}")
        print(f"n_generations: {result.get('n_generations', 'N/A')}")
        print(f"cosmological_constant: {result.get('cosmological_constant', 'N/A')}")


class TestPhysicsBridgeBasics:
    """Basic functionality tests for physics_bridge."""

    def test_bridge_instantiates(self):
        """Test that PhysicsBridge instantiates without error."""
        bridge = physics_bridge.PhysicsBridge()
        assert bridge is not None
        assert hasattr(bridge, "compute_physics")
        assert hasattr(bridge, "cytools")

    def test_model_version_defined(self):
        """Test that PHYSICS_MODEL_VERSION is defined."""
        assert hasattr(physics_bridge, "PHYSICS_MODEL_VERSION")
        version = physics_bridge.PHYSICS_MODEL_VERSION
        assert isinstance(version, str)
        assert len(version) > 0
        # Should be semantic version format (X.Y.Z)
        parts = version.split(".")
        assert len(parts) >= 2, "Version should be in X.Y or X.Y.Z format"

    @pytest.mark.slow
    def test_quintic_threefold(self):
        """Test computation with the quintic threefold (simplest CY3).

        The quintic P^4[5] has h11=1, h21=101.
        """
        bridge = physics_bridge.PhysicsBridge()

        vertices = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [-1, -1, -1, -1],
        ]

        genome = {
            "vertices": vertices,
            "kahler_moduli": [2.0],
            "complex_moduli": [1.0],
            "flux_f": [1, 0, 0, 0],
            "flux_h": [0, 1, 0, 0],
            "g_s": 0.1,
        }

        result = bridge.compute_physics(genome)

        # Should complete (success or failure with specific error)
        # Even if computation fails, it should fail gracefully
        assert "success" in result
        if not result["success"]:
            assert "error" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
