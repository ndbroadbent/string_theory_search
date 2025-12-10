"""
Test physics_bridge.py against gold standard McAllister data.

These tests validate that our physics_bridge produces results consistent
with published results from arXiv:2107.09064 (McAllister et al.).

If these tests fail, our GA results are suspect and should not be trusted.
"""

import json
import pytest
from pathlib import Path


def einstein_to_string_frame(v_einstein: float, g_s: float) -> float:
    """Convert CY volume from Einstein frame to string frame.

    V_string = V_einstein * g_s^(3/2)
    """
    return v_einstein * (g_s ** 1.5)

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
        assert len(data["kahler_moduli"]) == 4
        assert len(data["flux_k"]) == 4
        assert len(data["flux_m"]) == 4

    @pytest.mark.slow
    def test_full_physics_computation(self, mcallister_4_214_647):
        """Test full physics computation against McAllister published results.

        Uses kahler_mode="fixed" to use exact Kähler values from the fixture.

        McAllister paper (arXiv:2107.09064) reports:
        - W₀ = 2.30012e-90
        - CY Volume = 4711.83 (Einstein frame)
        - Cosmological constant ~ 10^-122

        If these tests fail, our physics_bridge implementation is broken.
        """
        data = mcallister_4_214_647

        bridge = physics_bridge.PhysicsBridge()

        # Build genome with FIXED mode - use exact Kähler values
        genome = {
            "vertices": data["vertices"],
            "kahler_moduli": data["kahler_moduli"],
            "complex_moduli": [1.0] * data["h21"],
            "flux_f": data["flux_k"],
            "flux_h": data["flux_m"],
            "g_s": data["g_s"],
            "kahler_mode": "fixed",  # Use exact values, don't raytrace
        }

        result = bridge.compute_physics(genome)

        # Must succeed
        assert result["success"] is True

        # McAllister expected values
        # W₀ = 2.30012e-90 (superpotential magnitude)
        # CY Volume = 4711.83 (Einstein frame)
        # CC ≈ 10^-122 (derived from W₀ via KKLT potential)
        expected_cy_volume_string = einstein_to_string_frame(data["expected"]["cy_volume"], data["g_s"])
        assert result["cy_volume"] == expected_cy_volume_string
        assert result["superpotential_abs"] == data["expected"]["w0_magnitude"]
        assert result["cosmological_constant"] == 1e-122

        # Hodge numbers
        assert result["h11"] == 4
        assert result["h21"] == 214


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
            "kahler_mode": "fixed",
        }

        result = bridge.compute_physics(genome)

        # Should complete successfully
        assert result["success"] is True

        # Quintic has h11=1, h21=101 (from CYTools)
        # Note: actual values depend on CYTools triangulation
        assert result["h11"] == 1
        assert result["h21"] == 101


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
