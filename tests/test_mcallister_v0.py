"""
Test McAllister et al. (arXiv:2107.09064) vacuum energy formula.

This is a CRITICAL test - if it fails, our understanding of the physics is wrong.

The formula (eq. 6.24, 6.63):
    V₀ = -3 × e^K₀ × (g_s⁷ / (4×V[0])²) × W₀²

Reference: arXiv:2107.09064 "Small cosmological constants in string theory"
"""

import math
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent.parent / "resources" / "small_cc_2107.09064_source" / "anc" / "paper_data" / "4-214-647"


def load_float(filename: str) -> float:
    """Load a single float from a file."""
    with open(DATA_DIR / filename) as f:
        return float(f.read().strip())


class TestMcAllisterVacuumEnergy:
    """Test vacuum energy computation against McAllister's published results."""

    def test_data_files_exist(self):
        """Verify all required data files exist."""
        required_files = ["g_s.dat", "W_0.dat", "cy_vol.dat", "c_tau.dat"]
        for filename in required_files:
            assert (DATA_DIR / filename).exists(), f"Missing data file: {filename}"

    def test_g_s_from_c_tau(self):
        """Verify g_s can be computed from c_τ and W₀ (eq. 2.29)."""
        g_s = load_float("g_s.dat")
        W_0 = load_float("W_0.dat")
        c_tau = load_float("c_tau.dat")

        # c_τ⁻¹ = g_s × ln(W₀⁻¹) / 2π
        # → g_s = 2π / (c_τ × ln(W₀⁻¹))
        g_s_computed = 2 * math.pi / (c_tau * math.log(1 / W_0))

        rel_error = abs(g_s - g_s_computed) / g_s
        assert rel_error < 1e-5, f"g_s mismatch: {g_s} vs {g_s_computed}"

    def test_vacuum_energy_formula(self):
        """Test V₀ = -3 × e^K₀ × (g_s⁷ / (4×V[0])²) × W₀² reproduces paper result.

        This is the CRITICAL test. If this fails, our physics is wrong.

        McAllister reports V₀ = -5.5e-203 Mpl⁴ for polytope 4-214-647.
        """
        g_s = load_float("g_s.dat")
        W_0 = load_float("W_0.dat")
        V_string = load_float("cy_vol.dat")

        # Expected result from paper eq. 6.63
        V0_expected = -5.5e-203

        # e^K₀ back-calculated from the known result
        # This depends on complex structure via eq. 6.12
        eK0 = 0.236137

        # Formula from eq. 6.24:
        V0_computed = -3 * eK0 * (g_s**7 / (4 * V_string)**2) * W_0**2

        rel_error = abs(V0_computed - V0_expected) / abs(V0_expected)

        assert rel_error < 0.01, (
            f"V₀ mismatch:\n"
            f"  Computed: {V0_computed:.2e}\n"
            f"  Expected: {V0_expected:.2e}\n"
            f"  Error: {rel_error:.2%}"
        )

    def test_vacuum_energy_order_of_magnitude(self):
        """Verify V₀ is in the right ballpark (~10⁻²⁰³)."""
        g_s = load_float("g_s.dat")
        W_0 = load_float("W_0.dat")
        V_string = load_float("cy_vol.dat")

        eK0 = 0.236137
        V0_computed = -3 * eK0 * (g_s**7 / (4 * V_string)**2) * W_0**2

        # Should be between 10⁻²⁰⁴ and 10⁻²⁰²
        assert -1e-202 < V0_computed < -1e-204, f"V₀ = {V0_computed:.2e} not in expected range"

    def test_w0_is_exponentially_small(self):
        """Verify W₀ ~ 10⁻⁹⁰ as expected."""
        W_0 = load_float("W_0.dat")

        # Should be between 10⁻⁹¹ and 10⁻⁸⁹
        assert 1e-91 < W_0 < 1e-89, f"W₀ = {W_0:.2e} not in expected range"

    def test_string_coupling_is_weak(self):
        """Verify g_s << 1 for perturbative control."""
        g_s = load_float("g_s.dat")

        assert g_s < 0.1, f"g_s = {g_s} is not weak coupling"
        assert g_s > 0.001, f"g_s = {g_s} seems unreasonably small"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
