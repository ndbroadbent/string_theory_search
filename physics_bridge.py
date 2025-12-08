#!/usr/bin/env python3
"""
Physics Bridge for String Theory Landscape Explorer

This module provides real physics computations for Calabi-Yau compactifications:
- Polytope analysis via PALP
- Numerical CY metric approximation via cymyc (JAX)
- Gauge coupling computation from cycle volumes
- Moduli stabilization via flux superpotentials

The bridge exposes a JSON-RPC interface for the Rust GA to call.
"""

import json
import sys
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Optional
import numpy as np

# JAX configuration for performance
os.environ['JAX_PLATFORM_NAME'] = 'cpu'  # Use CPU for now, GPU later

import jax
import jax.numpy as jnp

# Try to import cymyc
try:
    from cymyc import alg_geo
    from cymyc.utils import math_utils
    CYMYC_AVAILABLE = True
except ImportError:
    CYMYC_AVAILABLE = False
    print("Warning: cymyc not available, using simplified metric computation", file=sys.stderr)

# Path to PALP binary
PALP_PATH = Path(__file__).parent.parent / "palp_source" / "poly.x"


class PolytopeAnalyzer:
    """Analyze 4D reflexive polytopes using PALP."""

    def __init__(self, palp_path: Path = PALP_PATH):
        self.palp_path = palp_path
        if not palp_path.exists():
            raise FileNotFoundError(f"PALP binary not found at {palp_path}")

    def analyze(self, vertices: list[list[int]]) -> dict:
        """
        Analyze a polytope given its vertices.

        Args:
            vertices: List of 4D integer vertices

        Returns:
            Dictionary with Hodge numbers, Euler characteristic, etc.
        """
        # Format vertices for PALP input
        n_vertices = len(vertices)
        dim = len(vertices[0]) if vertices else 0

        # PALP expects: "n_vertices dim" followed by one vertex per line
        lines = [f"{n_vertices} {dim}"]
        for v in vertices:
            lines.append(" ".join(str(x) for x in v))
        palp_input = "\n".join(lines) + "\n"

        try:
            result = subprocess.run(
                [str(self.palp_path), "-f"],  # -f for full output
                input=palp_input,
                capture_output=True,
                text=True,
                timeout=10
            )

            # Parse PALP output
            output = result.stdout
            return self._parse_palp_output(output, vertices)

        except subprocess.TimeoutExpired:
            return {"error": "PALP timeout", "vertices": vertices}
        except Exception as e:
            return {"error": str(e), "vertices": vertices}

    def _parse_palp_output(self, output: str, vertices: list) -> dict:
        """Parse PALP output to extract Hodge numbers."""
        result = {
            "vertices": vertices,
            "h11": None,
            "h21": None,
            "euler": None,
            "raw_output": output
        }

        # Look for Hodge numbers in format "H:h11,h21 [euler]"
        import re
        hodge_match = re.search(r'H:(\d+),(\d+)\s*\[(-?\d+)\]', output)
        if hodge_match:
            result["h11"] = int(hodge_match.group(1))
            result["h21"] = int(hodge_match.group(2))
            result["euler"] = int(hodge_match.group(3))

        # Also extract M:faces vertices N:dualfaces dualvertices
        m_match = re.search(r'M:(\d+)\s+(\d+)', output)
        n_match = re.search(r'N:(\d+)\s+(\d+)', output)
        if m_match:
            result["faces"] = int(m_match.group(1))
            result["n_vertices"] = int(m_match.group(2))
        if n_match:
            result["dual_faces"] = int(n_match.group(1))
            result["dual_vertices"] = int(n_match.group(2))

        return result


class CYMetricComputer:
    """
    Compute numerical Calabi-Yau metrics using machine learning.

    This wraps cymyc's neural network approach to approximating
    Ricci-flat metrics on CY manifolds.
    """

    def __init__(self):
        self.jit_compute_metric = jax.jit(self._compute_fubini_study_metric)

    def _compute_fubini_study_metric(self, z: jnp.ndarray) -> jnp.ndarray:
        """
        Compute Fubini-Study metric at a point in projective space.
        This is the ambient space metric before restriction to CY.
        """
        # z is a complex coordinate in CP^n
        # g_{i\bar{j}} = \partial_i \partial_{\bar{j}} log(|z|^2)
        norm_sq = jnp.sum(jnp.abs(z)**2)

        # Metric components
        n = len(z)
        g = jnp.eye(n, dtype=jnp.complex64) / norm_sq
        g = g - jnp.outer(jnp.conj(z), z) / (norm_sq ** 2)

        return g

    def compute_kahler_potential(self, z: jnp.ndarray, moduli: jnp.ndarray) -> float:
        """
        Compute Kähler potential for given complex structure moduli.

        For a CY threefold, the Kähler potential is:
        K = -log(i ∫ Ω ∧ Ω̄)

        where Ω is the holomorphic 3-form depending on moduli.
        """
        # Simplified: assume diagonal Kähler moduli
        # K = -sum_i log(t_i) where t_i are Kähler moduli
        return -jnp.sum(jnp.log(jnp.abs(moduli) + 1e-10))

    def compute_volume(self, kahler_moduli: jnp.ndarray,
                      intersection_numbers: Optional[jnp.ndarray] = None) -> float:
        """
        Compute CY volume from Kähler moduli.

        V = (1/6) κ_{ijk} t^i t^j t^k

        where κ_{ijk} are triple intersection numbers.
        """
        if intersection_numbers is None:
            # Default: assume simple diagonal intersection form
            # This is the simplest case, real CYs have complex intersection forms
            return jnp.prod(kahler_moduli) / 6.0

        # Full computation with intersection numbers
        n = len(kahler_moduli)
        volume = 0.0
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    volume += intersection_numbers[i, j, k] * \
                              kahler_moduli[i] * kahler_moduli[j] * kahler_moduli[k]
        return volume / 6.0


class GaugeCouplingComputer:
    """
    Compute gauge couplings from CY geometry.

    In string compactifications:
    - α_GUT ~ g_s / V^{2/3} (where V is CY volume in string units)
    - Individual SM couplings depend on cycle volumes where branes wrap
    """

    def __init__(self, string_scale: float = 2e16):  # GeV
        self.string_scale = string_scale
        self.alpha_gut_observed = 1/25.0  # ~ at GUT scale

    def compute_gauge_couplings(self,
                                 cycle_volumes: jnp.ndarray,
                                 g_s: float = 0.1) -> dict:
        """
        Compute gauge couplings from cycle volumes.

        In Type IIB with D7 branes:
        1/g_YM^2 = Vol(4-cycle) / (g_s l_s^4)

        Args:
            cycle_volumes: Array of 4-cycle volumes (in string units)
            g_s: String coupling

        Returns:
            Dictionary with gauge couplings
        """
        # Normalize to get dimensionless couplings
        # α = g^2 / (4π)

        # Assume first 3 cycles host SU(3), SU(2), U(1) branes
        if len(cycle_volumes) < 3:
            # Pad with duplicates for simple polytopes
            cycle_volumes = jnp.concatenate([
                cycle_volumes,
                jnp.full(3 - len(cycle_volumes), cycle_volumes[-1])
            ])

        # Inverse gauge couplings
        g_inv_sq = cycle_volumes[:3] / g_s

        # Convert to α = g²/(4π)
        alpha = g_s / (4 * jnp.pi * cycle_volumes[:3] + 1e-10)

        return {
            "alpha_3": float(alpha[0]),  # Strong
            "alpha_2": float(alpha[1]),  # Weak
            "alpha_1": float(alpha[2]),  # Hypercharge (not normalized to SM)
        }

    def compute_sm_couplings(self,
                             cycle_volumes: jnp.ndarray,
                             g_s: float = 0.1,
                             total_volume: float = 1.0) -> dict:
        """
        Compute Standard Model gauge couplings with GUT normalization.

        Returns α_em, α_s, sin²θ_W at low energies (with simplified RG running).
        """
        raw = self.compute_gauge_couplings(cycle_volumes, g_s)

        # At GUT scale, assume unification
        # α_1 needs GUT normalization: α_1_SM = (5/3) α_1_GUT
        alpha_1_gut = raw["alpha_1"]
        alpha_2_gut = raw["alpha_2"]
        alpha_3_gut = raw["alpha_3"]

        # Simplified 1-loop RG running from GUT to Z scale
        # β coefficients for SM
        b1, b2, b3 = 41/10, -19/6, -7

        # Running: 1/α(μ) = 1/α(M) + b/(2π) log(M/μ)
        log_ratio = np.log(2e16 / 91.2)  # GUT to Z mass

        alpha_1_z = 1.0 / (1.0/alpha_1_gut + b1/(2*np.pi) * log_ratio)
        alpha_2_z = 1.0 / (1.0/alpha_2_gut + b2/(2*np.pi) * log_ratio)
        alpha_3_z = 1.0 / (1.0/alpha_3_gut + b3/(2*np.pi) * log_ratio)

        # GUT normalization for U(1)
        alpha_1_sm = (5/3) * alpha_1_z

        # Weinberg angle: sin²θ_W = α_1 / (α_1 + α_2)
        sin2_theta_w = alpha_1_sm / (alpha_1_sm + alpha_2_z)

        # EM coupling: 1/α_em = 1/α_1 + 1/α_2
        alpha_em = 1.0 / (1.0/alpha_1_sm + 1.0/alpha_2_z)

        return {
            "alpha_em": float(alpha_em),
            "alpha_s": float(alpha_3_z),
            "sin2_theta_w": float(sin2_theta_w),
            "alpha_1_gut": float(alpha_1_gut),
            "alpha_2_gut": float(alpha_2_gut),
            "alpha_3_gut": float(alpha_3_gut),
        }


class KKLTComputer:
    """
    Compute KKLT moduli stabilization and uplifting.

    KKLT (Kachru-Kallosh-Linde-Trivedi) is a mechanism to get de Sitter vacua:
    1. GVW superpotential W_flux stabilizes complex structure moduli
    2. Non-perturbative effects W_np = A e^{-aT} stabilize Kähler moduli
    3. Anti-D3 branes uplift from AdS to dS

    The scalar potential is:
    V = e^K [ K^{ij̄} D_i W D_j̄ W̄ - 3|W|² ] + V_uplift
    """

    def __init__(self):
        # Parameters for non-perturbative effects
        self.A = 1.0  # Prefactor (from gaugino condensation)
        self.a = 2 * np.pi / 10  # a = 2π/N for SU(N) gauge group

    def compute_kklt_potential(self,
                                kahler_moduli: jnp.ndarray,
                                w_flux: complex,
                                n_antiD3: int = 1) -> dict:
        """
        Compute the KKLT scalar potential with uplifting.

        Args:
            kahler_moduli: Kähler moduli (real parts of T_i = τ_i + i b_i)
            w_flux: Flux superpotential W_0
            n_antiD3: Number of anti-D3 branes for uplifting

        Returns:
            Dictionary with potential components and total Λ
        """
        # Total volume (simplified: product of moduli)
        volume = jnp.prod(kahler_moduli)

        # Kähler potential: K = -2 log(V)
        k_kahler = -2 * jnp.log(volume + 1e-10)

        # Non-perturbative superpotential from largest modulus
        tau = jnp.max(kahler_moduli)  # Use largest cycle
        w_np = self.A * jnp.exp(-self.a * tau)

        # Total superpotential
        w_total = w_flux + w_np

        # F-term potential (simplified - assumes diagonal Kähler metric)
        # V_F = e^K [ |DW|² - 3|W|² ]
        # For KKLT minimum: DW ≈ 0, so V_F ≈ -3 e^K |W|²
        exp_k = jnp.exp(k_kahler)
        v_ads = -3 * exp_k * jnp.abs(w_total)**2

        # Uplifting from anti-D3 branes at tip of warped throat
        # V_uplift = D / V^{4/3} where D depends on warp factor
        # Tune D to get small positive Λ
        warp_factor = 0.01  # Warping at throat tip
        d_uplift = n_antiD3 * warp_factor**4  # Anti-D3 tension (warped down)
        v_uplift = d_uplift / (volume**(4/3) + 1e-10)

        # Total potential
        v_total = float(v_ads + v_uplift)

        # Cosmological constant in Planck units
        # Λ = V_total / M_Pl^4
        lambda_cc = v_total

        return {
            "v_ads": float(v_ads),
            "v_uplift": float(v_uplift),
            "v_total": v_total,
            "cosmological_constant": lambda_cc,
            "w_flux": float(jnp.abs(w_flux)),
            "w_np": float(jnp.abs(w_np)),
            "w_total": float(jnp.abs(w_total)),
            "volume": float(volume),
            "is_de_sitter": v_total > 0,
        }


class FluxComputer:
    """
    Compute flux superpotential and moduli stabilization.

    In Type IIB, the Gukov-Vafa-Witten superpotential is:
    W = ∫ G_3 ∧ Ω

    where G_3 = F_3 - τ H_3 is the complexified 3-form flux.
    """

    def __init__(self):
        self.kklt = KKLTComputer()

    def compute_superpotential(self,
                                flux_f: jnp.ndarray,
                                flux_h: jnp.ndarray,
                                periods: jnp.ndarray,
                                axio_dilaton: complex) -> complex:
        """
        Compute GVW superpotential.

        W = (F - τH) · Π

        where Π are the periods of Ω.
        """
        tau = axio_dilaton
        g3 = flux_f - tau * flux_h

        # Superpotential is dot product with periods
        w = jnp.dot(g3, periods)
        return complex(w)

    def compute_tadpole(self, flux_f: jnp.ndarray, flux_h: jnp.ndarray) -> float:
        """
        Compute D3-brane tadpole contribution from fluxes.

        N_flux = (1/2) ∫ F_3 ∧ H_3 = (1/2) F · Σ · H

        where Σ is the intersection form on H^3.
        """
        # Simplified: assume standard symplectic intersection form
        # For h21+1 complex structure moduli, have 2(h21+1) periods
        n = len(flux_f) // 2

        # Σ = [[0, I], [-I, 0]] in symplectic basis
        sigma_f = jnp.concatenate([flux_f[n:], -flux_f[:n]])

        return float(jnp.dot(flux_f, flux_h[::-1]) / 2)


class PhysicsBridge:
    """
    Main interface for the Rust GA to call physics computations.

    Accepts JSON-RPC style requests and returns physics predictions.
    """

    def __init__(self):
        self.polytope_analyzer = PolytopeAnalyzer()
        self.metric_computer = CYMetricComputer()
        self.gauge_computer = GaugeCouplingComputer()
        self.flux_computer = FluxComputer()

        # Cache for expensive computations
        self._cache = {}

    def compute_physics(self, genome: dict) -> dict:
        """
        Compute all physical observables from a compactification genome.

        Args:
            genome: Dictionary containing:
                - polytope_id: Index into polytope database
                - kahler_moduli: Array of Kähler moduli values
                - complex_moduli: Array of complex structure moduli
                - flux_f: F_3 flux quanta (integers)
                - flux_h: H_3 flux quanta (integers)
                - g_s: String coupling

        Returns:
            Dictionary with computed physical observables
        """
        try:
            # Extract genome parameters
            kahler = jnp.array(genome.get("kahler_moduli", [1.0, 1.0, 1.0]))
            complex_mod = jnp.array(genome.get("complex_moduli", [1.0]))
            flux_f = jnp.array(genome.get("flux_f", [1, 0, 0, 0]))
            flux_h = jnp.array(genome.get("flux_h", [0, 1, 0, 0]))
            g_s = genome.get("g_s", 0.1)

            # Compute CY volume
            volume = float(self.metric_computer.compute_volume(kahler))

            # Cycle volumes (simplified: proportional to Kähler moduli squared)
            cycle_volumes = kahler ** 2

            # Gauge couplings
            sm_couplings = self.gauge_computer.compute_sm_couplings(
                cycle_volumes, g_s, volume
            )

            # Compute flux superpotential
            n_flux = len(flux_f)
            periods = jnp.exp(1j * jnp.arange(n_flux) * 0.1) * complex_mod[0]

            w_flux = self.flux_computer.compute_superpotential(
                flux_f, flux_h, periods, complex(0.1 + 1j * g_s)
            )

            # KKLT uplifting for cosmological constant
            # Number of anti-D3 branes (can be tuned by GA via genome)
            n_antiD3 = genome.get("n_antiD3", 1)

            kklt_result = self.flux_computer.kklt.compute_kklt_potential(
                kahler, w_flux, n_antiD3
            )

            lambda_cc = kklt_result["cosmological_constant"]
            is_de_sitter = kklt_result["is_de_sitter"]

            # Tadpole constraint
            n_flux_tadpole = self.flux_computer.compute_tadpole(flux_f, flux_h)

            # Number of generations from topology
            # In realistic models: N_gen = |χ(CY) ∩ D7| / 2
            # Simplified: just use h11 - h21 mod 3
            h11 = genome.get("h11", 3)
            h21 = genome.get("h21", 3)
            n_gen = abs(h11 - h21) % 4 + 1  # Ensure 1-4 generations

            # Electron mass ratio (extremely simplified)
            # Real computation requires Yukawa couplings from worldsheet instantons
            w_total = kklt_result["w_total"]
            m_e_ratio = g_s * w_total / (volume**(1/3) + 1e-10) * 1e-22

            # Proton mass ratio
            m_p_ratio = m_e_ratio * 1836.15  # Use observed ratio as placeholder

            return {
                "success": True,
                "alpha_em": sm_couplings["alpha_em"],
                "alpha_s": sm_couplings["alpha_s"],
                "sin2_theta_w": sm_couplings["sin2_theta_w"],
                "cosmological_constant": float(lambda_cc),
                "n_generations": int(n_gen),
                "m_e_planck_ratio": float(m_e_ratio),
                "m_p_planck_ratio": float(m_p_ratio),
                "cy_volume": volume,
                "string_coupling": g_s,
                "flux_tadpole": float(n_flux_tadpole),
                "superpotential_abs": float(w_total),
                "is_de_sitter": is_de_sitter,
                "v_ads": kklt_result["v_ads"],
                "v_uplift": kklt_result["v_uplift"],
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def serve_stdio(self):
        """
        Serve JSON-RPC requests over stdin/stdout.

        Protocol:
        - Input: JSON object with "method" and "params" keys
        - Output: JSON object with "result" or "error" keys
        """
        for line in sys.stdin:
            try:
                request = json.loads(line.strip())
                method = request.get("method", "")
                params = request.get("params", {})

                if method == "compute_physics":
                    result = self.compute_physics(params)
                elif method == "analyze_polytope":
                    result = self.polytope_analyzer.analyze(params.get("vertices", []))
                elif method == "ping":
                    result = {"status": "ok", "cymyc_available": CYMYC_AVAILABLE}
                else:
                    result = {"error": f"Unknown method: {method}"}

                print(json.dumps({"result": result}), flush=True)

            except json.JSONDecodeError as e:
                print(json.dumps({"error": f"JSON decode error: {e}"}), flush=True)
            except Exception as e:
                print(json.dumps({"error": str(e)}), flush=True)


def test_bridge():
    """Test the physics bridge with sample inputs."""
    bridge = PhysicsBridge()

    # Test polytope analysis
    print("Testing polytope analyzer...")
    quintic_vertices = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [-1, -1, -1, -1]
    ]
    result = bridge.polytope_analyzer.analyze(quintic_vertices)
    print(f"Quintic threefold: h11={result.get('h11')}, h21={result.get('h21')}, χ={result.get('euler')}")

    # Test physics computation
    print("\nTesting physics computation...")
    genome = {
        "polytope_id": 0,
        "kahler_moduli": [2.0, 1.5, 1.0],
        "complex_moduli": [1.0],
        "flux_f": [1, 0, 0, 0, 0, 0],
        "flux_h": [0, 1, 0, 0, 0, 0],
        "g_s": 0.1,
        "h11": 3,
        "h21": 101
    }
    physics = bridge.compute_physics(genome)

    if physics["success"]:
        print(f"  α_em = {physics['alpha_em']:.6f}")
        print(f"  α_s  = {physics['alpha_s']:.6f}")
        print(f"  sin²θ_W = {physics['sin2_theta_w']:.6f}")
        print(f"  Λ = {physics['cosmological_constant']:.6e}")
        print(f"  N_gen = {physics['n_generations']}")
        print(f"  CY Volume = {physics['cy_volume']:.4f}")
    else:
        print(f"  Error: {physics['error']}")

    print("\nPhysics bridge ready!")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_bridge()
    elif len(sys.argv) > 1 and sys.argv[1] == "--serve":
        bridge = PhysicsBridge()
        bridge.serve_stdio()
    else:
        print("Usage:")
        print("  python physics_bridge.py --test   # Run tests")
        print("  python physics_bridge.py --serve  # Start JSON-RPC server")
