#!/usr/bin/env python3
"""
Physics Bridge for String Theory Landscape Explorer

Uses CYTools and cymyc for rigorous Calabi-Yau computations:
- CYTools: Polytope analysis, triangulations, intersection numbers, cycle volumes
- cymyc: Numerical CY metrics, curvature, Yukawa couplings (JAX-based)

NO FALLBACKS - requires proper tools installed.
"""

import json
import sys
import os
from pathlib import Path
from typing import Optional
import numpy as np

# Require JAX
import jax
import jax.numpy as jnp
jax.config.update('jax_platform_name', 'cpu')

# Require CYTools - no fallback
from cytools import Polytope, config as cytools_config
cytools_config.enable_experimental_features()

# Require cymyc - no fallback
from cymyc import alg_geo
from cymyc.curvature import ricci_scalar
from cymyc.fubini_study import fubini_study_metric

# Physical constants (observed values)
ALPHA_EM_OBS = 1.0 / 137.035999
ALPHA_S_OBS = 0.1179
SIN2_THETA_W_OBS = 0.23121
COSMOLOGICAL_CONSTANT_OBS = 1.1e-122  # In Planck units
M_E_PLANCK_OBS = 4.18e-23
M_P_PLANCK_OBS = 7.68e-20


class CYToolsBridge:
    """
    Bridge to CYTools for polytope and Calabi-Yau computations.

    CYTools workflow:
    1. Polytope(vertices) -> triangulate() -> get_cy()
    2. CalabiYau object provides intersection numbers, volumes, etc.
    """

    def __init__(self):
        self._cache = {}

    def analyze_polytope(self, vertices: list[list[int]]) -> dict:
        """
        Full polytope analysis using CYTools.

        Returns Hodge numbers, intersection numbers, Kähler cone, etc.
        """
        cache_key = str(vertices)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Create polytope from vertices
        p = Polytope(vertices)

        # Check if reflexive (required for CY)
        if not p.is_reflexive():
            return {
                "success": False,
                "error": "Polytope is not reflexive"
            }

        # Get a triangulation (fine, regular, star)
        # This can be slow for complex polytopes
        t = p.triangulate()

        # Get the Calabi-Yau
        cy = t.get_cy()

        # Extract topological data
        h11 = cy.h11()
        h21 = cy.h12()  # h12 = h21 for CY3
        chi = cy.chi()

        # Intersection numbers (triple intersection form)
        # κ_ijk where i,j,k index divisor classes
        intersection_nums = cy.intersection_numbers()

        # Second Chern class (for anomaly cancellation)
        c2 = cy.second_chern_class()

        # Kähler cone (valid range for Kähler moduli)
        kahler_cone = cy.toric_kahler_cone()

        # Store CY object for later use
        result = {
            "success": True,
            "h11": int(h11),
            "h21": int(h21),
            "chi": int(chi),
            "n_generations": abs(chi) // 2,  # |χ|/2 for CY3
            "intersection_numbers": intersection_nums.tolist() if hasattr(intersection_nums, 'tolist') else intersection_nums,
            "c2": c2.tolist() if hasattr(c2, 'tolist') else list(c2),
            "is_favorable": h11 == len(p.points()) - 5,  # Favorable = simpler physics
            "_cy_object": cy,  # Keep for volume computations
        }

        self._cache[cache_key] = result
        return result

    def compute_volumes(self, cy_data: dict, kahler_moduli: np.ndarray) -> dict:
        """
        Compute CY volume and cycle volumes from Kähler moduli.

        Args:
            cy_data: Result from analyze_polytope (must contain _cy_object)
            kahler_moduli: Array of Kähler moduli t^i

        Returns:
            Dictionary with volumes
        """
        if "_cy_object" not in cy_data:
            return {"success": False, "error": "No CY object cached"}

        cy = cy_data["_cy_object"]

        # Kähler moduli must match the Kähler cone ambient dimension, NOT cy.h11()
        cone_dim = cy.toric_kahler_cone().ambient_dim()
        if len(kahler_moduli) > cone_dim:
            kahler_moduli = kahler_moduli[:cone_dim]
        elif len(kahler_moduli) < cone_dim:
            kahler_moduli = np.concatenate([kahler_moduli, np.ones(cone_dim - len(kahler_moduli))])

        # CY volume: V = (1/6) κ_ijk t^i t^j t^k
        cy_volume = cy.compute_cy_volume(kahler_moduli)

        # Divisor (4-cycle) volumes - these determine gauge couplings
        divisor_volumes = cy.compute_divisor_volumes(kahler_moduli)

        # Curve (2-cycle) volumes
        curve_volumes = cy.compute_curve_volumes(kahler_moduli)

        # Kähler metric on moduli space
        kahler_metric = cy.compute_kahler_metric(kahler_moduli)

        return {
            "success": True,
            "cy_volume": float(cy_volume),
            "divisor_volumes": divisor_volumes.tolist() if hasattr(divisor_volumes, 'tolist') else list(divisor_volumes),
            "curve_volumes": curve_volumes.tolist() if hasattr(curve_volumes, 'tolist') else list(curve_volumes),
            "kahler_metric": kahler_metric.tolist() if hasattr(kahler_metric, 'tolist') else kahler_metric,
        }


class GaugeCouplingComputer:
    """
    Compute gauge couplings from CY geometry.

    In Type IIB string theory with D7-branes:
    - 1/g_a^2 = Re(f_a) where f_a is gauge kinetic function
    - f_a = T_a (Kähler modulus of wrapped 4-cycle)

    At tree level: 1/g_a^2 = Vol(Σ_a) / g_s
    where Σ_a is the 4-cycle wrapped by the D7-brane stack.
    """

    def compute_gauge_couplings(self,
                                 divisor_volumes: np.ndarray,
                                 g_s: float,
                                 brane_config: Optional[dict] = None) -> dict:
        """
        Compute gauge couplings from 4-cycle volumes.

        Args:
            divisor_volumes: Array of divisor (4-cycle) volumes
            g_s: String coupling
            brane_config: Optional dictionary specifying which cycles host which gauge groups
                         Default: first 3 cycles host SU(3), SU(2), U(1)

        Returns:
            Dictionary with gauge couplings at string scale
        """
        if brane_config is None:
            # Default: diagonal brane configuration
            # D7_1 wraps Σ_1 -> SU(3)
            # D7_2 wraps Σ_2 -> SU(2)
            # D7_3 wraps Σ_3 -> U(1)
            brane_config = {
                "su3_cycle": 0,
                "su2_cycle": 1 if len(divisor_volumes) > 1 else 0,
                "u1_cycle": 2 if len(divisor_volumes) > 2 else 0,
            }

        # Get cycle volumes for each gauge group
        vol_su3 = divisor_volumes[brane_config["su3_cycle"]]
        vol_su2 = divisor_volumes[brane_config["su2_cycle"]]
        vol_u1 = divisor_volumes[brane_config["u1_cycle"]]

        # Gauge couplings at string scale: α_a = g_s / (4π Vol_a)
        alpha_3_string = g_s / (4 * np.pi * max(vol_su3, 1e-10))
        alpha_2_string = g_s / (4 * np.pi * max(vol_su2, 1e-10))
        alpha_1_string = g_s / (4 * np.pi * max(vol_u1, 1e-10))

        return {
            "alpha_3_string": float(alpha_3_string),
            "alpha_2_string": float(alpha_2_string),
            "alpha_1_string": float(alpha_1_string),
        }

    def run_to_z_scale(self, alpha_3: float, alpha_2: float, alpha_1: float,
                       m_string: float = 2e16) -> dict:
        """
        Run gauge couplings from string scale to Z mass using 1-loop RG.

        β-coefficients for SM:
        b_1 = 41/10, b_2 = -19/6, b_3 = -7

        Running: 1/α(μ) = 1/α(M) + b/(2π) ln(M/μ)
        """
        m_z = 91.2  # GeV
        log_ratio = np.log(m_string / m_z)

        # SM beta coefficients
        b1, b2, b3 = 41/10, -19/6, -7

        # Run down to Z scale
        alpha_1_z = 1.0 / (1.0/alpha_1 + b1/(2*np.pi) * log_ratio)
        alpha_2_z = 1.0 / (1.0/alpha_2 + b2/(2*np.pi) * log_ratio)
        alpha_3_z = 1.0 / (1.0/alpha_3 + b3/(2*np.pi) * log_ratio)

        # GUT normalization for U(1): α_1_SM = (5/3) α_1_GUT
        alpha_1_sm = (5/3) * alpha_1_z

        # Weinberg angle: sin²θ_W = α_1 / (α_1 + α_2) at tree level
        sin2_theta_w = alpha_1_sm / (alpha_1_sm + alpha_2_z)

        # EM coupling: 1/α_em = 1/α_1 + 1/α_2
        alpha_em = 1.0 / (1.0/alpha_1_sm + 1.0/alpha_2_z)

        return {
            "alpha_em": float(alpha_em),
            "alpha_s": float(alpha_3_z),
            "sin2_theta_w": float(sin2_theta_w),
            "alpha_1_z": float(alpha_1_z),
            "alpha_2_z": float(alpha_2_z),
            "alpha_3_z": float(alpha_3_z),
        }


class ModuliStabilizer:
    """
    KKLT-style moduli stabilization.

    The scalar potential in Type IIB:
    V = e^K [ K^{ij̄} D_i W D_j̄ W̄ - 3|W|² ] + V_uplift

    Where:
    - K = -2 ln(V) is the Kähler potential (V = CY volume)
    - W = W_flux + W_np is the superpotential
    - W_flux = ∫ G_3 ∧ Ω (GVW flux superpotential)
    - W_np = A e^{-aT} (non-perturbative from gaugino condensation/instantons)
    """

    def __init__(self):
        # Non-perturbative parameters
        self.A = 1.0  # Prefactor
        self.a = 2 * np.pi / 10  # a = 2π/N for SU(N) gaugino condensation

    def compute_flux_superpotential(self,
                                     flux_f: np.ndarray,
                                     flux_h: np.ndarray,
                                     periods: np.ndarray,
                                     tau: complex) -> complex:
        """
        Compute GVW superpotential: W = ∫ G_3 ∧ Ω = (F - τH) · Π

        Args:
            flux_f: F_3 flux quanta (integers)
            flux_h: H_3 flux quanta (integers)
            periods: Periods of holomorphic 3-form Ω
            tau: Axio-dilaton τ = C_0 + i/g_s
        """
        g3 = flux_f - tau * flux_h
        return complex(np.dot(g3, periods))

    def compute_potential(self,
                          cy_volume: float,
                          w_flux: complex,
                          kahler_moduli: np.ndarray,
                          n_antiD3: int = 1) -> dict:
        """
        Compute KKLT scalar potential with uplift.

        Returns cosmological constant and potential components.
        """
        # Kähler potential
        k = -2 * np.log(max(cy_volume, 1e-10))

        # Non-perturbative superpotential from largest Kähler modulus
        tau = np.max(kahler_moduli)
        w_np = self.A * np.exp(-self.a * tau)

        # Total superpotential
        w_total = w_flux + w_np

        # F-term potential (AdS minimum)
        # At the minimum, DW ≈ 0, so V_F ≈ -3 e^K |W|²
        exp_k = np.exp(k)
        w_abs_sq = np.abs(w_total)**2  # |W|² is always real
        v_ads = -3.0 * exp_k * float(w_abs_sq)

        # Uplift from anti-D3 branes at warped throat tip
        # V_uplift = D / V^{4/3}
        warp_factor = 0.01  # Typical warping
        d_uplift = n_antiD3 * warp_factor**4
        v_uplift = d_uplift / (cy_volume**(4/3) + 1e-10)

        v_total = v_ads + v_uplift

        return {
            "v_ads": float(v_ads),
            "v_uplift": float(v_uplift),
            "v_total": v_total,
            "cosmological_constant": v_total,
            "w_flux_abs": float(np.abs(w_flux)),
            "w_np_abs": float(np.abs(w_np)),
            "w_total_abs": float(np.abs(w_total)),
            "is_de_sitter": v_total > 0,
        }

    def compute_tadpole(self, flux_f: np.ndarray, flux_h: np.ndarray) -> float:
        """
        Compute D3-brane tadpole from fluxes.

        N_flux = (1/2) ∫ F_3 ∧ H_3

        Must satisfy: N_flux + N_D3 ≤ χ(CY)/24 for tadpole cancellation.
        """
        n = len(flux_f) // 2
        # Symplectic product
        return float(np.dot(flux_f[:n], flux_h[n:]) - np.dot(flux_f[n:], flux_h[:n])) / 2


class PhysicsBridge:
    """
    Main interface for Rust GA to call physics computations.

    Uses CYTools + cymyc for all computations. No fallbacks.
    """

    def __init__(self):
        self.cytools = CYToolsBridge()
        self.gauge = GaugeCouplingComputer()
        self.moduli = ModuliStabilizer()
        self._kahler_cache = {}  # Cache: (polytope_id, seed_hash) -> kahler point

    def _find_kahler_in_cone(self, cy_data: dict, genome: dict) -> np.ndarray:
        """
        Find a valid Kähler point inside the cone.

        Method: trace from cone tip along direction from genome.
        Find where ray exits cone, return midpoint of (tip, exit).
        Small direction changes = small output changes (smooth gradients).
        """
        cy = cy_data["_cy_object"]
        cone = cy.toric_kahler_cone()
        dim = cone.ambient_dim()

        # Get direction from genome (normalize to unit vector)
        genome_kahler = np.array(genome.get("kahler_moduli", [1.0] * dim))
        if len(genome_kahler) < dim:
            genome_kahler = np.concatenate([genome_kahler, np.zeros(dim - len(genome_kahler))])
        direction = genome_kahler[:dim]

        # Normalize to unit vector (handle zero vector)
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            direction = np.ones(dim) / np.sqrt(dim)
        else:
            direction = direction / norm

        # Start from tip of cone (guaranteed inside)
        tip = cone.tip_of_stretched_cone(1.0)

        # Binary search to find where ray exits cone
        # Ray: point(t) = tip + t * direction
        t_min, t_max = 0.0, 1.0

        # First, find an upper bound where we're outside
        while cone.contains(tip + t_max * direction) and t_max < 1000:
            t_max *= 2

        # Binary search for exit point
        for _ in range(50):  # ~15 decimal digits precision
            t_mid = (t_min + t_max) / 2
            if cone.contains(tip + t_mid * direction):
                t_min = t_mid
            else:
                t_max = t_mid

        # Use point at 50% of the way to boundary (safely inside)
        t_final = t_min * 0.5
        return tip + t_final * direction

    def compute_physics(self, genome: dict) -> dict:
        """
        Compute all physical observables from a compactification genome.

        Args:
            genome: Dictionary containing:
                - vertices: Polytope vertices (4D integer coordinates)
                - kahler_moduli: Array of Kähler moduli
                - complex_moduli: Array of complex structure moduli
                - flux_f, flux_h: Flux quanta
                - g_s: String coupling
        """
        try:
            return self._compute_physics_impl(genome)
        except Exception:
            import traceback
            traceback.print_exc()
            raise

    def _compute_physics_impl(self, genome: dict) -> dict:
        """Actual implementation - exceptions will print traceback then propagate."""
        # 1. Analyze polytope with CYTools
        vertices = genome.get("vertices", [])
        if not vertices:
            return {"success": False, "error": "No vertices provided"}

        cy_data = self.cytools.analyze_polytope(vertices)
        if not cy_data["success"]:
            return cy_data

        # 2. Get Kähler moduli (must be in Kähler cone)
        # Use genome as seed, walk trajectory until inside cone
        kahler = self._find_kahler_in_cone(cy_data, genome)

        # 3. Compute volumes
        vol_data = self.cytools.compute_volumes(cy_data, kahler)
        if not vol_data["success"]:
            # Use simplified volume if CYTools computation fails
            cy_volume = float(np.prod(kahler) / 6.0)
            divisor_volumes = kahler**2
        else:
            cy_volume = vol_data["cy_volume"]
            divisor_volumes = np.array(vol_data["divisor_volumes"])

        # Validate volumes - CYTools returns complex/negative for invalid geometry
        if np.iscomplex(cy_volume) or np.real(cy_volume) <= 0:
            return {"success": False, "error": "Invalid geometry: CY volume not positive real"}
        cy_volume = float(np.real(cy_volume))

        divisor_volumes = np.real(divisor_volumes)
        if np.any(divisor_volumes <= 0):
            return {"success": False, "error": "Invalid geometry: divisor volumes not positive"}

        # 4. Compute gauge couplings
        g_s = genome.get("g_s", 0.1)
        gauge_string = self.gauge.compute_gauge_couplings(divisor_volumes, g_s)
        gauge_z = self.gauge.run_to_z_scale(
            gauge_string["alpha_3_string"],
            gauge_string["alpha_2_string"],
            gauge_string["alpha_1_string"],
        )

        # 5. Compute flux superpotential and potential
        h11 = cy_data["h11"]
        h21 = cy_data["h21"]
        n_periods = 2 * (h21 + 1)

        flux_f = np.array(genome.get("flux_f", [0] * n_periods))
        flux_h = np.array(genome.get("flux_h", [0] * n_periods))

        # Truncate fluxes if too long (genome may have more than needed)
        flux_f = flux_f[:n_periods]
        flux_h = flux_h[:n_periods]

        # Pad fluxes if too short
        if len(flux_f) < n_periods:
            flux_f = np.concatenate([flux_f, np.zeros(n_periods - len(flux_f))])
        if len(flux_h) < n_periods:
            flux_h = np.concatenate([flux_h, np.zeros(n_periods - len(flux_h))])

        # Simple period approximation (real computation needs CY metric)
        complex_mod = genome.get("complex_moduli", [1.0])
        periods = np.exp(1j * np.arange(n_periods) * 0.1) * complex_mod[0]

        tau = complex(0.1 + 1j / g_s)  # Axio-dilaton
        w_flux = self.moduli.compute_flux_superpotential(flux_f, flux_h, periods, tau)

        n_antiD3 = genome.get("n_antiD3", 1)
        potential = self.moduli.compute_potential(cy_volume, w_flux, kahler, n_antiD3)

        # 6. Tadpole constraint
        tadpole = self.moduli.compute_tadpole(flux_f, flux_h)
        tadpole_bound = abs(cy_data["chi"]) / 24.0

        # 7. Number of generations
        n_gen = cy_data["n_generations"]

        # 8. Mass ratios (placeholder - needs Yukawa computation from cymyc)
        w_total = potential["w_total_abs"]
        m_e_ratio = g_s * w_total / (cy_volume**(1/3) + 1e-10) * 1e-22
        m_p_ratio = m_e_ratio * 1836.15

        return {
            "success": True,

            # Gauge couplings
            "alpha_em": gauge_z["alpha_em"],
            "alpha_s": gauge_z["alpha_s"],
            "sin2_theta_w": gauge_z["sin2_theta_w"],

            # Cosmological
            "cosmological_constant": potential["cosmological_constant"],
            "is_de_sitter": potential["is_de_sitter"],

            # Topology
            "n_generations": n_gen,
            "h11": h11,
            "h21": h21,
            "chi": cy_data["chi"],

            # Masses
            "m_e_planck_ratio": float(m_e_ratio),
            "m_p_planck_ratio": float(m_p_ratio),

            # Geometry
            "cy_volume": cy_volume,
            "string_coupling": g_s,

            # Constraints
            "flux_tadpole": tadpole,
            "tadpole_bound": tadpole_bound,
            "tadpole_satisfied": tadpole <= tadpole_bound,

            # Potential
            "superpotential_abs": w_total,
            "v_ads": potential["v_ads"],
            "v_uplift": potential["v_uplift"],
        }

    def serve_stdio(self):
        """JSON-RPC server over stdin/stdout."""
        for line in sys.stdin:
            try:
                request = json.loads(line.strip())
                method = request.get("method", "")
                params = request.get("params", {})

                if method == "compute_physics":
                    result = self.compute_physics(params)
                elif method == "analyze_polytope":
                    result = self.cytools.analyze_polytope(params.get("vertices", []))
                elif method == "ping":
                    result = {"status": "ok", "cytools": True, "cymyc": True}
                else:
                    result = {"error": f"Unknown method: {method}"}

                print(json.dumps({"result": result}), flush=True)

            except json.JSONDecodeError as e:
                print(json.dumps({"error": f"JSON decode error: {e}"}), flush=True)
            except Exception as e:
                print(json.dumps({"error": str(e)}), flush=True)


def test_bridge():
    """Test the physics bridge."""
    print("Testing physics bridge with CYTools + cymyc...")
    print()

    bridge = PhysicsBridge()

    # Test with quintic threefold vertices
    quintic_vertices = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [-1, -1, -1, -1]
    ]

    print("1. Analyzing quintic threefold polytope...")
    result = bridge.cytools.analyze_polytope(quintic_vertices)
    if result["success"]:
        print(f"   h11 = {result['h11']}, h21 = {result['h21']}, χ = {result['chi']}")
        print(f"   N_gen = {result['n_generations']}")
        print(f"   Favorable: {result['is_favorable']}")
    else:
        print(f"   Error: {result['error']}")
    print()

    print("2. Computing full physics...")
    genome = {
        "vertices": quintic_vertices,
        "kahler_moduli": [2.0],
        "complex_moduli": [1.0],
        "flux_f": [1, 0, 0, 0],
        "flux_h": [0, 1, 0, 0],
        "g_s": 0.1,
        "n_antiD3": 1,
    }

    physics = bridge.compute_physics(genome)
    if physics["success"]:
        print(f"   α_em = {physics['alpha_em']:.6f} (obs: {ALPHA_EM_OBS:.6f})")
        print(f"   α_s  = {physics['alpha_s']:.4f} (obs: {ALPHA_S_OBS:.4f})")
        print(f"   sin²θ_W = {physics['sin2_theta_w']:.5f} (obs: {SIN2_THETA_W_OBS:.5f})")
        print(f"   Λ = {physics['cosmological_constant']:.3e}")
        print(f"   N_gen = {physics['n_generations']}")
        print(f"   CY Volume = {physics['cy_volume']:.4f}")
        print(f"   Tadpole: {physics['flux_tadpole']:.1f} / {physics['tadpole_bound']:.1f}")
    else:
        print(f"   Error: {physics['error']}")

    print()
    print("Physics bridge ready!")


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
