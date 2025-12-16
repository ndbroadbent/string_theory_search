#!/usr/bin/env python3
"""
Full Pipeline: Compute V₀ from scratch - NO CHEATING.

INPUTS (model specification):
1. Polytope points (primal and dual)
2. Triangulation (simplices/heights)
3. K, M flux vectors
4. c_i values (orientifold choice)
5. Divisor basis indices

OUTPUTS (all computed from scratch):
- κ (intersection numbers)
- p (flat direction)
- e^K₀ (Kähler potential)
- GV invariants (computed via compute_gvs)
- g_s, W₀ (from racetrack)
- χ(D_i) (divisor characteristics)
- t (Kähler moduli)
- V_string (CY volume)
- V₀ (cosmological constant)

NO intermediate .dat files read - everything computed from geometry.
"""

import sys
from pathlib import Path
from decimal import Decimal, getcontext
from collections import defaultdict
import time

SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent.parent
DATA_BASE = ROOT_DIR / "resources/small_cc_2107.09064_source/anc/paper_data"
CYTOOLS_2021 = ROOT_DIR / "vendor/cytools_mcallister_2107"
CYTOOLS_LATEST = ROOT_DIR / "vendor/cytools_latest/src"

getcontext().prec = 250
sys.path.insert(0, str(CYTOOLS_2021))

import numpy as np
from scipy.special import zeta
from mpmath import mp, mpf, mpc, exp as mp_exp, log as mp_log, pi as mp_pi, polylog
mp.dps = 150

# Constant from eq. 2.22: ζ = 1/(2^{3/2} π^{5/2})
ZETA = mpf(1) / (mpf(2) ** mpf('1.5') * mp_pi ** mpf('2.5'))

from cytools import Polytope

# Import from sibling modules
from compute_basis_transform import compute_T_from_glsm, transform_fluxes
from compute_chi_divisor import compute_chi_divisor as compute_chi_divisor_func
from compute_target_tau import compute_c_tau as compute_c_tau_func
from compute_kklt_iterative import solve_kklt as solve_kklt_proper
from compute_kahler_param import (
    SparseKappa as KahlerSparseKappa,
    solve_kklt_path_following,
    compute_chi_for_kklt_divisors,
)

MCALLISTER_EXAMPLES = [
    ("4-214-647", 214, 4, True),
    ("5-113-4627-main", 113, 5, True),
    ("5-113-4627-alternative", 113, 5, True),
    ("5-81-3213", 81, 5, True),
    ("7-51-13590", 51, 7, False),
]


# =============================================================================
# INPUT LOADING
# =============================================================================


def load_model_inputs(example_name: str) -> dict:
    """Load model inputs - geometry + flux/orientifold choices."""
    data_dir = DATA_BASE / example_name

    def load_array(filename, dtype=int):
        lines = (data_dir / filename).read_text().strip().split('\n')
        return np.array([[dtype(x) for x in line.split(',')] for line in lines])

    def load_vector(filename, dtype=int):
        text = (data_dir / filename).read_text().strip()
        return np.array([dtype(x) for x in text.split(',')])

    def load_simplices(filename):
        lines = (data_dir / filename).read_text().strip().split('\n')
        return [[int(x) for x in line.split(',')] for line in lines]

    return {
        "primal_points": load_array("points.dat"),
        "dual_points": load_array("dual_points.dat"),
        "dual_simplices": load_simplices("dual_simplices.dat"),
        "primal_heights": load_vector("corrected_heights.dat", float),
        "K": load_vector("K_vec.dat"),
        "M": load_vector("M_vec.dat"),
        "c_i": load_vector("target_volumes.dat"),
        "basis": [int(x) for x in (data_dir / "basis.dat").read_text().strip().split(',')],
        "kklt_basis": [int(x) for x in (data_dir / "kklt_basis.dat").read_text().strip().split(',')],
    }


def load_expected(example_name: str) -> dict:
    """Load expected values for validation."""
    data_dir = DATA_BASE / example_name

    def safe_load(filename):
        path = data_dir / filename
        if path.exists():
            return float(path.read_text().strip())
        return None

    return {
        "g_s": safe_load("g_s.dat"),
        "W0": safe_load("W_0.dat"),
        "eK0": safe_load("eK0.dat"),
        "V_string": safe_load("cy_vol.dat"),
        "V0": safe_load("V_0.dat"),
    }


# =============================================================================
# GEOMETRY COMPUTATIONS
# =============================================================================


def build_cy_2021(points: np.ndarray, simplices: list = None, heights: np.ndarray = None):
    """Build CY using CYTools 2021."""
    poly = Polytope(points)
    if simplices is not None:
        tri = poly.triangulate(simplices=simplices, check_input_simplices=False)
    else:
        tri = poly.triangulate(heights=heights)
    return tri.get_cy()


def build_cy_latest(points: np.ndarray, simplices: list = None, heights: np.ndarray = None):
    """Build CY using latest CYTools (for compute_gvs and non-favorable)."""
    mods = [k for k in list(sys.modules.keys()) if 'cytools' in k]
    for m in mods:
        del sys.modules[m]

    if str(CYTOOLS_LATEST) in sys.path:
        sys.path.remove(str(CYTOOLS_LATEST))
    sys.path.insert(0, str(CYTOOLS_LATEST))

    from cytools import Polytope as PL
    from cytools import config
    config.enable_experimental_features()

    poly = PL(points)
    if simplices is not None:
        tri = poly.triangulate(simplices=simplices, check_input_simplices=False)
    else:
        tri = poly.triangulate(heights=heights)
    return tri.get_cy()


def restore_cytools_2021():
    """Restore CYTools 2021 after using latest."""
    mods = [k for k in list(sys.modules.keys()) if 'cytools' in k]
    for m in mods:
        del sys.modules[m]
    if str(CYTOOLS_LATEST) in sys.path:
        sys.path.remove(str(CYTOOLS_LATEST))
    sys.path.insert(0, str(CYTOOLS_2021))


def compute_kappa(cy) -> np.ndarray:
    """Compute symmetric intersection tensor."""
    h11 = len(cy.divisor_basis())
    kappa_sparse = cy.intersection_numbers(in_basis=True)
    kappa = np.zeros((h11, h11, h11))

    if hasattr(kappa_sparse, 'items'):
        for (i, j, k), val in kappa_sparse.items():
            for p in [(i,j,k), (i,k,j), (j,i,k), (j,k,i), (k,i,j), (k,j,i)]:
                kappa[p] = val
    else:
        for row in kappa_sparse:
            i, j, k, val = int(row[0]), int(row[1]), int(row[2]), row[3]
            for p in [(i,j,k), (i,k,j), (j,i,k), (j,k,i), (k,i,j), (k,j,i)]:
                kappa[p] = val
    return kappa


def compute_p(kappa: np.ndarray, K: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Compute p = N^{-1} K where N_ab = κ_abc M^c."""
    N = np.einsum('abc,c->ab', kappa, M)
    return np.linalg.solve(N, K)


def compute_eK0(kappa: np.ndarray, p: np.ndarray) -> float:
    """Compute e^{K_0} = (4/3 × κ_abc p^a p^b p^c)^{-1}."""
    kappa_ppp = np.einsum('abc,a,b,c->', kappa, p, p, p)
    return 1.0 / ((4.0/3.0) * kappa_ppp)


# =============================================================================
# GV AND RACETRACK
# =============================================================================


def compute_gv_invariants(dual_points: np.ndarray, dual_simplices: list) -> tuple:
    """
    Compute GV invariants from dual polytope using compute_gvs.

    Returns:
        (gv_dict, cy_latest, kappa_latest, basis_latest) - everything in latest basis
    """
    cy = build_cy_latest(dual_points, simplices=dual_simplices)
    gv_obj = cy.compute_gvs(min_points=20000)

    charges = gv_obj.charges()
    gvs = gv_obj.gvs
    gv_dict = {tuple(charges[i]): gvs[i] for i in range(len(gvs))}

    # Also compute kappa in this basis (no switching versions)
    kappa = compute_kappa(cy)
    basis = list(cy.divisor_basis())

    # DON'T restore_cytools_2021 - we need to stay in latest for basis transformation
    return gv_dict, cy, kappa, basis


def compute_racetrack_terms(gv_dict: dict, p: np.ndarray, M: np.ndarray) -> list:
    """Build racetrack terms from GV invariants."""
    terms = defaultdict(lambda: 0)

    for q_tuple, N_q in gv_dict.items():
        if N_q == 0:
            continue

        q = np.array(q_tuple)
        qp = float(np.dot(q, p))
        Mq = int(np.dot(M, q))
        coeff = Mq * N_q

        if abs(coeff) > 0 and qp > 0:
            terms[round(qp, 8)] += coeff

    return sorted([(e, c) for e, c in terms.items() if c != 0], key=lambda x: x[0])


def solve_racetrack(terms: list) -> dict:
    """Solve racetrack for g_s and W₀."""
    if len(terms) < 2:
        return {"success": False, "error": "Need >= 2 terms"}

    alpha, c_alpha = terms[0]
    beta, c_beta = terms[1]
    epsilon = beta - alpha

    if epsilon <= 0:
        return {"success": False, "error": "Invalid exponents"}

    d1 = c_alpha * alpha
    d2 = c_beta * beta

    if abs(d2) < 1e-10:
        return {"success": False, "error": "Zero denominator"}

    delta = -d1 / d2

    if delta == 0:
        return {"success": False, "error": "δ = 0"}

    Im_tau = np.log(1.0 / abs(delta)) / (2 * np.pi * epsilon)

    if Im_tau <= 0:
        return {"success": False, "error": f"Im(τ) = {Im_tau} <= 0"}

    g_s = 1.0 / Im_tau
    Re_tau = 1.0 / (2 * epsilon) if delta < 0 else 0.0

    # Compute W₀
    W0 = compute_W0(terms, Re_tau, Im_tau)

    return {"success": True, "g_s": g_s, "W0": W0, "Re_tau": Re_tau, "Im_tau": Im_tau}


def compute_W0(terms: list, Re_tau: float, Im_tau: float) -> mpf:
    """
    Compute W₀ = |W(τ)| where W = -ζ Σ (M·q) N_q Li₂(e^{2πiτ(q·p)}).

    For τ = x + iy:
        e^{2πiτ α} = e^{-2παy} × e^{2πiαx}
    """
    W_sum = mpc(0)
    Re_tau_mp = mpf(str(Re_tau))
    Im_tau_mp = mpf(str(Im_tau))

    for alpha, coeff in terms:
        alpha_mp = mpf(str(alpha))
        # e^{2πiτα} = e^{-2παy} × e^{2πiαx}
        magnitude = mp_exp(-2 * mp_pi * Im_tau_mp * alpha_mp)
        phase = 2 * mp_pi * alpha_mp * Re_tau_mp
        arg = magnitude * mp_exp(mpc(0, phase))  # e^{iθ} = cos(θ) + i*sin(θ)
        W_sum += mpf(str(float(coeff))) * polylog(2, arg)

    return abs(-ZETA * W_sum)


# =============================================================================
# KKLT AND VOLUME
# =============================================================================


def compute_c_tau(g_s: float, W0: mpf) -> float:
    """Compute c_τ = 2π / (g_s × ln(1/W₀))."""
    return float(2 * mp_pi / (mpf(g_s) * mp_log(1 / W0)))






def compute_V_string(kappa: np.ndarray, t: np.ndarray, h11: int, h21: int) -> float:
    """Compute V_string = (1/6)κt³ - BBHL."""
    V = np.einsum('ijk,i,j,k->', kappa, t, t, t) / 6.0
    chi = 2 * (h11 - h21)
    BBHL = zeta(3) * chi / (4 * (2 * np.pi) ** 3)
    return V - BBHL


def compute_V0(eK0: float, g_s: float, V_string: float, W0) -> Decimal:
    """Compute V₀ = -3 × e^K₀ × (g_s⁷/(4V)²) × W₀²."""
    return (Decimal(-3) * Decimal(str(eK0)) *
            (Decimal(str(g_s))**7 / (4 * Decimal(str(V_string)))**2) *
            Decimal(str(float(W0)))**2)


# =============================================================================
# FULL PIPELINE
# =============================================================================


def run_pipeline(example_name: str, h11_primal: int, h21_primal: int,
                 is_favorable: bool, verbose: bool = True) -> dict:
    """Run complete pipeline."""
    t0 = time.time()

    if verbose:
        print("=" * 70)
        print(f"FULL PIPELINE - {example_name}")
        print("=" * 70)

    # Load inputs
    inputs = load_model_inputs(example_name)
    if verbose:
        print(f"\n[1] Inputs: K={inputs['K']}, M={inputs['M']}")

    # Build dual CY (2021) and get kappa/basis
    if verbose:
        print("\n[2] Building dual CY (2021) for kappa...")

    cy_dual_2021 = build_cy_2021(inputs['dual_points'], simplices=inputs['dual_simplices'])
    kappa_dual_2021 = compute_kappa(cy_dual_2021)
    basis_2021 = list(cy_dual_2021.divisor_basis())

    # Compute e^K₀ directly with 2021 kappa and inputs (this is the invariant)
    p_2021 = compute_p(kappa_dual_2021, inputs['K'], inputs['M'])
    eK0 = compute_eK0(kappa_dual_2021, p_2021)

    if verbose:
        print(f"  basis_2021 = {basis_2021}")
        print(f"  p (2021) = {p_2021}")
        print(f"  e^K₀ = {eK0:.6f}")

    # Compute GV invariants (using latest CYTools) - returns everything in latest basis
    if verbose:
        print("\n[3] Computing GV invariants (CYTools latest)...")

    gv_dict, cy_latest, kappa_latest, basis_latest = compute_gv_invariants(
        inputs['dual_points'], inputs['dual_simplices']
    )

    if verbose:
        print(f"  Found {len(gv_dict)} curves")
        print(f"  basis_latest = {basis_latest}")

    # Transform K, M from 2021 basis to latest basis
    if verbose:
        print("\n[4] Transforming K, M to latest basis...")

    if basis_2021 != basis_latest:
        T = compute_T_from_glsm(cy_latest, basis_2021, basis_latest)
        K_latest, M_latest = transform_fluxes(inputs['K'], inputs['M'], T)
        if verbose:
            print(f"  K_latest = {K_latest}")
            print(f"  M_latest = {M_latest}")
    else:
        K_latest = inputs['K']
        M_latest = inputs['M']
        if verbose:
            print("  Bases identical - no transformation needed")

    # Compute p in latest basis
    p_latest = compute_p(kappa_latest, K_latest, M_latest)

    if verbose:
        print(f"  p (latest) = {p_latest}")

    # Verify e^K₀ is invariant
    eK0_check = compute_eK0(kappa_latest, p_latest)
    if verbose:
        print(f"  e^K₀ check = {eK0_check:.6f} (should match {eK0:.6f})")

    # Restore CYTools 2021 before racetrack (racetrack uses mpmath, not CYTools)
    restore_cytools_2021()

    # Racetrack - using p and M in LATEST basis (matching GV curves)
    if verbose:
        print("\n[5] Solving racetrack...")

    terms = compute_racetrack_terms(gv_dict, p_latest, M_latest)

    if verbose:
        print(f"  {len(terms)} racetrack terms")

    result = solve_racetrack(terms)

    if not result['success']:
        if verbose:
            print(f"  ERROR: {result['error']}")
        return {"success": False, "error": result['error'], "example_name": example_name}

    g_s = result['g_s']
    W0 = result['W0']

    if verbose:
        print(f"  g_s = {g_s:.6f}")
        print(f"  W₀ = {float(W0):.2e}")

    # Build primal CY
    if verbose:
        print("\n[6] Building primal CY...")

    if is_favorable:
        cy_primal = build_cy_2021(inputs['primal_points'], heights=inputs['primal_heights'])
        cy_primal.set_divisor_basis(inputs['basis'])
    else:
        cy_primal = build_cy_latest(inputs['primal_points'], heights=inputs['primal_heights'])
        cy_primal.set_divisor_basis(inputs['basis'])

    basis_size = len(cy_primal.divisor_basis())
    h11, h21 = cy_primal.h11(), cy_primal.h21()

    if not is_favorable:
        restore_cytools_2021()

    kappa_primal = compute_kappa(cy_primal) if is_favorable else np.zeros((basis_size,)*3)

    # Recompute kappa for non-favorable (CY object lost after restore)
    if not is_favorable:
        cy_primal = build_cy_latest(inputs['primal_points'], heights=inputs['primal_heights'])
        cy_primal.set_divisor_basis(inputs['basis'])
        kappa_primal = compute_kappa(cy_primal)
        basis_size = len(cy_primal.divisor_basis())
        h11, h21 = cy_primal.h11(), cy_primal.h21()

    if verbose:
        print(f"  h11={h11}, h21={h21}, basis={basis_size}")

    # Target τ using correct basis/kklt_basis handling
    if verbose:
        print("\n[7] Computing target τ...")

    c_tau = compute_c_tau(g_s, W0)

    # Get intersection numbers in both basis and all-divisor formats
    poly_primal = Polytope(inputs['primal_points'])
    kappa_basis = cy_primal.intersection_numbers(in_basis=True)
    kappa_all = cy_primal.intersection_numbers(in_basis=False)

    # Build sparse kappa with correct basis/kklt_basis handling
    basis = inputs['basis']
    kklt_basis = inputs['kklt_basis']
    kappa = KahlerSparseKappa(kappa_basis, kappa_all, basis, kklt_basis)

    # Compute χ(D) for KKLT divisors
    chi_kklt = compute_chi_for_kklt_divisors(poly_primal, kappa_all, kklt_basis)

    # Target τ for KKLT divisors
    target_tau = inputs['c_i'] / c_tau + chi_kklt / 24.0

    if verbose:
        print(f"  c_τ = {c_tau:.4f}")
        print(f"  χ range: [{chi_kklt.min():.1f}, {chi_kklt.max():.1f}]")
        print(f"  τ range: [{target_tau.min():.2f}, {target_tau.max():.2f}]")
        print(f"  basis: {len(basis)}, kklt_basis: {len(kklt_basis)}, shared: {len(set(basis) & set(kklt_basis))}")

    # KKLT - use path-following with Newton-Raphson fallback
    # Pass c_i to enable two-phase solver (first τ=c_i, then target τ)
    if verbose:
        print("\n[8] Solving KKLT...")

    t, tau_achieved, converged = solve_kklt_path_following(kappa, target_tau, c_i=inputs['c_i'])

    # Compute V_string
    # Build full tensor for V computation
    kappa_tensor = np.zeros((basis_size, basis_size, basis_size))
    for row in kappa_basis:
        i, j, k = int(row[0]), int(row[1]), int(row[2])
        val = row[3]
        for perm in [(i,j,k), (i,k,j), (j,i,k), (j,k,i), (k,i,j), (k,j,i)]:
            kappa_tensor[perm] = val

    V_classical = np.einsum('ijk,i,j,k->', kappa_tensor, t, t, t) / 6.0
    chi_euler = 2 * (h11 - h21)
    BBHL = zeta(3) * chi_euler / (4 * (2 * np.pi) ** 3)
    V_string = V_classical - BBHL

    if verbose:
        print(f"  V_classical = {V_classical:.4f}")
        print(f"  BBHL = {BBHL:.6f} (χ={chi_euler})")
        print(f"  V_string = {V_string:.4f}")
        print(f"  t range: [{t.min():.4f}, {t.max():.4f}]")
        print(f"  converged: {converged}")

    if verbose:
        print(f"\n[9] V_string = {V_string:.4f}")

    # V₀
    V0 = compute_V0(eK0, g_s, V_string, W0)
    V0_f = float(V0)

    if verbose:
        print(f"[10] V₀ = {V0_f:.2e}")

    # Validate
    expected = load_expected(example_name)
    ratios = {}
    for k, computed in [("g_s", g_s), ("W0", float(W0)), ("eK0", eK0),
                        ("V_string", V_string), ("V0", V0_f)]:
        exp = expected.get(k)
        if exp is not None and exp != 0:
            ratios[k] = computed / abs(exp) if k == "W0" else computed / exp
        else:
            ratios[k] = None

    if verbose:
        print(f"\n[11] Validation:")
        for k, r in ratios.items():
            if r is not None:
                print(f"  {k}: ratio = {r:.4f}")
            else:
                print(f"  {k}: no expected value")

    V0_ratio = ratios.get('V0')
    passed = V0_ratio is not None and abs(V0_ratio - 1.0) < 0.5  # 50% tolerance for now

    if verbose:
        print(f"\n{'PASS' if passed else 'FAIL'}: {example_name} ({time.time()-t0:.1f}s)")

    if not is_favorable:
        restore_cytools_2021()

    return {
        "success": True, "passed": passed, "example_name": example_name,
        "g_s": g_s, "W0": float(W0), "eK0": eK0, "V_string": V_string,
        "V0": V0_f, "ratios": ratios, "elapsed": time.time() - t0,
    }


def main():
    print("=" * 70)
    print("FULL PIPELINE - END TO END V₀")
    print("=" * 70)

    results = []
    for name, h11, h21, fav in MCALLISTER_EXAMPLES:
        r = run_pipeline(name, h11, h21, fav, verbose=True)
        results.append(r)
        print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for r in results:
        if r['success']:
            print(f"  {'PASS' if r['passed'] else 'FAIL'}: {r['example_name']:30s} V₀={r['V0']:.2e}")
        else:
            print(f"  FAIL: {r['example_name']:30s} {r.get('error','')}")

    return all(r.get('passed', False) for r in results)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
