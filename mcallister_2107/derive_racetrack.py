#!/usr/bin/env python3
"""
Derive the two-term racetrack from GV invariants.

This script:
1. Loads curve classes from dual_curves.dat (9D ambient basis)
2. Projects them to 4D h11 basis using CYTools
3. Computes q̃·p and M·q̃ for each curve
4. Builds W_flux(τ) and identifies the leading terms
5. Solves ∂W/∂τ = 0 numerically to get g_s and W₀

The goal is to reproduce eq. 6.59 from arXiv:2107.09064:
W_flux(τ) = 5ζ [-e^{2πiτ·(32/110)} + 512·e^{2πiτ·(33/110)}] + O(e^{2πiτ·(13/22)})
"""

import numpy as np
from fractions import Fraction
from pathlib import Path
from mpmath import mp, mpf, polylog, exp, pi, log, diff, findroot

from cytools import Polytope

# High precision for W₀ ~ 10^{-90}
mp.dps = 150

DATA_DIR = Path(__file__).parent.parent / "resources/small_cc_2107.09064_source/anc/paper_data/4-214-647"

# Constants
ZETA = 1.0 / (2**(3/2) * np.pi**(5/2))  # eq. 2.22


def load_mcallister_data():
    """Load flux vectors and compute flat direction."""
    K = np.array([-3, -5, 8, 6])
    M = np.array([10, 11, -11, -5])

    # Expected p from paper
    p_expected = np.array([
        float(Fraction(293, 110)),
        float(Fraction(163, 110)),
        float(Fraction(163, 110)),
        float(Fraction(13, 22))
    ])

    return K, M, p_expected


def load_dual_polytope():
    """Load dual polytope points."""
    lines = (DATA_DIR / "dual_points.dat").read_text().strip().split('\n')
    return np.array([[int(x) for x in line.split(',')] for line in lines])


def load_triangulation():
    """Load McAllister's triangulation."""
    lines = (DATA_DIR / "dual_simplices.dat").read_text().strip().split('\n')
    return [[int(x) for x in line.split(',')] for line in lines]


def load_curves_and_gv():
    """Load curve classes and GV invariants."""
    # Curves: 5177 x 9 integers
    curve_lines = (DATA_DIR / "dual_curves.dat").read_text().strip().split('\n')
    curves = np.array([[int(x) for x in line.split(',')] for line in curve_lines])

    # GV: comma-separated on single line
    gv_text = (DATA_DIR / "dual_curves_gv.dat").read_text().strip()
    gv = np.array([int(x) for x in gv_text.split(',')])

    print(f"Loaded {len(curves)} curves, {len(gv)} GV invariants")
    assert len(curves) == len(gv), "Curve/GV count mismatch!"

    return curves, gv


def setup_cytools():
    """Initialize CYTools with McAllister's geometry."""
    dual_points = load_dual_polytope()
    simplices = load_triangulation()

    poly = Polytope(dual_points)
    tri = poly.triangulate(simplices=simplices, check_input_simplices=False)
    cy = tri.get_cy()

    print(f"CY: h11={cy.h11()}, h21={cy.h21()}")
    print(f"Divisor basis: {cy.divisor_basis()}")

    return cy


def get_curve_projection_matrix(cy):
    """
    Get the matrix to project 9D ambient curves to 4D h11 basis.

    In toric geometry, curve classes in H_2(X) are dual to divisor classes in H^2(X).
    The GLSM linear relations tell us how ambient divisors relate to the basis.
    """
    # Get the linear relations among divisors
    # This is a matrix L where L @ D_ambient = 0 (linear dependencies)
    linrels = cy.polytope().glsm_linear_relations()
    print(f"GLSM linear relations shape: {linrels.shape}")

    # Get the divisor basis indices
    basis = cy.divisor_basis()
    print(f"Divisor basis: {basis}")

    # The ambient divisors D_i can be expressed in terms of basis divisors
    # We need to find this transformation

    # Actually, for curves, we use the dual approach:
    # A curve q in 9D ambient contracts with divisors: q·D_i gives intersection numbers
    # In the 4D basis, we want q_basis such that q_basis·D_basis = q·D_basis

    # The simplest approach: extract the columns corresponding to basis divisors
    # This gives the 4D curve class directly
    n_pts = cy.polytope().points().shape[0]
    print(f"Number of polytope points: {n_pts}")

    # The curve's 4D representation is just its components at the basis indices
    # But we need to be careful about the origin point

    return basis


def project_curves_to_basis(curves_9d, cy):
    """
    Project 9D ambient curves to 4D basis.

    Key insight: The curve class in the 4D basis is obtained by
    taking the intersection with the basis divisors.
    """
    basis = cy.divisor_basis()

    # The 9D curves are indexed by polytope points (excluding origin in some cases)
    # The basis divisors correspond to specific points

    # For the dual polytope with 12 points, indices 0-11 correspond to points
    # But CYTools may use a different indexing

    # Let's check what the curves look like
    print(f"Curves shape: {curves_9d.shape}")
    print(f"First few curves:\n{curves_9d[:5]}")

    # The 9D seems to be the full ambient toric variety
    # We need to extract the 4D components corresponding to h11 basis

    # Actually, the curves_9d columns likely correspond to:
    # - Vertices of the polytope (or some subset of lattice points)

    # Let's try a direct extraction based on basis indices
    # If basis = [3,4,5,8], extract those columns

    # But first verify the basis indices are within range
    assert all(b < curves_9d.shape[1] for b in basis), \
        f"Basis indices {basis} out of range for {curves_9d.shape[1]} columns"

    curves_4d = curves_9d[:, basis]
    print(f"Projected curves shape: {curves_4d.shape}")
    print(f"First few projected curves:\n{curves_4d[:5]}")

    return curves_4d


def compute_q_dot_p(curves_4d, p):
    """Compute q̃·p for each curve."""
    return curves_4d @ p


def compute_M_dot_q(curves_4d, M):
    """Compute M·q̃ for each curve."""
    return curves_4d @ M


def identify_leading_terms(q_dot_p, M_dot_q, gv, cutoff=1.0):
    """
    Identify the leading terms in the racetrack.

    Terms are ordered by q̃·p (smaller = more dominant at large Im(τ)).
    """
    # Filter for positive q·p < cutoff
    mask = (q_dot_p > 0) & (q_dot_p < cutoff)

    indices = np.where(mask)[0]

    # Build term list
    terms = []
    for i in indices:
        eff_coeff = M_dot_q[i] * gv[i]
        terms.append({
            'idx': i,
            'q_dot_p': q_dot_p[i],
            'M_dot_q': M_dot_q[i],
            'N_q': gv[i],
            'eff_coeff': eff_coeff
        })

    # Sort by q·p
    terms.sort(key=lambda t: t['q_dot_p'])

    return terms


def group_terms_by_exponent(terms, tol=1e-6):
    """
    Group terms with the same q·p exponent and sum their coefficients.

    This is important because multiple curves can contribute to the same
    exponential e^{2πiτ(q·p)}.
    """
    from collections import defaultdict

    groups = defaultdict(lambda: {'q_dot_p': None, 'eff_coeff': 0, 'count': 0, 'curves': []})

    for t in terms:
        # Round to find group key
        key = round(t['q_dot_p'], 6)

        if groups[key]['q_dot_p'] is None:
            groups[key]['q_dot_p'] = t['q_dot_p']

        groups[key]['eff_coeff'] += t['eff_coeff']
        groups[key]['count'] += 1
        groups[key]['curves'].append(t)

    # Convert to list and sort
    grouped = list(groups.values())
    grouped.sort(key=lambda g: g['q_dot_p'])

    return grouped


def build_W_flux(terms, tau, n_terms=10):
    """
    Build W_flux(τ) from the leading terms.

    W_flux(τ) = -ζ Σ (M·q̃) N_q̃ Li₂(e^{2πiτ(q̃·p)})

    At large Im(τ), Li₂(x) ≈ x for small x.
    """
    zeta = mpf(1) / (mpf(2)**mpf('1.5') * mp.pi**mpf('2.5'))

    W = mpf(0)
    for t in terms[:n_terms]:
        q_p = mpf(str(t['q_dot_p']))
        eff = mpf(str(t['eff_coeff']))

        # Li₂(e^{2πiτ·(q·p)})
        arg = exp(2 * mp.pi * mp.j * tau * q_p)
        li2 = polylog(2, arg)

        W += eff * li2

    return -zeta * W


def W_flux_approx(terms, tau, n_terms=10):
    """
    Approximate W_flux using leading exponentials (valid at large Im(τ)).

    W ≈ -ζ Σ (M·q̃) N_q̃ e^{2πiτ(q̃·p)}
    """
    zeta = mpf(1) / (mpf(2)**mpf('1.5') * mp.pi**mpf('2.5'))

    W = mpf(0)
    for t in terms[:n_terms]:
        q_p = mpf(str(t['q_dot_p']))
        eff = mpf(str(t['eff_coeff']))

        arg = exp(2 * mp.pi * mp.j * tau * q_p)
        W += eff * arg

    return -zeta * W


def solve_fterm(terms):
    """
    Solve the F-term equation ∂W/∂τ = 0 for τ.

    For 2-term racetrack with exponents α, β and coefficients A, B:
    W = ζ(A e^{2πiτα} + B e^{2πiτβ})
    ∂W/∂τ = 2πiζ(Aα e^{2πiτα} + Bβ e^{2πiτβ}) = 0

    Solution: e^{2πiτ(β-α)} = -Aα/(Bβ)
    """
    if len(terms) < 2:
        raise ValueError("Need at least 2 terms for racetrack")

    # Leading two terms
    t1, t2 = terms[0], terms[1]

    alpha = mpf(str(t1['q_dot_p']))
    beta = mpf(str(t2['q_dot_p']))
    A = mpf(str(t1['eff_coeff']))
    B = mpf(str(t2['eff_coeff']))

    print(f"\nTwo-term racetrack:")
    print(f"  α = {float(alpha):.6f} (q₁·p)")
    print(f"  β = {float(beta):.6f} (q₂·p)")
    print(f"  A = {float(A)} (eff coeff 1)")
    print(f"  B = {float(B)} (eff coeff 2)")

    # From ∂W/∂τ = 0:
    # Aα e^{2πiτα} + Bβ e^{2πiτβ} = 0
    # e^{2πiτ(β-α)} = -Aα/(Bβ)

    ratio = -A * alpha / (B * beta)
    print(f"  -Aα/(Bβ) = {float(ratio)}")

    # For real τ = i·y (pure imaginary), we need ratio > 0
    # Then: e^{-2πy(β-α)} = ratio
    # y = -ln(ratio) / (2π(β-α))

    if ratio <= 0:
        print("  Warning: ratio <= 0, no real solution for Im(τ)")
        # Try numerical solution
        return None

    delta = beta - alpha
    y = -log(ratio) / (2 * mp.pi * delta)

    print(f"  β - α = {float(delta)}")
    print(f"  Im(τ) = {float(y)}")

    # g_s = 1/Im(τ)
    g_s = 1 / y
    print(f"  g_s = 1/Im(τ) = {float(g_s)}")

    # Now compute W₀ = |W(τ_vev)|
    tau_vev = mp.j * y
    W_vev = W_flux_approx(terms, tau_vev, n_terms=2)
    W0 = abs(W_vev)

    print(f"  W₀ = |W(τ)| = {float(W0):.6e}")

    return float(y), float(g_s), float(W0)


def main():
    print("=" * 70)
    print("Deriving Racetrack from GV Invariants")
    print("=" * 70)

    # Load data
    K, M, p_expected = load_mcallister_data()
    curves_9d, gv = load_curves_and_gv()

    # Setup CYTools
    print("\nSetting up CYTools...")
    cy = setup_cytools()

    # Get intersection numbers and verify p
    kappa_result = cy.intersection_numbers(in_basis=True)
    h11 = cy.h11()

    # Convert to 3D array
    kappa = np.zeros((h11, h11, h11))
    for row in kappa_result:
        i, j, k, val = int(row[0]), int(row[1]), int(row[2]), row[3]
        for perm in [(i,j,k), (i,k,j), (j,i,k), (j,k,i), (k,i,j), (k,j,i)]:
            kappa[perm] = val

    # Compute p = N⁻¹K
    N = np.zeros((h11, h11))
    for a in range(h11):
        for b in range(h11):
            for c in range(h11):
                N[a, b] += kappa[a, b, c] * M[c]

    p = np.linalg.solve(N, K)
    print(f"\nFlat direction p = {p}")
    print(f"Expected:        {p_expected}")

    if not np.allclose(p, p_expected, rtol=1e-6):
        print("WARNING: p does not match expected!")

    # Project curves to 4D basis
    print("\n" + "=" * 70)
    print("Projecting curves to 4D basis")
    print("=" * 70)
    curves_4d = project_curves_to_basis(curves_9d, cy)

    # Compute q·p and M·q
    q_dot_p = compute_q_dot_p(curves_4d, p)
    M_dot_q = compute_M_dot_q(curves_4d, M)

    print(f"\nq·p range: [{q_dot_p.min():.4f}, {q_dot_p.max():.4f}]")
    print(f"M·q range: [{M_dot_q.min()}, {M_dot_q.max()}]")

    # Identify leading terms
    print("\n" + "=" * 70)
    print("Identifying leading racetrack terms")
    print("=" * 70)

    terms = identify_leading_terms(q_dot_p, M_dot_q, gv, cutoff=1.0)
    print(f"Found {len(terms)} individual curve contributions with 0 < q·p < 1")

    print("\nTop 10 individual terms by q·p:")
    print(f"{'idx':>5} {'q·p':>12} {'M·q':>8} {'N_q':>12} {'eff_coeff':>12}")
    print("-" * 55)
    for t in terms[:10]:
        print(f"{t['idx']:>5} {t['q_dot_p']:>12.6f} {t['M_dot_q']:>8} {t['N_q']:>12} {t['eff_coeff']:>12}")

    # Group terms by exponent
    print("\n" + "=" * 70)
    print("Grouping terms by exponent (summing coefficients)")
    print("=" * 70)

    grouped = group_terms_by_exponent(terms)
    print(f"Found {len(grouped)} distinct exponents")

    print("\nGrouped terms:")
    print(f"{'q·p':>12} {'net_coeff':>12} {'#curves':>8}")
    print("-" * 35)
    for g in grouped[:10]:
        print(f"{g['q_dot_p']:>12.6f} {g['eff_coeff']:>12} {g['count']:>8}")

    # Check for expected exponents 32/110 and 33/110
    exp1 = 32/110
    exp2 = 33/110
    print(f"\nExpected exponents from paper:")
    print(f"  32/110 = {exp1:.6f}")
    print(f"  33/110 = {exp2:.6f}")
    print(f"Expected coefficients: -1 (at 32/110), +512 (at 33/110)")

    # Find closest groups
    for g in grouped[:5]:
        if abs(g['q_dot_p'] - exp1) < 0.001:
            print(f"\nAt q·p ≈ 32/110: net_coeff = {g['eff_coeff']}, from {g['count']} curves")
            for c in g['curves']:
                print(f"    curve {c['idx']}: M·q={c['M_dot_q']}, N_q={c['N_q']}, contrib={c['eff_coeff']}")
        if abs(g['q_dot_p'] - exp2) < 0.001:
            print(f"\nAt q·p ≈ 33/110: net_coeff = {g['eff_coeff']}, from {g['count']} curves")
            for c in g['curves']:
                print(f"    curve {c['idx']}: M·q={c['M_dot_q']}, N_q={c['N_q']}, contrib={c['eff_coeff']}")

    # Solve F-term using grouped terms
    print("\n" + "=" * 70)
    print("Solving F-term equation (using grouped terms)")
    print("=" * 70)

    result = solve_fterm(grouped)

    if result:
        Im_tau, g_s, W0 = result

        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"Im(τ) = {Im_tau:.6f}")
        print(f"g_s   = {g_s:.8f}")
        print(f"W₀    = {W0:.6e}")

        # Expected values
        g_s_expected = 0.00911134
        W0_expected = 2.30012e-90

        print(f"\nExpected from McAllister:")
        print(f"g_s   = {g_s_expected}")
        print(f"W₀    = {W0_expected:.6e}")

        print(f"\nRatios:")
        print(f"g_s_computed / g_s_expected = {g_s / g_s_expected:.6f}")
        if W0 > 0:
            print(f"log10(W₀_computed) = {np.log10(W0):.2f}")
            print(f"log10(W₀_expected) = {np.log10(W0_expected):.2f}")


if __name__ == "__main__":
    main()
