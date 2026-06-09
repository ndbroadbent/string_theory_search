#!/usr/bin/env python3
"""
Independent verification that McAllister's corrected-chamber data for
4-214-647 (arXiv:2107.09064) is exactly reproduced by first-principles
geometry plus the shipped input-chamber GV data.

Checks performed (all against paper_data/4-214-647, CYTools-latest geometry):

1. FLOP CERTIFICATE: the corrected chamber (corrected_heights.dat FRST) is
   the input chamber (heights.dat FRST) after 10 conifold flops of GV=1
   curves, all rows of small_curves.dat. Verified integer-exactly via
   kappa_input - kappa_corrected == sum_C q_C^{x3} (198 entries, 0 mismatch).
   (Run identify_corrected_chamber_flop.py + decompose_kappa_flop_diff.py to
   regenerate the flop set; it is inlined below for self-containment.)

2. GAMMA (B-field): the O7 divisors are exactly the lattice points with
   parity p mod 2 == sigma, where sigma is the unique common parity of the
   declared c_i=6 so(8) divisors. For this model sigma=(1,0,0,0), giving 51
   O7 divisors: the 49 KKLT-basis stacks plus points 2 and 46.

3. TAU TARGETS: corrected_target_volumes.dat equals
   c_i/c_tau + chi(D_i)/24 - (1/4pi^2) sum_q N_q q_i Li2((-1)^{gamma.q} e^{-2pi q.t})
   at t = corrected_kahler_param.dat, with chi(D) the corrected-chamber Braun
   Euler characteristics (12 chi(O_D) - D^3), integers for all 214 divisors,
   up to the checkpoint's own solver tolerance (~5e-4).

4. VOLUME: corrected_cy_vol.dat = 4711.432499235554 is reproduced to ~1e-8
   by BOTH equivalent evaluations:
   (a) corrected-chamber: kappa' t^3/6 - BBHL + flop-mapped GV sum;
   (b) input-chamber analytic continuation: kappa t^3/6 - BBHL + raw
       small_curves sum with the ten crossed (odd-parity) curves' polylogs
       continued through their walls.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "vendor/cytools_latest/src"))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from scipy.special import spence, zeta
import mpmath

from cytools import Polytope
from compute_chi_divisor import compute_chi_holomorphic

DATA = Path(
    "/Users/ndbroadbent/code/string_theory/resources/small_cc_2107.09064_source"
    "/anc/paper_data/4-214-647"
)
N_COLS = 219

# Input-chamber-effective orientation of the ten flopped classes
# (sparse {file_point_index: coefficient}; origin coefficient is implied 0).
FLOPPED_CLASSES = [
    {2: 1, 33: 1, 37: -1, 178: -1},
    {6: -1, 51: 1, 199: -1, 210: 1},
    {7: -1, 54: 1, 207: -1, 218: 1},
    {22: 1, 102: -1, 115: -1, 144: 1},
    {32: 1, 35: -1, 41: 1, 148: -1},
    {33: 1, 39: -1, 167: -1, 182: 1},
    {48: -1, 188: -1, 191: 1, 205: 1},
    {58: -1, 204: -1, 216: 1, 217: 1},
    {2: 1, 18: 1, 107: -1, 120: -1},
    {15: -1, 81: 1, 90: 1, 127: -1},
]


def li3(args):
    return np.array([float(mpmath.re(mpmath.polylog(3, complex(a)))) for a in args])


def main():
    basis = np.loadtxt(DATA / "basis.dat", delimiter=",", dtype=np.int64, ndmin=1)
    kklt = np.loadtxt(DATA / "kklt_basis.dat", delimiter=",", dtype=np.int64, ndmin=1)
    c_i = np.loadtxt(DATA / "target_volumes.dat", delimiter=",", ndmin=1)
    t_chk = np.loadtxt(DATA / "corrected_kahler_param.dat", delimiter=",", ndmin=1)
    targets = np.loadtxt(DATA / "corrected_target_volumes.dat", delimiter=",", ndmin=1)
    curves = np.loadtxt(DATA / "small_curves.dat", delimiter=",", dtype=np.int64, ndmin=2)
    gv = np.loadtxt(DATA / "small_curves_gv.dat", delimiter=",", dtype=np.int64, ndmin=1)
    v_target = float((DATA / "corrected_cy_vol.dat").read_text().strip())
    g_s = float((DATA / "g_s.dat").read_text().strip())
    w_0 = float((DATA / "W_0.dat").read_text().strip())
    pts = np.loadtxt(DATA / "points.dat", delimiter=",", dtype=np.int64, ndmin=2)
    c_tau = 2 * np.pi / (g_s * np.log(1 / w_0))

    poly = Polytope(pts)

    # --- gamma from the involution parity
    o7_declared = kklt[c_i == 6]
    parities = {tuple(pts[i] % 2) for i in o7_declared}
    assert len(parities) == 1, f"so(8) divisors span parities {parities}"
    sigma = next(iter(parities))
    assert any(sigma), "trivial involution parity"
    gamma_pts = np.array([i for i in range(1, N_COLS) if tuple(pts[i] % 2) == sigma])
    extra = sorted(set(gamma_pts.tolist()) - set(kklt.tolist()))
    print(f"[gamma] sigma={sigma} O7 divisors={len(gamma_pts)} beyond KKLT basis: {extra}")
    assert len(gamma_pts) == 51 and extra == [2, 46]

    # --- flop map and chamber kappas
    curves_m = curves.copy()
    flop_rows = []
    for q in FLOPPED_CLASSES:
        full = np.zeros(N_COLS, dtype=np.int64)
        for lab, c in q.items():
            full[lab] = c
        full[0] = -sum(q.values())
        hits = np.where((curves == full).all(axis=1))[0]
        assert len(hits) == 1, f"flopped class {q} not a unique small_curves row"
        assert gv[hits[0]] == 1
        flop_rows.append(int(hits[0]))
        curves_m[hits[0]] = -curves_m[hits[0]]
    print(f"[flops] rows {sorted(flop_rows)} negated (all GV=1)")

    h_in = np.loadtxt(DATA / "heights.dat", delimiter=",", ndmin=1)
    h_co = np.loadtxt(DATA / "corrected_heights.dat", delimiter=",", ndmin=1)
    tri_in = poly.triangulate(heights=h_in, verbosity=0)
    tri_co = poly.triangulate(heights=h_co, verbosity=0)
    cy_in, cy_co = tri_in.get_cy(), tri_co.get_cy()

    def normalize(k):
        out = {}
        for idx, val in k.items():
            key = tuple(sorted(int(i) for i in idx))
            if abs(float(val)) > 1e-12:
                out[key] = float(val)
        return out

    k_in = normalize(cy_in.intersection_numbers(in_basis=False, format="dok"))
    k_co = normalize(cy_co.intersection_numbers(in_basis=False, format="dok"))
    pred = {}
    for q in FLOPPED_CLASSES:
        labs = sorted(q)
        for i in range(len(labs)):
            for j in range(i, len(labs)):
                for k in range(j, len(labs)):
                    key = (labs[i], labs[j], labs[k])
                    pred[key] = pred.get(key, 0) + q[labs[i]] * q[labs[j]] * q[labs[k]]
    pred = {k: v for k, v in pred.items() if v != 0}
    mism = sum(
        1
        for key in set(k_in) | set(k_co) | set(pred)
        if abs(k_in.get(key, 0.0) - k_co.get(key, 0.0) - pred.get(key, 0)) > 1e-9
    )
    print(f"[flop certificate] kappa_in - kappa_corr == sum q^3: mismatches={mism}")
    assert mism == 0

    # --- shared evaluation pieces
    def evaluate(cset, gvs):
        v = cset[:, basis] @ t_chk
        par = cset[:, gamma_pts].sum(axis=1) % 2
        arg = np.where(par == 1, -1.0, 1.0) * np.exp(-2 * np.pi * v)
        assert (arg <= 1.0).all(), "even-parity branch cut hit"
        d2 = spence(1.0 - arg)
        d3 = li3(arg)
        tau_corr = (gvs * d2) @ cset[:, kklt] / (4 * np.pi**2)
        vol_corr = float(np.sum(gvs * (d3 + 2 * np.pi * v * d2))) / (2 * (2 * np.pi) ** 3)
        return tau_corr, vol_corr

    def v_classical(cy):
        cy.set_divisor_basis([int(b) for b in basis])
        kap = cy.intersection_numbers(in_basis=True, format="dok")
        out = 0.0
        for (a, b, c), val in kap.items():
            mult = {1: 1, 2: 3, 3: 6}[len({a, b, c})]
            out += mult * float(val) * t_chk[a] * t_chk[b] * t_chk[c]
        return out / 6.0

    bbhl = zeta(3) * 420 / (4 * (2 * np.pi) ** 3)

    # --- tau targets with corrected-chamber Braun chi
    chi_o = compute_chi_holomorphic(poly)
    d3_co = {
        int(a): float(val)
        for (a, b, c), val in cy_co.intersection_numbers(in_basis=False, format="dok").items()
        if a == b == c
    }
    chi_corr = np.array([12 * chi_o[int(p)]["chi_O"] - d3_co.get(int(p), 0.0) for p in kklt])
    tau_corr_m, vol_corr_m = evaluate(curves_m, gv.astype(float))
    resid_tau = c_i / c_tau + chi_corr / 24.0 - tau_corr_m - targets
    print(
        f"[tau targets] max_abs residual = {np.max(np.abs(resid_tau)):.3e} "
        f"(checkpoint solver tolerance ~5.6e-4)"
    )
    assert np.max(np.abs(resid_tau)) < 1e-3

    # --- volume, both evaluations
    v_flopped = v_classical(cy_co) - bbhl + vol_corr_m
    tau_corr_i, vol_corr_i = evaluate(curves, gv.astype(float))
    v_continued = v_classical(cy_in) - bbhl + vol_corr_i
    print(f"[volume] corrected-chamber evaluation: {v_flopped:.12f} (target {v_target})")
    print(f"[volume] input-chamber continuation:   {v_continued:.12f} (target {v_target})")
    print(
        f"[volume] residuals: flopped={v_flopped - v_target:+.3e} "
        f"continued={v_continued - v_target:+.3e}"
    )
    assert abs(v_flopped - v_target) < 1e-6
    assert abs(v_continued - v_target) < 1e-6
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
