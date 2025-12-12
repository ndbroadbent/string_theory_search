#!/usr/bin/env python3
"""Debug W0 computation differences."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "vendor/cytools_latest/src"))

from mpmath import mp, mpf, polylog, pi as mp_pi, exp as mp_exp
mp.dps = 150

from compute_triangulation import load_example_points, load_example_model_choices
from compute_derived_racetrack import load_simplices_list
from compute_gv_invariants import compute_gv_invariants
from compute_basis_transform import load_mcallister_example, compute_T_from_glsm, transform_fluxes
import numpy as np

ZETA = mpf(1) / (mpf(2) ** mpf('1.5') * mp_pi ** mpf('2.5'))

def debug_example(example):
    print(f'\n{"="*60}')
    print(f'{example}')
    print(f'{"="*60}')

    dual_pts = load_example_points(example, which='dual')

    sys.path.insert(0, str(Path(__file__).parent.parent / "vendor/cytools_mcallister_2107"))
    mods = [k for k in list(sys.modules.keys()) if 'cytools' in k]
    for m in mods: del sys.modules[m]
    from cytools import Polytope as P2021
    old_basis = list(P2021(dual_pts).triangulate().get_cy().divisor_basis())

    mods = [k for k in list(sys.modules.keys()) if 'cytools' in k]
    for m in mods: del sys.modules[m]
    sys.path.remove(str(Path(__file__).parent.parent / "vendor/cytools_mcallister_2107"))

    from cytools import Polytope
    poly = Polytope(dual_pts)
    simplices = load_simplices_list(example)
    tri = poly.triangulate(simplices=simplices, check_input_simplices=False)
    cy = tri.get_cy()
    new_basis = list(cy.divisor_basis())

    example_data = load_mcallister_example(example)
    K_old, M_old = example_data['K'], example_data['M']
    if old_basis != new_basis:
        T = compute_T_from_glsm(cy, old_basis, new_basis)
        K_new, M_new = transform_fluxes(K_old, M_old, T)
    else:
        K_new, M_new = K_old, M_old

    kappa_dict = cy.intersection_numbers(in_basis=True)
    h11 = cy.h11()
    kappa = np.zeros((h11, h11, h11))
    for (i,j,k), val in kappa_dict.items():
        for perm in [(i,j,k), (i,k,j), (j,i,k), (j,k,i), (k,i,j), (k,j,i)]:
            kappa[perm] = val
    N = np.einsum('abc,c->ab', kappa, M_new)
    p = np.linalg.solve(N, K_new)

    gv = compute_gv_invariants(cy, min_points=100)
    model = load_example_model_choices(example)
    Im_tau = 1.0 / model['g_s']

    print(f'Im(tau) = {Im_tau:.4f}')
    print(f'Expected W0 = {model["W0"]:.2e}')

    # Compute W sum showing individual terms
    terms = []
    for q_tuple, N_q in gv.items():
        q = np.array(q_tuple)
        qp = float(np.dot(q, p))
        Mq = int(np.dot(M_new, q))
        coeff = Mq * N_q
        if abs(coeff) > 0 and qp > 0:
            arg = mp_exp(-2 * mp_pi * mpf(str(Im_tau)) * mpf(str(qp)))
            term = mpf(str(float(coeff))) * polylog(2, arg)
            terms.append((qp, coeff, float(term)))

    terms.sort(key=lambda x: x[0])
    print(f'\nTop 5 terms (by q.p):')
    for qp, coeff, term_val in terms[:5]:
        print(f'  q.p={qp:.4f}, coeff={coeff:6d}, term={term_val:.2e}')

    W_sum = sum(t[2] for t in terms)
    W0_computed = abs(-float(ZETA) * W_sum)
    print(f'\nSum of all terms: {W_sum:.2e}')
    print(f'Computed W0 = {W0_computed:.2e}')
    print(f'Ratio computed/expected = {W0_computed / model["W0"]:.2f}')


if __name__ == "__main__":
    for example in ['4-214-647', '5-113-4627-main', '5-113-4627-alternative', '5-81-3213', '7-51-13590']:
        debug_example(example)
