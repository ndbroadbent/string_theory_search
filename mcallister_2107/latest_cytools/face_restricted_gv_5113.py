#!/usr/bin/env python3
"""
Face-restricted GV analysis for the three exotic origin-circuit curves of
5-113-4627 (the only classes in the example set whose GVs resist both toric
formulas and degree-bounded cygv).

The classes (ambient coords over the 118 not-facet points, col 0 = origin):
  A = {0:-1, 8:1, 9:-2, 46:1, 117:1}    their GV = -2, grading degree 2
  B = {0:-1, 8:1, 44:2, 46:-3, 117:1}   their GV =  3, grading degree 10
  C = {0:-1, 3:1, 43:-3, 45:2, 113:1}   their GV =  3, grading degree 24

FINDINGS (2026-06-10):

1. All three lie on LOW-DIMENSIONAL faces of the CYTools toric Mori-cone cap
   (727 extremal rays, in-basis): A spans a single extremal ray, B a 6-ray
   face, C a 12-ray face (per-ray strict-separation LPs below).

2. Face-restricted cygv (compute_gvs with mcap_generators = face rays +
   face lattice points and the full-cone grading vector) REPRODUCES every
   known sub-cutoff class inside B's face exactly (+1, +1, and A's -2 at
   degree 2) and gives GV(A) = -2 — A's value is domain-independent anyway
   (degree-2 minimal: no decompositions exist), so A is SOLVED and verified.

3. But the same face computation yields nothing (i.e. zero) for B and C,
   while McAllister's file says 3. Since all of their published sub-cutoff
   classes below B lie INSIDE the cap-face (checked), and within-cap
   decomposition channels cannot leave a true face, the missing channels
   must involve effective classes OUTSIDE the toric cap cone altogether.
   No cap-based CYTools/cygv call in this phase can see them: this is
   exactly the situation the paper resolves by "finding an appropriate
   phase where a given face of M_inf(X) is also a face of M(X)" — i.e. the
   computation must be done in a different (flopped) chamber and mapped
   back. That remains the open implementation task for 5-113-4627.

Note for the Cyrus side: Cyrus's raw ambient cap list (137k unreduced
relations) makes naive strict-separation extremality LPs infeasible
(suspected antipodal/oriented duplicates); an extremal-ray reduction of the
projected cone (CYTools gets 727) is prerequisite for in-Rust face
certificates.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "vendor/cytools_latest/src"))

import numpy as np
from scipy.optimize import linprog
from cytools import Polytope

DATA = Path(
    "/Users/ndbroadbent/code/string_theory/resources/small_cc_2107.09064_source"
    "/anc/paper_data/5-113-4627-main"
)

MISSING = [
    ("A", {0: -1, 8: 1, 9: -2, 46: 1, 117: 1}, -2),
    ("B", {0: -1, 8: 1, 44: 2, 46: -3, 117: 1}, 3),
    ("C", {0: -1, 3: 1, 43: -3, 45: 2, 113: 1}, 3),
]


def main():
    pts = np.loadtxt(DATA / "points.dat", delimiter=",", dtype=np.int64, ndmin=2)
    h = np.loadtxt(DATA / "heights.dat", delimiter=",", ndmin=1)
    poly = Polytope(pts)
    tri = poly.triangulate(heights=h, verbosity=0)
    cy = tri.get_cy()
    mc = cy.toric_mori_cone(in_basis=True)
    rays = np.array(mc.rays())
    dbasis = list(cy.divisor_basis())
    grading = np.array(mc.find_grading_vector())
    print(f"mori cap extremal rays (in basis): {rays.shape}")

    def minimal_face(q):
        keep = []
        for i in range(rays.shape[0]):
            a_ub = np.vstack([-rays, -rays[i]])
            b_ub = np.append(np.zeros(rays.shape[0]), -1.0)
            res = linprog(
                np.zeros(rays.shape[1]),
                A_ub=a_ub,
                b_ub=b_ub,
                A_eq=q.reshape(1, -1).astype(float),
                b_eq=[0.0],
                bounds=[(None, None)] * rays.shape[1],
                method="highs",
            )
            if not res.success:
                keep.append(i)
        return keep

    for name, sparse, want in MISSING:
        v = np.zeros(118, dtype=np.int64)
        for i, x in sparse.items():
            v[i] = x
        q = np.array([v[d] for d in dbasis], dtype=np.int64)
        face = minimal_face(q)
        gens = rays[face]
        if len(face) > 1:
            from cytools.cone import Cone

            lattice = np.array(Cone(rays=gens).find_lattice_points(min_points=300))
            gens = np.unique(np.vstack([gens, lattice]), axis=0)
            gens = gens[(gens @ grading) > 0]
        qdeg = int(q @ grading)
        t0 = time.time()
        gvs = cy.compute_gvs(
            mcap_generators=gens, grading_vec=grading, max_deg=qdeg + 2
        )
        table = dict(gvs.dok if isinstance(getattr(gvs, "dok", None), dict) else gvs.dok())
        got = table.get(tuple(int(x) for x in q), "ABSENT(=0 in face domain)")
        print(
            f"[{name}] face_rays={len(face)} gens={len(gens)} q_degree={qdeg} "
            f"face-domain GV={got} (their file: {want}) [{time.time()-t0:.1f}s]"
        )


if __name__ == "__main__":
    main()
