#!/usr/bin/env python3
"""
Globally decompose kappa_input - kappa_corrected as an integer combination
sum_C n_C * (q_C)^{\otimes 3} over circuit candidates, certifying the full
flop set relating the two chambers.

Iterative strategy: fit candidates, subtract exact integer part, mine new
circuit candidates on the remainder support, repeat.

Reads /tmp/flop_identification_4-214-647.json; writes
/tmp/flop_decomposition_4-214-647.json.
"""

import itertools
import json
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "vendor/cytools_latest/src"))

import numpy as np
from cytools import Polytope

DATA = Path(
    "/Users/ndbroadbent/code/string_theory/resources/small_cc_2107.09064_source"
    "/anc/paper_data/4-214-647"
)
FIX = Path("/tmp/flop_identification_4-214-647.json")
OUT = Path("/tmp/flop_decomposition_4-214-647.json")


def integer_circuit(points):
    pts = np.asarray(points, dtype=np.int64)
    hom = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.int64)]).T
    u, s, vh = np.linalg.svd(hom.astype(np.float64))
    rank = int(np.sum(s > 1e-9))
    null_dim = pts.shape[0] - rank
    if null_dim != 1:
        return None
    v = vh[-1]
    nz = np.abs(v) > 1e-9
    if not nz.any():
        return None
    v = v / np.min(np.abs(v[nz]))
    v_int = np.round(v).astype(np.int64)
    if not np.all(np.abs(v - v_int) < 1e-6):
        return None
    if not np.all(hom @ v_int == 0):
        return None
    g = np.gcd.reduce(np.abs(v_int[v_int != 0]))
    v_int //= max(g, 1)
    first = v_int[v_int != 0][0]
    if first < 0:
        v_int = -v_int
    return v_int


def cube_entries(q_sparse):
    ent = {}
    labs = sorted(q_sparse)
    for i in range(len(labs)):
        for j in range(i, len(labs)):
            for k in range(j, len(labs)):
                key = tuple(sorted((labs[i], labs[j], labs[k])))
                ent[key] = q_sparse[labs[i]] * q_sparse[labs[j]] * q_sparse[labs[k]]
    return ent


def mine_candidates(coords, support, max_size=6, max_subsets=2_000_000):
    """All primitive circuits among subsets of `support` of size 4..max_size."""
    out = {}
    support = sorted(support)
    count = 0
    for size in range(4, max_size + 1):
        for subset in itertools.combinations(support, size):
            count += 1
            if count > max_subsets:
                print(f"  (subset budget hit at size {size})")
                return out
            rel = integer_circuit([coords[lab] for lab in subset])
            if rel is None:
                continue
            if np.count_nonzero(rel) != size:
                continue  # circuit is on a proper subset; found at smaller size
            q = tuple(sorted((lab, int(c)) for lab, c in zip(subset, rel)))
            out[q] = True
    return out


def main():
    fix = json.loads(FIX.read_text())
    file_points = np.loadtxt(DATA / "points.dat", delimiter=",", dtype=np.int64, ndmin=2)
    poly = Polytope(file_points)
    labels_file = list(poly.points_to_labels(file_points))
    coords = {lab: file_points[i] for i, lab in enumerate(labels_file)}
    origin = fix["label_origin"]

    kdiff = {}
    for key, val in fix["kappa_diff_nonorigin_entries"].items():
        idx = tuple(int(x) for x in key.split(","))
        v = Fraction(val).limit_denominator(10**6)
        assert v.denominator == 1, (key, val)
        kdiff[idx] = int(v)

    # initial candidates from per-region supports
    candidates = {}
    for region in fix["flip_regions"]:
        support = [lab for lab in region["support_labels"] if lab != origin]
        candidates.update(mine_candidates(coords, support))
    print(f"initial circuit candidates: {len(candidates)}")

    found = {}  # q_tuple -> n
    remainder = dict(kdiff)

    for iteration in range(6):
        cand_list = sorted(candidates)
        cubes = [cube_entries(dict(q)) for q in cand_list]
        all_keys = sorted(set(remainder) | {k for c in cubes for k in c})
        key_pos = {k: i for i, k in enumerate(all_keys)}
        a_mat = np.zeros((len(all_keys), len(cand_list)))
        for c_idx, ent in enumerate(cubes):
            for k, v in ent.items():
                a_mat[key_pos[k], c_idx] = v
        b_vec = np.zeros(len(all_keys))
        for k, v in remainder.items():
            b_vec[key_pos[k]] = v

        sol, _, rank, _ = np.linalg.lstsq(a_mat, b_vec, rcond=None)
        sol_int = np.round(sol).astype(np.int64)
        resid = b_vec - a_mat @ sol_int
        max_resid = float(np.max(np.abs(resid))) if len(resid) else 0.0
        print(
            f"iter {iteration}: candidates={len(cand_list)} rank={rank} "
            f"max_resid={max_resid}"
        )

        # accept the integer part, update remainder
        for i, n in enumerate(sol_int):
            if n != 0:
                q = cand_list[i]
                found[q] = found.get(q, 0) + int(n)
        new_remainder = {}
        for pos, k in enumerate(all_keys):
            v = resid[pos]
            vi = int(round(v))
            assert abs(v - vi) < 1e-6
            if vi != 0:
                new_remainder[k] = vi
        remainder = new_remainder
        print(f"  remainder entries: {len(remainder)}")
        if not remainder:
            break

        rem_support = sorted({i for k in remainder for i in k})
        print(f"  remainder support: {rem_support}")
        before = len(candidates)
        candidates.update(mine_candidates(coords, rem_support, max_size=6))
        print(f"  candidates: {before} -> {len(candidates)}")
        if len(candidates) == before:
            print("  no new candidates; stopping")
            break

    exact = not remainder
    found = {q: n for q, n in found.items() if n != 0}
    print(f"EXACT integer decomposition: {exact}; flop terms: {len(found)}")

    small_curves = np.loadtxt(DATA / "small_curves.dat", delimiter=",", dtype=np.int64, ndmin=2)
    small_gv = np.loadtxt(DATA / "small_curves_gv.dat", delimiter=",", dtype=np.int64, ndmin=1)
    label_to_col = {labels_file[row]: row for row in range(small_curves.shape[1])}

    flops = []
    for q, n in sorted(found.items()):
        qd = dict(q)
        full = np.zeros(small_curves.shape[1], dtype=np.int64)
        for lab, c in qd.items():
            full[label_to_col[lab]] = c
        full[label_to_col[origin]] = -sum(qd.values())
        m = []
        for sign in (1, -1):
            hits = np.where((small_curves == sign * full).all(axis=1))[0]
            for h in hits:
                m.append({"sign": sign, "row": int(h), "gv": int(small_gv[h])})
        flops.append(
            {
                "q_sparse": {str(lab): int(c) for lab, c in q},
                "origin_coeff": int(-sum(qd.values())),
                "n": n,
                "small_curves_matches": m,
            }
        )
        print(json.dumps(flops[-1]))

    OUT.write_text(
        json.dumps(
            {"exact": exact, "remainder": {",".join(map(str, k)): v for k, v in remainder.items()}, "flops": flops},
            indent=1,
        )
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
