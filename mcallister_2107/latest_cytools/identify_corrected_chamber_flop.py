#!/usr/bin/env python3
"""
Identify the flop(s) relating McAllister's input chamber (heights.dat) to the
corrected chamber (corrected_heights.dat) for 4-214-647, and verify the
intersection-number flop transformation kappa' = kappa - sum_C n_C q^3.

Conventions: points.dat rows are the full 294-point list in the paper-data
(CYTools 2021) ordering. heights.dat / corrected_heights.dat have 219 entries
covering the points NOT interior to facets, in file order. small_curves.dat
columns are the 219 not-interior points in file order (col 0 = origin).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "vendor/cytools_latest/src"))

import numpy as np
from cytools import Polytope

DATA = Path(
    "/Users/ndbroadbent/code/string_theory/resources/small_cc_2107.09064_source"
    "/anc/paper_data/4-214-647"
)
OUT = Path("/tmp/flop_identification_4-214-647.json")


def load_csv_int(path):
    return np.loadtxt(path, delimiter=",", dtype=np.int64, ndmin=2)


def load_csv_float_line(path):
    return np.loadtxt(path, delimiter=",", dtype=np.float64, ndmin=1)


def main():
    file_points = load_csv_int(DATA / "points.dat")  # 294 x 4, row 0 = origin
    h_input = load_csv_float_line(DATA / "heights.dat")
    h_corr = load_csv_float_line(DATA / "corrected_heights.dat")
    n_pts = file_points.shape[0]

    poly = Polytope(file_points)
    labels_file = list(poly.points_to_labels(file_points))
    not_facet = set(poly.labels_not_facet)
    file_rows_not_facet = [i for i, lab in enumerate(labels_file) if lab in not_facet]
    print(f"points: total={n_pts} not_interior_to_facets={len(file_rows_not_facet)}")
    assert len(file_rows_not_facet) == h_input.shape[0] == h_corr.shape[0]
    # Verify the not-facet rows are the first 219 rows of the file (expected
    # from CYTools 2021 ordering); if not, the positional mapping below still
    # holds since heights follow file order restricted to not-facet rows.
    print(f"not-facet rows are first {len(file_rows_not_facet)}: "
          f"{file_rows_not_facet == list(range(len(file_rows_not_facet)))}")

    # Triangulation point labels are sorted(labels_not_facet)
    triang_labels = sorted(not_facet)
    label_to_height_pos = {
        labels_file[row]: k for k, row in enumerate(file_rows_not_facet)
    }
    # file row for each label (for reporting and curve columns)
    label_to_file_row = {labels_file[row]: row for row in range(n_pts)}
    # height-position (== small_curves column) for each label
    label_to_col = label_to_height_pos

    def heights_for(h_file):
        return np.array([h_file[label_to_height_pos[lab]] for lab in triang_labels])

    tris = {}
    for name, h in (("input", h_input), ("corrected", h_corr)):
        tri = poly.triangulate(heights=heights_for(h), verbosity=0)
        tris[name] = tri
        print(
            f"[{name}] simplices={len(tri.simplices())} fine={tri.is_fine()} "
            f"star={tri.is_star()} regular={tri.is_regular()}"
        )

    s_input = {frozenset(int(x) for x in s) for s in tris["input"].simplices().tolist()}
    s_corr = {frozenset(int(x) for x in s) for s in tris["corrected"].simplices().tolist()}
    removed = s_input - s_corr
    added = s_corr - s_input
    print(
        f"common={len(s_input & s_corr)} removed(input-only)={len(removed)} "
        f"added(corrected-only)={len(added)}"
    )

    # Group changed simplices into connected flip regions (shared facet = 4 pts)
    changed = [set(s) for s in list(removed) + list(added)]
    regions: list[set] = []
    for s in changed:
        overlapping = [r for r in regions if len(r & s) >= 4]
        merged = set(s)
        for r in overlapping:
            merged |= r
            regions.remove(r)
        regions.append(merged)

    print(f"flip regions: {len(regions)}")
    label_origin = poly.label_origin
    pts_by_label = {lab: poly.points(which=[lab])[0] for lab in triang_labels}

    small_curves = load_csv_int(DATA / "small_curves.dat")
    small_gv = load_csv_int(DATA / "small_curves_gv.dat").ravel()
    print(f"small_curves: {small_curves.shape}, gv: {small_gv.shape}")
    n_cols = small_curves.shape[1]

    region_reports = []
    for region in regions:
        support = sorted(region)
        support_no_origin = [lab for lab in support if lab != label_origin]
        pts = np.array([pts_by_label[lab] for lab in support_no_origin])
        hom = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.int64)])
        u, s, vh = np.linalg.svd(hom.T.astype(np.float64))
        rank = int(np.sum(s > 1e-9))
        null_dim = pts.shape[0] - rank
        report = {
            "support_labels": support,
            "support_file_rows": [label_to_file_row[lab] for lab in support],
            "origin_in_support": label_origin in region,
            "n_support_points": len(support_no_origin),
            "nullspace_dim": int(null_dim),
            "removed_simplices_in_region": sum(
                1 for s_ in removed if set(s_) <= region
            ),
            "added_simplices_in_region": sum(1 for s_ in added if set(s_) <= region),
        }
        if null_dim == 1:
            v = vh[-1]
            v = v / np.min(np.abs(v[np.abs(v) > 1e-9]))
            v_int = np.round(v).astype(np.int64)
            if np.all(np.abs(v - v_int) < 1e-6) and np.all((hom.T @ v_int) == 0):
                full = np.zeros(n_cols, dtype=np.int64)
                for lab, coeff in zip(support_no_origin, v_int):
                    full[label_to_col[lab]] = int(coeff)
                full[label_to_col[label_origin]] = -int(np.sum(v_int))
                matches = []
                for sign in (1, -1):
                    cand = sign * full
                    hit = np.where((small_curves == cand).all(axis=1))[0]
                    for hidx in hit:
                        matches.append(
                            {
                                "sign": int(sign),
                                "small_curves_row": int(hidx),
                                "gv": int(small_gv[hidx]),
                            }
                        )
                report["relation_sparse_cols"] = {
                    str(int(i)): int(c) for i, c in enumerate(full) if c != 0
                }
                report["small_curves_matches"] = matches
        region_reports.append(report)
        print(json.dumps(report))

    # Intersection-number flop check
    print("computing intersection numbers (input chamber)...")
    cy1 = tris["input"].get_cy()
    k1 = cy1.intersection_numbers(in_basis=False, format="dok")
    print("computing intersection numbers (corrected chamber)...")
    cy2 = tris["corrected"].get_cy()
    k2 = cy2.intersection_numbers(in_basis=False, format="dok")

    sample_key = next(iter(k1))
    print(f"kappa key sample: {sample_key} (indices are triangulation point labels)")

    def normalize(k):
        out = {}
        for idx, val in k.items():
            key = tuple(sorted(int(i) for i in idx))
            v = float(val)
            if abs(v) > 1e-12:
                out[key] = v
        return out

    k1n, k2n = normalize(k1), normalize(k2)
    diff = {}
    for key in set(k1n) | set(k2n):
        d = k1n.get(key, 0.0) - k2n.get(key, 0.0)
        if abs(d) > 1e-9:
            diff[key] = d
    diff_support = sorted({i for key in diff for i in key})
    print(f"kappa nonzeros: input={len(k1n)} corrected={len(k2n)} diff_entries={len(diff)}")
    print(f"kappa-diff support (labels): {diff_support}")

    # Check diff == sum over flip regions of n_C * q_a q_b q_c (prime divisors)
    # using each region's relation restricted to non-origin labels.
    checks = []
    for report in region_reports:
        rel = report.get("relation_sparse_cols")
        if rel is None:
            continue
        col_to_label = {v: k for k, v in label_to_col.items()}
        q_by_label = {
            col_to_label[int(c)]: coeff
            for c, coeff in rel.items()
            if col_to_label[int(c)] != label_origin
        }
        for match in report.get("small_curves_matches", []) or [
            {"sign": 1, "gv": None}
        ]:
            n_c = match["gv"]
            if n_c is None:
                continue
            sign = match["sign"]
            pred = {}
            labs = sorted(q_by_label)
            for i, la in enumerate(labs):
                for lb in labs[i:]:
                    for lc in labs[labs.index(lb):]:
                        key = tuple(sorted((la, lb, lc)))
                        # multiplicity handling: iterate unordered triples once
            # build unordered triples properly
            pred = {}
            for ii in range(len(labs)):
                for jj in range(ii, len(labs)):
                    for kk in range(jj, len(labs)):
                        key = (labs[ii], labs[jj], labs[kk])
                        qprod = (
                            q_by_label[labs[ii]]
                            * q_by_label[labs[jj]]
                            * q_by_label[labs[kk]]
                        )
                        # sign convention: q effective in corrected chamber is
                        # sign*full; kappa_input - kappa_corr = n * q^3 with q
                        # the class shrinking in input chamber (effective in
                        # corrected); try both orientations.
                        pred[key] = n_c * qprod
            # restrict diff to non-origin keys for comparison
            diff_no0 = {
                k: v for k, v in diff.items() if label_origin not in k
            }
            max_err_plus = max(
                (
                    abs(diff_no0.get(k, 0.0) - pred.get(k, 0.0))
                    for k in set(diff_no0) | set(pred)
                ),
                default=0.0,
            )
            max_err_minus = max(
                (
                    abs(diff_no0.get(k, 0.0) + pred.get(k, 0.0))
                    for k in set(diff_no0) | set(pred)
                ),
                default=0.0,
            )
            checks.append(
                {
                    "small_curves_row": match.get("small_curves_row"),
                    "n_c": n_c,
                    "match_sign": sign,
                    "max_err_kappa_diff_eq_plus_nq3": max_err_plus,
                    "max_err_kappa_diff_eq_minus_nq3": max_err_minus,
                }
            )
            print(json.dumps(checks[-1]))

    fixture = {
        "n_points": int(n_pts),
        "triang_labels": [int(x) for x in triang_labels],
        "label_origin": int(label_origin),
        "simplices": {
            "input": len(s_input),
            "corrected": len(s_corr),
            "removed": len(removed),
            "added": len(added),
        },
        "flip_regions": region_reports,
        "kappa_diff_nonorigin_entries": {
            ",".join(map(str, k)): v
            for k, v in sorted(diff.items())
            if int(poly.label_origin) not in k
        },
        "kappa_diff_origin_entries": {
            ",".join(map(str, k)): v
            for k, v in sorted(diff.items())
            if int(poly.label_origin) in k
        },
        "flop_checks": checks,
    }
    OUT.write_text(json.dumps(fixture, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
