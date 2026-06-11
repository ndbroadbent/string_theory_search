#!/usr/bin/env python3
"""Build a cyrus-ga pool (JSONL) from the local KS parquet database.

Usage: make_pool.py <h21_min> <h21_max> <out.jsonl>
Records: {"name": "ks_<h21>_<n>", "h11", "h21", "points": [vertices]}
(vertices suffice: cyrus computes lattice points natively). Sorted by
ascending h21 to match the bandit first-visit order.
"""
import glob, json, sys
import pyarrow.parquet as pq

h21_min, h21_max, out_path = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
counters = {}
records = []
for path in sorted(glob.glob("/data/polytopes/parquet/*.parquet")):
    t = pq.read_table(path, columns=["vertices", "h11", "h12"],
                      filters=[("h12", ">=", h21_min), ("h12", "<=", h21_max)])
    for verts, h11, h21 in zip(t["vertices"].to_pylist(),
                               t["h11"].to_pylist(), t["h12"].to_pylist()):
        n = counters.get(h21, 0)
        counters[h21] = n + 1
        records.append((h21, {"name": "ks_%d_%d" % (h21, n), "h11": h11,
                              "h21": h21, "points": verts}))
    base = path.split("/")[-1]
    print(base + ": total " + str(len(records)), flush=True)
records.sort(key=lambda r: r[0])
with open(out_path, "w") as f:
    for _, rec in records:
        f.write(json.dumps(rec) + "\n")
print("wrote %d -> %s; per h21: %s" % (len(records), out_path, dict(sorted(counters.items()))))
