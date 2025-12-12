# Primal Polytope Triangulation (h11=214)

## Problem

McAllister's primal polytope (4-214-647) has h11=214. To compute V_string, we need the correct triangulation which determines the intersection numbers κ_ijk.

## Solution: Use heights.dat

CYTools can compute triangulations from "heights" - values that determine how the polytope is lifted before projection.

```python
from cytools import Polytope

poly = Polytope(points)
tri = poly.triangulate(heights=heights)  # Use McAllister's heights.dat
```

## Key Findings

| Property | Default | With heights.dat |
|----------|---------|------------------|
| Simplices | 1104 | 1011 |
| Common simplices | 47 | 47 |
| h11 | 214 | 214 |
| h21 | 4 | 4 |

**These are completely different triangulations** - only 47 of ~1000 simplices are shared. They represent different "phases" of the Kähler cone.

## Data File Correspondence

- `points.dat`: 294 lattice points
- `heights.dat`: 219 values

CYTools uses 219 points for triangulation (excludes points interior to facets). This matches heights.dat exactly.

## Code

```python
import numpy as np
from cytools import Polytope

# Load data
lines = open("points.dat").read().strip().split('\n')
points = np.array([[int(x) for x in line.split(',')] for line in lines])

heights = np.array([float(x) for x in open("heights.dat").read().strip().split(',')])

# Create triangulation with McAllister's heights
poly = Polytope(points)
tri = poly.triangulate(heights=heights)  # 1011 simplices
cy = tri.get_cy()  # h11=214, h21=4
```

## Remaining Issue

Even with the correct triangulation, V_string = (1/6) κ_ijk t^i t^j t^k gives 17901 instead of 4711.

This is a **separate problem** - likely basis indexing mismatch between:
- McAllister's `basis.dat` (214 divisor indices)
- McAllister's `kahler_param.dat` (214 t values)
- CYTools divisor basis ordering

The triangulation itself is correct.
