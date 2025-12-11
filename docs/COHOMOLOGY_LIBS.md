# Libraries for Computing Line Bundle Cohomology

## Overview

Computing divisor cohomology h^i(D, O_D) on toric Calabi-Yau manifolds requires specialized algorithms. This document catalogs available tools.

---

## 1. cohomCalg (C++)

**The reference implementation** for line bundle cohomology on toric varieties.

- **Paper**: arXiv:1003.5217 (Blumenhagen, Jurke, Rahn, Roschy)
- **Download**: http://wwwth.mpp.mpg.de/members/blumenha/cohomcalg/
- **Language**: C++ (no Python bindings)

### Features
- Computes H^i(X, O(D)) for any line bundle D on a toric variety X
- Uses Stanley-Reisner ideal and GLSM charges
- Includes "HodgeDiamond" module for divisor Hodge numbers

### Usage
```bash
# Compile
tar -xzf cohomCalg.tar.gz
cd cohomCalg
make

# Run (example input file)
./cohomCalg example.in
```

### Input Format
```
# Example: P^2 with line bundle O(3)
vertices = [[1,0],[0,1],[-1,-1]]
GLSM = [[1,1,1]]
monomials = [3]
```

### Limitations
- C++ only, no Python wrapper
- Manual input file creation required
- No direct integration with CYTools

---

## 2. SageMath

SageMath has toric geometry modules with some cohomology capabilities.

- **Documentation**: https://doc.sagemath.org/html/en/reference/schemes/sage/schemes/toric/variety.html
- **Language**: Python (SageMath environment)

### Available Methods
```python
from sage.schemes.toric.variety import ToricVariety

# Create toric variety from fan
X = ToricVariety(fan)

# Cohomology ring (NOT line bundle cohomology)
X.cohomology_ring()

# Integration (top degree)
X.integrate(cohomology_class)

# Chern classes
X.Chern_class()
```

### Limitations
- **Does NOT compute H^i(X, O(D))** for arbitrary line bundles
- `cohomology_ring()` gives the ring structure, not dimensions
- No direct cohomCalg equivalent
- Would need to implement rationom counting manually

---

## 3. Altman Database (Precomputed)

For h11 <= 6, use precomputed values instead of computing from scratch.

- **Paper**: arXiv:2111.03078
- **Database**: http://www.rossealtman.com/toriccy/
- **Format**: JSON files with DIVCOHOM field

### Data Files
- `*.triang.json`: Contains `DIVCOHOM` for ALL toric divisors
- `*.invol.json`: Contains `INVOLDIVCOHOM` for involution-related divisors only

### DIVCOHOM Format
```json
{
  "POLYID": 4,
  "H11": 4,
  "DIVCOHOM": "{{1,0,0,2},{1,0,0,12},{1,0,1,21},{1,0,0,8},{1,0,2,29},{1,0,0,9},{1,0,3,38},{1,0,12,97}}"
}
```
Each entry is `{h^0, h^1, h^2, h^{1,1}}`.

### Python Loader
```python
import json
import re

def load_divcohom(json_path):
    """Load DIVCOHOM from Altman triang.json file."""
    with open(json_path) as f:
        records = json.load(f)

    for rec in records:
        if "DIVCOHOM" not in rec:
            continue

        # Parse "{{1,0,0,2},{1,0,0,12},...}"
        s = rec["DIVCOHOM"]
        matches = re.findall(r'\{(\d+),(\d+),(\d+),(\d+)\}', s)
        cohom = [[int(x) for x in m] for m in matches]
        return cohom

    return None
```

### Advantages
- Validated, peer-reviewed data
- Covers all h11 <= 6 polytopes
- Fast lookup vs computation

### Limitations
- Only h11 <= 6 available
- ~5GB for h11=6 data
- No h11 > 6 data (must compute)

---

## 4. CYTools

CYTools does NOT compute line bundle cohomology directly.

### What CYTools Provides
```python
from cytools import Polytope

poly = Polytope(points)
tri = poly.triangulate()
cy = tri.get_cy()

# Stanley-Reisner ideal (input for cohomCalg)
sr = tri.sr_ideal()

# GLSM charges (input for cohomCalg)
glsm = poly.glsm_linear_relations()

# Hodge numbers (CY, not divisors)
h11, h21 = cy.h11(), cy.h21()

# Intersection numbers
kappa = cy.intersection_numbers()
```

### What CYTools Does NOT Provide
- `cy.line_bundle_cohomology()` - does not exist
- `cy.divisor_hodge_numbers()` - does not exist
- No cohomCalg integration

---

## 5. Implementation Path for h11 > 6

For polytopes with h11 > 6 (like McAllister's h11=214), options are:

### Option A: Wrap cohomCalg
1. Download and compile cohomCalg
2. Write Python subprocess wrapper
3. Generate input files from CYTools data
4. Parse output

```python
import subprocess
import tempfile

def cohomcalg_wrapper(sr_ideal, glsm_charges, line_bundle):
    """Call cohomCalg via subprocess."""
    # Generate input file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.in') as f:
        f.write(format_cohomcalg_input(sr_ideal, glsm_charges, line_bundle))
        f.flush()

        # Run cohomCalg
        result = subprocess.run(
            ['./cohomCalg', f.name],
            capture_output=True, text=True
        )

        # Parse output
        return parse_cohomcalg_output(result.stdout)
```

### Option B: Implement Algorithm in Python
The cohomCalg algorithm (arXiv:1003.5217) can be implemented:

1. **Input**: Stanley-Reisner ideal SR, GLSM charges Q, line bundle degree D
2. **For each Q in powerset(SR)**:
   - Compute c-degree N = |union(Q)| - |Q|
   - Count rationoms T(x)/[Q * W(y)] matching D
   - Weight by remnant cohomology
3. **Sum contributions** to get H^i dimensions

Key functions:
- `count_monomials()`: Lattice point counting (Barvinok)
- `count_rationoms()`: Negative degree counting
- `remnant_cohomology()`: Combinatorial factor

### Option C: Use Koszul Sequence
For divisor cohomology specifically:
1. Compute ambient variety line bundle cohomology
2. Chase the long exact sequence from Koszul resolution
3. Extract h^i(D|X, O_D)

---

## Comparison Table

| Tool | H^i(X,O(D)) | h^i(D,O_D) | Python | h11 limit |
|------|-------------|------------|--------|-----------|
| cohomCalg | Yes | Via Koszul | No (C++) | None |
| SageMath | No | No | Yes | N/A |
| Altman DB | Precomputed | Precomputed | Yes | <= 6 |
| CYTools | No | No | Yes | None |

---

## Recommended Approach

1. **h11 <= 6**: Use Altman database (fast, validated)
2. **h11 > 6**:
   - Option A: Compile cohomCalg + Python wrapper (most reliable)
   - Option B: Implement algorithm from scratch (educational)
3. **Validation**: Compare computed values against Altman DB for h11 <= 6

---

## References

1. arXiv:1003.5217 - cohomCalg algorithm
2. arXiv:2111.03078 - Altman orientifold database
3. arXiv:2211.03823 - CYTools paper
4. http://www.rossealtman.com/toriccy/ - Database download
5. http://wwwth.mpp.mpg.de/members/blumenha/cohomcalg/ - cohomCalg download
