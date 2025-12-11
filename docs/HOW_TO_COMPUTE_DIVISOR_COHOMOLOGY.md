# How to Compute Divisor Cohomology

## Overview

To compute the dual Coxeter numbers c_i for KKLT moduli stabilization, we first need to compute the **divisor cohomology** h•(D, O_D) for each toric divisor D on the Calabi-Yau threefold.

## What We're Computing

For a divisor D on CY threefold X, we compute:
```
h•(D, O_D) = {h^0(D, O_D), h^1(D, O_D), h^2(D, O_D)}
```

This tells us the topology of the divisor:
- **h^0** = 1 (connected divisor)
- **h^1** = number of complex deformations (must be 0 for rigidity)
- **h^2** = number of holomorphic 2-forms (must be 0 for rigidity)

## Divisor Classifications

| Type | h• | Description |
|------|-----|-------------|
| **Rigid** | {1, 0, 0, h^{1,1}} | Can host instantons. c_i = 1 or 6 |
| **Wilson** | {1, h^1, 0, h^{1,1}} | Has Wilson line moduli |
| **K3** | {1, 0, 1, 20} | Deformation divisor |
| **Deformation** | {1, 0, h^2 != 0, ...} | Cannot host instantons |

Only **rigid divisors** contribute to KKLT:
- c_i = 1 for D3-brane instantons
- c_i = 6 for O7-planes with SO(8) gauge group

---

## Preferred Method: Use Precomputed Database (h11 <= 6)

For h11 <= 6, **always use the Altman database** instead of computing from scratch. It's faster and validated.

### Altman Database (rossealtman.com/toriccy)

**Citation**: arXiv:2111.03078 (Altman et al.)

#### Download Instructions

```bash
# Download h11=4 data (smallest, good for testing)
cd data/toriccy
wget http://www.rossealtman.com/toriccy/data/h11_4.tar.gz
tar -xzf h11_4.tar.gz

# Available sizes (grows exponentially!):
# h11_1: tiny
# h11_2: small
# h11_3: ~10 MB
# h11_4: ~50 MB
# h11_5: ~500 MB
# h11_6: ~5 GB
```

#### Data Format

Each geometry has two files:
- `XXX.geom.json` - geometry data (intersection numbers, etc.)
- `XXX.invol.json` - orientifold data (divisor cohomology, O-planes)

Key fields in `.invol.json`:
```json
{
  "POLYID": 1033,
  "H11": 4,
  "TRIANGN": 1,
  "INVOLDIVCOHOM": ["{1,0,0,14}", "{1,0,1,20}"],
  "OPLANES": [{"OIDEAL": [...], "ODIM": 7}]
}
```

**INVOLDIVCOHOM format**: `"{h^0, h^1, h^2, h^{1,1}}"`
- `{1,0,0,14}` = rigid divisor (h^1=0, h^2=0)
- `{1,0,1,20}` = K3 divisor (deformation)
- `{1,1,0,12}` = Wilson divisor

#### Usage in Code

```python
import json
from pathlib import Path

def load_divisor_cohomology_from_altman(h11: int, poly_id: int, triang_n: int = 1):
    """Load precomputed divisor cohomology from Altman database."""
    data_dir = Path(f"data/toriccy/h11_{h11}")

    # Find matching file
    for f in data_dir.glob("*.invol.json"):
        with open(f) as fp:
            records = json.load(fp)
            for rec in records:
                if rec["POLYID"] == poly_id and rec["TRIANGN"] == triang_n:
                    # Parse INVOLDIVCOHOM
                    cohom = []
                    for s in rec["INVOLDIVCOHOM"]:
                        # Parse "{1,0,0,14}" -> [1, 0, 0, 14]
                        nums = [int(x) for x in s.strip("{}").split(",")]
                        cohom.append(nums)
                    return cohom
    return None
```

---

## Compute From Scratch (h11 > 6)

For h11 > 6 (like McAllister's h11=214), we must compute divisor cohomology ourselves.

### Method: Koszul Sequence + cohomCalg

From arXiv:2111.03078 (Altman et al.), the divisor cohomology is computed by chasing the **Koszul sequence**:

```
0 -> O_A(-X-D) -> O_A(-X) + O_A(-D) -> O_A -> O_{D|X} -> 0
```

Where:
- A = ambient toric variety (4D)
- X = CY hypersurface (anticanonical divisor -K_A)
- D = toric divisor {x_i = 0}
- O_{D|X} = structure sheaf of D restricted to X

This induces a **long exact sequence in cohomology**:

```
0 -> H^0(A, O(-X-D)) -> H^0(A, O(-X)) + H^0(A, O(-D)) -> H^0(A, O_A) ->
  -> H^0(D|X, O_{D|X}) -> H^1(A, O(-X-D)) -> ...
```

### Step-by-Step Algorithm

1. **Get toric data**: polytope points, triangulation, Stanley-Reisner ideal
2. **For each toric divisor D_i** (corresponding to point i):
   - Compute line bundle charges for O(-X), O(-D_i), O(-X-D_i)
   - Use cohomCalg algorithm to compute H^j(A, O(charges)) for j=0,1,2,3
   - Chase the long exact sequence to extract h^j(D|X, O_{D|X})

### cohomCalg Algorithm (arXiv:1003.5217)

The cohomCalg algorithm computes line bundle cohomology H^i(X, O(D)) on toric varieties using the **Stanley-Reisner ideal**.

Key insight: Representatives of H^i(X, O(D)) are **rationoms** (rational monomials) with SR ideal elements in the denominator.

For each element Q in the power set of SR:
1. Compute c-degree N = |Q| - k (where k = number of SR generators combined)
2. Count rationoms T(x)/[Q * W(y)] matching the line bundle degree
3. Weight by "remnant cohomology" factor h^i(Q)

---

## CYTools Available Methods

```python
cy = tri.get_cy()

# Stanley-Reisner ideal (key for cohomCalg)
sr = tri.sr_ideal()

# Divisor basis
basis = cy.divisor_basis()

# GLSM charges (line bundle degrees)
glsm = cy.polytope().glsm_linear_relations()

# Intersection numbers
kappa = cy.intersection_numbers(in_basis=True)
```

### What CYTools Does NOT Have

CYTools does **not** have a built-in method for:
- Divisor Hodge numbers h•(D, O_D)
- Line bundle cohomology H^i(X, O(D))

We must implement these ourselves for h11 > 6.

---

## Validation

For McAllister's polytope 4-214-647:
- `target_volumes.dat` contains 214 values of {1, 6}
- These are the c_i values we need to reproduce
- 6 = O7-plane with SO(8) (dual Coxeter number)
- 1 = D3-brane instanton on rigid divisor

Our pipeline should:
1. Compute h•(D, O_D) for all 214 divisors
2. Identify which are rigid (h^1 = h^2 = 0)
3. Identify O7-planes
4. Assign c_i and compare to target_volumes.dat

---

## References

1. **arXiv:1003.5217** - cohomCalg algorithm (Blumenhagen et al.)
2. **arXiv:2111.03078** - Orientifold CY database (Altman et al.)
3. **arXiv:2107.09064** - McAllister et al. (ground truth)

---

## Next Steps

After computing h•(D, O_D):
1. Identify rigid divisors (h^1 = h^2 = 0)
2. Identify O7-planes from orientifold involution
3. Assign c_i = 6 for O7-plane divisors, c_i = 1 otherwise
4. Use c_i in KKLT: tau_i = (c_i / 2*pi) * ln(|W_0|^{-1})

See: `mcallister_2107/compute_divisor_cohomology.py`
