# McAllister Section 6.4: Vacuum with (h21, h11) = (4, 214)

Extracted from arXiv:2107.09064, lines 3448-3599.

## Key Equations

### Flux Choice (eq. 6.55)
```
M = (10, 11, -11, -5)^T
K = (-3, -5, 8, 6)^T
```

### Flat Direction p (eq. 6.56)
```
p = (293/110, 163/110, 163/110, 13/22)
  = (2.6636, 1.4818, 1.4818, 0.5909)
```

This satisfies z = p*tau along the flat direction.

### Leading Instanton Charges (eq. 6.57)
The leading curves q̃ are given by the columns of:
```
 1   0  -1   0  -1   0   1   1
-1   1   1   0   1  -2   0  -2
 ?   ?   ?   ?   ?   ?   ?   ?   (rows 3-4 unclear from text)
 ?   ?   ?   ?   ?   ?   ?   ?
```
Note: This should be a 4x8 matrix (h21=4, 8 leading curves).

### GV Invariants (eq. 6.58)
```
N_q̃ = (1, -2, 252, -2, ...)
```
These are the Gopakumar-Vafa invariants for the leading curves.

### Flux Superpotential (eq. 6.59)
```
W_flux(τ) = 5ζ × (-e^{2πiτ·33/110} + 512·e^{2πiτ·32/110} + O(e^{2πiτ·13/22}))
```
Where ζ = 1/(2^{3/2} π^{5/2}) from eq. 2.23.

### Two Leading Terms (racetrack structure)
- α = 32/110 ≈ 0.2909, coefficient involves N_q̃ = 252 (or 512?)
- β = 33/110 = 0.30, coefficient involves N_q̃ = -1

The ratio 32/110 : 33/110 creates the racetrack with ε = (33-32)/110 = 1/110.

### Stabilized g_s (eq. 6.60)
```
g_s ≈ 2π / (110 × log(528)) ≈ 0.009
```
Here 110 is the D3 tadpole, and 528 = 2 × 252 + 24 (from GV hierarchy).

### Flux Superpotential vev (eq. 6.61)
```
W₀ ≈ 80 × ζ × 528^{-33} ≈ 2.3 × 10^{-90}
```

### Kähler Volume (eq. 6.78 reference)
```
V[0] ≈ 4711 (string frame)
V_E ≈ 5.4 × 10^6 (Einstein frame)
```

### Final Vacuum Energy (eq. 6.63)
```
V₀ = -3 e^K |W|² ≈ -5.5 × 10^{-203} Mpl⁴
```

## Critical Physics Point

The GV invariants and curves q̃ are on the MIRROR X̃, not X itself.
- The primal polytope defines X with (h11, h21) = (214, 4)
- The dual polytope defines X̃ with (h11, h21) = (4, 214)
- The flux superpotential formula (eq. 2.22) uses curves on X̃
- p lies in the Kähler cone of X̃ (eq. 2.21)

So when we use dual_points.dat and compute GV invariants, we ARE working
on the mirror X̃, which is correct for the racetrack computation.

## The q̃·p Product

For the racetrack to work, we need q̃·p = 32/110 and 33/110 for the two
leading curves. With p = (2.6636, 1.4818, 1.4818, 0.5909), the curves q̃
must be specific 4-component integer vectors.

32/110 ≈ 0.2909
33/110 = 0.30

These are SMALL exponents. If our q̃·p values are ~0.009, something is wrong
with the coordinate system - likely a factor of ~30-33 off.
