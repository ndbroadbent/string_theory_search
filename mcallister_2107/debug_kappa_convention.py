#!/usr/bin/env python3
"""
Debug κ̃ convention by using EXPLICIT values from the paper.

From eq 6.5-6.6, the paper gives explicit κ̃_abc values.
From eq 6.8, the paper gives explicit p values.
From eq 6.12, the paper gives e^K₀ = 1170672/12843563.

Goal: Figure out the exact contraction convention.
"""

import numpy as np
from fractions import Fraction

# From eq 6.8: p = (7/58, 15/58, 101/116, 151/58, -13/116)
# Paper uses 1-indexed, we use 0-indexed
p_exact = [
    Fraction(7, 58),
    Fraction(15, 58),
    Fraction(101, 116),
    Fraction(151, 58),
    Fraction(-13, 116),
]
p = np.array([float(x) for x in p_exact])

print("p (from eq 6.8):")
for i, (pe, pf) in enumerate(zip(p_exact, p)):
    print(f"  p[{i}] = {pe} = {pf:.6f}")

# From eq 6.5-6.6, the κ̃ values (1-indexed in paper, converting to 0-indexed)
# Paper shows κ̃_1ab as 5x5 symmetric matrix:
#   89  0  16  12   7
#       ?   ?   ?   ?
#           0   3   ?
#               3  -3
#                  -3

# Let me re-extract from the text more carefully...
# Line 2541: "89 0 16 12 7" - this is κ̃_1ab diagonal
# Then κ̃_2ab, κ̃_3ab, etc.

# From CYTools output for 5-113-4627-main:
#   κ_{000} = 89
#   κ_{002} = 16
#   κ_{003} = 12
#   κ_{004} = 7
#   κ_{023} = 3
#   κ_{034} = 3
#   κ_{044} = -3
#   κ_{111} = 8
#   κ_{112} = -2
#   κ_{113} = -2
#   κ_{114} = -2
#   κ_{123} = 1
#   κ_{134} = 1
#   κ_{444} = -1

# Build the full tensor (0-indexed)
h11 = 5
kappa = np.zeros((h11, h11, h11))

# Fill from CYTools output (these are exact integers)
kappa_values = {
    (0, 0, 0): 89,
    (0, 0, 2): 16,
    (0, 0, 3): 12,
    (0, 0, 4): 7,
    (0, 2, 3): 3,
    (0, 3, 4): 3,
    (0, 4, 4): -3,
    (1, 1, 1): 8,
    (1, 1, 2): -2,
    (1, 1, 3): -2,
    (1, 1, 4): -2,
    (1, 2, 3): 1,
    (1, 3, 4): 1,
    (4, 4, 4): -1,
}

# Symmetrize
for (i, j, k), val in kappa_values.items():
    for perm in [(i, j, k), (i, k, j), (j, i, k), (j, k, i), (k, i, j), (k, j, i)]:
        kappa[perm] = val

print("\nκ tensor values (non-zero, sorted indices):")
for i in range(h11):
    for j in range(i, h11):
        for k in range(j, h11):
            val = kappa[i, j, k]
            if val != 0:
                print(f"  κ_{{{i}{j}{k}}} = {int(val)}")

# Compute κ_p3 using einsum (sums over ALL a,b,c)
kappa_p3_einsum = np.einsum('abc,a,b,c->', kappa, p, p, p)
print(f"\nκ_p3 (einsum over all a,b,c) = {kappa_p3_einsum:.6f}")

# Compute κ_p3 by explicit sum over unique triplets with multiplicity
kappa_p3_explicit = 0
for i in range(h11):
    for j in range(i, h11):
        for k in range(j, h11):
            val = kappa[i, j, k]
            if val != 0:
                if i == j == k:
                    mult = 1
                elif i == j or j == k or i == k:
                    mult = 3
                else:
                    mult = 6
                term = val * mult * p[i] * p[j] * p[k]
                kappa_p3_explicit += term
                if abs(term) > 0.01:
                    print(f"  κ_{{{i}{j}{k}}} × mult × p^{i} p^{j} p^{k} = {val} × {mult} × {p[i]:.4f} × {p[j]:.4f} × {p[k]:.4f} = {term:.4f}")

print(f"\nκ_p3 (explicit with multiplicity) = {kappa_p3_explicit:.6f}")

# These should be EQUAL
print(f"\nDifference einsum - explicit = {kappa_p3_einsum - kappa_p3_explicit:.2e}")

# Now check against paper's e^K₀
eK0_paper = Fraction(1170672, 12843563)
print(f"\nPaper's e^K₀ = {eK0_paper} = {float(eK0_paper):.6f}")

# What κ_p3 does the paper imply?
kappa_p3_implied = (4/3) / float(eK0_paper)
print(f"Implied κ_p3 from (4/3)/e^K₀ = {kappa_p3_implied:.6f}")

# What do we get?
eK0_from_einsum = (4/3) / kappa_p3_einsum
print(f"\ne^K₀ from (4/3)/κ_p3 (einsum) = {eK0_from_einsum:.6f}")
print(f"Ratio our/paper = {eK0_from_einsum / float(eK0_paper):.6f}")

# Try other factors
print("\nTrying different factors:")
for num in [1, 2, 3, 4]:
    for denom in [1, 2, 3, 4, 6]:
        factor = num / denom
        eK0_test = factor / kappa_p3_einsum
        ratio = eK0_test / float(eK0_paper)
        if 0.99 < ratio < 1.01:
            print(f"  e^K₀ = ({num}/{denom}) / κ_p3 = {eK0_test:.6f} → ratio = {ratio:.6f} ✓")

# Maybe the formula is e^K₀ = 1 / (4/3 × κ_p3) = 3 / (4 × κ_p3)?
print("\nTrying inverse formulas:")
eK0_inv1 = 1 / ((4/3) * kappa_p3_einsum)
print(f"  e^K₀ = 1 / ((4/3) × κ_p3) = {eK0_inv1:.6f} → ratio = {eK0_inv1 / float(eK0_paper):.6f}")

eK0_inv2 = 3 / (4 * kappa_p3_einsum)
print(f"  e^K₀ = 3 / (4 × κ_p3) = {eK0_inv2:.6f} → ratio = {eK0_inv2 / float(eK0_paper):.6f}")

# Try with a different p contraction - maybe they use (κ_abc p^a p^b p^c)^{-1}
# without the factor (4/3)?
eK0_no_factor = 1 / kappa_p3_einsum
print(f"\n  e^K₀ = 1 / κ_p3 = {eK0_no_factor:.6f} → ratio = {eK0_no_factor / float(eK0_paper):.6f}")

# The ratio is ~1.78 = 16/9 = (4/3)^2
# So maybe the paper uses a different contraction...

# What if the paper contracts WITHOUT symmetrizing - i.e., only summing
# over sorted indices a ≤ b ≤ c with no multiplicity factor?
kappa_p3_nosym = 0
for i in range(h11):
    for j in range(i, h11):
        for k in range(j, h11):
            val = kappa[i, j, k]
            term = val * p[i] * p[j] * p[k]  # NO multiplicity
            kappa_p3_nosym += term

print(f"\nκ_p3 (sorted indices, no multiplicity) = {kappa_p3_nosym:.6f}")
eK0_nosym = (4/3) / kappa_p3_nosym
print(f"e^K₀ from (4/3)/κ_p3_nosym = {eK0_nosym:.6f}")
print(f"Ratio our/paper = {eK0_nosym / float(eK0_paper):.6f}")

# What about (4/3) / (6 × κ_p3_nosym)?
eK0_with_6 = (4/3) / (6 * kappa_p3_nosym)
print(f"\ne^K₀ from (4/3)/(6 × κ_p3_nosym) = {eK0_with_6:.6f}")
print(f"Ratio = {eK0_with_6 / float(eK0_paper):.6f}")

# Final check: what multiplier M satisfies (4/3)/(M × κ_p3_einsum) = eK0_paper?
M_needed = (4/3) / (float(eK0_paper) * kappa_p3_einsum)
print(f"\nMultiplier needed: M = {M_needed:.6f}")
print(f"If M = 16/9 = {16/9:.6f}: matches!")
