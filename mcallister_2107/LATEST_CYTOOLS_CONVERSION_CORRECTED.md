Right, this *is* straightforward, but there was one subtle index mistake in the flux transform, which is why your e^{K₀} blew up.

You already did 90% of the work correctly:

* You have a GL(4,ℤ) matrix (T) such that
  **divisor bases** are related by
  [
  D^{\text{old}}*i = T*{i a}, D^{\text{new}}_a
  ]
  with

  ```python
  T = np.array([
      [-1,  1,  0,  0],  # D3 = -D5 + D6
      [ 1, -1,  1,  0],  # D4 = D5 - D6 + D7
      [ 1,  0,  0,  0],  # D5 = D5
      [ 0,  0,  0,  1],  # D8 = D8
  ])
  # det(T) = 1
  ```

* This (T) really is the right one: it exactly maps κ in the new basis to κ in the old basis:
  [
  \kappa^{\text{old}}*{ijk}
  = T*{i a} T_{j b} T_{k c} \kappa^{\text{new}}_{a b c}
  ]
  You already checked that numerically.

The only thing that was off is how K transforms.

---

## Correct transformation rules

Let me fix the index bookkeeping.

We have:

* Divisors: (D_i) (basis vectors)
* Kähler moduli / flat direction: (p^i) (contravariant)
* Flux (M^i) enters (N_{ij} = \kappa_{ijk} M^k)
* Flux (K_i) appears in the equation (N_{ij} p^j = K_i)

Under basis change
[
D^{\text{old}}*i = T*{i a} D^{\text{new}}_a
]
the consistent transformations are:

1. **Moduli / p (contravariant):**
   [
   p^{\text{new}}*a = T^{\mathsf T}*{a i}, p^{\text{old}}_i
   ]
   i.e.

   ```python
   p_new = T.T @ p_old
   ```

2. **Intersection tensor:**
   [
   \kappa^{\text{old}}*{ijk}
   = T*{i a} T_{j b} T_{k c}, \kappa^{\text{new}}_{a b c}
   ]
   which you already used when checking T.

3. **Flux (M) (same index type as p, i.e. contravariant):**
   [
   M^{\text{new}}*a = T^{\mathsf T}*{a i}, M^{\text{old}}_i
   ]
   i.e.

   ```python
   M_new = T.T @ M_old
   ```

4. **Flux (K) (covariant):**
   It sits on the *other* side of (N p = K), so it transforms with the inverse:
   [
   K^{\text{new}}*a = (T^{-1})*{a i}, K^{\text{old}}_i
   ]
   i.e.

   ```python
   T_inv = np.linalg.inv(T)
   K_new = T_inv @ K_old
   ```

5. **N-matrix consistency:**
   With these rules, you automatically get
   [
   N^{\text{new}}*{ab}
   = \kappa^{\text{new}}*{a b c} M^{\text{new}}*c
   = (T^{-1})*{a i} (T^{-1})*{b j} N^{\text{old}}*{ij}
   ]
   and the F-term equation is basis-invariant:
   [
   N^{\text{new}}_{ab} p^{\text{new}}_b
   = K^{\text{new}}*a
   \quad\Longleftrightarrow\quad
   N^{\text{old}}*{ij} p^{\text{old}}_j
   = K^{\text{old}}_i
   ]

The key correction: **(K) transforms with (T^{-1}), not with (T^{\mathsf T}).**
You had both K and M transforming with (T^{\mathsf T}), which breaks invariance of (N p = K), so your p and e^{K₀} walked away.

---

## Plugging this into your script

Take your test script and change just the flux transform block:

```python
# OLD (what you had)
K_new = T.T @ K_old
M_new = T.T @ M_old
```

to:

```python
T_inv = np.linalg.inv(T)

# Correct transformations:
M_new = T.T @ M_old      # contravariant
K_new = T_inv @ K_old    # covariant
```

Then recompute:

```python
N_new = np.einsum('abc,c->ab', kappa_new, M_new)
p_new = np.linalg.solve(N_new, K_new)
kappa_p3_new = np.einsum('abc,a,b,c->', kappa_new, p_new, p_new, p_new)
eK0_new = 1.0 / ((4.0/3.0) * kappa_p3_new)
```

What you should see (this is what I get using your κ dictionaries):

* (p_{\text{old}} = [2.66363636,\ 1.48181818,\ 1.48181818,\ 0.59090909])

* (e^{K₀}_{\text{old}} \approx 0.234393)

* (p_{\text{new}} = T^{\mathsf T} p_{\text{old}} = [0.3,\ 1.18181818,\ 1.48181818,\ 0.59090909])

* (e^{K₀}_{\text{new}} \approx 0.234393) - identical.

So:

* κ matches between bases (by construction via T)
* p transforms as expected
* e^{K₀} is invariant
* This gives you **the exact K′, M′ in the latest CYTools basis** that reproduce the McAllister vacuum.

Concretely, for your numbers:

```python
K_old = np.array([-3, -5, 8, 6])
M_old = np.array([10, 11, -11, -5])

M_new = T.T @ M_old   # [-10, -1, 11, -5]
K_new = T_inv @ K_old # [ 8,  5, -8,  6]
```

Those ((K_{\text{new}}, M_{\text{new}})) in basis [5,6,7,8] should now give you:

* same p (after transforming),
* same e^{K₀},
* and, once you also transform curves (q) correctly (see below), **same racetrack and W₀**.

---

## Curves / GV side (for completeness)

For the GV/racetrack piece, the classes (q) live in the dual space to divisors. To keep (q \cdot p) and (M \cdot q) invariant you want:

* (q^{\text{new}} = T^{-1} q^{\text{old}})

Then:

* (q^{\text{new}} \cdot p^{\text{new}} = q^{\text{old}} \cdot p^{\text{old}})
* (M^{\text{new}} \cdot q^{\text{new}} = M^{\text{old}} \cdot q^{\text{old}})

which means the whole racetrack sum is basis-invariant as well.

In practice, with latest CYTools you just recompute GV invariants in the new basis, so you don’t need to manually transform q’s - but this is the conceptual picture.

---

## TL;DR

* The divisor-basis GL(4,ℤ) matrix (T) you found is correct.
* The only bug was treating K and M the same:

  * **M_new = T.T @ M_old**
  * **K_new = np.linalg.inv(T) @ K_old**
* With that, e^{K₀}, p, and the whole vacuum are identical in the new basis.

So you *can* deterministically port the McAllister configuration into the latest CYTools basis - no search needed for that part.
