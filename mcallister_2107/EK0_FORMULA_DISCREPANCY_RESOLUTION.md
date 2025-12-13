### 1) The “16/9 discrepancy” is a parenthesis issue, not a real mismatch

In the PDF, eq. (6.12) defines the constant as the **inverse of the whole product**:
[
e^{K_0} := \left(\frac{4}{3},\tilde\kappa_{abc} p^a p^b p^c\right)^{-1}.
]
You can see the `−1` applying to the entire ( \frac{4}{3}\tilde\kappa_{abc}p^ap^bp^c ) expression in the printed equation.

That is algebraically
[
e^{K_0}=\frac{3}{4},\left(\tilde\kappa_{abc}p^ap^bp^c\right)^{-1},
]
which is exactly your empirically “working” formula.

So the factor (16/9) appears only if you (mis)read the paper as
[
e^{K_0}=\left(\frac{4}{3}\right)\left(\tilde\kappa_{abc}p^ap^bp^c\right)^{-1},
]
instead of
[
e^{K_0}=\left[\left(\frac{4}{3}\right)\left(\tilde\kappa_{abc}p^ap^bp^c\right)\right]^{-1}.
]

A quick consistency check using the paper’s own number for your 5-113-4627-main case:

* paper: (e^{K_0} = 1170672/12843563)
* implies
  [
  \tilde\kappa p^3=\frac{3}{4}\frac{1}{e^{K_0}}=\frac{12843563}{1560896}=8.22832719\ldots
  ]
  which matches your computed (8.228327).

So Q1 and Q5 are basically resolved: no typo needed, just correct grouping of the inverse.

---

### 2) Why 7-51-13590 “doesn’t match”: you’re using the wrong “paper e^{K0}” value

Your table’s “paper e^{K0} = 0.2719” for **7-51-13590** is almost certainly coming from the **second vacuum in the (5,113) geometry**, where the paper explicitly states
[
e^{K_0} = 5488000/20186543 \approx 0.2719
]
in eq. (6.30).

For the **(7,51)** example (section 6.2), the paper gives (g_s), (W_0), (V^{[0]}), and (V_0), but it does **not** print an explicit (e^{K_0}) value there.

If you *infer* (e^{K_0}) for (7,51) using their approximation (6.24),
[
V_0 \approx -3,e^{K_0},\frac{g_s^7}{(4V^{[0]})^2},W_0^2,
]
(which they write explicitly for the first (5,113) vacuum)
and plug in the (7,51) numbers they report ((g_s \approx 0.040), (W_0 \approx 4.1\times10^{-21}), (V^{[0]}\approx 141.4), (V_0\approx-3.1\times10^{-57})),
you get (e^{K_0}\approx 0.11), which is consistent with your computed (0.1157) from (\frac{3}{4}/(\kappa p^3)).

So the “7-51 mismatch” is coming from comparing against an (e^{K_0}) value that belongs to a different example.

---

### 3) Contraction convention for (\kappa_{abc}p^ap^bp^c)

Nothing in the paper suggests a nonstandard contraction here. The combination appearing in (6.12) is literally (\tilde\kappa_{abc}p^ap^bp^c) inside the parentheses that get inverted.

Your implementation is consistent:

* Building a fully symmetric tensor then doing `einsum('abc,a,b,c->', ...)` is correct.
* Summing only over (i\le j\le k) requires multiplicity factors (1,3,6), and you confirmed that matches the einsum result.

Also, this cubic form is basis-invariant if you transform (\kappa) and (p) consistently, so you should not be able to “fix” a constant factor like (16/9) via a basis change.

---

### 4) Relation to the (1/6) in the prepotential / volume normalization

The paper’s LCS prepotential normalization is the standard one:
[
F_{\rm poly}(z) = -\frac{1}{3!}\tilde\kappa_{abc}z^az^bz^c + \cdots
]
i.e. the (1/6) is already built in.

The familiar (4/3) arises when you plug this cubic prepotential into the special-geometry expression for (K_{cs}) and keep the leading term at large complex structure; schematically one gets
[
-i\int \Omega\wedge\bar\Omega \sim \frac{4}{3}\tilde\kappa_{abc}y^ay^by^c + \cdots,\quad y^a=\mathrm{Im}(z^a).
]
Then because (K_{cs}=-\log(\cdots)), the exponential (e^{K_{cs}}) carries the reciprocal, hence the ((\frac{4}{3}\tilde\kappa y^3)^{-1}) structure, matching (6.12).

So: the (1/6) is not a missing extra factor you need to apply later. It is exactly what produces the standard (4/3) in the period norm and therefore the overall inverse factor in (e^{K_0}).

---

## Direct answers to your numbered questions

1. **Why 3/4 instead of 4/3?**
   Because (6.12) is (e^{K_0} = \left(\frac{4}{3}\tilde\kappa p^3\right)^{-1}), which equals (\frac{3}{4}(\tilde\kappa p^3)^{-1}).
   The (16/9) comes from inverting only (\tilde\kappa p^3) instead of inverting the full product.

2. **Why doesn’t 7-51-13590 match?**
   Because the “paper e^{K_0}=0.2719” value is from the *second (5,113) vacuum* (eq. 6.30), not from the (7,51) example.
   For (7,51), the paper prints (g_s,W_0,V^{[0]},V_0)  but not (e^{K_0}). If you infer (e^{K_0}) from those, you land near (\sim 0.11), consistent with your (\sim 0.116).

3. **Different convention for (\kappa_{abc}p^ap^bp^c)?**
   No evidence of that in the paper’s definition. The contraction is the standard Einstein summation appearing directly in (6.12).

4. **Is this related to the (1/6) in the prepotential / volume?**
   The (1/6) is already in their prepotential normalization (F_{\rm poly} = -\frac{1}{3!}\tilde\kappa z^3 + \cdots).
   Combining that with the special-geometry formula is what yields the (4/3) inside the inverse in (6.12), not an extra factor you should insert by hand.

5. **Could it be a typo in eq 6.12?**
   Unlikely. The printed equation is consistent with the numeric example and with the standard LCS asymptotics, as long as you read it as (\left(\frac{4}{3}\tilde\kappa p^3\right)^{-1}).

---

## If anything still looks off after this

The only remaining “moving parts” I’d want to see (for any example that still mismatches) are:

* exactly what you’re calling “paper’s (e^{K_0})” for that example (is it printed in the PDF, or inferred, or pulled from a different directory),
* the triangulation / phase you used (but you’re already loading the simplices from their ancillary data, so that’s probably fine),
* a quick check that your (p) matches the PDF for that example (like you did for 5-113).

If you paste your 7-51-13590 computed (\kappa p^3) and where you sourced “0.2719”, I can sanity-check it against the correct section immediately.
