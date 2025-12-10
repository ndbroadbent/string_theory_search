# CRITICAL: Placeholder and Garbage Calculations

This document lists ALL fake/placeholder computations in the codebase that MUST be fixed before any results can be trusted.

## physics_bridge.py

### 1. PERIODS ARE COMPLETELY FAKE (Line 566-568)
```python
# Simple period approximation (real computation needs CY metric)
complex_mod = genome.get("complex_moduli", [1.0])
periods = np.exp(1j * np.arange(n_periods) * 0.1) * complex_mod[0]
```
**Impact**: W₀ (flux superpotential) is GARBAGE. We get ~650 instead of 10⁻⁹⁰.
**Fix**: Compute actual periods by solving Picard-Fuchs equations OR use cymyc to compute CY metric and extract periods from holomorphic 3-form.

### 2. COMPLEX MODULI ARE IGNORED (Line 567)
```python
complex_mod = genome.get("complex_moduli", [1.0])
```
We pass `[1.0] * h21` (214 ones) but only use `complex_mod[0]` to scale fake periods.
**Impact**: Complex structure moduli determine periods which determine W₀. Ignoring them = garbage W₀.
**Fix**: Implement proper complex structure moduli dependence via period computation.

### 3. MASS RATIOS ARE PLACEHOLDER (Line 583-586)
```python
# 8. Mass ratios (placeholder - needs Yukawa computation from cymyc)
w_total = potential["w_total_abs"]
m_e_ratio = g_s * w_total / (cy_volume**(1/3) + 1e-10) * 1e-22
m_p_ratio = m_e_ratio * 1836.15
```
**Impact**: Electron and proton mass ratios are meaningless.
**Fix**: Use cymyc to compute Yukawa couplings from CY metric, then derive physical masses.

### 4. WARP FACTOR IS HARDCODED (Line 326)
```python
warp_factor = 0.01  # Typical warping
```
**Impact**: Uplift potential V_uplift is wrong. Affects cosmological constant.
**Fix**: Compute warp factor from throat geometry or use flux-dependent calculation.

### 5. NON-PERTURBATIVE PARAMETERS ARE HARDCODED (Line 277-279)
```python
self.A = 1.0  # Prefactor
self.a = 2 * np.pi / 10  # a = 2π/N for SU(N) gaugino condensation
```
**Impact**: W_np (non-perturbative superpotential) uses arbitrary constants.
**Fix**: A should be computed from one-loop determinants. N should come from actual brane configuration.

### 6. BRANE CONFIGURATION IS ARBITRARY (Line 195-204)
```python
if brane_config is None:
    # Default: diagonal brane configuration
    brane_config = {
        "su3_cycle": 0,
        "su2_cycle": 1 if len(divisor_volumes) > 1 else 0,
        "u1_cycle": 2 if len(divisor_volumes) > 2 else 0,
    }
```
**Impact**: Gauge couplings depend on which cycles D7-branes wrap. Arbitrary assignment = wrong couplings.
**Fix**: Either search over brane configurations or use anomaly cancellation to constrain.

### 7. AXIO-DILATON REAL PART HARDCODED (Line 570)
```python
tau = complex(0.1 + 1j / g_s)  # Axio-dilaton
```
The real part (C_0 axion) is hardcoded to 0.1.
**Impact**: Affects flux superpotential W_flux = (F - τH) · Π
**Fix**: C_0 should be a modulus that gets stabilized, not hardcoded.

### 8. NUMBER OF ANTI-D3 BRANES DEFAULTS TO 1 (Line 573)
```python
n_antiD3 = genome.get("n_antiD3", 1)
```
**Impact**: Affects uplift potential.
**Fix**: Should be determined by tadpole cancellation constraint.

### 9. TADPOLE COMPUTATION ASSUMES SIMPLE FLUX STRUCTURE (Line 343-353)
```python
def compute_tadpole(self, flux_f: np.ndarray, flux_h: np.ndarray) -> float:
    n = len(flux_f) // 2
    # Symplectic product
    return float(np.dot(flux_f[:n], flux_h[n:]) - np.dot(flux_f[n:], flux_h[:n])) / 2
```
**Impact**: Assumes fluxes are in symplectic basis. May not match actual CY cohomology structure.
**Fix**: Use intersection matrix from CYTools to compute proper flux tadpole.

### 10. GENERATION COUNT IS OVERSIMPLIFIED (Line 112)
```python
"n_generations": abs(chi) // 2,  # |χ|/2 for CY3
```
**Impact**: Real generation count depends on bundle/brane configuration, not just Euler characteristic.
**Fix**: Compute index of Dirac operator for actual gauge bundle.

## Summary: What Actually Works

**REAL (from CYTools):**
- Polytope analysis (reflexivity, vertices)
- Hodge numbers h11, h21
- Euler characteristic χ
- Intersection numbers κ_ijk
- Kähler cone
- CY volume computation V = (1/6) κ_ijk t^i t^j t^k
- Divisor volumes
- Curve volumes

**GARBAGE:**
- Periods (fake exponentials)
- W₀ flux superpotential (depends on fake periods)
- W_np non-perturbative superpotential (hardcoded A, a)
- Cosmological constant (depends on garbage W)
- Mass ratios (placeholder formula)
- Gauge couplings (arbitrary brane assignment)

## What Must Be Implemented

1. **Period computation**: Either:
   - Solve Picard-Fuchs differential equations for the periods Π(z) as functions of complex structure moduli
   - Use cymyc to compute numerical CY metric, extract holomorphic 3-form Ω, integrate to get periods

2. **Yukawa couplings**: Use cymyc's capabilities (it claims to do this - see README)

3. **Proper brane configuration search**: The GA should search over which cycles branes wrap

4. **Moduli stabilization**: Actually minimize the scalar potential to find the vacuum

5. **Warp factor computation**: From throat geometry in flux compactification

## References for Implementation

- cymyc documentation: https://justin-tan.github.io/cymyc/
- cymyc paper: arxiv:2410.19728
- Picard-Fuchs equations: standard algebraic geometry / mirror symmetry literature
- KKLT moduli stabilization: arxiv:hep-th/0301240
- McAllister small CC paper: arxiv:2107.09064 (what we're trying to reproduce)
