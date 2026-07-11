#!/usr/bin/env python3
"""
Feasibility probe for "CY Heuristics": does the sampled Calabi-Yau point
cloud's SHAPE respond to the complex-structure moduli (the polynomial
coefficients), and do simple shape heuristics capture that response?

This does NOT yet use flux-stabilized moduli (that needs the leading-LCS
mirror map). It varies the coefficients and measures whether the cloud
shape + heuristics move at all. If they barely move, the (polytope, K, M)
-> CY-shape idea has no signal and isn't worth the rigorous build.

For each of several coefficient sets (random seeds, and a couple of
structured ones) on the SAME polytope, we sample the CY and compute:
  sphericity   - 1 - spread of PCA eigenvalues (1 = isotropic blob)
  spikiness    - 99th-pct radius / median radius (tails)
  symmetry     - 1 - |mean|/std per axis, averaged (reflection symmetry)
  entropy      - Shannon entropy of the radius histogram (normalized)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "vendor" / "cytools_latest"))
from cytools import Polytope

DATA = Path(__file__).parent.parent / "resources" / "small_cc_2107.09064_source" / "anc" / "paper_data"


def load_monomials(example):
    pts = [[int(x) for x in l.split(",")] for l in open(DATA / example / "points.dat") if l.strip()]
    return np.array(Polytope(pts).dual().points())


def sample_cy(monomials, coeffs, resolution=40):
    """Sample {P=0} by sweeping phases and solving for z0 (reuses the
    existing visualizer's structured-grid approach)."""
    pts = []
    z0_pow = monomials[:, 0]
    lo, hi = int(z0_pow.min()), int(z0_pow.max())
    deg = hi - lo
    if deg < 1:
        return np.zeros((0, 6))
    thetas = np.linspace(0, 2 * np.pi, resolution)
    for th3 in [0, np.pi / 2, np.pi, 3 * np.pi / 2]:
        for r in [0.8, 1.0, 1.2]:
            for t1 in thetas:
                for t2 in thetas:
                    z = np.array([1.0, r * np.exp(1j * t1), r * np.exp(1j * t2), r * np.exp(1j * th3)])
                    pc = np.zeros(deg + 1, dtype=complex)
                    for mono, c in zip(monomials, coeffs):
                        term = c
                        for i in range(1, 4):
                            if mono[i]:
                                term *= z[i] ** mono[i]
                        pc[hi - mono[0]] += term
                    try:
                        roots = np.roots(pc)
                    except Exception:
                        continue
                    valid = [w for w in roots if np.isfinite(w) and 0.1 < abs(w) < 10]
                    if not valid:
                        continue
                    z0 = min(valid, key=lambda w: abs(abs(w) - 1))
                    pts.append([z0.real, z0.imag, z[1].real, z[1].imag, z[2].real, z[2].imag])
    return np.array(pts)


def heuristics(cloud):
    if len(cloud) < 50:
        return None
    c = cloud - cloud.mean(0)
    # sphericity via PCA eigenvalue evenness
    eig = np.linalg.eigvalsh(np.cov(c.T))
    eig = np.clip(eig, 1e-12, None)
    sphericity = 1.0 - eig.std() / eig.mean()
    # spikiness: tail radius vs typical radius
    rad = np.linalg.norm(c, axis=1)
    spikiness = np.percentile(rad, 99) / (np.median(rad) + 1e-12)
    # reflection symmetry per axis
    sym = np.mean([1.0 - abs(c[:, i].mean()) / (c[:, i].std() + 1e-12) for i in range(c.shape[1])])
    # radius-histogram entropy (normalized to [0,1])
    h, _ = np.histogram(rad, bins=24, density=True)
    h = h[h > 0]
    p = h / h.sum()
    entropy = -(p * np.log(p)).sum() / np.log(len(p))
    return dict(n=len(cloud), sphericity=sphericity, spikiness=spikiness,
               symmetry=sym, entropy=entropy)


def main():
    example = sys.argv[1] if len(sys.argv) > 1 else "5-81-3213"
    mono = load_monomials(example)
    n = len(mono)
    print(f"{example}: {n} monomials\n")
    print(f"{'coeff set':<16}{'n':>6}{'spheric':>9}{'spiky':>8}{'symm':>8}{'entropy':>9}")

    coeff_sets = {}
    for seed in (1, 2, 3, 42):
        rng = np.random.default_rng(seed)
        c = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        coeff_sets[f"random#{seed}"] = c / np.abs(c)
    # structured: decay the higher-index (deformation-like) monomials, as a
    # crude stand-in for an LCS-suppressed coefficient pattern.
    base = coeff_sets["random#42"].copy()
    for tag, decay in [("decayed-0.5", 0.5), ("decayed-0.1", 0.1)]:
        idx = np.arange(n)
        coeff_sets[tag] = base * np.exp(-decay * (idx - idx.mean()).clip(0))

    rows = {}
    for tag, coeffs in coeff_sets.items():
        h = heuristics(sample_cy(mono, coeffs))
        if h is None:
            print(f"{tag:<16}  (too few points)")
            continue
        rows[tag] = h
        print(f"{tag:<16}{h['n']:>6}{h['sphericity']:>9.3f}{h['spikiness']:>8.2f}"
              f"{h['symmetry']:>8.3f}{h['entropy']:>9.3f}")

    # Signal check: how much do heuristics vary across coefficient sets?
    if len(rows) >= 2:
        print("\nspread across coefficient sets (max - min):")
        for k in ("sphericity", "spikiness", "symmetry", "entropy"):
            vals = [r[k] for r in rows.values()]
            print(f"  {k:<12} {max(vals) - min(vals):.3f}   (range {min(vals):.3f}..{max(vals):.3f})")


if __name__ == "__main__":
    main()
