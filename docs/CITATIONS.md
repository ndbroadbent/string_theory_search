  # Citations and References

This document tracks all papers, databases, and tools used in this project. **Always cite when using data or methods from these sources.**

---

## Primary References

### McAllister et al. - Small Cosmological Constants (Ground Truth)
- **arXiv**: [2107.09064](https://arxiv.org/abs/2107.09064)
- **Title**: "Small cosmological constants in string theory"
- **Local PDF**: `resources/small_cc_2107.09064.pdf`
- **Used for**:
  - Ground truth validation (polytope 4-214-647)
  - KKLT moduli stabilization formulas
  - Racetrack mechanism for W₀ computation
  - V₀ = -3 e^{K₀} (g_s^7 / (4V[0])²) W₀² formula
- **Key data**: `resources/small_cc_2107.09064_source/anc/paper_data/4-214-647/`

---

## Orientifold and Divisor Topology

### Altman et al. - Orientifold CY3 Database
- **arXiv**: [2111.03078](https://arxiv.org/abs/2111.03078)
- **Title**: "Orientifold Calabi-Yau Threefolds with Divisor Involutions and String Landscape"
- **Local PDF**: `resources/orientifold_cy_divisor_involutions_2111.03078.pdf`
- **Database**: http://www.rossealtman.com/toriccy/
- **Used for**:
  - Computing Hodge numbers of individual divisors h•(D, O_D)
  - Identifying rigid divisors (h• = {1, 0, 0, h^{1,1}})
  - O7-plane identification from orientifold involutions
  - Determining c_i (dual Coxeter numbers) for KKLT
- **Key method**: cohomCalg + Koszul sequence for divisor cohomology
- **Bulk data download**: http://www.rossealtman.com/toriccy/ (*.invol.json files)

---

## Tools

### CYTools
- **arXiv**: [2211.03823](https://arxiv.org/abs/2211.03823)
- **Title**: "CYTools: A Software Package for Analyzing Calabi-Yau Manifolds"
- **Local PDF**: `resources/cytools_paper_2211.03823.pdf`
- **Repository**: https://github.com/LiamMcAllisterGroup/cytools
- **Used for**:
  - Polytope analysis, triangulations
  - Hodge numbers (h¹¹, h²¹)
  - Intersection numbers κᵢⱼₖ
  - GV invariants via `cy.compute_gvs()`
  - Kähler cone computations

### cohomCalg
- **arXiv**: [1003.5217](https://arxiv.org/abs/1003.5217)
- **Title**: "Cohomology of Line Bundles: A Computational Algorithm"
- **Authors**: Ralph Blumenhagen, Benjamin Jurke, Thorsten Rahn, Helmut Roschy
- **Repository**: https://github.com/BenjaminJurke/cohomCalg
- **Local**: `vendor/cohomCalg/`
- **License**: GPL v3
- **Used for**:
  - Line bundle cohomology H^i(X, O(D)) on toric varieties
  - Divisor cohomology via Koszul extension
- **Algorithm proofs**: arXiv:1006.2392, arXiv:1006.0780
- **Koszul extension**: arXiv:1010.3717

### cymyc
- **arXiv**: [2410.19728](https://arxiv.org/abs/2410.19728)
- **Repository**: https://github.com/Justin-Tan/cymyc
- **Used for**:
  - Numerical CY metrics (JAX-based)
  - Yukawa coupling calculations

### PALP
- **Repository**: https://gitlab.com/stringstuwien/PALP
- **Used for**:
  - Polytope analysis
  - Hodge number computation from weights

---

## Flux Vacua and Moduli Stabilization

### Demirtas et al. - Vacua with Small W₀
- **arXiv**: [1912.10047](https://arxiv.org/abs/1912.10047)
- **Local PDF**: `resources/vacua_small_W0_1912.10047.pdf`
- **Used for**: Methods for finding flux vacua with exponentially small W₀

### Demirtas et al. - GA for Flux Vacua
- **arXiv**: [1907.10072](https://arxiv.org/abs/1907.10072)
- **Local PDF**: `resources/ga_flux_vacua_1907.10072.pdf`
- **Used for**: Genetic algorithm approach to searching flux landscape

### Conifold Vacua
- **arXiv**: [2009.03312](https://arxiv.org/abs/2009.03312)
- **Local PDF**: `resources/conifold_vacua_2009.03312.pdf`

---

## Databases

### Kreuzer-Skarke Polytope Database
- **Original**: http://hep.itp.tuwien.ac.at/~kreuzer/CY/
- **HuggingFace mirror**: Used for bulk download
- **Size**: ~473M reflexive 4D polytopes

### Ross Altman Toric CY Database
- **URL**: http://www.rossealtman.com/toriccy/
- **Contains**:
  - Calabi-Yau threefolds with h¹¹ ≤ 6
  - Hodge diamonds for all toric divisors
  - Orientifold involutions and O-plane data (*.invol.json)
- **Citation**: Cite arXiv:2111.03078 when using this data

---

## Cosmological Parameters

### Particle Data Group - Cosmological Parameters
- **Local PDF**: `resources/rpp2024-rev-cosmological-parameters.pdf`
- **Used for**: Current observational values of cosmological constant

---

## Other References

### 1807.06209v4
- **Local PDF**: `resources/1807.06209v4.pdf`
- **TODO**: Add description

---

## Citation Format

When publishing or sharing results, cite:

```bibtex
@article{McAllister:2021oto,
    author = "Demirtas, Mehmet and Kim, Manki and McAllister, Liam and Moritz, Jakob and Rios-Tascon, Andres",
    title = "{Small cosmological constants in string theory}",
    eprint = "2107.09064",
    archivePrefix = "arXiv",
    primaryClass = "hep-th",
    year = "2021"
}

@article{Gao:2021xbs,
    author = "Gao, Xin and Shukla, Pramod and Leung, Simon and Altman, Ross",
    title = "{Orientifold Calabi-Yau threefolds with divisor involutions and string landscape}",
    eprint = "2111.03078",
    archivePrefix = "arXiv",
    primaryClass = "hep-th",
    year = "2021"
}

@article{Demirtas:2022cytools,
    author = "Demirtas, Mehmet and McAllister, Liam and Rios-Tascon, Andres",
    title = "{CYTools: A Software Package for Analyzing Calabi-Yau Manifolds}",
    eprint = "2211.03823",
    archivePrefix = "arXiv",
    primaryClass = "hep-th",
    year = "2022"
}

@article{Blumenhagen:2010pv,
    author = "Blumenhagen, Ralph and Jurke, Benjamin and Rahn, Thorsten and Roschy, Helmut",
    title = "{Cohomology of Line Bundles: A Computational Algorithm}",
    journal = "J. Math. Phys.",
    volume = "51",
    pages = "103525",
    issue = "10",
    year = "2010",
    doi = "10.1063/1.3501132",
    eprint = "1003.5217",
    archivePrefix = "arXiv",
    primaryClass = "hep-th"
}

@misc{cohomCalg:Implementation,
    title = "{cohomCalg package}",
    howpublished = "\\url{https://github.com/BenjaminJurke/cohomCalg}",
    note = "High-performance line bundle cohomology computation based on arXiv:1003.5217",
    year = "2010"
}
```
