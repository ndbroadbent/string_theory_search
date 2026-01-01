# ExoPriors Research Findings for String Theory Landscape Explorer

*Generated: 2026-01-01*
*Source: ExoPriors Alignment Research Corpus (60M documents)*

## Executive Summary

This document indexes relevant papers and discussions found via the ExoPriors corpus that relate directly to our genetic algorithm approach for searching Calabi-Yau compactifications.

**KEY FINDING: There are TWO major prior works using genetic algorithms for flux vacua search:**
1. arXiv:1302.0529 (Damian et al., 2013) - MATLAB-based GA, non-geometric flux
2. arXiv:1907.10072 (Cole, Schachner & Shiu, 2019) - **Comprehensive study, already in our resources!**

The Cole et al. paper is particularly valuable - it provides a complete dictionary mapping GA terminology to landscape concepts and compares GAs to other search methods.

---

## 1. CRITICAL: Genetic Algorithm for Flux Vacua (Direct Precedents!)

### arXiv:1907.10072 - Cole, Schachner & Shiu (2019) ⭐ ALREADY HAVE
**"In this paper, we employ genetic algorithms to explore the landscape of type IIB flux vacua. We show that genetic algorithms can efficiently scan the landscape for viable solutions."**

- URL: https://arxiv.org/abs/1907.10072
- Authors: Alex Cole, Andreas Schachner, Gary Shiu (Wisconsin)
- **Status**: Already in `resources/ga_flux_vacua_1907.10072.tex`
- **Relevance**: **CRITICAL** - Comprehensive GA methodology for flux vacua

**Key insights from this paper:**
- Dictionary: individuals = flux vacua, chromosome = flux vector, phenotype = masses/couplings
- Fitness = function of physical observables
- Roulette-wheel selection for parent pairing
- Crossover + mutation for breeding
- Compared to simulated annealing - GA exploits landscape structure better
- Applied to: T⁶ symmetric torus AND Calabi-Yau hypersurface in WP⁴_{1,1,1,1,4}
- Key observation: GAs NOT effective for "needle-in-a-haystack" problems; need pseudo-continuous fitness landscape

### arXiv:1302.0529 - Damian et al. (2013)
**"By implementing a genetic algorithm we search for stable vacua in Type IIB non-geometric flux compactification on an isotropic torus with orientifold 3-planes."**

- URL: https://arxiv.org/abs/1302.0529
- Authors: Cesar Damian et al.
- **Status**: Downloaded to `resources/ga_flux_vacua_1302.0529.tex`
- **Relevance**: HIGH - Earlier work, uses MATLAB's `ga()` function

**Key insights from this paper:**
- MATLAB implementation with `gaoptimset('Generations',4000,'TolFun',1e-20,'PopulationSize',100)`
- Uses `mutationadaptfeasible` mutation function
- Hybrid with `fmincon` for local optimization
- Found ~2000 vacua, 15 dS and rest AdS
- Limitation: Cannot find metastable vacua (finds global minimum in neighborhood)
- Limitation: Cannot find Minkowski vacua (numerical approximation)
- Finding: Number of stable dS and AdS vacua are same order

### arXiv:2306.06160 - Dubey et al. (2023)
**"We apply our techniques to the search of Type IIB flux vacua in Calabi-Yau orientifold compactifications. We argue that our methods only scale mildly with the Hodge numbers making exhaustive searches tractable."**

- URL: https://arxiv.org/abs/2306.06160
- **Status**: Downloaded to `resources/scalable_flux_vacua_2306.06160.pdf`
- **Relevance**: HIGH - Scalability analysis for large h11 (like our h11=214 case)

---

## 2. McAllister Group Papers (Our Ground Truth)

### arXiv:2206.08400 - Blumenhagen et al.
**References "the mechanism proposed by Demirtas, Kim, McAllister" for generating small W₀**

- URL: https://arxiv.org/abs/2206.08400
- **Relevance**: Cites our primary reference, may have implementation details

### Key McAllister Papers (already in our resources/):
- arXiv:2107.09064 - Small cosmological constants (our primary reference)
- arXiv:1912.10047 - Demirtas small W₀ computations

---

## 3. Machine Learning in String Theory

### arXiv:2403.03245 - Lanza et al.
**"The landscape of low-energy effective field theories stemming from string theory is too vast for a systematic exploration. However, the meadows of the string landscape may be fertile ground for [ML]"**

- URL: https://arxiv.org/abs/2403.03245
- **Relevance**: HIGH - ML landscape exploration methodology

### arXiv:1707.00655 - Carifio et al.
**"We utilize machine learning to study the string landscape. Deep data dives and conjecture generation are proposed as useful frameworks."**

- URL: https://arxiv.org/abs/1707.00655
- Authors: Jonathan Carifio et al.
- Date: 2017-07-03
- **Relevance**: HIGH - ML methodology for landscape

### arXiv:2204.08073 - Constantin et al.
**"The goal of identifying the Standard Model of particle physics and its extensions within string theory... Recently, the [ML approaches]"**

- URL: https://arxiv.org/abs/2204.08073
- **Relevance**: HIGH - Standard Model from strings using ML

### arXiv:2402.13321 - Gukov et al.
**"rigor ranging from string theory to the smooth 4d Poincaré conjecture... building direct bridges between machine learning theory and [physics]"**

- URL: https://arxiv.org/abs/2402.13321
- **Relevance**: MEDIUM - Theoretical ML-physics connections

### arXiv:2011.14442 - He
**"how string theory led theoretical physics first to precise problems in algebraic and differential geometry, and thence to computational geometry"**

- URL: https://arxiv.org/abs/2011.14442
- Author: Yang-Hui He
- **Relevance**: MEDIUM - Computational geometry overview

---

## 4. String Landscape & Swampland

### arXiv:2212.06187 - Agmon et al.
**"We provide an overview of the string landscape and the Swampland program. Our review covers worldsheet and spacetime perspectives, including vacua and string dualities."**

- URL: https://arxiv.org/abs/2212.06187
- Date: 2023-03-05
- **Relevance**: HIGH - Comprehensive landscape review

### arXiv:hep-th/0509212 - Vafa
**"Recent developments in string theory suggest that string theory landscape of vacua is vast. Is this landscape as vast as allowed by consistent-looking effective field theories?"**

- URL: https://arxiv.org/abs/hep-th/0509212
- Author: Cumrun Vafa
- **Relevance**: HIGH - Original swampland paper

### arXiv:1711.06685 - Carifio et al.
**"We introduce network science as a framework for studying the string landscape. Two large networks of string geometries are constructed."**

- URL: https://arxiv.org/abs/1711.06685
- **Relevance**: MEDIUM - Network/graph approach to landscape

---

## 5. KKLT and Moduli Stabilization

### arXiv:2101.05281 - Carta et al.
**"a region of O(1) Einstein frame volume... generically takes up an O(1) fraction of the compactification in a KKLT de Sitter vacuum we argue that a small flux [superpotential]"**

- URL: https://arxiv.org/abs/2101.05281
- **Relevance**: HIGH - KKLT volume constraints

### arXiv:1211.6858 - Sumitomo et al.
**"We study the probability distribution P(Λ) of the cosmological constant Λ in a specific set of KKLT type models"**

- URL: https://arxiv.org/abs/1211.6858
- **Relevance**: HIGH - Statistical analysis of Λ in KKLT

### arXiv:1306.1237 - Pedro et al.
**"For KKLT-like scenarios we find that consistency of the action imposes an upper bound on the flux superpotential |W₀| ≲ 10⁻³"**

- URL: https://arxiv.org/abs/1306.1237
- **Relevance**: HIGH - W₀ bounds (compare to McAllister's 10⁻⁹⁰!)

---

## 6. Type IIB Flux Vacua (Technical)

### arXiv:2310.06040 - Plauschinn et al.
**"We determine all flux vacua with flux numbers N_flux ≤ 10 for a type IIB orientifold-compactification on the mirror-octic three-fold."**

- URL: https://arxiv.org/abs/2310.06040
- **Relevance**: HIGH - Exhaustive flux vacua enumeration

### arXiv:1912.10047 - Demirtas et al.
**"an orientifold of a Calabi-Yau hypersurface with (h¹¹,h²¹)=(2,272), at large complex structure and weak string coupling"**

- URL: https://arxiv.org/abs/1912.10047
- **Relevance**: CRITICAL - Already in our resources, our methodology source

### arXiv:0806.0192 - Chen et al.
**"We study the flux parameter spaces for semi-realistic supersymmetric Pati-Salam models"**

- URL: https://arxiv.org/abs/0806.0192
- **Relevance**: MEDIUM - Pati-Salam models from flux

---

## 7. Toric Geometry & Kreuzer-Skarke

### arXiv:1811.04947 - Huang et al.
**"through toric geometry to the line bundle -6K_B. The Kreuzer-Skarke database includes all these examples"**

- URL: https://arxiv.org/abs/1811.04947
- **Relevance**: HIGH - Kreuzer-Skarke database usage

### arXiv:2512.14817 - MacFadden
**"We then apply this theory to the Kreuzer-Skarke (KS) database, where we encounter both FRSTs and vex triangulations."**

- URL: https://arxiv.org/abs/2512.14817
- Title: "Calabi-Yau Threefolds from Vex Triangulations"
- Date: 2025-12-16 (very recent!)
- **Relevance**: HIGH - New triangulation methods for KS database

---

## 8. Cosmological Constant Problem

### arXiv:0908.4324 - Fujii
**"The accelerating universe is closely related to today's version of the cosmological constant problem; fine-tuning and coincidence problems."**

- URL: https://arxiv.org/abs/0908.4324
- **Relevance**: MEDIUM - CC problem review

### arXiv:1402.0828 - Banks
**"I review three attempts to explain the small value of the cosmological constant... The String Landscape, SLED, and [others]"**

- URL: https://arxiv.org/abs/1402.0828
- Author: T. Banks
- **Relevance**: HIGH - Landscape approach to CC

---

## 9. Swampland Conjectures

### arXiv:1809.04512 - Danielsson et al.
**"We propose a quantum version of the swampland conjecture. Quantum instabilities of de Sitter space are directly related to [swampland]"**

- URL: https://arxiv.org/abs/1809.04512
- **Relevance**: MEDIUM - Quantum swampland

### arXiv:1906.05225 - Lust et al.
**"We study aspects of anti-de Sitter space in the context of the Swampland. We conjecture that the near-flat limit of pure AdS belongs to the Swampland."**

- URL: https://arxiv.org/abs/1906.05225
- **Relevance**: MEDIUM - AdS swampland

---

## 10. LessWrong/Rationalist Discussions

### Tegmark Multiverse & Mathematical Universe
Several LessWrong posts discuss Tegmark's multiverse classification and its relationship to string theory landscape:

- **"Pluralistic Existence in Many Many-Worlds"** - LW post comparing Tegmark's 4-level multiverse
- **"Shock Level 5: Big Worlds and Modal Realism"** - Roko's post on Mathematical Universe Hypothesis
- **"Towards a New Decision Theory"** - Wei Dai's post referencing Level IV multiverse

These connect to anthropic reasoning about why we observe our specific physical constants.

---

## Action Items

### Completed Downloads
- [x] arXiv:1907.10072 - Cole et al. GA flux vacua (already had!)
- [x] arXiv:1302.0529 - Damian et al. GA flux vacua (tex + pdf)
- [x] arXiv:2306.06160 - Scalable flux vacua search
- [x] arXiv:1707.00655 - ML string landscape methodology

### Still Needed
1. [ ] Download arXiv:2512.14817 (vex triangulations - very recent)
2. [ ] Download arXiv:2403.03245 (ML landscape exploration)
3. [ ] Download arXiv:2212.06187 (landscape/swampland review)

### Research Questions to Investigate
- [x] How does Damian et al.'s GA approach compare to ours? → They use MATLAB's built-in GA with hybrid local optimization
- [x] What fitness function did they use? → Minimize scalar potential V, looking for stable dS vacua
- [ ] How does Cole et al.'s fitness-distance correlation concept apply to our problem?
- [ ] Can we use roulette-wheel selection instead of tournament selection?
- [ ] Should we implement hybrid GA + local optimizer like Damian et al.?

### Key Lessons from Prior Work
1. **Fitness landscape matters**: GAs fail on "needle-in-a-haystack" - need smooth fitness landscape
2. **Hybrid approach**: Damian uses GA + fmincon hybrid for local refinement
3. **Population size**: ~100 individuals seems common
4. **Constraint handling**: Tadpole/Bianchi constraints handled via rejection or penalty
5. **Breeding**: Crossover operates on flux vectors; mutation modifies individual fluxes

---

## Query Examples for Future Research

```bash
# Search for specific author
cat > /tmp/query.json << 'JSONEOF'
{"sql": "SELECT * FROM alignment.search('Vafa swampland', kinds => ARRAY['paper'], limit_n => 30)"}
JSONEOF
curl -s -X POST https://api.exopriors.com/v1/alignment/query \
  -H "Authorization: Bearer exopriors_public_readonly_v1_2025" \
  -H "Content-Type: application/json" \
  -d @/tmp/query.json | jq '.rows'

# Search with metadata
cat > /tmp/query.json << 'JSONEOF'
{"sql": "SELECT s.*, e.metadata->>'abstract' FROM alignment.search('flux vacua statistics', kinds => ARRAY['paper'], limit_n => 20) s JOIN alignment.entities e ON e.id = s.id"}
JSONEOF
```

---

*This index will be updated as we discover more relevant papers.*
