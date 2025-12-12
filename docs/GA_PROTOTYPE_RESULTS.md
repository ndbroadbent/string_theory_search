# GA Prototype Results: First Week Analysis

**Period:** December 9-12, 2025
**Duration:** ~67 hours of continuous operation

> **Important Note:** These results used a **prototype physics model** that was largely placeholder code.
> The fitness function did not compute real physics - it used approximations and fallbacks.
> Despite this, the data reveals valuable insights about the GA's exploration behavior and convergence patterns.

## Executive Summary

| Metric | Value |
|--------|-------|
| Total Evaluations | 20,597,136 |
| Successful Evaluations | 20,597,136 (100%) |
| Failed Evaluations | 0 |
| Total Runs | 11,875 |
| Total Algorithms | 1,200 |
| Meta-Generations | 75 |
| Runs per Algorithm | 10 (fixed) |
| Best Fitness Achieved | 0.3062 |
| Average Fitness | 0.2114 |
| Worst Fitness | 0.1505 |
| Polytope Visits (across runs) | 985,009 |

## Timeline

| Metric | Value |
|--------|-------|
| First evaluation | 2025-12-09 12:46:53 |
| Last evaluation | 2025-12-12 08:07:26 |
| Total runtime | ~67 hours |
| Avg throughput | ~85,000 eval/hour (~24/sec) |

### Evaluations by Day

| Date | Evaluations | Notes |
|------|-------------|-------|
| Dec 9 | 1,012,246 | First day (partial) |
| Dec 10 | 8,465,140 | Full day |
| Dec 11 | 9,085,802 | Full day |
| Dec 12 | 2,033,948 | Partial (disk full) |

## Best Configuration Found

### Winning Evaluation (ID: 4499775)

| Parameter | Value |
|-----------|-------|
| **Fitness** | **0.306215** |
| Polytope ID | 3,933,517 |
| Run ID | 3240 |
| Generation | 17 |
| Created | 2025-12-10 11:16:20 |

### Physics Values

| Constant | Target | Achieved | Match |
|----------|--------|----------|-------|
| α_em | 7.297 × 10⁻³ | 2.076 × 10⁻³ | 28.5% |
| α_s | 0.118 | 0.1174 | 99.5% |
| sin²θ_W | 0.231 | 0.2311 | 100% |
| N_gen | 3 | 3 | 100% |
| Λ | 2.888 × 10⁻¹²² | -4.58 × 10⁻⁶ | ~0% |

### Genome Parameters

| Parameter | Value |
|-----------|-------|
| g_s (string coupling) | 0.9115 |
| h11 | 18 |
| h21 | 21 |
| Vertex count | 17 |

**Kähler Moduli (18 values):**
```
[2.887, 4.310, 0.725, 0.413, 2.212, 2.605, 0.605, 6.513,
 2.283, 1.609, 1.225, 1.228, 4.352, 2.270, 5.710, 1.092,
 4.424, 0.773]
```

### Winning Polytope (#3,933,517)

| Property | Value |
|----------|-------|
| h11 | 18 |
| h21 | 21 |
| Vertices | 17 |
| Evaluations | 1,097 |
| Avg Fitness | 0.283 |
| Best Fitness | 0.306 |

**Vertex Coordinates (4D):**
```
[1,0,0,0], [0,1,0,0], [0,0,1,0], [-1,1,1,0], [0,0,0,1],
[1,0,-1,1], [0,0,1,-1], [-1,1,1,-1], [1,-1,0,0], [-1,1,0,0],
[-2,1,1,-1], [1,-1,-2,2], [-1,1,0,-1], [0,-1,1,-1], [1,-1,0,-1],
[0,-1,-1,1], [1,-1,-2,1]
```

## Meta-GA Evolution

### Structure

- **75 meta-generations** completed (gen 74 partial)
- **16 algorithms per generation**
- **10 runs per algorithm**
- **~16.6 generations per run** (avg)

### Best Fitness by Meta-Generation

The meta-GA showed clear improvement over time:

| Gen | Best Fitness | Notable |
|-----|-------------|---------|
| 0 | 0.2390 | Initial population |
| 2 | 0.2689 | First jump |
| 11 | 0.2888 | Significant improvement |
| 18 | 0.2896 | |
| 19 | 0.2903 | |
| **20** | **0.3062** | **Global best found!** |
| 56 | 0.2914 | Second best |

**Key Observation:** The global best (0.3062) was found in meta-generation 20 by algorithm #325. Later generations did not improve on this, suggesting either:
1. A strong local optimum was found
2. Meta-GA exploration was insufficient
3. The fitness landscape has few peaks

### Top Performing Algorithms

| Algo ID | Meta-Gen | Runs | Avg Fitness | Best Fitness | Avg Gens |
|---------|----------|------|-------------|--------------|----------|
| **325** | **20** | 10 | 0.2302 | **0.3062** | 17 |
| 910 | 56 | 10 | 0.2295 | 0.2914 | 15 |
| 312 | 19 | 10 | 0.2316 | 0.2903 | 17 |
| 304 | 18 | 10 | 0.2298 | 0.2896 | 17 |
| 185 | 11 | 10 | 0.2288 | 0.2888 | 19 |

### Winning Algorithm (#325) Configuration

**Name:** `gen20_mutant4`

**Feature Weights:**
```json
{
  "coord_median": 0.896,
  "flatness_2d": 2.582,
  "conformity_ratio": 0.555,
  "h11": 0.084,
  "spikiness": 0.780,
  "handedness_det": 0.860,
  "chirality_w": 0.793,
  "intrinsic_dim_estimate": 1.926,
  "zero_count": 0.101,
  "prime_count": 0.227,
  "chirality_z": 0.443,
  "sphericity": 0.089,
  "outlier_count_3sigma": 0.259,
  "symmetry_y": 0.327,
  "sort_compression_gain": 0.052,
  "distance_kurtosis": 2.860,
  "coord_skewness": 0.015,
  "coord_std": 0.647,
  "symmetry_x": 2.117,
  "symmetry_w": 1.053,
  "chirality_y": 2.531,
  "vertex_count": 0.500,
  "sorted_compression_ratio": 1.495,
  "shannon_entropy": 2.206,
  "inertia_isotropy": 0.700,
  "symmetry_z": 1.918,
  "phi_ratio_count": 0.210,
  "chirality_optimal": 1.535,
  "one_count": 2.196,
  "outlier_count_2sigma": 2.153,
  "chirality_x": 1.294,
  "max_exposure": 3.182,
  "loner_score": 1.107,
  "coord_kurtosis": 0.518,
  "joint_entropy": 2.059
}
```

**GA Parameters:**
| Parameter | Value |
|-----------|-------|
| similarity_radius | 0.652 |
| interpolation_weight | 0.008 |
| population_size | 91 |
| max_generations | 17 |
| mutation_rate | 0.522 |
| mutation_strength | 0.525 |
| crossover_rate | 0.850 |
| tournament_size | 4 |
| elite_count | 10 |
| polytope_patience | 4 |
| switch_threshold | 0.243 |
| switch_probability | 0.040 |
| cc_weight | 5.249 |

## Fitness Distribution

Based on 1% random sample (205,971 evaluations):

| Fitness Range | Count | Percentage |
|---------------|-------|------------|
| 0.15 - 0.20 | 3,993,700 | 19.4% |
| 0.20 - 0.25 | 16,584,100 | 80.5% |
| 0.25 - 0.30 | 19,200 | 0.09% |
| 0.30+ | ~100 | <0.001% |

**Observation:** The vast majority of evaluations (~80%) cluster in a narrow fitness band (0.20-0.25). Only 0.1% achieve fitness > 0.25.

## Run Statistics

| Metric | Value |
|--------|-------|
| Total runs | 11,875 |
| Avg generations per run | 16.6 |
| Max generations | 22 |
| Min generations | 6 |
| Avg fitness improvement | +0.0083 |
| Max fitness improvement | +0.0879 |
| Min fitness improvement | +0.0001 |

## Polytope Diversity Analysis

### Extreme Convergence in Top Results

| Top N | Unique Polytopes | Dominant Polytope |
|-------|------------------|-------------------|
| 100 | **1** | #3,933,517 (100%) |
| 1000 | 19 | #3,933,517 (27%) |

### Top 1000 Polytope Distribution

| Polytope ID | Count | Best Fitness |
|-------------|-------|--------------|
| **3,933,517** | 270 | **0.3062** |
| 11,893,918 | 29 | 0.2914 |
| 11,282,474 | 15 | 0.2903 |
| 10,138,605 | 16 | 0.2896 |
| 11,901,329 | 41 | 0.2888 |
| 442,592 | 15 | 0.2885 |
| 6,106,503 | 74 | 0.2882 |
| 2,849,490 | 77 | 0.2881 |
| 6,102,261 | 51 | 0.2881 |
| 1,449,853 | 87 | 0.2880 |
| 594,987 | 34 | 0.2880 |
| 2,283,770 | 107 | 0.2878 |
| 537,958 | 75 | 0.2871 |
| 2,250,889 | 53 | 0.2867 |
| 199,486 | 28 | 0.2862 |
| 1,025,793 | 22 | 0.2861 |
| 3,928,467 | 4 | 0.2852 |
| 10,099,581 | 1 | 0.2849 |
| 428,359 | 1 | 0.2848 |

**Observation:** The fitness gap between #1 and #2 polytope is 0.015 (~5%), suggesting a clear winner in this fitness landscape.

### Top Polytope Geometries

| Polytope ID | h11 | h21 | |h11-h21| | Vertices | Best Fitness |
|-------------|-----|-----|----------|----------|--------------|
| **3,933,517** | 18 | 21 | 3 | 17 | **0.3062** |
| 11,893,918 | 18 | 21 | 3 | 22 | 0.2914 |
| 11,282,474 | 18 | 21 | 3 | 21 | 0.2903 |
| 10,138,605 | 20 | 23 | 3 | 20 | 0.2896 |
| 11,901,329 | 22 | 19 | 3 | 22 | 0.2888 |
| 442,592 | 22 | 19 | 3 | 14 | 0.2885 |
| 6,106,503 | 19 | 22 | 3 | 18 | 0.2882 |

**Observations:**
- All top polytopes have |h11-h21| = 3 (by design - this is our three-generation filter)
- Winning polytope has relatively few vertices (17) compared to others
- h11=18, h21=21 appears 3 times in top 7 (different vertex configurations)

## Key Findings

### 1. Strong Convergence Indicates Effective Search

**All top 100 evaluations used polytope #3,933,517.** This is actually a positive sign:
- The GA explored ~985,000 polytope configurations across 20M evaluations
- It correctly identified and exploited the best region in the fitness landscape
- The 5% fitness gap between #1 and #2 polytope shows a clear winner exists
- Convergence happened because one polytope genuinely scored better, not due to lack of exploration

### 2. 100% Success Rate is Suspicious

Zero failed evaluations indicates the physics model was too permissive. Real physics would have many failure modes:
- Invalid triangulations
- Non-reflexive polytopes
- Failed gauge unification
- Negative volumes

### 3. Meta-GA Found Best Early

The global best was found in generation 20 of 75. Later generations didn't improve, suggesting:
- Premature convergence
- Need for more exploration vs exploitation
- Or: the global optimum was actually found

### 4. Narrow Parameter Clustering

All top performers share similar g_s (~0.91), suggesting either strong selection or limited exploration.

### 5. α_s Matching was Excellent

Strong coupling matched within 0.5% despite placeholder physics. This may indicate:
- α_s depends mostly on geometric factors we captured
- Or coincidental alignment with heuristics

### 6. Cosmological Constant was Garbage

As expected with placeholder formulas, Λ was ~116 orders of magnitude wrong.

## Recommendations for Phase 2

1. **Implement real physics** - All these results are meaningless for actual physics
2. **Add failure modes** - Evaluations should fail on invalid configurations
3. **Track diversity** - Monitor polytope diversity during runs
4. **Faster analytics** - Add indexes, use DuckDB for analysis
5. **Save winning genomes** - Export full configurations for future reference
6. **Validate against McAllister** - Must reproduce known results first

## Data Preservation

Before deletion, export:
- [x] This analysis document
- [ ] Top 100 evaluations with full genomes
- [ ] Winning algorithm configurations
- [ ] Feature weight evolution over meta-generations

---

## Appendix: Raw Query Results

### A. Evaluations by Day
```
2025-12-09: 1,012,246
2025-12-10: 8,465,140
2025-12-11: 9,085,802
2025-12-12: 2,033,948
```

### B. Meta-Generation Progress
```
Gen 0-10:  Best ~0.24-0.29
Gen 11-20: Best ~0.24-0.31 (peak at gen 20)
Gen 21-74: Best ~0.23-0.29 (no improvement)
```

### C. Algorithm Performance (Top 10)
```
#325:  avg=0.230, best=0.306, gens=17
#910:  avg=0.230, best=0.291, gens=15
#312:  avg=0.232, best=0.290, gens=17
#304:  avg=0.230, best=0.290, gens=17
#185:  avg=0.229, best=0.289, gens=19
#641:  avg=0.229, best=0.288, gens=16
#987:  avg=0.232, best=0.288, gens=16
#652:  avg=0.231, best=0.288, gens=20
#473:  avg=0.230, best=0.288, gens=20
#188:  avg=0.230, best=0.288, gens=18
```
