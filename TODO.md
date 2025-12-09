# String Theory Landscape Explorer - TODO

## Visualization Ideas

### Genealogy Trees
- [ ] Track ancestry for each genome with unique IDs
- [ ] Parent ID references for lineage tracking
- [ ] Export to DOT graph format
- [ ] Convert to PNG with graphviz
- [ ] Tree visualization: single root at top, branching downward
- [ ] Color nodes by fitness (gradient from red→yellow→green)
- [ ] Node size proportional to fitness improvement
- [ ] Optional: toggle ancestry tracking for performance
- [ ] Show mutations/crossovers as edge labels

### Polytope Renders
- [ ] 2D projections of 4D polytope vertices
- [ ] Batch render current generation's polytopes
- [ ] Cool visual style (glow effects, color by Hodge numbers)
- [ ] Grid layout showing population diversity
- [ ] Animate evolution over generations (GIF/video)

### Dashboard / Live View
- [ ] Real-time fitness plot over generations
- [ ] Population diversity metrics
- [ ] Best candidate physics values vs targets
- [ ] Moduli space coverage heatmap

## Search Algorithm Improvements

### Adaptive Strategies
- [ ] Track which moduli mutations improve fitness most
- [ ] Learn promising polytope regions
- [ ] Multi-objective optimization (Pareto front)

### Parallelization
- [ ] Distribute across multiple servers
- [ ] Shared hall of fame between nodes
- [ ] Island model with migration

## Physics Bridge

### CYTools Integration
- [ ] Cache expensive computations (Kähler cone, etc.)
- [ ] Batch physics evaluations
- [ ] Pre-filter polytopes by geometric constraints

### Accuracy
- [ ] Loop corrections for gauge couplings
- [ ] Threshold corrections
- [ ] Moduli stabilization validation

## Data Management

- [ ] Better result organization by fitness score
- [ ] Searchable database of found compactifications
- [ ] Compare runs across servers
- [ ] Export best results to standardized format

## CLI Improvements

- [ ] `--visualize` flag to generate graphs during run
- [ ] `--ancestry` flag to enable lineage tracking
- [ ] `--render-interval` for periodic snapshots
- [ ] Resume from saved state
