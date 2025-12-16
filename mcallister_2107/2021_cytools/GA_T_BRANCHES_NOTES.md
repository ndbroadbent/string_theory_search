⏺ Current data:
  - With frontier=8, population=24: ~20-25s per epoch, found 41 branches in 60s (3 epochs)
  - With frontier=16, population=64: ~50-60s per epoch, found 162 branches in 300s (5-6 epochs)

  For a 10 minute (600s) run:

  Option A: frontier=8, population=24
  - ~25s per epoch → ~24 epochs in 10 min
  - Faster iterations, more evolution cycles
  - But smaller population = less diversity per epoch

  Option B: frontier=16, population=64
  - ~55s per epoch → ~11 epochs in 10 min
  - More diversity per epoch
  - But fewer evolution cycles

  Option C: frontier=12, population=48 (middle ground)
  - ~40s per epoch → ~15 epochs
  - Balance of diversity and iteration speed

  For branch discovery, I think more evaluations with moderate diversity is better than fewer evaluations with high diversity. The GA is already converging well with the smaller population.

  I'd suggest: frontier=12, population=48 as a balanced choice, or stick with frontier=8, population=32 for more epochs.

  Actually, let me reconsider. The key metrics are:
  1. Total unique branches found
  2. Quality of best branch (proximity to McAllister)

  In 60s with frontier=8, pop=24 we got 41 branches and 80 evaluations.
  In 300s with frontier=16, pop=64 we got 162 branches and 336 evaluations.

  The ratio is similar - about 0.5 branches per evaluation. So the key is maximizing total evaluations while maintaining enough diversity to explore.

  For 10 minutes, I'd recommend:
  - frontier=10, population=40 - ~30s per epoch, ~20 epochs, good balance
  - Or frontier=8, population=32 - ~25s per epoch, ~24 epochs, more iterations

  For 10 minutes, I'd recommend frontier=10, population=40:

  | Config | Epoch time | Epochs in 10min | Evals | Diversity   |
  |--------|------------|-----------------|-------|-------------|
  | 8/24   | ~22s       | ~27             | ~650  | Low         |
  | 10/40  | ~32s       | ~19             | ~760  | Medium      |
  | 12/48  | ~38s       | ~16             | ~768  | Medium-High |
  | 16/64  | ~55s       | ~11             | ~704  | High        |

  The 10/40 config gives the most total evaluations while maintaining reasonable diversity. Let me run it:

⏺ Bash(rm -f mcallister_2107/2021_cytools/ga_branches_output/5-81-3213_branches.jsonl 2>/dev/null; timeout 660 uv run python -u mcallister_2107/2021_cytools/explore_t_…) timeout: 12m 0s
  ⎿  Init 8/10...
     Initial best fitness: 58.46
     Initial best V_string: 563.78, n_steps: 32
       Epoch 1 starting...
       Generating 4 children per parent...
     +16 more lines (36s)
