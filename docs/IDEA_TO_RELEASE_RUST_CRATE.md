Calabi-Yau Manifold Tools
Idea: Build an extremely performant Rust toolkit for evaluating CY manifolds.

Even if I will never be personally eligible for a Nobel Prize, by contributing such a toolkit to the HET physics community, I might one day be able to get a citation in the paper that actually does win a Nobel Prize.




Take the Frankenstein pipeline I built with a mix of Rust, Python, and C++, and reimplement every algorithm from scratch in Rust.

Take all the principles from rack-gateway - ultra-strict linting rules, ultra clean code, but even further - massive benchmarking suite.

Idea: Every single function will have completely configurable caching options, with the capability of automatically determining the optimal strategy based on runtime analysis depending on use-case (and being able to evolve caching techniques automatically via genetic algorithms throughout the entire pipeline).

Caching strategies:
No caching
Cache in memory
Cache on disk
Load from precomputed database if exists
Caches can also have indexing and compression settings for both keys and values.
No index or compression
Build index from parameters
Multiple cache storage tree structures
 Plain storage
linked list from first parameter => second parameter, etc.
Compression with zstd for larger values
Automatically build a zstd compression table from all values in the current cache, use it to optimize storage automatically
Caching sets of data in blocks
auto-cluster cached data that is commonly used together in blocks, e.g. 250MB chunks for certain types of polytopes, or with certain cohomologies
Auto store and retrieve from memory => local disk => server disk
Strategy: cache on first use, or use large bloom filters in front of all functions to automatically cache on second use. E.g. one ultra efficient data structure that automatically only caches things when they’re actually used more than once. Plus a meta-cache to transform the function inputs into a bloom filter bit index. Could even do bloom filter bit index for each function parameter individually, and then XOR them together.

Cache metrics - search for functions that have extremely low or no cache hits and disable the cache for them (by default - can still be enabled by the user).

Optimal caching/storage/compression settings can be derived via thorough benchmarking and genetic algorithms.

Then all you need to do is say:
I have 16GB of RAM
512GB of NVMe storage
1TB of space on HDDs for long term storage

Then the entire system as a whole can intelligently learn how to best use those available storage tiers, either live, or with upfront pre-computation. With some cache storage being permanent, others being weighted by general importance or how often they were used.

And you can just load in all the existing data you already have into the cache, which also can serve the same purpose as a database. E.g. polytope vertices are also stored in there, indexed by a single incrementing id.



 Also machine learning. Just slow our slow pipeline to produce a huge amount of training data - every single number across the whole stack - every vertice, all the calculations we run throughout that whole pipeline (especially the cheap ones), plus all our random heuristics, and just run some machine learning on my MacBook GPU to see if it’s all random noise or if anything stands out. Include e.g. 1000 runs of our advanced genetic algo as well, and include sampled steps across fitness - e.g. the input tokens should include the full set of calculations for all 10 sampled steps across the run, and it should also include synthesized data like mean, min, max, std dev, rate of improvement (and statistics about that rate - e.g. smoothness)
