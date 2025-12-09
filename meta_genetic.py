#!/usr/bin/env python3
"""
Meta-Genetic Algorithm for evolving search strategies.

The outer GA evolves "algorithm genomes" - configurations that define
how the inner physics search operates. The fitness of an algorithm
is the rate of improvement it achieves on the physics fitness.

Key insight: the embedding weights ARE the genome.
"""

import json
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional
from pathlib import Path
import time


@dataclass
class AlgorithmGenome:
    """
    Genome for a search algorithm configuration.

    These parameters define HOW the inner GA searches.
    The meta-GA evolves these to find what works best.
    """

    # Embedding weights (the key innovation)
    # Higher weight = that embedding type matters more for similarity
    embedding_weights: dict[str, float] = field(default_factory=lambda: {
        "hodge": 1.0,
        "vertex_stats": 1.0,
        "vertex_coords_flat": 0.5,
        "coord_histogram": 0.5,
        "f_vector": 0.5,
        "geometric": 1.0,
        "algebraic": 0.5,
    })

    # Polytope selection strategy weights
    # These control how we pick the next polytope to evaluate
    strategy_weights: dict[str, float] = field(default_factory=lambda: {
        "random": 0.2,           # Pure random from database
        "nearest_neighbor": 0.5, # Query similar to current best
        "explore_cluster": 0.2,  # UCB on clusters
        "transition_walk": 0.1,  # Follow transition graph (if available)
    })

    # Mutation parameters
    mutation_rate: float = 0.1
    mutation_magnitude: float = 0.2  # Std dev for gaussian mutations
    adaptive_mutation: bool = True   # Increase mutation when stuck

    # Which parameters to mutate (weights)
    mutate_kahler: float = 1.0
    mutate_complex: float = 1.0
    mutate_flux: float = 0.5
    mutate_gs: float = 0.3

    # Neighbor query parameters
    neighbor_k: int = 50           # How many neighbors to consider
    neighbor_eval_fraction: float = 0.1  # Fraction to actually evaluate

    # Exploration vs exploitation
    exploration_rate: float = 0.3  # Probability of exploring vs exploiting
    temperature: float = 1.0       # Softmax temperature for selection

    # Meta-fitness tracking (filled in during evaluation)
    meta_fitness: Optional[float] = None
    fitness_history: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "embedding_weights": self.embedding_weights,
            "strategy_weights": self.strategy_weights,
            "mutation_rate": self.mutation_rate,
            "mutation_magnitude": self.mutation_magnitude,
            "adaptive_mutation": self.adaptive_mutation,
            "mutate_kahler": self.mutate_kahler,
            "mutate_complex": self.mutate_complex,
            "mutate_flux": self.mutate_flux,
            "mutate_gs": self.mutate_gs,
            "neighbor_k": self.neighbor_k,
            "neighbor_eval_fraction": self.neighbor_eval_fraction,
            "exploration_rate": self.exploration_rate,
            "temperature": self.temperature,
            "meta_fitness": self.meta_fitness,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AlgorithmGenome":
        genome = cls()
        for key, value in d.items():
            if hasattr(genome, key):
                setattr(genome, key, value)
        return genome

    @classmethod
    def random(cls) -> "AlgorithmGenome":
        """Create a random algorithm genome."""
        genome = cls()

        # Randomize embedding weights
        for key in genome.embedding_weights:
            genome.embedding_weights[key] = np.random.uniform(0, 2)

        # Randomize strategy weights (must sum to 1)
        raw = {k: np.random.uniform(0, 1) for k in genome.strategy_weights}
        total = sum(raw.values())
        genome.strategy_weights = {k: v/total for k, v in raw.items()}

        # Randomize other parameters
        genome.mutation_rate = np.random.uniform(0.01, 0.3)
        genome.mutation_magnitude = np.random.uniform(0.05, 0.5)
        genome.adaptive_mutation = np.random.random() > 0.5

        genome.mutate_kahler = np.random.uniform(0, 2)
        genome.mutate_complex = np.random.uniform(0, 2)
        genome.mutate_flux = np.random.uniform(0, 2)
        genome.mutate_gs = np.random.uniform(0, 2)

        genome.neighbor_k = int(np.random.uniform(10, 200))
        genome.neighbor_eval_fraction = np.random.uniform(0.05, 0.3)

        genome.exploration_rate = np.random.uniform(0.1, 0.5)
        genome.temperature = np.random.uniform(0.1, 3.0)

        return genome


def mutate_algorithm(genome: AlgorithmGenome, rate: float = 0.2) -> AlgorithmGenome:
    """Mutate an algorithm genome."""
    new = AlgorithmGenome.from_dict(genome.to_dict())

    # Mutate embedding weights
    for key in new.embedding_weights:
        if np.random.random() < rate:
            new.embedding_weights[key] *= np.random.uniform(0.5, 2.0)
            new.embedding_weights[key] = max(0, min(5, new.embedding_weights[key]))

    # Mutate strategy weights
    for key in new.strategy_weights:
        if np.random.random() < rate:
            new.strategy_weights[key] *= np.random.uniform(0.5, 2.0)
    # Renormalize
    total = sum(new.strategy_weights.values())
    if total > 0:
        new.strategy_weights = {k: v/total for k, v in new.strategy_weights.items()}

    # Mutate scalars
    if np.random.random() < rate:
        new.mutation_rate *= np.random.uniform(0.5, 2.0)
        new.mutation_rate = max(0.01, min(0.5, new.mutation_rate))

    if np.random.random() < rate:
        new.mutation_magnitude *= np.random.uniform(0.5, 2.0)
        new.mutation_magnitude = max(0.01, min(1.0, new.mutation_magnitude))

    if np.random.random() < rate:
        new.neighbor_k = int(new.neighbor_k * np.random.uniform(0.5, 2.0))
        new.neighbor_k = max(5, min(500, new.neighbor_k))

    if np.random.random() < rate:
        new.exploration_rate *= np.random.uniform(0.5, 2.0)
        new.exploration_rate = max(0.05, min(0.9, new.exploration_rate))

    if np.random.random() < rate:
        new.temperature *= np.random.uniform(0.5, 2.0)
        new.temperature = max(0.1, min(10, new.temperature))

    return new


def crossover_algorithms(a: AlgorithmGenome, b: AlgorithmGenome) -> AlgorithmGenome:
    """Crossover two algorithm genomes."""
    child = AlgorithmGenome()

    # Blend embedding weights
    for key in child.embedding_weights:
        alpha = np.random.random()
        child.embedding_weights[key] = alpha * a.embedding_weights[key] + (1-alpha) * b.embedding_weights[key]

    # Blend strategy weights
    for key in child.strategy_weights:
        alpha = np.random.random()
        child.strategy_weights[key] = alpha * a.strategy_weights[key] + (1-alpha) * b.strategy_weights[key]
    # Renormalize
    total = sum(child.strategy_weights.values())
    child.strategy_weights = {k: v/total for k, v in child.strategy_weights.items()}

    # Random parent for other params
    parent = a if np.random.random() > 0.5 else b
    child.mutation_rate = parent.mutation_rate
    child.mutation_magnitude = parent.mutation_magnitude
    child.adaptive_mutation = parent.adaptive_mutation
    child.mutate_kahler = parent.mutate_kahler
    child.mutate_complex = parent.mutate_complex
    child.mutate_flux = parent.mutate_flux
    child.mutate_gs = parent.mutate_gs
    child.neighbor_k = parent.neighbor_k
    child.neighbor_eval_fraction = parent.neighbor_eval_fraction
    child.exploration_rate = parent.exploration_rate
    child.temperature = parent.temperature

    return child


def compute_meta_fitness(fitness_history: list[float], method: str = "improvement_rate") -> float:
    """
    Compute meta-fitness from a fitness curve.

    Methods:
    - improvement_rate: (final - initial) / generations
    - area_under_curve: total integral of fitness
    - weighted_improvement: early improvements weighted more
    """
    if not fitness_history or len(fitness_history) < 2:
        return 0.0

    history = np.array(fitness_history)

    if method == "improvement_rate":
        return (history[-1] - history[0]) / len(history)

    elif method == "area_under_curve":
        return np.trapz(history) / len(history)

    elif method == "weighted_improvement":
        # Weight early improvements more (exponential decay)
        improvements = np.diff(history)
        weights = np.exp(-np.arange(len(improvements)) * 0.01)
        return np.sum(improvements * weights) / np.sum(weights)

    else:
        raise ValueError(f"Unknown method: {method}")


class MetaGA:
    """
    Meta-Genetic Algorithm that evolves search algorithms.
    """

    def __init__(
        self,
        population_size: int = 20,
        inner_generations: int = 100,
        run_inner_ga: Optional[Callable] = None,
        output_dir: str = "./meta_ga_results",
    ):
        self.population_size = population_size
        self.inner_generations = inner_generations
        self.run_inner_ga = run_inner_ga  # Callback to run inner GA
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize population
        self.population: list[AlgorithmGenome] = []
        self.generation = 0
        self.best_ever: Optional[AlgorithmGenome] = None

    def initialize_population(self):
        """Create initial population of random algorithms."""
        self.population = []

        # A few hand-designed "reasonable" starting points
        defaults = AlgorithmGenome()
        self.population.append(defaults)

        # Emphasis on different embeddings
        for emb_name in defaults.embedding_weights.keys():
            genome = AlgorithmGenome()
            for k in genome.embedding_weights:
                genome.embedding_weights[k] = 0.1
            genome.embedding_weights[emb_name] = 2.0
            self.population.append(genome)

        # Fill rest with random
        while len(self.population) < self.population_size:
            self.population.append(AlgorithmGenome.random())

    def evaluate_population(self):
        """Run inner GA for each algorithm and compute meta-fitness."""
        for i, genome in enumerate(self.population):
            print(f"\n=== Evaluating algorithm {i+1}/{len(self.population)} ===")
            print(f"Embedding weights: {genome.embedding_weights}")
            print(f"Strategy: {genome.strategy_weights}")

            if self.run_inner_ga:
                # Run the actual inner GA
                fitness_history = self.run_inner_ga(genome, self.inner_generations)
            else:
                # Placeholder: simulate a fitness curve
                # In real usage, this calls the Rust GA
                fitness_history = self._simulate_inner_ga(genome)

            genome.fitness_history = fitness_history
            genome.meta_fitness = compute_meta_fitness(fitness_history, "weighted_improvement")

            print(f"Meta-fitness: {genome.meta_fitness:.6f}")
            print(f"Final fitness: {fitness_history[-1]:.4f}")

    def _simulate_inner_ga(self, genome: AlgorithmGenome) -> list[float]:
        """Placeholder simulation of inner GA for testing."""
        # Simulate: better embedding weights → faster improvement
        quality = sum(genome.embedding_weights.values()) / len(genome.embedding_weights)
        noise = np.random.randn(self.inner_generations) * 0.01

        curve = []
        fitness = 0.3 + np.random.random() * 0.1
        for i in range(self.inner_generations):
            # Improvement rate depends on genome quality
            improvement = 0.001 * quality * (1 - fitness) + noise[i]
            fitness = max(0, min(1, fitness + improvement))
            curve.append(fitness)

        return curve

    def select_and_reproduce(self):
        """Select best algorithms and create next generation."""
        # Sort by meta-fitness
        self.population.sort(key=lambda g: g.meta_fitness or 0, reverse=True)

        # Track best ever
        if self.best_ever is None or (self.population[0].meta_fitness or 0) > (self.best_ever.meta_fitness or 0):
            self.best_ever = self.population[0]

        # Keep top 25%
        n_keep = max(2, self.population_size // 4)
        survivors = self.population[:n_keep]

        # Create offspring
        new_population = list(survivors)  # Keep survivors

        while len(new_population) < self.population_size:
            if np.random.random() < 0.7:
                # Crossover
                p1, p2 = np.random.choice(survivors, size=2, replace=False)
                child = crossover_algorithms(p1, p2)
            else:
                # Mutation only
                parent = np.random.choice(survivors)
                child = mutate_algorithm(parent, rate=0.3)

            # Always mutate a bit
            child = mutate_algorithm(child, rate=0.1)
            new_population.append(child)

        self.population = new_population
        self.generation += 1

    def save_state(self):
        """Save current state to disk."""
        state = {
            "generation": self.generation,
            "population": [g.to_dict() for g in self.population],
            "best_ever": self.best_ever.to_dict() if self.best_ever else None,
        }

        path = self.output_dir / f"meta_ga_gen_{self.generation:04d}.json"
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

        # Also save best ever separately
        if self.best_ever:
            best_path = self.output_dir / "best_algorithm.json"
            with open(best_path, "w") as f:
                json.dump(self.best_ever.to_dict(), f, indent=2)

    def run(self, meta_generations: int = 10):
        """Run the meta-GA."""
        print(f"Starting Meta-GA with population={self.population_size}")
        print(f"Each algorithm runs for {self.inner_generations} inner generations")

        self.initialize_population()

        for gen in range(meta_generations):
            print(f"\n{'='*60}")
            print(f"META-GENERATION {gen + 1}/{meta_generations}")
            print(f"{'='*60}")

            self.evaluate_population()

            # Log best
            best = max(self.population, key=lambda g: g.meta_fitness or 0)
            print(f"\nBest this generation: meta_fitness={best.meta_fitness:.6f}")
            print(f"Best embedding weights: {best.embedding_weights}")

            self.save_state()
            self.select_and_reproduce()

        print(f"\n{'='*60}")
        print("META-GA COMPLETE")
        print(f"Best algorithm found:")
        print(json.dumps(self.best_ever.to_dict(), indent=2))


if __name__ == "__main__":
    # Test run with simulated inner GA
    meta = MetaGA(
        population_size=10,
        inner_generations=50,
    )
    meta.run(meta_generations=5)
