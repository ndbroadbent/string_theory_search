#!/usr/bin/env python3
"""
Genetic Algorithm search for (K, M) pairs on McAllister's polytope.

Compare GA vs random sampling for finding small V₀.

Genome: [K_0, K_1, K_2, K_3, M_0, M_1, M_2, M_3] (8 integers)
Fitness: -log10(|V_0|) (higher = smaller V₀ = better)

Based on: Cole, Schachner, Shiu (arXiv:1907.10072)
"""
import sys
sys.path.insert(0, str(__file__).replace('/mcallister_2107/search_km_via_genetic_algo.py', '/vendor/cytools_latest/src'))

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import time


@dataclass
class Individual:
    """A single (K, M) configuration."""
    K: np.ndarray
    M: np.ndarray
    fitness: float = float('-inf')
    # Cached physics
    p: Optional[np.ndarray] = None
    eK0: Optional[float] = None
    g_s: Optional[float] = None
    W_0: Optional[float] = None
    V_0: Optional[float] = None


class GeneticAlgorithm:
    def __init__(self, kappa, gv_invariants, h11=4,
                 population_size=1000,
                 flux_range=(-15, 16),
                 mutation_rate=0.1,
                 mutation_strength=3,
                 crossover_rate=0.7,
                 elite_fraction=0.1,
                 V_CY=4711.83):
        self.kappa = kappa
        self.gv_invariants = gv_invariants
        self.h11 = h11
        self.population_size = population_size
        self.flux_range = flux_range
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.crossover_rate = crossover_rate
        self.elite_count = max(1, int(population_size * elite_fraction))
        self.V_CY = V_CY

        self.population = []
        self.generation = 0
        self.best_ever = None
        self.stats = {
            'evaluated': 0,
            'valid': 0,
            'has_racetrack': 0,
        }

    def random_flux(self):
        """Generate random flux vector."""
        return np.array([np.random.randint(*self.flux_range) for _ in range(self.h11)])

    def create_individual(self, K=None, M=None):
        """Create and evaluate an individual."""
        if K is None:
            K = self.random_flux()
        if M is None:
            M = self.random_flux()

        ind = Individual(K=K.copy(), M=M.copy())
        self.evaluate(ind)
        return ind

    def check_N_invertible(self, M):
        """Check if N = κ_abc M^c is invertible."""
        N = np.einsum('abc,c->ab', self.kappa, M)
        det = np.linalg.det(N)
        if abs(det) < 1e-10:
            return False, None
        return True, N

    def check_tadpole(self, K, M, Q_D3=500):
        """Check tadpole constraint."""
        return -0.5 * np.dot(M, K) <= Q_D3

    def compute_racetrack(self, p, M):
        """Compute g_s and W_0 from racetrack mechanism."""
        curves_by_action = {}
        for q, N_q in self.gv_invariants.items():
            q = np.array(q)
            action = np.dot(q, p)
            if action > 0:
                M_dot_q = np.dot(M, q)
                if M_dot_q != 0:
                    key = round(action, 6)
                    if key not in curves_by_action:
                        curves_by_action[key] = []
                    curves_by_action[key].append({
                        'q': q, 'N_q': N_q, 'M_dot_q': M_dot_q, 'action': action,
                    })

        if len(curves_by_action) < 2:
            return None

        sorted_actions = sorted(curves_by_action.keys())
        action1, action2 = sorted_actions[0], sorted_actions[1]

        def sum_coeff(action):
            return sum(c['M_dot_q'] * c['N_q'] * c['action'] for c in curves_by_action[action])

        c1, c2 = sum_coeff(action1), sum_coeff(action2)
        if c1 == 0 or c2 == 0:
            return None

        delta_action = action2 - action1
        if delta_action <= 0:
            return None

        ratio = -c2 / c1
        if ratio <= 0:
            return None

        g_s = 2 * np.pi * delta_action / np.log(ratio)
        if g_s <= 0 or g_s > 1:
            return None

        exponent = -2 * np.pi * action1 / g_s
        W_0 = 0.0 if exponent < -500 else abs(c1) * np.exp(exponent)

        return {'g_s': g_s, 'W_0': W_0}

    def evaluate(self, ind: Individual):
        """Evaluate fitness of an individual."""
        self.stats['evaluated'] += 1

        # Check N invertible
        invertible, N = self.check_N_invertible(ind.M)
        if not invertible:
            ind.fitness = -1000
            return

        # Compute p
        try:
            p = np.linalg.solve(N, ind.K)
        except np.linalg.LinAlgError:
            ind.fitness = -1000
            return

        # Check p positive (Kähler cone)
        if not np.all(p > 0):
            ind.fitness = -500
            return

        # Check tadpole
        if not self.check_tadpole(ind.K, ind.M):
            ind.fitness = -500
            return

        # Compute e^K0
        kappa_p3 = np.einsum('abc,a,b,c->', self.kappa, p, p, p)
        if abs(kappa_p3) < 1e-10 or kappa_p3 < 0:
            ind.fitness = -100
            return
        eK0 = 1.0 / ((4.0/3.0) * kappa_p3)
        if eK0 < 0:
            ind.fitness = -100
            return

        self.stats['valid'] += 1
        ind.p = p
        ind.eK0 = eK0

        # Compute racetrack
        racetrack = self.compute_racetrack(p, ind.M)
        if racetrack is None:
            # Valid but no racetrack - give small positive fitness
            ind.fitness = 0
            return

        self.stats['has_racetrack'] += 1
        ind.g_s = racetrack['g_s']
        ind.W_0 = racetrack['W_0']

        # Compute V_0
        V_0 = -3 * eK0 * (ind.g_s**7 / (4 * self.V_CY)**2) * ind.W_0**2
        ind.V_0 = V_0

        # Fitness = -log10(|V_0|) - higher is better (smaller V_0)
        # Use log of W_0 as proxy since V_0 underflows
        if ind.W_0 == 0 or ind.W_0 < 1e-300:
            # Use g_s as tiebreaker - smaller g_s generally means smaller W_0
            ind.fitness = 300 + (1.0 / (ind.g_s + 0.001))  # Bonus for small g_s
        else:
            # Approximate log10(|V_0|) using the formula components
            # V_0 = -3 * eK0 * (g_s^7 / (4*V_CY)^2) * W_0^2
            log_V0 = (np.log10(3) + np.log10(eK0) + 7*np.log10(ind.g_s)
                      - 2*np.log10(4*self.V_CY) + 2*np.log10(ind.W_0))
            ind.fitness = -log_V0

    def initialize_population(self):
        """Create initial random population."""
        print(f"Initializing population of {self.population_size}...")
        self.population = []
        while len(self.population) < self.population_size:
            ind = self.create_individual()
            self.population.append(ind)

        self.population.sort(key=lambda x: x.fitness, reverse=True)
        self.best_ever = self.population[0]
        print(f"Initial best fitness: {self.best_ever.fitness:.2f}")

    def select_parent(self):
        """Tournament selection."""
        tournament_size = 5
        candidates = np.random.choice(len(self.population), tournament_size, replace=False)
        best_idx = max(candidates, key=lambda i: self.population[i].fitness)
        return self.population[best_idx]

    def crossover(self, parent1: Individual, parent2: Individual):
        """Two-point crossover on the combined (K, M) genome."""
        if np.random.random() > self.crossover_rate:
            # No crossover - return copies of parents
            return (
                self.create_individual(parent1.K.copy(), parent1.M.copy()),
                self.create_individual(parent2.K.copy(), parent2.M.copy())
            )

        # Combine K and M into single genome
        genome1 = np.concatenate([parent1.K, parent1.M])
        genome2 = np.concatenate([parent2.K, parent2.M])

        # Two-point crossover
        points = sorted(np.random.choice(len(genome1), 2, replace=False))

        child1_genome = genome1.copy()
        child2_genome = genome2.copy()

        child1_genome[points[0]:points[1]] = genome2[points[0]:points[1]]
        child2_genome[points[0]:points[1]] = genome1[points[0]:points[1]]

        # Split back into K and M
        child1 = self.create_individual(child1_genome[:self.h11], child1_genome[self.h11:])
        child2 = self.create_individual(child2_genome[:self.h11], child2_genome[self.h11:])

        return child1, child2

    def mutate(self, ind: Individual):
        """Mutate an individual's genome."""
        if np.random.random() > self.mutation_rate:
            return ind

        # Mutate K
        K_new = ind.K.copy()
        M_new = ind.M.copy()

        # Randomly mutate some components
        for i in range(self.h11):
            if np.random.random() < 0.25:  # 25% chance per component
                delta = np.random.randint(-self.mutation_strength, self.mutation_strength + 1)
                K_new[i] = np.clip(K_new[i] + delta, *self.flux_range)

        for i in range(self.h11):
            if np.random.random() < 0.25:
                delta = np.random.randint(-self.mutation_strength, self.mutation_strength + 1)
                M_new[i] = np.clip(M_new[i] + delta, *self.flux_range)

        return self.create_individual(K_new, M_new)

    def evolve_generation(self):
        """Evolve one generation."""
        self.generation += 1

        # Elitism - keep best individuals
        new_population = self.population[:self.elite_count]

        # Generate children
        while len(new_population) < self.population_size:
            parent1 = self.select_parent()
            parent2 = self.select_parent()

            child1, child2 = self.crossover(parent1, parent2)

            child1 = self.mutate(child1)
            child2 = self.mutate(child2)

            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)

        # Sort by fitness
        new_population.sort(key=lambda x: x.fitness, reverse=True)
        self.population = new_population[:self.population_size]

        # Update best ever
        if self.population[0].fitness > self.best_ever.fitness:
            self.best_ever = self.population[0]

    def run(self, generations=100, verbose=True):
        """Run the GA for specified generations."""
        self.initialize_population()

        for gen in range(generations):
            self.evolve_generation()

            if verbose and (gen + 1) % 10 == 0:
                best = self.population[0]
                avg_fitness = np.mean([ind.fitness for ind in self.population])
                print(f"Gen {gen+1:4d}: best={best.fitness:8.2f}, avg={avg_fitness:8.2f}, "
                      f"best_ever={self.best_ever.fitness:.2f}")
                if self.best_ever.V_0 is not None:
                    print(f"         V_0={self.best_ever.V_0:.2e}, "
                          f"W_0={self.best_ever.W_0:.2e}, g_s={self.best_ever.g_s:.6f}")

        return self.best_ever


def random_search(kappa, gv_invariants, n_samples, h11=4, V_CY=4711.83):
    """Baseline random search for comparison."""
    flux_range = (-15, 16)
    best = None
    best_fitness = float('-inf')
    stats = {'evaluated': 0, 'valid': 0, 'has_racetrack': 0}

    for _ in range(n_samples):
        K = np.array([np.random.randint(*flux_range) for _ in range(h11)])
        M = np.array([np.random.randint(*flux_range) for _ in range(h11)])

        stats['evaluated'] += 1

        # Check N invertible
        N = np.einsum('abc,c->ab', kappa, M)
        if abs(np.linalg.det(N)) < 1e-10:
            continue

        try:
            p = np.linalg.solve(N, K)
        except np.linalg.LinAlgError:
            continue

        if not np.all(p > 0):
            continue

        if -0.5 * np.dot(M, K) > 500:
            continue

        kappa_p3 = np.einsum('abc,a,b,c->', kappa, p, p, p)
        if abs(kappa_p3) < 1e-10 or kappa_p3 < 0:
            continue
        eK0 = 1.0 / ((4.0/3.0) * kappa_p3)
        if eK0 < 0:
            continue

        stats['valid'] += 1

        # Compute racetrack (simplified)
        curves_by_action = {}
        for q, N_q in gv_invariants.items():
            q = np.array(q)
            action = np.dot(q, p)
            if action > 0:
                M_dot_q = np.dot(M, q)
                if M_dot_q != 0:
                    key = round(action, 6)
                    if key not in curves_by_action:
                        curves_by_action[key] = []
                    curves_by_action[key].append({'N_q': N_q, 'M_dot_q': M_dot_q, 'action': action})

        if len(curves_by_action) < 2:
            continue

        sorted_actions = sorted(curves_by_action.keys())
        action1, action2 = sorted_actions[0], sorted_actions[1]

        def sum_coeff(action):
            return sum(c['M_dot_q'] * c['N_q'] * c['action'] for c in curves_by_action[action])

        c1, c2 = sum_coeff(action1), sum_coeff(action2)
        if c1 == 0 or c2 == 0:
            continue

        delta_action = action2 - action1
        if delta_action <= 0:
            continue

        ratio = -c2 / c1
        if ratio <= 0:
            continue

        g_s = 2 * np.pi * delta_action / np.log(ratio)
        if g_s <= 0 or g_s > 1:
            continue

        exponent = -2 * np.pi * action1 / g_s
        W_0 = 0.0 if exponent < -500 else abs(c1) * np.exp(exponent)

        stats['has_racetrack'] += 1

        V_0 = -3 * eK0 * (g_s**7 / (4 * V_CY)**2) * W_0**2

        # Use log formula to avoid underflow
        if W_0 == 0 or W_0 < 1e-300:
            fitness = 300 + (1.0 / (g_s + 0.001))
        else:
            log_V0 = (np.log10(3) + np.log10(eK0) + 7*np.log10(g_s)
                      - 2*np.log10(4*V_CY) + 2*np.log10(W_0))
            fitness = -log_V0

        if fitness > best_fitness:
            best_fitness = fitness
            best = {
                'K': K.tolist(), 'M': M.tolist(), 'p': p.tolist(),
                'eK0': eK0, 'g_s': g_s, 'W_0': W_0, 'V_0': V_0, 'fitness': fitness
            }

    return best, stats


def sparse_to_dense(sparse, h11):
    kappa = np.zeros((h11, h11, h11))
    for (i, j, k), val in sparse.items():
        for perm in [(i,j,k), (i,k,j), (j,i,k), (j,k,i), (k,i,j), (k,j,i)]:
            kappa[perm] = val
    return kappa


def main():
    print("=" * 70)
    print("Genetic Algorithm vs Random Search for (K, M) pairs")
    print("=" * 70)

    # Load McAllister's polytope
    DATA_DIR = Path(__file__).parent.parent / "resources/small_cc_2107.09064_source/anc/paper_data/4-214-647"
    dual_points = np.loadtxt(DATA_DIR / "dual_points.dat", delimiter=',').astype(int)
    with open(DATA_DIR / "dual_simplices.dat") as f:
        simplices = [[int(x) for x in line.strip().split(',')] for line in f]

    # Compute GV invariants
    print("\nLoading geometry and computing GV invariants...")
    from cytools import Polytope
    poly = Polytope(dual_points)
    tri = poly.triangulate(simplices=simplices)
    cy = tri.get_cy()

    gv_obj = cy.compute_gvs(min_points=100)
    gv_invariants = {}
    for q, N_q in gv_obj.dok.items():
        if N_q != 0:
            gv_invariants[tuple(q)] = int(N_q)
    print(f"Computed {len(gv_invariants)} non-zero GV invariants")

    # Get intersection numbers
    kappa_sparse = cy.intersection_numbers(in_basis=True)
    h11 = cy.h11()
    kappa = sparse_to_dense(kappa_sparse, h11)
    print(f"h11 = {h11}, basis = {list(cy.divisor_basis())}")

    # Target for reference
    TARGET_V0 = -5.5e-203
    print(f"\nTarget V_0 = {TARGET_V0:.2e} (fitness = {-np.log10(abs(TARGET_V0)):.2f})")

    # Parameters
    N_EVALUATIONS = 100_000  # Total evaluations for fair comparison
    POP_SIZE = 500
    GENERATIONS = N_EVALUATIONS // POP_SIZE

    print(f"\nTotal evaluations: {N_EVALUATIONS:,}")
    print(f"GA: population={POP_SIZE}, generations={GENERATIONS}")

    # Run GA
    print("\n" + "=" * 70)
    print("GENETIC ALGORITHM")
    print("=" * 70)

    np.random.seed(42)
    ga = GeneticAlgorithm(
        kappa=kappa,
        gv_invariants=gv_invariants,
        h11=h11,
        population_size=POP_SIZE,
        mutation_rate=0.3,  # Higher mutation for exploration
        mutation_strength=3,
        crossover_rate=0.5,  # Lower crossover (paper found it often hurts)
        elite_fraction=0.1,
    )

    start = time.time()
    best_ga = ga.run(generations=GENERATIONS, verbose=True)
    ga_time = time.time() - start

    print(f"\nGA completed in {ga_time:.1f}s")
    print(f"Evaluations: {ga.stats['evaluated']:,}")
    print(f"Valid: {ga.stats['valid']:,} ({100*ga.stats['valid']/ga.stats['evaluated']:.1f}%)")
    print(f"Has racetrack: {ga.stats['has_racetrack']:,}")

    if best_ga.V_0 is not None:
        print(f"\nBest GA result:")
        print(f"  K = {best_ga.K.tolist()}")
        print(f"  M = {best_ga.M.tolist()}")
        print(f"  V_0 = {best_ga.V_0:.2e}")
        print(f"  W_0 = {best_ga.W_0:.2e}")
        print(f"  g_s = {best_ga.g_s:.6f}")
        print(f"  fitness = {best_ga.fitness:.2f}")

    # Run random search
    print("\n" + "=" * 70)
    print("RANDOM SEARCH (baseline)")
    print("=" * 70)

    np.random.seed(42)
    start = time.time()
    best_random, random_stats = random_search(kappa, gv_invariants, N_EVALUATIONS, h11)
    random_time = time.time() - start

    print(f"\nRandom search completed in {random_time:.1f}s")
    print(f"Evaluations: {random_stats['evaluated']:,}")
    print(f"Valid: {random_stats['valid']:,} ({100*random_stats['valid']/random_stats['evaluated']:.1f}%)")
    print(f"Has racetrack: {random_stats['has_racetrack']:,}")

    if best_random:
        print(f"\nBest random result:")
        print(f"  K = {best_random['K']}")
        print(f"  M = {best_random['M']}")
        print(f"  V_0 = {best_random['V_0']:.2e}")
        print(f"  W_0 = {best_random['W_0']:.2e}")
        print(f"  g_s = {best_random['g_s']:.6f}")
        print(f"  fitness = {best_random['fitness']:.2f}")

    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    ga_fitness = best_ga.fitness if best_ga.V_0 is not None else 0
    random_fitness = best_random['fitness'] if best_random else 0

    print(f"GA best fitness:     {ga_fitness:.2f}")
    print(f"Random best fitness: {random_fitness:.2f}")
    print(f"Target fitness:      {-np.log10(abs(TARGET_V0)):.2f}")

    if ga_fitness > random_fitness:
        print(f"\n>>> GA wins by {ga_fitness - random_fitness:.2f} (10^{ga_fitness - random_fitness:.1f}x smaller V_0)")
    elif random_fitness > ga_fitness:
        print(f"\n>>> Random wins by {random_fitness - ga_fitness:.2f} (10^{random_fitness - ga_fitness:.1f}x smaller V_0)")
    else:
        print(f"\n>>> Tie!")


if __name__ == '__main__':
    main()
