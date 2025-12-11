#!/usr/bin/env python3
"""
Advanced Genetic Algorithm for (K, M) flux search.

Techniques from genetic_logic_shapes:
1. Frontier-based selection - maintain diverse top N, not just one best
2. Adaptive mutation - small when improving, large when stagnant
3. Stagnation tracking per lineage - kill branches that aren't improving
4. ASTEROID IMPACT - when globally stuck, reset from Hall of Fame
5. Hall of Fame - store best solutions ever found
6. Parsimony pressure - prefer simpler solutions (smaller |K|, |M|)
"""
import sys
sys.path.insert(0, str(__file__).replace('/mcallister_2107/search_km_via_advanced_genetic_algo.py', '/vendor/cytools_latest/src'))

import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import time


@dataclass
class Individual:
    """A single (K, M) configuration with lineage tracking."""
    K: np.ndarray
    M: np.ndarray
    fitness: float = float('-inf')
    last_improved_epoch: int = 0
    # Cached physics
    p: Optional[np.ndarray] = None
    eK0: Optional[float] = None
    g_s: Optional[float] = None
    W_0: Optional[float] = None
    V_0: Optional[float] = None
    # For diversity
    complexity: int = 0  # sum of |K| + |M|

    def __hash__(self):
        return hash((tuple(self.K), tuple(self.M)))

    def __eq__(self, other):
        return np.array_equal(self.K, other.K) and np.array_equal(self.M, other.M)


class AdvancedGA:
    def __init__(self, kappa, gv_invariants, h11=4,
                 frontier_size=64,
                 population_size=1024,
                 flux_range=(-20, 21),
                 max_stagnation=15,
                 asteroid_threshold=25,
                 V_CY=4711.83):
        self.kappa = kappa
        self.gv_invariants = gv_invariants
        self.h11 = h11
        self.frontier_size = frontier_size
        self.population_size = population_size
        self.flux_range = flux_range
        self.max_stagnation = max_stagnation
        self.asteroid_threshold = asteroid_threshold
        self.V_CY = V_CY

        self.frontier: List[Individual] = []
        self.hall_of_fame: List[Individual] = []
        self.epoch = 0
        self.best_ever: Optional[Individual] = None
        self.epochs_since_global_improvement = 0

        self.stats = {
            'evaluated': 0,
            'valid': 0,
            'has_racetrack': 0,
            'asteroids': 0,
            'pruned': 0,
        }

    def random_flux(self):
        return np.array([np.random.randint(*self.flux_range) for _ in range(self.h11)])

    def create_individual(self, K=None, M=None, epoch=0) -> Individual:
        if K is None:
            K = self.random_flux()
        if M is None:
            M = self.random_flux()

        ind = Individual(K=K.copy(), M=M.copy(), last_improved_epoch=epoch)
        ind.complexity = np.sum(np.abs(K)) + np.sum(np.abs(M))
        self.evaluate(ind)
        return ind

    def check_N_invertible(self, M):
        N = np.einsum('abc,c->ab', self.kappa, M)
        det = np.linalg.det(N)
        if abs(det) < 1e-10:
            return False, None
        return True, N

    def compute_racetrack(self, p, M):
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
        if -0.5 * np.dot(ind.M, ind.K) > 500:
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
            ind.fitness = 0
            return

        self.stats['has_racetrack'] += 1
        ind.g_s = racetrack['g_s']
        ind.W_0 = racetrack['W_0']

        # Compute V_0 via log formula to avoid underflow
        V_0 = -3 * eK0 * (ind.g_s**7 / (4 * self.V_CY)**2) * ind.W_0**2
        ind.V_0 = V_0

        if ind.W_0 == 0 or ind.W_0 < 1e-300:
            # Use g_s as proxy - smaller g_s → smaller W_0 → smaller V_0
            ind.fitness = 300 + (1.0 / (ind.g_s + 0.0001))
        else:
            log_V0 = (np.log10(3) + np.log10(eK0) + 7*np.log10(ind.g_s)
                      - 2*np.log10(4*self.V_CY) + 2*np.log10(ind.W_0))
            ind.fitness = -log_V0

    def get_adaptive_mutation_rate(self, stagnation: int) -> float:
        """Adaptive mutation rate based on stagnation."""
        roll = np.random.random()
        if stagnation < 3:
            # Recently improved - small mutations
            if roll < 0.6:
                return 0.1  # 10% of components
            elif roll < 0.9:
                return 0.2
            else:
                return 0.3
        elif stagnation < 7:
            # Moderately stagnant
            if roll < 0.3:
                return 0.1
            elif roll < 0.7:
                return 0.3
            else:
                return 0.5
        elif stagnation < 12:
            # Very stagnant
            if roll < 0.2:
                return 0.3
            elif roll < 0.6:
                return 0.5
            else:
                return 0.7
        else:
            # Extremely stagnant - aggressive mutation
            if roll < 0.3:
                return 0.5
            elif roll < 0.7:
                return 0.7
            else:
                return 1.0  # Full randomization

    def get_adaptive_mutation_strength(self, stagnation: int) -> int:
        """Adaptive mutation strength based on stagnation."""
        if stagnation < 3:
            return np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
        elif stagnation < 7:
            return np.random.choice([2, 3, 5], p=[0.4, 0.4, 0.2])
        elif stagnation < 12:
            return np.random.choice([3, 5, 8], p=[0.3, 0.4, 0.3])
        else:
            return np.random.choice([5, 8, 12], p=[0.3, 0.4, 0.3])

    def mutate(self, parent: Individual) -> Individual:
        """Mutate with adaptive rate based on parent's stagnation."""
        stagnation = self.epoch - parent.last_improved_epoch
        rate = self.get_adaptive_mutation_rate(stagnation)
        strength = self.get_adaptive_mutation_strength(stagnation)

        K_new = parent.K.copy()
        M_new = parent.M.copy()

        for i in range(self.h11):
            if np.random.random() < rate:
                delta = np.random.randint(-strength, strength + 1)
                K_new[i] = np.clip(K_new[i] + delta, *self.flux_range)

        for i in range(self.h11):
            if np.random.random() < rate:
                delta = np.random.randint(-strength, strength + 1)
                M_new[i] = np.clip(M_new[i] + delta, *self.flux_range)

        return self.create_individual(K_new, M_new, parent.last_improved_epoch)

    def crossover(self, p1: Individual, p2: Individual) -> Individual:
        """Uniform crossover - each component independently from either parent."""
        K_new = np.array([p1.K[i] if np.random.random() < 0.5 else p2.K[i] for i in range(self.h11)])
        M_new = np.array([p1.M[i] if np.random.random() < 0.5 else p2.M[i] for i in range(self.h11)])

        # Inherit best parent's stagnation
        best_parent = p1 if p1.fitness > p2.fitness else p2
        return self.create_individual(K_new, M_new, best_parent.last_improved_epoch)

    def asteroid_impact(self):
        """ASTEROID IMPACT - Reset population from Hall of Fame."""
        print(f"\n☄️  ASTEROID IMPACT! Global stagnation at epoch {self.epoch}. Rebuilding... ☄️")
        self.stats['asteroids'] += 1

        # Add current best to Hall of Fame
        if self.best_ever and self.best_ever not in self.hall_of_fame:
            self.hall_of_fame.append(self.best_ever)
            # Keep Hall of Fame bounded
            if len(self.hall_of_fame) > 20:
                self.hall_of_fame = sorted(self.hall_of_fame, key=lambda x: x.fitness, reverse=True)[:20]

        print(f"   Hall of Fame size: {len(self.hall_of_fame)}")

        self.frontier.clear()

        # 30% Fresh random (exploration)
        random_quota = self.frontier_size * 3 // 10
        for _ in range(random_quota):
            self.frontier.append(self.create_individual(epoch=self.epoch))

        # 70% Mutations of Hall of Fame members
        while len(self.frontier) < self.frontier_size:
            if self.hall_of_fame:
                parent = np.random.choice(self.hall_of_fame)
                # Heavy mutation for diversity
                K_new = parent.K + np.random.randint(-8, 9, self.h11)
                M_new = parent.M + np.random.randint(-8, 9, self.h11)
                K_new = np.clip(K_new, *self.flux_range)
                M_new = np.clip(M_new, *self.flux_range)
                self.frontier.append(self.create_individual(K_new, M_new, self.epoch))
            else:
                self.frontier.append(self.create_individual(epoch=self.epoch))

        self.epochs_since_global_improvement = 0
        self.frontier.sort(key=lambda x: x.fitness, reverse=True)

    def initialize(self):
        """Initialize frontier with random individuals."""
        print(f"Initializing frontier of {self.frontier_size}...")
        self.frontier = [self.create_individual(epoch=0) for _ in range(self.frontier_size)]
        self.frontier.sort(key=lambda x: x.fitness, reverse=True)
        self.best_ever = self.frontier[0]
        print(f"Initial best fitness: {self.best_ever.fitness:.2f}")

    def evolve_generation(self):
        """Evolve one generation."""
        self.epoch += 1

        # Check for asteroid
        if self.epochs_since_global_improvement >= self.asteroid_threshold:
            self.asteroid_impact()
            return

        # Generate children
        children_per_parent = self.population_size // self.frontier_size
        candidates = []

        # Keep elites
        for p in self.frontier:
            candidates.append(p)

        # Generate children via mutation and crossover
        for parent in self.frontier:
            for _ in range(children_per_parent):
                if np.random.random() < 0.3 and len(self.frontier) > 1:
                    # Crossover with another frontier member
                    other = np.random.choice([f for f in self.frontier if f != parent])
                    child = self.crossover(parent, other)
                else:
                    # Mutation
                    child = self.mutate(parent)

                # Update last_improved_epoch if child is better
                if child.fitness > parent.fitness:
                    child.last_improved_epoch = self.epoch

                candidates.append(child)

        # Prune stagnant branches
        before_count = len(candidates)
        candidates = [c for c in candidates if (self.epoch - c.last_improved_epoch) <= self.max_stagnation]
        self.stats['pruned'] += before_count - len(candidates)

        # Sort by fitness (higher is better)
        candidates.sort(key=lambda x: x.fitness, reverse=True)

        # Remove duplicates
        seen = set()
        unique = []
        for c in candidates:
            key = (tuple(c.K), tuple(c.M))
            if key not in seen:
                seen.add(key)
                unique.append(c)
        candidates = unique

        # Truncate to frontier size
        self.frontier = candidates[:self.frontier_size]

        # Refill if needed
        while len(self.frontier) < self.frontier_size:
            self.frontier.append(self.create_individual(epoch=self.epoch))
        self.frontier.sort(key=lambda x: x.fitness, reverse=True)

        # Update global best
        if self.frontier[0].fitness > self.best_ever.fitness:
            self.best_ever = self.frontier[0]
            self.epochs_since_global_improvement = 0
        else:
            self.epochs_since_global_improvement += 1

    def run(self, epochs=200, verbose=True):
        """Run the GA."""
        self.initialize()

        for _ in range(epochs):
            self.evolve_generation()

            if verbose and self.epoch % 10 == 0:
                best = self.frontier[0]
                avg_fitness = np.mean([ind.fitness for ind in self.frontier])
                stagnation = self.epoch - best.last_improved_epoch
                print(f"Epoch {self.epoch:4d}: best={best.fitness:8.2f}, avg={avg_fitness:8.2f}, "
                      f"stale={stagnation:2d}, global_stale={self.epochs_since_global_improvement:2d}")
                if self.best_ever.g_s is not None:
                    print(f"         g_s={self.best_ever.g_s:.6f}, "
                          f"K={self.best_ever.K.tolist()}, M={self.best_ever.M.tolist()}")

        return self.best_ever


def sparse_to_dense(sparse, h11):
    kappa = np.zeros((h11, h11, h11))
    for (i, j, k), val in sparse.items():
        for perm in [(i,j,k), (i,k,j), (j,i,k), (j,k,i), (k,i,j), (k,j,i)]:
            kappa[perm] = val
    return kappa


def random_search(kappa, gv_invariants, n_samples, h11=4, V_CY=4711.83):
    """Baseline random search."""
    flux_range = (-20, 21)
    best = None
    best_fitness = float('-inf')

    for _ in range(n_samples):
        K = np.array([np.random.randint(*flux_range) for _ in range(h11)])
        M = np.array([np.random.randint(*flux_range) for _ in range(h11)])

        # Quick filter chain
        N = np.einsum('abc,c->ab', kappa, M)
        if abs(np.linalg.det(N)) < 1e-10:
            continue

        try:
            p = np.linalg.solve(N, K)
        except:
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

        # Racetrack
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

        # Fitness
        if W_0 == 0 or W_0 < 1e-300:
            fitness = 300 + (1.0 / (g_s + 0.0001))
        else:
            log_V0 = (np.log10(3) + np.log10(eK0) + 7*np.log10(g_s)
                      - 2*np.log10(4*V_CY) + 2*np.log10(W_0))
            fitness = -log_V0

        if fitness > best_fitness:
            best_fitness = fitness
            best = {'K': K.tolist(), 'M': M.tolist(), 'g_s': g_s, 'W_0': W_0, 'fitness': fitness}

    return best


def main():
    print("=" * 70)
    print("Advanced GA vs Random Search for (K, M) pairs")
    print("=" * 70)

    # Load geometry
    DATA_DIR = Path(__file__).parent.parent / "resources/small_cc_2107.09064_source/anc/paper_data/4-214-647"
    dual_points = np.loadtxt(DATA_DIR / "dual_points.dat", delimiter=',').astype(int)
    with open(DATA_DIR / "dual_simplices.dat") as f:
        simplices = [[int(x) for x in line.strip().split(',')] for line in f]

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

    kappa_sparse = cy.intersection_numbers(in_basis=True)
    h11 = cy.h11()
    kappa = sparse_to_dense(kappa_sparse, h11)
    print(f"h11 = {h11}, basis = {list(cy.divisor_basis())}")

    TARGET_V0 = -5.5e-203
    print(f"\nTarget V_0 = {TARGET_V0:.2e} (fitness = {-np.log10(abs(TARGET_V0)):.2f})")

    # Parameters - same total evaluations
    N_EVALUATIONS = 100_000
    FRONTIER_SIZE = 64
    POP_SIZE = 1024
    EPOCHS = N_EVALUATIONS // POP_SIZE

    print(f"\nTotal evaluations budget: ~{N_EVALUATIONS:,}")
    print(f"Advanced GA: frontier={FRONTIER_SIZE}, pop={POP_SIZE}, epochs={EPOCHS}")

    # Run Advanced GA
    print("\n" + "=" * 70)
    print("ADVANCED GENETIC ALGORITHM")
    print("=" * 70)

    np.random.seed(42)
    ga = AdvancedGA(
        kappa=kappa,
        gv_invariants=gv_invariants,
        h11=h11,
        frontier_size=FRONTIER_SIZE,
        population_size=POP_SIZE,
        max_stagnation=15,
        asteroid_threshold=25,
    )

    start = time.time()
    best_ga = ga.run(epochs=EPOCHS, verbose=True)
    ga_time = time.time() - start

    print(f"\nAdvanced GA completed in {ga_time:.1f}s")
    print(f"Evaluations: {ga.stats['evaluated']:,}")
    print(f"Valid: {ga.stats['valid']:,}")
    print(f"Has racetrack: {ga.stats['has_racetrack']:,}")
    print(f"Asteroids: {ga.stats['asteroids']}")
    print(f"Pruned: {ga.stats['pruned']:,}")

    if best_ga.g_s is not None:
        print(f"\nBest Advanced GA result:")
        print(f"  K = {best_ga.K.tolist()}")
        print(f"  M = {best_ga.M.tolist()}")
        print(f"  g_s = {best_ga.g_s:.6f}")
        print(f"  fitness = {best_ga.fitness:.2f}")

    # Run random search
    print("\n" + "=" * 70)
    print("RANDOM SEARCH (baseline)")
    print("=" * 70)

    np.random.seed(42)
    start = time.time()
    best_random = random_search(kappa, gv_invariants, N_EVALUATIONS, h11)
    random_time = time.time() - start

    print(f"\nRandom search completed in {random_time:.1f}s")

    if best_random:
        print(f"\nBest random result:")
        print(f"  K = {best_random['K']}")
        print(f"  M = {best_random['M']}")
        print(f"  g_s = {best_random['g_s']:.6f}")
        print(f"  fitness = {best_random['fitness']:.2f}")

    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    ga_fitness = best_ga.fitness if best_ga.g_s is not None else 0
    random_fitness = best_random['fitness'] if best_random else 0

    print(f"Advanced GA best fitness: {ga_fitness:.2f}")
    print(f"Random best fitness:      {random_fitness:.2f}")
    print(f"Target fitness:           {-np.log10(abs(TARGET_V0)):.2f}")

    diff = ga_fitness - random_fitness
    if diff > 0:
        print(f"\n>>> Advanced GA wins by {diff:.2f} (10^{diff:.1f}x smaller V_0)")
    elif diff < 0:
        print(f"\n>>> Random wins by {-diff:.2f} (10^{-diff:.1f}x smaller V_0)")
    else:
        print(f"\n>>> Tie!")


if __name__ == '__main__':
    main()
