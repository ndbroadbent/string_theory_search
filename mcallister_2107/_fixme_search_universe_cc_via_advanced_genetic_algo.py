#!/usr/bin/env python3
"""
Advanced Genetic Algorithm to search for our Universe's Cosmological Constant.

Target: Λ = 2.888 × 10⁻¹²² (Planck units)

This searches for flux configurations (K, M) that produce V₀ ≈ -10⁻¹²²
(the AdS vacuum energy before uplift to de Sitter).

Techniques from genetic_logic_shapes:
1. Frontier-based selection - maintain diverse top N, not just one best
2. Adaptive mutation - small when improving, large when stagnant
3. Stagnation tracking per lineage - kill branches that aren't improving
4. ASTEROID IMPACT - when globally stuck, reset from Hall of Fame
5. Hall of Fame - store best solutions ever found
"""
import sys
sys.path.insert(0, str(__file__).replace('/mcallister_2107/search_universe_cc_via_advanced_genetic_algo.py', '/vendor/cytools_latest/src'))

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List
import time

# TARGET: Our Universe's Cosmological Constant
# From Planck 2018 (arXiv:1807.06209):
#   Ω_Λ = 0.6847 ± 0.0073
#   H₀ = 67.4 ± 0.5 km/s/Mpc
#
# Derived cosmological constant in Planck units:
#   ρ_Λ = Ω_Λ × 3H₀²/(8πG) = 5.85 × 10⁻²⁷ kg/m³
#   ρ_Planck = c⁵/(ℏG²) = 5.18 × 10⁹⁶ kg/m³
#   Λ = 8π × ρ_Λ/ρ_Planck = 2.846 × 10⁻¹²² (Planck units)
#
# We search for AdS vacuum with |V₀| ≈ Λ (before uplift to dS)
UNIVERSE_LAMBDA = 2.846e-122  # Planck 2018 best fit
UNIVERSE_LAMBDA_UNCERTAINTY = 0.06e-122  # ~2% from Ω_Λ and H₀ uncertainties
TARGET_LOG_V0 = np.log10(UNIVERSE_LAMBDA)  # = -121.546


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
    log_V0: Optional[float] = None

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

        # Track closest to target
        self.closest_to_target: Optional[Individual] = None
        self.closest_distance = float('inf')

    def random_flux(self):
        return np.array([np.random.randint(*self.flux_range) for _ in range(self.h11)])

    def create_individual(self, K=None, M=None, epoch=0) -> Individual:
        if K is None:
            K = self.random_flux()
        if M is None:
            M = self.random_flux()

        ind = Individual(K=K.copy(), M=M.copy(), last_improved_epoch=epoch)
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

        # TIER 1: Invalid geometry = very bad
        invertible, N = self.check_N_invertible(ind.M)
        if not invertible:
            ind.fitness = -2000
            return

        try:
            p = np.linalg.solve(N, ind.K)
        except np.linalg.LinAlgError:
            ind.fitness = -2000
            return

        # TIER 2: p not in Kähler cone = bad
        if not np.all(p > 0):
            ind.fitness = -1500
            return

        # TIER 3: Tadpole too large = bad
        if -0.5 * np.dot(ind.M, ind.K) > 500:
            ind.fitness = -1500
            return

        # TIER 4: eK0 invalid = bad
        kappa_p3 = np.einsum('abc,a,b,c->', self.kappa, p, p, p)
        if abs(kappa_p3) < 1e-10 or kappa_p3 < 0:
            ind.fitness = -1000
            return
        eK0 = 1.0 / ((4.0/3.0) * kappa_p3)
        if eK0 < 0:
            ind.fitness = -1000
            return

        self.stats['valid'] += 1
        ind.p = p
        ind.eK0 = eK0

        # TIER 5: No racetrack = less bad than geometry issues
        racetrack = self.compute_racetrack(p, ind.M)
        if racetrack is None:
            ind.fitness = -500
            return

        # TIER 6: Has racetrack - use target-based fitness
        self.stats['has_racetrack'] += 1
        ind.g_s = racetrack['g_s']
        ind.W_0 = racetrack['W_0']

        # Compute log10(|V_0|)
        if ind.W_0 == 0 or ind.W_0 < 1e-300:
            log_V0 = (np.log10(3) + np.log10(eK0) + 7*np.log10(ind.g_s)
                      - 2*np.log10(4*self.V_CY) + 2*(-300))
        else:
            log_V0 = (np.log10(3) + np.log10(eK0) + 7*np.log10(ind.g_s)
                      - 2*np.log10(4*self.V_CY) + 2*np.log10(ind.W_0))

        ind.log_V0 = log_V0

        # TARGET-BASED FITNESS: 1000 - distance from target
        distance = abs(log_V0 - TARGET_LOG_V0)
        ind.fitness = 1000.0 - distance

        # Track closest to target
        if distance < self.closest_distance:
            self.closest_distance = distance
            self.closest_to_target = ind

    def get_adaptive_mutation_rate(self, stagnation: int) -> float:
        """Adaptive mutation rate based on stagnation."""
        roll = np.random.random()
        if stagnation < 3:
            if roll < 0.6:
                return 0.1
            elif roll < 0.9:
                return 0.2
            else:
                return 0.3
        elif stagnation < 7:
            if roll < 0.3:
                return 0.1
            elif roll < 0.7:
                return 0.3
            else:
                return 0.5
        elif stagnation < 12:
            if roll < 0.2:
                return 0.3
            elif roll < 0.6:
                return 0.5
            else:
                return 0.7
        else:
            if roll < 0.3:
                return 0.5
            elif roll < 0.7:
                return 0.7
            else:
                return 1.0

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

        best_parent = p1 if p1.fitness > p2.fitness else p2
        return self.create_individual(K_new, M_new, best_parent.last_improved_epoch)

    def asteroid_impact(self):
        """ASTEROID IMPACT - Reset population from Hall of Fame."""
        print(f"\n☄️  ASTEROID IMPACT! Global stagnation at epoch {self.epoch}. Rebuilding... ☄️")
        self.stats['asteroids'] += 1

        if self.best_ever and self.best_ever not in self.hall_of_fame:
            self.hall_of_fame.append(self.best_ever)
            if len(self.hall_of_fame) > 20:
                self.hall_of_fame = sorted(self.hall_of_fame, key=lambda x: x.fitness, reverse=True)[:20]

        print(f"   Hall of Fame size: {len(self.hall_of_fame)}")

        self.frontier.clear()

        random_quota = self.frontier_size * 3 // 10
        for _ in range(random_quota):
            self.frontier.append(self.create_individual(epoch=self.epoch))

        while len(self.frontier) < self.frontier_size:
            if self.hall_of_fame:
                parent = np.random.choice(self.hall_of_fame)
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

        if self.epochs_since_global_improvement >= self.asteroid_threshold:
            self.asteroid_impact()
            return

        children_per_parent = self.population_size // self.frontier_size
        candidates = []

        for p in self.frontier:
            candidates.append(p)

        for parent in self.frontier:
            for _ in range(children_per_parent):
                if np.random.random() < 0.3 and len(self.frontier) > 1:
                    other = np.random.choice([f for f in self.frontier if f != parent])
                    child = self.crossover(parent, other)
                else:
                    child = self.mutate(parent)

                if child.fitness > parent.fitness:
                    child.last_improved_epoch = self.epoch

                candidates.append(child)

        before_count = len(candidates)
        candidates = [c for c in candidates if (self.epoch - c.last_improved_epoch) <= self.max_stagnation]
        self.stats['pruned'] += before_count - len(candidates)

        candidates.sort(key=lambda x: x.fitness, reverse=True)

        seen = set()
        unique = []
        for c in candidates:
            key = (tuple(c.K), tuple(c.M))
            if key not in seen:
                seen.add(key)
                unique.append(c)
        candidates = unique

        self.frontier = candidates[:self.frontier_size]

        while len(self.frontier) < self.frontier_size:
            self.frontier.append(self.create_individual(epoch=self.epoch))
        self.frontier.sort(key=lambda x: x.fitness, reverse=True)

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
                    print(f"         log10(|V₀|)={self.best_ever.log_V0:.2f}, g_s={self.best_ever.g_s:.6f}")
                    print(f"         K={self.best_ever.K.tolist()}, M={self.best_ever.M.tolist()}")

        return self.best_ever


def sparse_to_dense(sparse, h11):
    kappa = np.zeros((h11, h11, h11))
    for (i, j, k), val in sparse.items():
        for perm in [(i,j,k), (i,k,j), (j,i,k), (j,k,i), (k,i,j), (k,j,i)]:
            kappa[perm] = val
    return kappa


def main():
    print("=" * 70)
    print("🌌 SEARCHING FOR OUR UNIVERSE'S COSMOLOGICAL CONSTANT 🌌")
    print("=" * 70)
    print(f"Target: Λ = {UNIVERSE_LAMBDA:.3e} (Planck units)")
    print(f"Target log10(|V₀|) = {TARGET_LOG_V0:.2f}")

    # Load geometry (using McAllister's polytope as test case)
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

    print(f"\nSearch space: K, M ∈ [-20, 20]^4 = {41**8:,} possibilities")

    # Parameters
    N_EVALUATIONS = 1_000_000
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

    print(f"\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\nGA completed in {ga_time:.1f}s")
    print(f"Evaluations: {ga.stats['evaluated']:,}")
    print(f"Valid: {ga.stats['valid']:,}")
    print(f"Has racetrack: {ga.stats['has_racetrack']:,}")
    print(f"Asteroids: {ga.stats['asteroids']}")

    print(f"\n🎯 Target log10(|V₀|) = {TARGET_LOG_V0:.2f}")
    print(f"   Closest achieved:   {ga.closest_distance:.2f} away")

    if ga.closest_to_target:
        ind = ga.closest_to_target
        print(f"\n📊 Best solution found:")
        print(f"   K = {ind.K.tolist()}")
        print(f"   M = {ind.M.tolist()}")
        print(f"   g_s = {ind.g_s:.6f}")
        print(f"   log10(|V₀|) = {ind.log_V0:.2f}")
        print(f"   |V₀| ≈ 10^{ind.log_V0:.1f}")

        # Compare to target
        ratio = 10**(ind.log_V0 - TARGET_LOG_V0)
        if ratio > 1:
            print(f"\n   This is {ratio:.1e}x LARGER than our universe's Λ")
        else:
            print(f"\n   This is {1/ratio:.1e}x SMALLER than our universe's Λ")


if __name__ == '__main__':
    main()
