#!/usr/bin/env python3
"""
Genetic Algorithm for exploring KKLT solution branches via t_init evolution.

GENOME: (t_init, n_steps)
- t_init: starting point in Kähler cone (h11 floats)
- n_steps: predictor-corrector step count (16-128)

Uses predictor-corrector solver (NOT pure Newton!) with GA techniques:
1. Frontier-based selection - maintain diverse top N
2. Adaptive mutation - small when improving, large when stagnant
3. Hall of Fame - store best/unique branches ever found
4. ASTEROID IMPACT - reset when globally stuck
5. Branch diversity - track unique V_string values

Based on findings from explore_t_branches_via_adaptive_step_size.py:
- n_steps=16: 40% success (minimum working)
- n_steps=32: 70% success (optimal)
- n_steps=64+: 60-67% (diminishing returns)
"""

import sys
import time
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import numpy as np

# Paths
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_BASE = ROOT_DIR / "resources/small_cc_2107.09064_source/anc/paper_data"
CYTOOLS_2021 = ROOT_DIR / "vendor/cytools_mcallister_2107"

sys.path.insert(0, str(CYTOOLS_2021))

from cytools import Polytope
from compute_kahler_param import SparseKappa, _path_follow
from compute_t_uncorrected import _compute_V_from_kappa

# Constants
LAMBDA_OBS = 2.888e-122
STEP_RANGE = (16, 128)  # Genome range for n_steps


@dataclass
class Individual:
    """A (t_init, n_steps) configuration with lineage tracking."""
    t_init: np.ndarray
    n_steps: int = 32  # Part of genome!
    fitness: float = float('-inf')
    last_improved_epoch: int = 0
    # Results from solver
    t_result: Optional[np.ndarray] = None
    V_uncorrected: Optional[float] = None
    V_string: Optional[float] = None
    V0: Optional[float] = None
    converged: bool = False

    def __hash__(self):
        if self.V_string is not None:
            return hash(round(self.V_string, 1))
        return hash(id(self))


def predictor_corrector_solve(kappa: SparseKappa, tau_target: np.ndarray,
                               t_init: np.ndarray, n_steps: int,
                               tol: float = 1e-6, newton_tol: float = 1e-8,
                               max_newton: int = 5) -> dict:
    """
    Predictor-corrector solver for τ(t) = τ_target.

    Each step has:
    - Predictor: Euler step toward τ_step
    - Corrector: Newton iterations to converge exactly

    Returns dict with success, t, V, residual.
    """
    t = t_init.copy()
    tau_init = kappa.compute_tau_kklt(t)
    tau_target_norm = np.linalg.norm(tau_target)

    for m in range(n_steps):
        alpha = (m + 1) / n_steps
        tau_step = (1 - alpha) * tau_init + alpha * tau_target
        tau_step_norm = np.linalg.norm(tau_step)

        # Predictor: Euler step
        tau_current = kappa.compute_tau_kklt(t)
        delta_tau = tau_step - tau_current
        J = kappa.compute_jacobian_kklt(t)

        try:
            epsilon, _, _, _ = np.linalg.lstsq(J, delta_tau, rcond=1e-10)
        except np.linalg.LinAlgError:
            return {"success": False, "t": t, "V": None, "residual": np.inf}

        t_pred = t + epsilon

        # Corrector: Newton iterations
        t_corr = t_pred.copy()
        for _ in range(max_newton):
            tau_corr = kappa.compute_tau_kklt(t_corr)
            residual = tau_corr - tau_step
            if np.linalg.norm(residual) / tau_step_norm < newton_tol:
                break
            J_corr = kappa.compute_jacobian_kklt(t_corr)
            try:
                delta, _, _, _ = np.linalg.lstsq(J_corr, -residual, rcond=1e-10)
            except np.linalg.LinAlgError:
                break
            # Damping
            step_norm = np.linalg.norm(delta)
            if step_norm > 1.0:
                delta = delta / step_norm
            t_corr = t_corr + delta

        t = t_corr

        # Check for divergence
        if np.any(np.isnan(t)) or np.any(np.abs(t) > 1e6):
            return {"success": False, "t": t, "V": None, "residual": np.inf}

    # Final check
    tau_final = kappa.compute_tau_kklt(t)
    residual = np.linalg.norm(tau_final - tau_target) / tau_target_norm
    V = _compute_V_from_kappa(kappa, t)

    return {
        "success": residual < tol and V > 0,
        "t": t,
        "V": V,
        "residual": residual,
    }


class TInitGA:
    """Genetic Algorithm for finding diverse KKLT branches via (t_init, n_steps) evolution."""

    def __init__(self, kappa: SparseKappa, c_i: np.ndarray,
                 c_tau: float, g_s: float, W0: float,
                 h11: int, h21: int, e_K0: float = 0.0912,
                 mcallister_V_string: float = 198.31,
                 frontier_size: int = 32,
                 population_size: int = 128,
                 t_range: tuple = (0.1, 50.0),
                 max_stagnation: int = 20,
                 asteroid_threshold: int = 30):

        self.kappa = kappa
        self.c_i = c_i
        self.c_tau = c_tau
        self.g_s = g_s
        self.W0 = W0
        self.h11 = h11
        self.h21 = h21
        self.e_K0 = e_K0
        self.mcallister_V_string = mcallister_V_string

        self.frontier_size = frontier_size
        self.population_size = population_size
        self.t_range = t_range
        self.max_stagnation = max_stagnation
        self.asteroid_threshold = asteroid_threshold

        self.frontier: List[Individual] = []
        self.hall_of_fame: Dict[float, Individual] = {}
        self.epoch = 0
        self.best_ever: Optional[Individual] = None
        self.epochs_since_global_improvement = 0

        self.stats = {
            'evaluated': 0,
            'converged': 0,
            'V_positive': 0,
            'unique_branches': 0,
            'asteroids': 0,
        }

        # Output directory for saving branches
        self.output_dir = Path(__file__).parent / "ga_branches_output"
        self.output_dir.mkdir(exist_ok=True)
        self.example_name = "unknown"  # Will be set by caller

    def save_branch(self, ind: Individual, branch_key: float):
        """Save a branch to disk immediately when discovered."""
        branch_data = {
            "branch_key": branch_key,
            "V_uncorrected": float(ind.V_uncorrected) if ind.V_uncorrected else None,
            "n_steps": ind.n_steps,
            "fitness": float(ind.fitness),
            "t_init": ind.t_init.tolist(),
            "t_result": ind.t_result.tolist() if ind.t_result is not None else None,
        }

        # Append to single file
        branches_file = self.output_dir / f"{self.example_name}_branches.jsonl"
        with open(branches_file, 'a') as f:
            f.write(json.dumps(branch_data) + '\n')

    def random_t_init(self) -> np.ndarray:
        """Generate random t_init with various patterns."""
        pattern = np.random.randint(5)

        if pattern == 0:
            scale = 10 ** np.random.uniform(-0.5, 1.5)
            return np.ones(self.h11) * scale
        elif pattern == 1:
            scale = 10 ** np.random.uniform(-0.5, 1.5)
            return np.abs(np.random.randn(self.h11)) * scale + 0.1
        elif pattern == 2:
            t = np.ones(self.h11) * 0.1
            n_large = np.random.randint(1, max(2, self.h11 // 10))
            large_idx = np.random.choice(self.h11, n_large, replace=False)
            t[large_idx] = np.random.uniform(1, 25, n_large)
            return t
        elif pattern == 3:
            return 10 ** np.random.uniform(-0.5, 1.5, self.h11)
        else:
            t = np.ones(self.h11) * 0.5
            cluster_size = self.h11 // 4
            start = np.random.randint(0, self.h11 - cluster_size)
            t[start:start+cluster_size] = np.random.uniform(5, 20, cluster_size)
            return t

    def random_n_steps(self) -> int:
        """Generate random n_steps - HARDCODED to 32 for now."""
        return 32  # Optimal value from testing

    def evaluate(self, ind: Individual) -> None:
        """Evaluate individual by running predictor-corrector solver."""
        self.stats['evaluated'] += 1

        # Scale t_init to match τ magnitude
        t_start = ind.t_init.copy()
        tau_init = self.kappa.compute_tau_kklt(t_start)
        tau_target = self.c_i.astype(float)
        mean_tau_init = np.mean(np.abs(tau_init))
        mean_tau_target = np.mean(np.abs(tau_target))
        if mean_tau_init > 0 and mean_tau_target > 0:
            scale = np.sqrt(mean_tau_target / mean_tau_init)
            if 0.01 < scale < 100:
                t_start = t_start * scale

        # Run predictor-corrector solver with genome's n_steps
        result = predictor_corrector_solve(
            self.kappa, tau_target, t_start, ind.n_steps
        )

        if not result["success"]:
            ind.fitness = -1000 + (ind.n_steps / STEP_RANGE[1]) * 10  # Slight bonus for trying fewer steps
            ind.converged = False
            return

        self.stats['converged'] += 1
        ind.converged = True
        ind.t_result = result["t"]
        ind.V_uncorrected = result["V"]

        if result["V"] <= 0:
            ind.fitness = -500
            return

        self.stats['V_positive'] += 1

        # Use V_uncorrected as proxy (skip expensive Phase 2 for now)
        V_unc = result["V"]
        ind.V_string = V_unc  # Store V_uncorrected here

        # Fitness: reward proximity to McAllister's expected value
        # Ratio V_uncorrected/V_string = 2.5597757210 (verified from kahler_param.dat)
        mcallister_V_unc = self.mcallister_V_string * 2.5597757210

        # Exponential fitness: each decimal place of precision = 10x bonus
        # |error| < 1% → bonus ~50, |error| < 0.1% → bonus ~100, exact match → bonus ~150
        rel_error = abs(V_unc - mcallister_V_unc) / mcallister_V_unc
        if rel_error > 0:
            # -log10(rel_error): 1% → 2, 0.1% → 3, 0.01% → 4, etc.
            precision_score = -np.log10(rel_error + 1e-10)
            mcallister_bonus = max(0, precision_score * 25)  # 25 points per decimal place
        else:
            mcallister_bonus = 150  # Perfect match

        # Also reward larger V (diversity)
        log_V = np.log10(V_unc)

        ind.fitness = log_V * 5 + mcallister_bonus

        # Track unique branch by V_uncorrected
        branch_key = round(V_unc, 0)
        if branch_key not in self.hall_of_fame:
            self.hall_of_fame[branch_key] = ind
            self.stats['unique_branches'] += 1
            self.save_branch(ind, branch_key)

    def get_adaptive_mutation_rate(self, stagnation: int) -> float:
        if stagnation < 5:
            return np.random.choice([0.1, 0.2, 0.3], p=[0.5, 0.3, 0.2])
        elif stagnation < 10:
            return np.random.choice([0.2, 0.4, 0.6], p=[0.3, 0.4, 0.3])
        elif stagnation < 15:
            return np.random.choice([0.4, 0.6, 0.8], p=[0.3, 0.4, 0.3])
        else:
            return np.random.choice([0.6, 0.8, 1.0], p=[0.3, 0.4, 0.3])

    def get_adaptive_mutation_strength(self, stagnation: int) -> float:
        if stagnation < 5:
            return np.random.choice([0.1, 0.2, 0.3])
        elif stagnation < 10:
            return np.random.choice([0.2, 0.5, 1.0])
        elif stagnation < 15:
            return np.random.choice([0.5, 1.0, 2.0])
        else:
            return np.random.choice([1.0, 2.0, 5.0])

    def mutate(self, parent: Individual) -> Individual:
        """Mutate (t_init, n_steps) with adaptive rate."""
        stagnation = self.epoch - parent.last_improved_epoch
        rate = self.get_adaptive_mutation_rate(stagnation)
        strength = self.get_adaptive_mutation_strength(stagnation)

        # Mutate t_init
        t_new = parent.t_init.copy()
        for i in range(self.h11):
            if np.random.random() < rate:
                factor = np.exp(np.random.randn() * strength)
                t_new[i] = np.clip(t_new[i] * factor, self.t_range[0], self.t_range[1])

        # Mutate n_steps (occasionally)
        n_steps_new = parent.n_steps
        if np.random.random() < 0.2:  # 20% chance to mutate n_steps
            if np.random.random() < 0.5:
                # Multiply/divide by 2
                if np.random.random() < 0.5 and n_steps_new > STEP_RANGE[0]:
                    n_steps_new = max(STEP_RANGE[0], n_steps_new // 2)
                else:
                    n_steps_new = min(STEP_RANGE[1], n_steps_new * 2)
            else:
                # Random perturbation
                delta = np.random.choice([-16, -8, 8, 16])
                n_steps_new = np.clip(n_steps_new + delta, STEP_RANGE[0], STEP_RANGE[1])

        child = Individual(
            t_init=t_new,
            n_steps=int(n_steps_new),
            last_improved_epoch=parent.last_improved_epoch
        )
        self.evaluate(child)
        return child

    def crossover(self, p1: Individual, p2: Individual) -> Individual:
        """Uniform crossover of t_init, random choice of n_steps."""
        t_new = np.array([
            p1.t_init[i] if np.random.random() < 0.5 else p2.t_init[i]
            for i in range(self.h11)
        ])

        # n_steps: take from better parent or average
        if np.random.random() < 0.7:
            n_steps = p1.n_steps if p1.fitness > p2.fitness else p2.n_steps
        else:
            n_steps = (p1.n_steps + p2.n_steps) // 2

        best_parent = p1 if p1.fitness > p2.fitness else p2
        child = Individual(
            t_init=t_new,
            n_steps=int(n_steps),
            last_improved_epoch=best_parent.last_improved_epoch
        )
        self.evaluate(child)
        return child

    def asteroid_impact(self):
        """Reset population from Hall of Fame."""
        print(f"\n*** ASTEROID IMPACT at epoch {self.epoch}! Rebuilding... ***")
        self.stats['asteroids'] += 1

        self.frontier.clear()

        hof_list = list(self.hall_of_fame.values())
        hof_quota = int(self.frontier_size * 0.4)
        for _ in range(min(hof_quota, len(hof_list) * 3)):
            if hof_list:
                parent = np.random.choice(hof_list)
                t_new = parent.t_init * np.exp(np.random.randn(self.h11) * 1.0)
                t_new = np.clip(t_new, self.t_range[0], self.t_range[1])
                # Also vary n_steps
                n_steps = parent.n_steps
                if np.random.random() < 0.5:
                    n_steps = np.clip(n_steps + np.random.choice([-16, -8, 8, 16]),
                                       STEP_RANGE[0], STEP_RANGE[1])
                child = Individual(t_init=t_new, n_steps=int(n_steps), last_improved_epoch=self.epoch)
                self.evaluate(child)
                self.frontier.append(child)

        while len(self.frontier) < self.frontier_size:
            ind = Individual(
                t_init=self.random_t_init(),
                n_steps=self.random_n_steps(),
                last_improved_epoch=self.epoch
            )
            self.evaluate(ind)
            self.frontier.append(ind)

        self.epochs_since_global_improvement = 0
        self.frontier.sort(key=lambda x: x.fitness, reverse=True)

    def initialize(self):
        """Initialize frontier."""
        print(f"Initializing frontier of {self.frontier_size}...", flush=True)
        self.frontier = []
        for i in range(self.frontier_size):
            if i % 8 == 0:
                print(f"  Init {i}/{self.frontier_size}...", flush=True)
            ind = Individual(
                t_init=self.random_t_init(),
                n_steps=self.random_n_steps(),
                last_improved_epoch=0
            )
            self.evaluate(ind)
            self.frontier.append(ind)

        self.frontier.sort(key=lambda x: x.fitness, reverse=True)
        self.best_ever = self.frontier[0]
        print(f"Initial best fitness: {self.best_ever.fitness:.2f}", flush=True)
        if self.best_ever.V_string:
            print(f"Initial best V_string: {self.best_ever.V_string:.2f}, n_steps: {self.best_ever.n_steps}", flush=True)

    def evolve_generation(self):
        """Evolve one generation."""
        self.epoch += 1
        print(f"  Epoch {self.epoch} starting...", flush=True)

        if self.epochs_since_global_improvement >= self.asteroid_threshold:
            self.asteroid_impact()
            return

        children_per_parent = self.population_size // self.frontier_size
        candidates = list(self.frontier)
        print(f"  Generating {children_per_parent} children per parent...", flush=True)

        for parent in self.frontier:
            for _ in range(children_per_parent):
                if np.random.random() < 0.3 and len(self.frontier) > 1:
                    other = np.random.choice([f for f in self.frontier if f is not parent])
                    child = self.crossover(parent, other)
                else:
                    child = self.mutate(parent)

                if child.fitness > parent.fitness:
                    child.last_improved_epoch = self.epoch

                candidates.append(child)

        # Prune stagnant
        candidates = [c for c in candidates
                      if (self.epoch - c.last_improved_epoch) <= self.max_stagnation]

        # Sort and deduplicate
        candidates.sort(key=lambda x: x.fitness, reverse=True)
        seen_V = set()
        unique = []
        for c in candidates:
            if c.V_string is not None:
                key = round(c.V_string, 0)
                if key in seen_V:
                    continue
                seen_V.add(key)
            unique.append(c)
        candidates = unique

        self.frontier = candidates[:self.frontier_size]

        while len(self.frontier) < self.frontier_size:
            ind = Individual(
                t_init=self.random_t_init(),
                n_steps=self.random_n_steps(),
                last_improved_epoch=self.epoch
            )
            self.evaluate(ind)
            self.frontier.append(ind)

        self.frontier.sort(key=lambda x: x.fitness, reverse=True)

        if self.frontier[0].fitness > self.best_ever.fitness:
            self.best_ever = self.frontier[0]
            self.epochs_since_global_improvement = 0
        else:
            self.epochs_since_global_improvement += 1

    def run(self, max_time_seconds: float = 60, verbose: bool = True):
        """Run GA for specified time."""
        self.initialize()
        start_time = time.time()

        while time.time() - start_time < max_time_seconds:
            self.evolve_generation()

            if verbose:
                elapsed = time.time() - start_time
                best = self.frontier[0]
                V_str = f"{best.V_string:.1f}" if best.V_string else "N/A"
                print(f"[{elapsed:5.1f}s] Epoch {self.epoch:3d}: "
                      f"fitness={best.fitness:6.1f}, V_string={V_str}, n_steps={best.n_steps}, "
                      f"branches={self.stats['unique_branches']}, "
                      f"conv={self.stats['converged']}/{self.stats['evaluated']}", flush=True)

        return self.hall_of_fame


def load_example(example_name: str) -> dict:
    """Load all data for an example."""
    data_dir = DATA_BASE / example_name

    points = np.array([[int(x) for x in line.split(',')]
                       for line in (data_dir / "points.dat").read_text().strip().split('\n')])
    heights = np.array([float(x) for x in (data_dir / "corrected_heights.dat").read_text().strip().split(',')])
    basis = [int(x) for x in (data_dir / "basis.dat").read_text().strip().split(',')]
    kklt_basis = [int(x) for x in (data_dir / "kklt_basis.dat").read_text().strip().split(',')]
    c_i = np.array([int(x) for x in (data_dir / "target_volumes.dat").read_text().strip().split(',')])

    g_s = float((data_dir / "g_s.dat").read_text().strip())
    W0 = float((data_dir / "W_0.dat").read_text().strip())
    c_tau = float((data_dir / "c_tau.dat").read_text().strip())
    cy_vol_expected = float((data_dir / "cy_vol.dat").read_text().strip())

    poly = Polytope(points)
    tri = poly.triangulate(heights=heights)
    cy = tri.get_cy()
    cy.set_divisor_basis(basis)

    kappa_basis = cy.intersection_numbers(in_basis=True)
    kappa_all = cy.intersection_numbers(in_basis=False)
    kappa = SparseKappa(kappa_basis, kappa_all, basis, kklt_basis)

    return {
        "kappa": kappa,
        "c_i": c_i,
        "g_s": g_s,
        "W0": W0,
        "c_tau": c_tau,
        "cy_vol_expected": cy_vol_expected,
        "h11": cy.h11(),
        "h21": cy.h21(),
    }


def main():
    import argparse
    import sys

    # Ensure output is not buffered
    sys.stdout.reconfigure(line_buffering=True)

    print("Starting GA exploration...", flush=True)

    parser = argparse.ArgumentParser(description="GA-based KKLT branch exploration")
    parser.add_argument("--example", default="5-81-3213", help="Example name")
    parser.add_argument("--time", type=float, default=60, help="Max time in seconds")
    parser.add_argument("--frontier", type=int, default=32, help="Frontier size")
    parser.add_argument("--population", type=int, default=128, help="Population size")

    args = parser.parse_args()

    print("=" * 70, flush=True)
    print(f"GA BRANCH EXPLORATION - {args.example}", flush=True)
    print("=" * 70, flush=True)
    print(flush=True)
    print("GENOME: (t_init[h11], n_steps)", flush=True)
    print(f"  t_init: {args.example} has h11 floats", flush=True)
    print(f"  n_steps: {STEP_RANGE[0]}-{STEP_RANGE[1]} (predictor-corrector steps)", flush=True)
    print(flush=True)

    print("Loading data...", flush=True)
    data = load_example(args.example)
    print("Data loaded!", flush=True)

    print(f"h11 = {data['h11']}, h21 = {data['h21']}", flush=True)
    print(f"McAllister V_string = {data['cy_vol_expected']:.2f}", flush=True)
    print(flush=True)

    ga = TInitGA(
        kappa=data["kappa"],
        c_i=data["c_i"],
        c_tau=data["c_tau"],
        g_s=data["g_s"],
        W0=data["W0"],
        h11=data["h11"],
        h21=data["h21"],
        mcallister_V_string=data["cy_vol_expected"],
        frontier_size=args.frontier,
        population_size=args.population,
    )
    ga.example_name = args.example
    print(f"Branches will be saved to: {ga.output_dir / f'{args.example}_branches.jsonl'}", flush=True)

    print(f"Running GA for {args.time}s...")
    print("-" * 70)

    hall_of_fame = ga.run(max_time_seconds=args.time, verbose=True)

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Total evaluations: {ga.stats['evaluated']}")
    print(f"Converged: {ga.stats['converged']} ({ga.stats['converged']/ga.stats['evaluated']*100:.1f}%)")
    print(f"V > 0: {ga.stats['V_positive']}")
    print(f"Unique branches: {ga.stats['unique_branches']}")
    print(f"Asteroids: {ga.stats['asteroids']}")
    print()

    if hall_of_fame:
        branches = sorted(hall_of_fame.values(), key=lambda x: x.V_string or 0)

        print(f"All {len(branches)} branches found (sorted by V_uncorrected):")
        print("-" * 60)
        print(f"{'#':>3} {'n_steps':>8} {'V_unc':>12} {'fitness':>8}")
        print("-" * 60)

        for i, ind in enumerate(branches):
            print(f"{i+1:3d} {ind.n_steps:8d} {ind.V_string:12.2f} {ind.fitness:8.1f}")

        print("-" * 60)
        print()

        # Find a few interesting ones
        smallest_V = min(branches, key=lambda x: x.V_string)
        largest_V = max(branches, key=lambda x: x.V_string)
        print(f"Smallest V_unc: {smallest_V.V_string:.2f}")
        print(f"Largest V_unc: {largest_V.V_string:.2f}")
        print(f"Range: {largest_V.V_string / smallest_V.V_string:.1f}x")


if __name__ == "__main__":
    main()
