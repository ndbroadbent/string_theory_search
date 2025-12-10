#!/usr/bin/env python3
"""
Reproduce McAllister et al. CY volume (4711.83) using CYTools.

McAllister uses a FRST (Fine Regular Star Triangulation) with 218 prime toric
divisors, indexed 1-218 in basis.dat. CYTools with the 12-vertex dual polytope
only has 8 prime toric divisors.

Since we can't directly map McAllister's 214 ambient parameters, we use the
"safer route" from MCALLISTER_SMALL_CC_DETAILS.md: solve for t such that
the CY volume matches their published value.

Reference: arXiv:2107.09064 "Small cosmological constants in string theory"
"""

import sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Full precision for numpy printing
np.set_printoptions(precision=20, suppress=False)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "resources" / "small_cc_2107.09064_source" / "anc" / "paper_data" / "4-214-647"


def load_points(filename: str) -> list[list[int]]:
    """Load points from a CSV file."""
    points = []
    with open(DATA_DIR / filename) as f:
        for line in f:
            line = line.strip()
            if line:
                points.append([int(x) for x in line.split(",")])
    return points


def load_simplices(filename: str) -> list[list[int]]:
    """Load simplices from a CSV file."""
    simplices = []
    with open(DATA_DIR / filename) as f:
        for line in f:
            line = line.strip()
            if line:
                simplices.append([int(x) for x in line.split(",")])
    return simplices


def load_float(filename: str) -> float:
    """Load a single float from a file."""
    with open(DATA_DIR / filename) as f:
        return float(f.read().strip())


def run_single_optimization(args):
    """Worker function for parallel optimization. Creates its own CYTools objects."""
    idx, start_point, worker_data = args

    try:
        from cytools import Polytope
        import numpy as np
        from scipy.optimize import minimize

        # Create fresh CYTools objects in this worker
        poly = Polytope(worker_data['dual_points'])
        triang = poly.triangulate(simplices=worker_data['dual_simplices'])
        cy = triang.get_cy()
        cone = cy.toric_kahler_cone()
        H = np.array(cone.hyperplanes())
        target = worker_data['target_V_string']

        def objective(t):
            V = cy.compute_cy_volume(t)
            return (V - target) ** 2

        def cone_constraint(t):
            return H @ t

        start = np.array(start_point)
        res = minimize(
            objective,
            start,
            method='SLSQP',
            constraints={'type': 'ineq', 'fun': cone_constraint},
            options={'ftol': 1e-30, 'maxiter': 10000}
        )
        return (idx, res.fun, res.x.tolist())
    except Exception as e:
        return (idx, None, None)


def main():
    print("=" * 70)
    print("Reproducing McAllister CY Volume (4-214-647)")
    print("Using volume-matching optimization (not GLSM projection)")
    print("=" * 70)
    print()

    try:
        from cytools import Polytope
    except ImportError:
        raise RuntimeError("CYTools not installed")

    # Load McAllister data
    print("Loading McAllister data files...")
    dual_points = load_points("dual_points.dat")
    dual_simplices = load_simplices("dual_simplices.dat")
    g_s = load_float("g_s.dat")
    expected_vol_einstein = load_float("cy_vol.dat")

    print(f"  g_s: {g_s}")
    print(f"  expected CY vol (Einstein): {expected_vol_einstein}")
    print()

    # Create polytope with McAllister's triangulation
    print("Creating polytope with McAllister's triangulation...")
    poly = Polytope(dual_points)
    triang = poly.triangulate(simplices=dual_simplices)
    cy = triang.get_cy()

    h11 = cy.h11()
    h21 = cy.h21()
    print(f"  h11={h11}, h21={h21}")
    print(f"  prime toric divisors: {cy.prime_toric_divisors()}")
    print()

    # Get Kähler cone
    cone = cy.toric_kahler_cone()
    tip = cone.tip_of_stretched_cone(1.0)
    print(f"Kähler cone tip: {tip}")

    # Target volume in string frame
    target_V_string = expected_vol_einstein * (g_s ** 1.5)
    print(f"Target V_string: {target_V_string:.6f}")
    print()

    # Get hyperplane matrix for cone constraint
    H = np.array(cone.hyperplanes())
    print(f"Cone hyperplanes shape: {H.shape}")

    def objective(t):
        """Minimize |V(t) - V_target|^2"""
        V = cy.compute_cy_volume(t)
        return (V - target_V_string) ** 2

    def cone_constraint(t):
        """H @ t >= 0 for interior of cone"""
        return H @ t

    # Initial guess: scale tip to approximate target volume
    V_tip = cy.compute_cy_volume(tip)
    scale_init = (target_V_string / V_tip) ** (1/3)
    t_init = tip * scale_init

    print(f"Initial guess: {t_init}")
    print(f"V at initial: {cy.compute_cy_volume(t_init):.6f}")
    print()

    # Generate starting points
    n_starts = 100
    np.random.seed(42)  # Reproducibility
    starting_points = [t_init.tolist()]  # Include original
    for i in range(n_starts - 1):
        perturbed = t_init * (1 + 0.05 * np.random.randn(len(t_init)))
        perturbed = np.maximum(perturbed, 0.01)
        starting_points.append(perturbed.tolist())

    # Parallel optimization - each worker creates its own CYTools objects
    n_workers = multiprocessing.cpu_count()
    print(f"Optimizing to match target volume (parallel search)...")
    print(f"  Using {n_workers} CPU cores")
    print(f"  Running {n_starts} optimization starts...")
    print()

    # Save data needed by workers
    worker_data = {
        'dual_points': dual_points,
        'dual_simplices': dual_simplices,
        'target_V_string': target_V_string,
    }

    with multiprocessing.Pool(n_workers) as pool:
        args_list = [(i, start, worker_data) for i, start in enumerate(starting_points)]
        results = []
        for result in pool.imap_unordered(run_single_optimization, args_list):
            idx, obj_val, moduli = result
            results.append((idx, obj_val, moduli))
            if obj_val is not None:
                print(f"  [{idx+1}/{n_starts}] obj={obj_val:.6e}", flush=True)

    # Find best result
    valid_results = [(i, o, m) for i, o, m in results if o is not None]
    if not valid_results:
        raise RuntimeError("All optimizations failed")

    best_idx, best_fun, best_moduli = min(valid_results, key=lambda x: x[1])
    print()
    print(f"  Best objective value: {best_fun:.6e} (from start {best_idx})")

    # Re-run best to get full result object
    result = minimize(
        objective,
        best_moduli,
        method='SLSQP',
        constraints={'type': 'ineq', 'fun': cone_constraint},
        options={'ftol': 1e-30, 'maxiter': 10000}
    )

    if result is None or (not result.success and result.fun > 1e-20):
        raise RuntimeError(f"Optimization failed: {result.message if result else 'No valid result'}")

    t_opt = result.x
    V_opt = cy.compute_cy_volume(t_opt)
    V_E_opt = V_opt * (g_s ** (-1.5))

    print()
    print("=" * 70)
    print("RESULTS (FULL PRECISION)")
    print("=" * 70)
    print(f"  Kähler moduli t (list): {t_opt.tolist()}")
    print(f"  Kähler moduli t (repr): {repr(t_opt.tolist())}")
    print()
    print(f"  V_string computed:  {V_opt!r}")
    print(f"  V_string target:    {target_V_string!r}")
    print(f"  V_string diff:      {abs(V_opt - target_V_string)!r}")
    print(f"  V_string match:     {V_opt == target_V_string}")
    print()
    print(f"  V_einstein computed: {V_E_opt!r}")
    print(f"  V_einstein expected: {expected_vol_einstein!r}")
    print(f"  V_einstein diff:     {abs(V_E_opt - expected_vol_einstein)!r}")
    print(f"  Error %: {abs(V_E_opt - expected_vol_einstein) / expected_vol_einstein * 100!r}")
    print()

    # Verify cone containment
    in_cone = cone.contains(t_opt)
    print(f"  cone.contains(t): {in_cone}")

    if not in_cone:
        # Check hyperplane constraints
        Ht = H @ t_opt
        print(f"  Hyperplane values (should be > 0):")
        for i, val in enumerate(Ht):
            status = "OK" if val > 0 else "VIOLATED"
            print(f"    H[{i}] @ t = {val:.4f} ({status})")
        raise ValueError(
            f"Optimized moduli {t_opt.tolist()} are OUTSIDE Kähler cone. "
            f"Cannot reproduce McAllister volume with valid Kähler moduli."
        )

    # Compute divisor volumes
    div_vols = cy.compute_divisor_volumes(t_opt)
    print(f"  Divisor volumes: {div_vols}")

    if any(v < 0 for v in div_vols):
        raise ValueError(f"Negative divisor volumes: {div_vols}")

    print()
    print("SUCCESS: Found valid Kähler moduli reproducing McAllister volume.")
    return t_opt, V_E_opt, expected_vol_einstein


if __name__ == "__main__":
    main()
