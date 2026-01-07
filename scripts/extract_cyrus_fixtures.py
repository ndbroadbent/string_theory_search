#!/usr/bin/env python3
"""
Extract intermediate data from CYTools intersection number computation.
This serves as the ground truth for Rust unit tests.
"""

import sys
import os
import json
import numpy as np
from pathlib import Path

# Add CYTools to path (adjust as needed for your env)
CYTOOLS_PATH = Path(__file__).parent.parent / "vendor/cytools_mcallister_2107"
sys.path.insert(0, str(CYTOOLS_PATH))

from cytools import Polytope
from cytools.utils import filter_tensor_indices

def main():
    # Load 4-214-647 Dual Polytope
    data_dir = Path(__file__).parent.parent / "resources/small_cc_2107.09064_source/anc/paper_data/4-214-647"
    points = np.loadtxt(data_dir / "dual_points.dat", dtype=int, delimiter=",")
    # heights = np.loadtxt(data_dir / "heights.dat", delimiter=",") # Heights are for primal
    
    print(f"Loaded {len(points)} points")
    
    p = Polytope(points)
    t = p.triangulate() # Default triangulation for dual
    v = t.get_cy()
    
    print("Computing intersection numbers...")
    # Force computation to populate internal cache
    _ = v.intersection_numbers(in_basis=False)
    
    # Extract internal data structures
    # Note: These are private attributes, so we access them directly for debugging/extraction
    
    # 1. Linear Relations (Origin excluded)
    lin_rels = v.glsm_linear_relations(include_origin=False)
    print(f"Linear Relations shape: {lin_rels.shape}")
    
    # 2. Equation Construction (re-run logic to get intermediates)
    # We replicate the logic from _construct_intnum_equations_4d to capture M, C, vars
    
    # Access the cached result from the computation we just ran
    # The cache key is (zero_as_anticanonical, in_basis, exact_arithmetic, format)
    # Default: (False, False, False, "dok")
    
    # BUT, we want M and C. These aren't stored in the object, they are local to the function.
    # We must invoke the private method to get them.
    
    Mat, C, distintnum_array, variable_array = v._construct_intnum_equations_4d()
    
    print(f"Matrix M shape: {Mat.shape}")
    print(f"Vector C shape: {C.shape}")
    print(f"Variables count: {len(variable_array)}")
    print(f"Distinct knowns count: {len(distintnum_array)}")
    
    # 3. Solve
    # We want the solution vector x. We can get it by solving again or inferring from result.
    # Let's solve it again using the util
    from cytools.utils import solve_linear_system
    solution = solve_linear_system(Mat, C)
    
    solution_list = solution if isinstance(solution, list) else solution.tolist()
    
    # 4. Origin Handling
    # The full tensor (including origin) is what v.intersection_numbers() returned
    full_tensor_list = v.intersection_numbers(in_basis=False)
    full_tensor = {tuple(int(x) for x in item[:-1]): float(item[-1]) for item in full_tensor_list}
    
    # 5. Basis
    basis = v.divisor_basis().tolist()
    basis_tensor_list = v.intersection_numbers(in_basis=True)
    basis_tensor = {tuple(int(x) for x in item[:-1]): float(item[-1]) for item in basis_tensor_list}
    
    # Prepare Output
    output = {
        "linear_relations": lin_rels.tolist(),
        "known_intersections": [
            {"indices": [int(x) for x in item[:-1]], "value": float(item[-1])}
            for item in distintnum_array
        ],
        "variables": [
            [int(x) for x in var] for var in variable_array
        ],
        "equations": {
            "M_shape": list(Mat.shape),
            "M_indices": [[int(r), int(c)] for r, c in zip(*Mat.nonzero())],
            "M_values": Mat.data.tolist(),
            "C": C.tolist()
        },
        "solution": solution_list,
        "full_tensor": {
            str(k): v for k, v in full_tensor.items()
        },
        "basis": basis,
        "basis_tensor": {
            str(k): v for k, v in basis_tensor.items()
        }
    }
    
    outfile = Path("tests/fixtures/mcallister_4_214_647_cytools_internals.json")
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)
        
    print(f"Saved fixtures to {outfile}")

if __name__ == "__main__":
    main()