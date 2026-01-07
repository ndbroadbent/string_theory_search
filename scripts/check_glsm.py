import numpy as np
import sys
from pathlib import Path

# Add latest cytools to path
sys.path.insert(0, str(Path("vendor/cytools_latest/src")))
from cytools import Polytope

def check_glsm():
    # P1 x P1 rays: (1,0), (-1,0), (0,1), (0,-1)
    # These are the vertices of the reflexive polytope for P1xP1
    points = np.array([
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1]
    ])
    
    # In CYTools, GLSM charge matrix comes from the polytope
    p = Polytope(points)
    q = p.glsm_charge_matrix()
    print("CYTools GLSM Charge Matrix (P1xP1):")
    print(q)
    print(f"Shape: {q.shape}")
    print(f"Sum of charges per row: {np.sum(q, axis=1)}")

    # Simple P2
    points_p2 = np.array([
        [1, 0],
        [0, 1],
        [-1, -1]
    ])
    p2 = Polytope(points_p2)
    q2 = p2.glsm_charge_matrix()
    print("\nCYTools GLSM Charge Matrix (P2):")
    print(q2)
    print(f"Shape: {q2.shape}")
    print(f"Sum of charges per row: {np.sum(q2, axis=1)}")

if __name__ == "__main__":
    check_glsm()
