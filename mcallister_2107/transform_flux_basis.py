#!/usr/bin/env python3
"""
Transform flux vectors between CYTools basis versions.

Given old_basis and new_basis indices, compute T such that:
    D_old[i] = sum_j T[i,j] * D_new[j]

Then transform fluxes:
    K_new = T^{-1} @ K_old  (covariant)
    M_new = T^T @ M_old     (contravariant)

The transformation T is derived from GLSM linear relations.

Reference: mcallister_2107/LATEST_CYTOOLS_CONVERSION_RESULT.md
"""
import numpy as np
from typing import Tuple


def compute_T_from_glsm(cy, old_basis: list, new_basis: list, verbose: bool = False) -> np.ndarray:
    """
    Compute transformation matrix T from GLSM linear relations.

    Args:
        cy: CalabiYau object
        old_basis: list of divisor indices in old basis
        new_basis: list of divisor indices in new basis
        verbose: print debug info

    Returns:
        T: transformation matrix where D_old = T @ D_new
    """
    h = len(old_basis)
    assert len(new_basis) == h

    # Get GLSM linear relations: L @ D = 0
    L = cy.triangulation().polytope().glsm_linear_relations()

    if verbose:
        print(f"GLSM linear relations shape: {L.shape}")
        print(f"L =\n{L}\n")

    T = np.zeros((h, h))

    for i, old_idx in enumerate(old_basis):
        if old_idx in new_basis:
            # Old basis divisor is also in new basis - just identity
            j = new_basis.index(old_idx)
            T[i, j] = 1
            if verbose:
                print(f"D_{old_idx} = D_{old_idx} (in both bases)")
        else:
            # Need to express D_old_idx in terms of new_basis using GLSM relations
            # Find a relation involving D_old_idx
            for row_idx, row in enumerate(L):
                coeff = row[old_idx]
                if abs(coeff) > 1e-10:
                    # Found a relation: coeff * D_old_idx + sum(row[j] * D_j) = 0
                    # => D_old_idx = -sum(row[j] * D_j) / coeff (for j != old_idx)

                    if verbose:
                        print(f"Using GLSM row {row_idx}: {row}")

                    expr = []
                    for j, new_idx in enumerate(new_basis):
                        if new_idx != old_idx:
                            c = -row[new_idx] / coeff
                            T[i, j] = c
                            if abs(c) > 1e-10:
                                expr.append(f"{c:+.0f}*D_{new_idx}")

                    if verbose:
                        print(f"D_{old_idx} = {' '.join(expr) if expr else '0'}")
                    break

    return T


def transform_fluxes(K_old: np.ndarray, M_old: np.ndarray, T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform fluxes from old basis to new basis.

    K is covariant: K_new = T^{-1} @ K_old
    M is contravariant: M_new = T^T @ M_old

    Args:
        K_old: flux vector K in old basis
        M_old: flux vector M in old basis
        T: transformation matrix (D_old = T @ D_new)

    Returns:
        K_new, M_new: transformed flux vectors
    """
    T_inv = np.linalg.inv(T)
    K_new = T_inv @ K_old
    M_new = T.T @ M_old
    return np.round(K_new).astype(int), np.round(M_new).astype(int)


def transform_fluxes_for_cy(
    cy,
    K_old: np.ndarray,
    M_old: np.ndarray,
    old_basis: list,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Transform K, M from old basis to CYTools' current basis.

    Args:
        cy: CalabiYau object
        K_old: flux vector K in old basis
        M_old: flux vector M in old basis
        old_basis: list of divisor indices in old basis
        verbose: print debug info

    Returns:
        K_new, M_new, T: transformed fluxes and transformation matrix
    """
    new_basis = list(cy.divisor_basis())

    # If bases are identical, no transformation needed
    if list(old_basis) == list(new_basis):
        return K_old.copy(), M_old.copy(), np.eye(len(old_basis), dtype=int)

    T = compute_T_from_glsm(cy, old_basis, new_basis, verbose=verbose)
    K_new, M_new = transform_fluxes(K_old, M_old, T)

    return K_new, M_new, T
