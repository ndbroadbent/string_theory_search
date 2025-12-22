#!/usr/bin/env python3
"""
Visualization of toric Calabi-Yau manifolds from McAllister's paper.

The CY is defined as the zero locus of a Laurent polynomial in (C*)^4.
We sample points by solving the constraint numerically on a structured grid.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "vendor" / "cytools_latest"))
from cytools import Polytope


def load_polytope_and_dual(example_name="5-81-3213"):
    """Load polytope and get dual (which defines the polynomial)."""
    data_dir = Path(__file__).parent.parent / "resources" / "small_cc_2107.09064_source" / "anc" / "paper_data" / example_name

    points = []
    with open(data_dir / "points.dat") as f:
        for line in f:
            points.append([int(x) for x in line.strip().split(",")])

    p = Polytope(points)
    dual = p.dual()
    return p, dual


def evaluate_laurent_poly(z, monomials, coeffs):
    """
    Evaluate Laurent polynomial at point z in (C*)^4.
    """
    result = 0.0 + 0.0j
    for mono, coeff in zip(monomials, coeffs):
        term = coeff
        for i, exp in enumerate(mono):
            if exp != 0:
                term *= z[i] ** exp
        result += term
    return result


def sample_toric_cy_structured(monomials, resolution=50, seed=42):
    """
    Sample points on the toric CY on a structured grid.

    This reveals the surface structure by sampling nearby points
    rather than random scattered points.
    """
    np.random.seed(seed)
    n_terms = len(monomials)
    coeffs = np.random.randn(n_terms) + 1j * np.random.randn(n_terms)
    coeffs = coeffs / np.abs(coeffs)

    points_6d = []

    # Sample on a regular grid of phases (like latitude/longitude)
    theta1_vals = np.linspace(0, 2*np.pi, resolution)
    theta2_vals = np.linspace(0, 2*np.pi, resolution)

    # Fix theta3 and r values to get 2D slices at different "depths"
    for theta3_base in [0, np.pi/2, np.pi, 3*np.pi/2]:
        for r_scale in [0.8, 1.0, 1.2]:
            for theta1 in theta1_vals:
                for theta2 in theta2_vals:
                    z1 = r_scale * np.exp(1j * theta1)
                    z2 = r_scale * np.exp(1j * theta2)
                    z3 = r_scale * np.exp(1j * theta3_base)

                    z_partial = np.array([1.0, z1, z2, z3])
                    z0_powers = monomials[:, 0]
                    min_pow = int(z0_powers.min())
                    max_pow = int(z0_powers.max())

                    poly_coeffs = {}
                    for mono, coeff in zip(monomials, coeffs):
                        z0_pow = mono[0]
                        contrib = coeff
                        for i in range(1, 4):
                            if mono[i] != 0:
                                contrib *= z_partial[i] ** mono[i]
                        if z0_pow not in poly_coeffs:
                            poly_coeffs[z0_pow] = 0.0 + 0.0j
                        poly_coeffs[z0_pow] += contrib

                    degree = max_pow - min_pow
                    if degree < 1:
                        continue

                    root_coeffs = np.zeros(degree + 1, dtype=complex)
                    for pow_val, coeff_val in poly_coeffs.items():
                        idx = max_pow - pow_val
                        root_coeffs[idx] = coeff_val

                    try:
                        roots = np.roots(root_coeffs)
                    except:
                        continue

                    # Take the root closest to unit circle
                    valid_roots = [z0 for z0 in roots if np.isfinite(z0) and 0.1 < np.abs(z0) < 10]
                    if not valid_roots:
                        continue

                    z0 = min(valid_roots, key=lambda x: abs(np.abs(x) - 1))

                    z_full = np.array([z0, z1, z2, z3])
                    val = evaluate_laurent_poly(z_full, monomials, coeffs)
                    if np.abs(val) > 1e-6:
                        continue

                    point_6d = [
                        np.real(z0), np.imag(z0),
                        np.real(z1), np.imag(z1),
                        np.real(z2), np.imag(z2)
                    ]
                    points_6d.append(point_6d)

    return np.array(points_6d)


def rotation_matrix_6d(i, j, angle):
    """Create a 6D rotation matrix in the (i,j) plane."""
    R = np.eye(6)
    c, s = np.cos(angle), np.sin(angle)
    R[i, i] = c
    R[i, j] = -s
    R[j, i] = s
    R[j, j] = c
    return R


def project_6d_to_dual_3d(points_6d, R_6d):
    """Apply 6D rotation, then project to two 3D views."""
    rotated = (R_6d @ points_6d.T).T
    left_3d = rotated[:, :3]
    right_3d = rotated[:, 3:]
    return left_3d, right_3d


def create_toric_cy_still(example_name="5-81-3213", save_path=None, resolution=60):
    """Create a still image of the toric CY."""

    print(f"Loading polytope: {example_name}")
    p, dual = load_polytope_and_dual(example_name)
    monomials = dual.points()

    parts = example_name.split("-")
    h21 = int(parts[0])
    h11 = int(parts[1])
    print(f"  h¹¹ = {h11}, h²¹ = {h21}")
    print(f"  {len(monomials)} monomials in defining polynomial")

    print(f"Sampling structured points (resolution={resolution})...")
    points_6d = sample_toric_cy_structured(monomials, resolution=resolution)
    print(f"  Got {len(points_6d)} valid points")

    if len(points_6d) < 100:
        print("ERROR: Too few points sampled")
        return None

    # Set up figure
    fig = plt.figure(figsize=(16, 7))
    fig.patch.set_facecolor('black')

    ax_left = fig.add_subplot(121, projection='3d')
    ax_right = fig.add_subplot(122, projection='3d')

    for ax in [ax_left, ax_right]:
        ax.set_facecolor('black')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('none')
        ax.yaxis.pane.set_edgecolor('none')
        ax.zaxis.pane.set_edgecolor('none')
        ax.grid(False)
        ax.tick_params(colors='#333333', labelsize=7)

    # Apply a 6D rotation for interesting view
    R_6d = rotation_matrix_6d(0, 3, 0.4) @ rotation_matrix_6d(1, 4, 0.3)
    left_3d, right_3d = project_6d_to_dual_3d(points_6d, R_6d)

    # Color by position in hidden dimensions
    color_left = points_6d[:, 4]
    color_right = points_6d[:, 1]

    ax_left.scatter(left_3d[:, 0], left_3d[:, 1], left_3d[:, 2],
        c=color_left, cmap='inferno', s=0.5, alpha=0.7)
    ax_right.scatter(right_3d[:, 0], right_3d[:, 1], right_3d[:, 2],
        c=color_right, cmap='magma', s=0.5, alpha=0.7)

    ax_left.set_title('Dimensions 1, 2, 3', color='#666666', fontsize=10)
    ax_right.set_title('Dimensions 4, 5, 6', color='#666666', fontsize=10)

    for ax in [ax_left, ax_right]:
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')

    plt.suptitle(f'Toric CY: {example_name} (h¹¹={h11}, h²¹={h21})',
                 color='#888888', fontsize=12)
    plt.tight_layout()

    if save_path:
        print(f"Saving to {save_path}...")
        plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
        print(f"Saved: {save_path}")

    return fig, points_6d


def create_toric_cy_animation(example_name="5-81-3213", frames=120, save_path=None, resolution=50):
    """Create animation of toric CY with 6D rotations."""

    print(f"Loading polytope: {example_name}")
    p, dual = load_polytope_and_dual(example_name)
    monomials = dual.points()

    parts = example_name.split("-")
    h21 = int(parts[0])
    h11 = int(parts[1])
    print(f"  h¹¹ = {h11}, h²¹ = {h21}")

    print(f"Sampling structured points (resolution={resolution})...")
    points_6d = sample_toric_cy_structured(monomials, resolution=resolution)
    print(f"  Got {len(points_6d)} valid points")

    if len(points_6d) < 100:
        print("ERROR: Too few points sampled")
        return None, None

    # Set up figure
    fig = plt.figure(figsize=(16, 7))
    fig.patch.set_facecolor('black')

    ax_left = fig.add_subplot(121, projection='3d')
    ax_right = fig.add_subplot(122, projection='3d')

    for ax in [ax_left, ax_right]:
        ax.set_facecolor('black')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('none')
        ax.yaxis.pane.set_edgecolor('none')
        ax.zaxis.pane.set_edgecolor('none')
        ax.grid(False)
        ax.tick_params(colors='#333333', labelsize=7)

    # Find bounds
    max_extent = np.abs(points_6d).max() * 1.2

    # Color arrays (fixed)
    color_left = points_6d[:, 4]
    color_right = points_6d[:, 1]

    scatter_left = [None]
    scatter_right = [None]

    def init():
        for ax in [ax_left, ax_right]:
            ax.set_xlim(-max_extent, max_extent)
            ax.set_ylim(-max_extent, max_extent)
            ax.set_zlim(-max_extent, max_extent)
        return []

    def animate(frame):
        # Clear previous
        if scatter_left[0] is not None:
            scatter_left[0].remove()
        if scatter_right[0] is not None:
            scatter_right[0].remove()

        # 6D rotation
        frames_per_phase = max(1, frames // 4)
        phase = frame // frames_per_phase
        t = (frame % frames_per_phase) / frames_per_phase
        angle = 2 * np.pi * t

        if phase == 0:
            R_6d = rotation_matrix_6d(0, 3, angle)
        elif phase == 1:
            R_6d = rotation_matrix_6d(1, 4, angle)
        elif phase == 2:
            R_6d = rotation_matrix_6d(2, 5, angle)
        else:
            R_6d = rotation_matrix_6d(0, 4, angle) @ rotation_matrix_6d(1, 5, angle * 0.7)

        left_3d, right_3d = project_6d_to_dual_3d(points_6d, R_6d)

        scatter_left[0] = ax_left.scatter(
            left_3d[:, 0], left_3d[:, 1], left_3d[:, 2],
            c=color_left, cmap='inferno', s=0.5, alpha=0.7
        )
        scatter_right[0] = ax_right.scatter(
            right_3d[:, 0], right_3d[:, 1], right_3d[:, 2],
            c=color_right, cmap='magma', s=0.5, alpha=0.7
        )

        ax_left.set_title('Dimensions 1, 2, 3', color='#666666', fontsize=10)
        ax_right.set_title('Dimensions 4, 5, 6', color='#666666', fontsize=10)

        return [scatter_left[0], scatter_right[0]]

    plt.suptitle(f'Toric CY: {example_name} (h¹¹={h11}, h²¹={h21})',
                 color='#888888', fontsize=12)
    plt.tight_layout()

    print(f"Creating animation with {frames} frames...")
    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=frames, interval=50, blit=False
    )

    if save_path:
        print(f"Saving to {save_path}...")
        anim.save(save_path, writer='ffmpeg', fps=30, dpi=150,
                  savefig_kwargs={'facecolor': fig.get_facecolor()})
        print(f"Saved: {save_path}")

    return fig, anim


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Toric CY visualization")
    parser.add_argument("--example", type=str, default="5-81-3213",
                       choices=["4-214-647", "5-81-3213", "5-113-4627-main", "7-51-13590"],
                       help="Which McAllister example to visualize")
    parser.add_argument("--frames", type=int, default=120, help="Animation frames (0 for still)")
    parser.add_argument("--resolution", type=int, default=60, help="Grid resolution for sampling")
    parser.add_argument("--png", action="store_true", help="Output PNG still image")
    parser.add_argument("--show", action="store_true", help="Show interactive")

    args = parser.parse_args()

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    if args.png or args.frames == 0:
        save_path = output_dir / f"toric_cy_{args.example}.png"
        fig, _ = create_toric_cy_still(args.example, save_path, args.resolution)
    else:
        save_path = output_dir / f"toric_cy_{args.example}.mp4"
        fig, anim = create_toric_cy_animation(args.example, args.frames, save_path, args.resolution)

    if args.show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()
