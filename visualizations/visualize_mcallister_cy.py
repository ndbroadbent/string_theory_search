#!/usr/bin/env python3
"""
Visualization of Calabi-Yau manifolds from McAllister's small cosmological constant paper.

Uses the actual polytope data from arXiv:2107.09064 to visualize the CY geometry.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from pathlib import Path
import sys

# Add vendor path for cytools
sys.path.insert(0, str(Path(__file__).parent.parent / "vendor" / "cytools_latest"))

from cytools import Polytope


def load_mcallister_polytope(example_name="5-81-3213"):
    """Load polytope vertices from McAllister's paper data."""
    data_dir = Path(__file__).parent.parent / "resources" / "small_cc_2107.09064_source" / "anc" / "paper_data" / example_name

    points_file = data_dir / "points.dat"
    points = []
    with open(points_file) as f:
        for line in f:
            coords = [int(x) for x in line.strip().split(",")]
            points.append(coords)

    return np.array(points)


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
    shape = points_6d.shape[:-1]
    flat = points_6d.reshape(-1, 6)
    rotated = (R_6d @ flat.T).T.reshape(*shape, 6)
    left_3d = rotated[..., :3]
    right_3d = rotated[..., 3:]
    return left_3d, right_3d


def sample_cy_from_toric(polytope_points, n_samples=5000):
    """
    Sample points from a Calabi-Yau defined by a toric polytope.

    The CY is a hypersurface in the toric variety. We sample points
    by using the Kähler quotient structure of the toric variety.

    For visualization, we embed into 6D using:
    - First 3 dims: combination of toric coordinates
    - Last 3 dims: different combination to capture full structure
    """
    # The polytope defines the fan structure
    # We sample using the moment map image

    n_vertices = len(polytope_points)
    dim = polytope_points.shape[1]  # Should be 4 for CY3

    # Sample random points in the toric ambient space
    # Use exponential coordinates: |z_i|^2 parametrized by moment map

    # Generate random moment map values (in the polytope)
    # The polytope itself lives in R^dim, moment map maps to R^(n_vertices - dim - 1)

    # For visualization, we'll create a 6D embedding
    # by projecting the toric structure

    points_6d = []

    # Sample uniformly in parameter space
    for _ in range(n_samples):
        # Random phases for each coordinate
        phases = np.random.uniform(0, 2*np.pi, n_vertices)

        # Random moduli (|z_i|^2 values) - constrained by moment map
        # For simplicity, sample on a sphere in the polytope
        moduli = np.abs(np.random.randn(dim))
        moduli = moduli / np.linalg.norm(moduli)

        # Project polytope vertices with these moduli as weights
        weighted_point = np.zeros(dim)
        for i, v in enumerate(polytope_points):
            weight = np.exp(-0.5 * np.linalg.norm(moduli - v/np.max(np.abs(polytope_points)))**2)
            weighted_point += weight * v

        # Create 6D embedding
        # Use first 3 coordinates directly
        x1, x2, x3 = weighted_point[0], weighted_point[1], weighted_point[2]

        # Use phase information for other 3 coordinates
        phase_combo = phases[:3]
        x4 = np.sin(phase_combo[0]) * np.cos(phase_combo[1])
        x5 = np.sin(phase_combo[1]) * np.cos(phase_combo[2])
        x6 = np.cos(phase_combo[0]) * np.sin(phase_combo[2])

        # Scale by moduli
        scale = np.linalg.norm(moduli)
        points_6d.append([x1*scale, x2*scale, x3, x4*scale, x5*scale, x6])

    return np.array(points_6d)


def sample_cy_surface_patches(polytope_points, resolution=15):
    """
    Generate surface patches for the CY manifold visualization.

    Uses a parametric approach based on the polytope structure.
    """
    dim = polytope_points.shape[1]

    # Normalize polytope to unit scale
    pts_normalized = polytope_points / (np.max(np.abs(polytope_points)) + 1e-10)

    # Generate patches around each vertex
    patches = []

    u = np.linspace(0, 2*np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    U, V = np.meshgrid(u, v)

    # Group vertices by proximity for patch generation
    n_patches = min(25, len(polytope_points))  # Limit patches for performance

    # Use k-means-like selection of representative vertices
    indices = np.linspace(0, len(polytope_points)-1, n_patches, dtype=int)

    for idx in indices:
        vertex = pts_normalized[idx]

        # Find nearby vertices to define local patch
        distances = np.linalg.norm(pts_normalized - vertex, axis=1)
        nearby_idx = np.argsort(distances)[1:min(5, len(polytope_points))]

        # Create tangent directions from nearby vertices
        if len(nearby_idx) >= 2:
            t1 = pts_normalized[nearby_idx[0]] - vertex
            t2 = pts_normalized[nearby_idx[1]] - vertex

            # Orthogonalize
            t1 = t1 / (np.linalg.norm(t1) + 1e-10)
            t2 = t2 - np.dot(t2, t1) * t1
            t2 = t2 / (np.linalg.norm(t2) + 1e-10)
        else:
            # Default tangent directions
            t1 = np.zeros(dim)
            t2 = np.zeros(dim)
            t1[0] = 1
            t2[1] = 1

        # Parametric surface patch
        # x(u,v) = vertex + r(u,v) * (sin(v)*cos(u)*t1 + sin(v)*sin(u)*t2 + cos(v)*n)

        r = 0.3  # Patch radius

        # Surface in 4D parameter space
        patch_4d = np.zeros((resolution, resolution, dim))
        for i in range(resolution):
            for j in range(resolution):
                ui, vj = U[i, j], V[i, j]
                point = vertex + r * (np.sin(vj) * np.cos(ui) * t1 +
                                     np.sin(vj) * np.sin(ui) * t2)
                patch_4d[i, j] = point

        # Embed 4D patch into 6D
        # Use complex structure: (Re, Im) pairs for each complex coordinate
        patch_6d = np.zeros((resolution, resolution, 6))

        # Map 4D -> 6D using CY-like structure
        # z1 = x0 + i*x1, z2 = x2 + i*x3, z3 = cross terms
        patch_6d[..., 0] = patch_4d[..., 0]  # Re(z1)
        patch_6d[..., 1] = patch_4d[..., 1]  # Im(z1)
        patch_6d[..., 2] = patch_4d[..., 2]  # Re(z2)
        patch_6d[..., 3] = patch_4d[..., 3]  # Im(z2)

        # z3 comes from CY constraint - use combination
        z1_sq = patch_4d[..., 0]**2 + patch_4d[..., 1]**2
        z2_sq = patch_4d[..., 2]**2 + patch_4d[..., 3]**2
        patch_6d[..., 4] = np.sin(np.arctan2(patch_4d[..., 1], patch_4d[..., 0])) * (1 - z1_sq - z2_sq)**0.5
        patch_6d[..., 5] = np.cos(np.arctan2(patch_4d[..., 3], patch_4d[..., 2])) * (1 - z1_sq - z2_sq)**0.5

        # Handle NaN from sqrt of negative
        patch_6d = np.nan_to_num(patch_6d, nan=0.0)

        patches.append(patch_6d)

    return patches


def create_mcallister_still(example_name="5-81-3213", save_path=None, rotation_angle=0.3):
    """Create a single still image of McAllister CY manifold."""

    print(f"Loading polytope: {example_name}")
    polytope_points = load_mcallister_polytope(example_name)
    print(f"  Vertices: {len(polytope_points)}, dimension: {polytope_points.shape[1]}")

    parts = example_name.split("-")
    h21 = int(parts[0])
    h11 = int(parts[1])
    print(f"  h¹¹ = {h11}, h²¹ = {h21}")

    print("Generating surface patches...")
    patches_6d = sample_cy_surface_patches(polytope_points, resolution=20)
    print(f"  Generated {len(patches_6d)} patches")

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
        ax.xaxis.line.set_color('#222222')
        ax.yaxis.line.set_color('#222222')
        ax.zaxis.line.set_color('#222222')

    cmap_left = cm.inferno
    cmap_right = cm.magma

    all_points = np.concatenate([p.reshape(-1, 6) for p in patches_6d])
    max_extent = np.abs(all_points[np.isfinite(all_points)]).max() * 1.2
    if not np.isfinite(max_extent) or max_extent == 0:
        max_extent = 2.0

    for ax in [ax_left, ax_right]:
        ax.set_xlim(-max_extent, max_extent)
        ax.set_ylim(-max_extent, max_extent)
        ax.set_zlim(-max_extent, max_extent)

    # Cross-view 6D rotation
    R_6d = rotation_matrix_6d(0, 3, rotation_angle) @ rotation_matrix_6d(1, 4, rotation_angle * 0.7)

    for patch_6d in patches_6d:
        left_3d, right_3d = project_6d_to_dual_3d(patch_6d, R_6d)

        color_left = patch_6d[..., 3]
        color_right = patch_6d[..., 0]

        if color_left.max() == color_left.min():
            color_left = np.zeros_like(color_left)
        if color_right.max() == color_right.min():
            color_right = np.zeros_like(color_right)

        norm_left = Normalize(vmin=color_left.min() - 0.01, vmax=color_left.max() + 0.01)
        norm_right = Normalize(vmin=color_right.min() - 0.01, vmax=color_right.max() + 0.01)

        try:
            ax_left.plot_surface(
                left_3d[..., 0], left_3d[..., 1], left_3d[..., 2],
                facecolors=cmap_left(norm_left(color_left)),
                alpha=0.85, linewidth=0, antialiased=True, shade=False
            )
            ax_right.plot_surface(
                right_3d[..., 0], right_3d[..., 1], right_3d[..., 2],
                facecolors=cmap_right(norm_right(color_right)),
                alpha=0.85, linewidth=0, antialiased=True, shade=False
            )
        except:
            pass

    ax_left.set_title('Dimensions 1, 2, 3', color='#666666', fontsize=10)
    ax_right.set_title('Dimensions 4, 5, 6', color='#666666', fontsize=10)

    for ax in [ax_left, ax_right]:
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')

    plt.suptitle(f'McAllister CY: {example_name} (h¹¹={h11}, h²¹={h21})',
                 color='#888888', fontsize=12)
    plt.tight_layout()

    if save_path:
        print(f"Saving to {save_path}...")
        plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
        print(f"Saved: {save_path}")

    return fig


def create_mcallister_animation(example_name="5-81-3213", frames=180, save_path=None):
    """Create animation of McAllister CY manifold with 6D rotations."""

    print(f"Loading polytope: {example_name}")
    polytope_points = load_mcallister_polytope(example_name)
    print(f"  Vertices: {len(polytope_points)}, dimension: {polytope_points.shape[1]}")

    # Parse h11, h21 from name
    parts = example_name.split("-")
    h21 = int(parts[0])
    h11 = int(parts[1])
    print(f"  h¹¹ = {h11}, h²¹ = {h21}")

    # Generate surface patches
    print("Generating surface patches...")
    patches_6d = sample_cy_surface_patches(polytope_points, resolution=20)
    print(f"  Generated {len(patches_6d)} patches")

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
        ax.xaxis.line.set_color('#222222')
        ax.yaxis.line.set_color('#222222')
        ax.zaxis.line.set_color('#222222')

    # Colormaps
    cmap_left = cm.inferno
    cmap_right = cm.magma

    # Find global bounds
    all_points = np.concatenate([p.reshape(-1, 6) for p in patches_6d])
    max_extent = np.abs(all_points[np.isfinite(all_points)]).max() * 1.2
    if not np.isfinite(max_extent) or max_extent == 0:
        max_extent = 2.0

    surfaces_left = []
    surfaces_right = []

    def init():
        for ax in [ax_left, ax_right]:
            ax.set_xlim(-max_extent, max_extent)
            ax.set_ylim(-max_extent, max_extent)
            ax.set_zlim(-max_extent, max_extent)
        return []

    def animate(frame):
        nonlocal surfaces_left, surfaces_right

        # Clear previous
        for surf in surfaces_left + surfaces_right:
            try:
                surf.remove()
            except:
                pass
        surfaces_left = []
        surfaces_right = []

        # 6D rotation - cycle through cross-view planes
        frames_per_phase = max(1, frames // 4)
        phase = frame // frames_per_phase
        t_in_phase = (frame % frames_per_phase) / max(1, frames_per_phase)
        angle = 2 * np.pi * t_in_phase

        if phase == 0:
            R_6d = rotation_matrix_6d(0, 3, angle)
        elif phase == 1:
            R_6d = rotation_matrix_6d(1, 4, angle)
        elif phase == 2:
            R_6d = rotation_matrix_6d(2, 5, angle)
        else:
            R_6d = rotation_matrix_6d(0, 4, angle) @ rotation_matrix_6d(1, 5, angle * 0.7)

        # Plot each patch
        for patch_6d in patches_6d:
            left_3d, right_3d = project_6d_to_dual_3d(patch_6d, R_6d)

            # Color by hidden dimensions
            color_left = patch_6d[..., 3]
            color_right = patch_6d[..., 0]

            # Handle all-same values
            if color_left.max() == color_left.min():
                color_left = np.zeros_like(color_left)
            if color_right.max() == color_right.min():
                color_right = np.zeros_like(color_right)

            norm_left = Normalize(vmin=color_left.min() - 0.01, vmax=color_left.max() + 0.01)
            norm_right = Normalize(vmin=color_right.min() - 0.01, vmax=color_right.max() + 0.01)

            try:
                surf_l = ax_left.plot_surface(
                    left_3d[..., 0], left_3d[..., 1], left_3d[..., 2],
                    facecolors=cmap_left(norm_left(color_left)),
                    alpha=0.85, linewidth=0, antialiased=True, shade=False
                )
                surfaces_left.append(surf_l)

                surf_r = ax_right.plot_surface(
                    right_3d[..., 0], right_3d[..., 1], right_3d[..., 2],
                    facecolors=cmap_right(norm_right(color_right)),
                    alpha=0.85, linewidth=0, antialiased=True, shade=False
                )
                surfaces_right.append(surf_r)
            except Exception as e:
                pass

        ax_left.set_title('Dimensions 1, 2, 3', color='#666666', fontsize=10)
        ax_right.set_title('Dimensions 4, 5, 6', color='#666666', fontsize=10)

        for ax in [ax_left, ax_right]:
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_zlabel('')

        return surfaces_left + surfaces_right

    plt.suptitle(f'McAllister CY: {example_name} (h¹¹={h11}, h²¹={h21})',
                 color='#888888', fontsize=12)
    plt.tight_layout()

    print(f"Creating animation with {frames} frames...")
    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=frames, interval=50, blit=False
    )

    if save_path:
        print(f"Saving to {save_path}...")
        if str(save_path).endswith('.mp4'):
            anim.save(save_path, writer='ffmpeg', fps=30, dpi=150,
                      savefig_kwargs={'facecolor': fig.get_facecolor()})
        else:
            anim.save(save_path, writer='pillow', fps=24,
                      savefig_kwargs={'facecolor': fig.get_facecolor()})
        print(f"Saved: {save_path}")

    return fig, anim


def main():
    import argparse

    parser = argparse.ArgumentParser(description="McAllister CY visualization")
    parser.add_argument("--example", type=str, default="5-81-3213",
                       choices=["4-214-647", "5-81-3213", "5-113-4627-main", "7-51-13590"],
                       help="Which McAllister example to visualize")
    parser.add_argument("--frames", type=int, default=240, help="Animation frames (0 for still image)")
    parser.add_argument("--show", action="store_true", help="Show interactive")
    parser.add_argument("--png", action="store_true", help="Output PNG still image")

    args = parser.parse_args()

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    if args.png or args.frames == 0:
        # Single still image
        save_path = output_dir / f"mcallister_{args.example}.png"
        fig = create_mcallister_still(example_name=args.example, save_path=save_path)
    else:
        # Animation
        save_path = output_dir / f"mcallister_{args.example}.mp4"
        fig, anim = create_mcallister_animation(
            example_name=args.example,
            frames=args.frames,
            save_path=save_path
        )

    if args.show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()
