#!/usr/bin/env python3
"""
Visualize 4D polytopes with the 4th dimension as color/heatmap.

Techniques:
1. Project 4D → 3D via rotation matrix
2. Color points by their w-coordinate (4th dimension)
3. Animate rotation in 4D space
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.animation as animation
from matplotlib.colors import Normalize
import matplotlib.cm as cm

# Add CYTools
ROOT_DIR = Path(__file__).parent.parent
CYTOOLS_2021 = ROOT_DIR / "vendor/cytools_mcallister_2107"
DATA_BASE = ROOT_DIR / "resources/small_cc_2107.09064_source/anc/paper_data"

sys.path.insert(0, str(CYTOOLS_2021))


def rotation_matrix_4d(angle, plane='xw'):
    """Create a 4D rotation matrix in the specified plane."""
    c, s = np.cos(angle), np.sin(angle)
    R = np.eye(4)

    planes = {
        'xy': (0, 1), 'xz': (0, 2), 'xw': (0, 3),
        'yz': (1, 2), 'yw': (1, 3), 'zw': (2, 3)
    }
    i, j = planes[plane]
    R[i, i] = c
    R[i, j] = -s
    R[j, i] = s
    R[j, j] = c
    return R


def project_4d_to_3d(points_4d, angle_xw=0, angle_yw=0, angle_zw=0):
    """
    Project 4D points to 3D using rotation then orthographic projection.

    The rotation angles control how we "view" the 4D object.
    """
    # Apply 4D rotations
    R = rotation_matrix_4d(angle_xw, 'xw') @ \
        rotation_matrix_4d(angle_yw, 'yw') @ \
        rotation_matrix_4d(angle_zw, 'zw')

    rotated = (R @ points_4d.T).T

    # Project: just take x, y, z (drop w)
    points_3d = rotated[:, :3]
    w_values = rotated[:, 3]  # Keep w for coloring

    return points_3d, w_values


def get_polytope_edges(points, threshold=1.5):
    """
    Find edges of a polytope by connecting nearby points.

    For reflexive polytopes, lattice points at distance ~1-sqrt(2) are connected.
    """
    from scipy.spatial import Delaunay, ConvexHull

    edges = set()
    n = len(points)

    # Use convex hull to find edges
    try:
        if points.shape[1] == 4:
            # For 4D, use distance-based approach
            for i in range(n):
                for j in range(i+1, n):
                    dist = np.linalg.norm(points[i] - points[j])
                    if dist <= threshold:
                        edges.add((i, j))
        else:
            # For 3D projected points, use convex hull
            hull = ConvexHull(points)
            for simplex in hull.simplices:
                for k in range(len(simplex)):
                    i, j = simplex[k], simplex[(k+1) % len(simplex)]
                    edges.add((min(i,j), max(i,j)))
    except:
        # Fallback: connect points within threshold distance
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(points[i] - points[j])
                if dist <= threshold:
                    edges.add((i, j))

    return list(edges)


def load_mcallister_polytope(example_name="5-81-3213"):
    """Load a polytope from McAllister's data."""
    data_dir = DATA_BASE / example_name
    points_file = data_dir / "points.dat"

    points = np.array([
        [int(x) for x in line.split(',')]
        for line in points_file.read_text().strip().split('\n')
    ])

    return points


def visualize_static(points_4d, title="4D Polytope", save_path=None):
    """Create a static visualization with w-dimension as color."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Project to 3D
    points_3d, w_values = project_4d_to_3d(points_4d, angle_xw=0.3, angle_yw=0.2)

    # Normalize w for coloring
    norm = Normalize(vmin=w_values.min(), vmax=w_values.max())
    cmap = cm.plasma  # viridis, plasma, magma, inferno are good choices
    colors = cmap(norm(w_values))

    # Plot points
    scatter = ax.scatter(
        points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
        c=w_values, cmap=cmap, s=100, edgecolors='white', linewidths=0.5,
        alpha=0.9
    )

    # Add edges
    edges = get_polytope_edges(points_4d, threshold=1.5)
    edge_lines = []
    edge_colors = []
    for i, j in edges:
        edge_lines.append([points_3d[i], points_3d[j]])
        # Color edge by average w of endpoints
        avg_w = (w_values[i] + w_values[j]) / 2
        edge_colors.append(cmap(norm(avg_w)))

    if edge_lines:
        lc = Line3DCollection(edge_lines, colors=edge_colors, linewidths=0.8, alpha=0.6)
        ax.add_collection3d(lc)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, label='4th dimension (w)')

    # Style
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'{title}\n{len(points_4d)} lattice points, color = w coordinate')

    # Equal aspect ratio
    max_range = np.max([
        points_3d[:, 0].max() - points_3d[:, 0].min(),
        points_3d[:, 1].max() - points_3d[:, 1].min(),
        points_3d[:, 2].max() - points_3d[:, 2].min()
    ]) / 2
    mid = points_3d.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#0f0f1a')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
        print(f"Saved: {save_path}")

    return fig, ax


def visualize_animated(points_4d, title="4D Polytope", save_path=None, frames=120):
    """Create an animated visualization rotating in 4D."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Style
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#0f0f1a')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    cmap = cm.plasma
    edges_4d = get_polytope_edges(points_4d, threshold=1.5)

    # Set fixed axis limits based on max possible extent
    max_coord = np.abs(points_4d).max() * 1.2
    ax.set_xlim(-max_coord, max_coord)
    ax.set_ylim(-max_coord, max_coord)
    ax.set_zlim(-max_coord, max_coord)

    scatter = None
    lines = None

    def init():
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        return []

    def animate(frame):
        nonlocal scatter, lines

        # Clear previous
        if scatter:
            scatter.remove()
        if lines:
            lines.remove()

        # 4D rotation angles
        angle_xw = 2 * np.pi * frame / frames
        angle_yw = np.pi * frame / frames
        angle_zw = 0.5 * np.pi * frame / frames

        # Also rotate in 3D for better viewing
        ax.view_init(elev=20, azim=frame * 3)

        # Project
        points_3d, w_values = project_4d_to_3d(points_4d, angle_xw, angle_yw, angle_zw)

        # Normalize w
        if w_values.max() != w_values.min():
            norm = Normalize(vmin=w_values.min(), vmax=w_values.max())
        else:
            norm = Normalize(vmin=-1, vmax=1)

        # Plot points
        scatter = ax.scatter(
            points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
            c=w_values, cmap=cmap, s=80, edgecolors='white', linewidths=0.3,
            alpha=0.9, norm=norm
        )

        # Plot edges
        edge_lines = []
        edge_colors = []
        for i, j in edges_4d:
            edge_lines.append([points_3d[i], points_3d[j]])
            avg_w = (w_values[i] + w_values[j]) / 2
            edge_colors.append(cmap(norm(avg_w)))

        if edge_lines:
            lines = Line3DCollection(edge_lines, colors=edge_colors, linewidths=0.6, alpha=0.5)
            ax.add_collection3d(lines)
        else:
            lines = None

        ax.set_title(f'{title}\nFrame {frame}/{frames} - 4D rotation', color='white')

        return [scatter] if scatter else []

    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=frames, interval=50, blit=False
    )

    if save_path:
        print(f"Saving animation to {save_path}...")
        anim.save(save_path, writer='pillow', fps=24,
                  savefig_kwargs={'facecolor': fig.get_facecolor()})
        print(f"Saved: {save_path}")

    return fig, anim


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Visualize 4D polytopes")
    parser.add_argument("--example", default="5-81-3213", help="McAllister example name")
    parser.add_argument("--animate", action="store_true", help="Create animation")
    parser.add_argument("--frames", type=int, default=120, help="Animation frames")
    parser.add_argument("--show", action="store_true", help="Show interactive plot")

    args = parser.parse_args()

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    print(f"Loading polytope: {args.example}")
    points = load_mcallister_polytope(args.example)
    print(f"  {len(points)} lattice points in {points.shape[1]}D")

    # Ensure 4D (pad with zeros if needed)
    if points.shape[1] < 4:
        points = np.hstack([points, np.zeros((len(points), 4 - points.shape[1]))])
    elif points.shape[1] > 4:
        points = points[:, :4]  # Take first 4 coords

    title = f"Polytope {args.example}"

    if args.animate:
        save_path = output_dir / f"{args.example}_4d_rotation.gif"
        fig, anim = visualize_animated(points, title, save_path, args.frames)
    else:
        save_path = output_dir / f"{args.example}_4d_static.png"
        fig, ax = visualize_static(points, title, save_path)

    if args.show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()
