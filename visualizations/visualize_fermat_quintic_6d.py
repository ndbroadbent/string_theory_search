#!/usr/bin/env python3
"""
Full 6D visualization of the Fermat Quintic Calabi-Yau manifold.

Two 3D plots side by side with INDEPENDENT rotations:
- Left: dimensions (1, 2, 3) - rotates in its own 3D subspace
- Right: dimensions (4, 5, 6) - rotates in its own 3D subspace

This efficiently shows all 6D structure with no wasted frames.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from pathlib import Path


def rotation_matrix_3d(angle_xy, angle_xz, angle_yz):
    """Create a 3D rotation matrix from Euler-like angles."""
    c, s = np.cos(angle_xy), np.sin(angle_xy)
    Rxy = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    c, s = np.cos(angle_xz), np.sin(angle_xz)
    Rxz = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])

    c, s = np.cos(angle_yz), np.sin(angle_yz)
    Ryz = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    return Rxy @ Rxz @ Ryz


def rotation_matrix_6d(i, j, angle):
    """Create a 6D rotation matrix in the (i,j) plane."""
    R = np.eye(6)
    c, s = np.cos(angle), np.sin(angle)
    R[i, i] = c
    R[i, j] = -s
    R[j, i] = s
    R[j, j] = c
    return R


def calabi_yau_6d_points(n, resolution=25):
    """
    Generate 6D points on the Fermat quintic.

    Returns points in R^6 = (Re(z1), Im(z1), Re(z2), Im(z2), Re(z3), Im(z3))
    where z1^n + z2^n + z3^n = 1 (a 2D slice of the full CY).
    """
    points = []

    x = np.linspace(0.01, np.pi/2 - 0.01, resolution)
    y = np.linspace(-np.pi/2 + 0.01, np.pi/2 - 0.01, resolution)

    for k1 in range(n):
        for k2 in range(n):
            for k3 in range(n):
                # Use spherical-like parametrization for z1^n + z2^n + z3^n = 1
                for xi in x[::2]:  # Subsample for performance
                    for yi in y[::2]:
                        u = xi + 1j * yi

                        # Parametrize satisfying constraint
                        phase1 = np.exp(2j * np.pi * k1 / n)
                        phase2 = np.exp(2j * np.pi * k2 / n)
                        phase3 = np.exp(2j * np.pi * k3 / n)

                        # z1^n + z2^n = cos^2(u), z3^n = sin^2(u)
                        z1 = phase1 * (np.cos(u) * np.cos(yi)) ** (2/n)
                        z2 = phase2 * (np.cos(u) * np.sin(yi + 0.1)) ** (2/n)
                        z3 = phase3 * (np.sin(u)) ** (2/n)

                        # 6D point
                        point = [np.real(z1), np.imag(z1),
                                np.real(z2), np.imag(z2),
                                np.real(z3), np.imag(z3)]

                        if all(np.isfinite(point)):
                            points.append(point)

    return np.array(points)


def calabi_yau_surface_patch(n, k1, k2, resolution=30):
    """
    Generate a surface patch for visualization.

    Returns 6D coordinates: (Re(z1), Im(z1), Re(z2), Im(z2), projection_dim5, projection_dim6)
    """
    x = np.linspace(0.001, np.pi/2 - 0.001, resolution)
    y = np.linspace(-np.pi/2 + 0.001, np.pi/2 - 0.001, resolution)
    X, Y = np.meshgrid(x, y)
    u = X + 1j * Y

    phase1 = np.exp(2j * np.pi * k1 / n)
    phase2 = np.exp(2j * np.pi * k2 / n)

    z1 = phase1 * np.cos(u) ** (2/n)
    z2 = phase2 * np.sin(u) ** (2/n)

    # 6D: (Re(z1), Im(z1), Re(z2), Im(z2), mix1, mix2)
    # For the 5th and 6th dims, we use combinations to show structure
    dim1 = np.real(z1)
    dim2 = np.imag(z1)
    dim3 = np.real(z2)
    dim4 = np.imag(z2)
    dim5 = np.real(z1 * z2)  # Cross term for extra dimension
    dim6 = np.imag(z1 * z2)  # Cross term for extra dimension

    return np.stack([dim1, dim2, dim3, dim4, dim5, dim6], axis=-1)


def project_6d_to_dual_3d(points_6d, R_6d):
    """
    Apply 6D rotation, then project to two 3D views.

    Args:
        points_6d: (..., 6) array
        R_6d: 6x6 rotation matrix

    Returns:
        left_3d, right_3d: (..., 3) arrays - projections of the SAME rotated points
    """
    shape = points_6d.shape[:-1]
    flat = points_6d.reshape(-1, 6)

    # Apply single 6D rotation
    rotated = (R_6d @ flat.T).T.reshape(*shape, 6)

    # Project to both views (same points, different coords)
    left_3d = rotated[..., :3]   # dims 1,2,3
    right_3d = rotated[..., 3:]  # dims 4,5,6

    return left_3d, right_3d


def create_dual_animation(n=5, frames=180, save_path=None):
    """
    Create animation with independent rotations in each 3D subspace.
    """
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

    # Generate surface patches
    print(f"Generating n={n} Fermat quintic ({n**2} patches)...")
    patches_6d = []
    for k1 in range(n):
        for k2 in range(n):
            patch = calabi_yau_surface_patch(n, k1, k2, resolution=20)
            patches_6d.append(patch)

    # Hot colormaps on black - inferno/magma give a nice glow effect
    cmap_left = cm.inferno
    cmap_right = cm.magma

    # Find global bounds
    all_points = np.concatenate([p.reshape(-1, 6) for p in patches_6d])
    max_extent = np.abs(all_points).max() * 1.1

    surfaces_left = []
    surfaces_right = []

    def init():
        ax_left.set_xlim(-max_extent, max_extent)
        ax_left.set_ylim(-max_extent, max_extent)
        ax_left.set_zlim(-max_extent, max_extent)
        ax_right.set_xlim(-max_extent, max_extent)
        ax_right.set_ylim(-max_extent, max_extent)
        ax_right.set_zlim(-max_extent, max_extent)
        return []

    def animate(frame):
        nonlocal surfaces_left, surfaces_right

        # Clear previous surfaces
        for surf in surfaces_left + surfaces_right:
            try:
                surf.remove()
            except:
                pass
        surfaces_left = []
        surfaces_right = []

        # TRUE 6D rotation - one rotation affects BOTH views
        #
        # 15 rotation planes in SO(6):
        #   Within-view: (0-1), (0-2), (1-2), (3-4), (3-5), (4-5)
        #   Cross-view:  (0-3), (0-4), (0-5), (1-3), (1-4), (1-5), (2-3), (2-4), (2-5)
        #
        # Cross-view rotations are the interesting ones - both views warp together!

        # 4 phases showing different rotation planes
        frames_per_phase = frames // 4
        phase = frame // frames_per_phase
        t_in_phase = (frame % frames_per_phase) / frames_per_phase
        angle = 2 * np.pi * t_in_phase

        # Build 6D rotation matrix
        if phase == 0:
            # Cross-view rotation: (0-3) plane - mixes dim1 with dim4
            R_6d = rotation_matrix_6d(0, 3, angle)
        elif phase == 1:
            # Cross-view rotation: (1-4) plane - mixes dim2 with dim5
            R_6d = rotation_matrix_6d(1, 4, angle)
        elif phase == 2:
            # Cross-view rotation: (2-5) plane - mixes dim3 with dim6
            R_6d = rotation_matrix_6d(2, 5, angle)
        else:
            # Compound cross-view: (0-4) and (1-5) together
            R_6d = rotation_matrix_6d(0, 4, angle) @ rotation_matrix_6d(1, 5, angle * 0.7)

        # Project and plot each patch
        for patch_6d in patches_6d:
            left_3d, right_3d = project_6d_to_dual_3d(patch_6d, R_6d)

            # Color by "depth" in the hidden dimensions
            color_left = patch_6d[..., 3]  # dim 4 colors left plot
            color_right = patch_6d[..., 0]  # dim 1 colors right plot

            norm_left = Normalize(vmin=color_left.min(), vmax=color_left.max())
            norm_right = Normalize(vmin=color_right.min(), vmax=color_right.max())

            try:
                surf_l = ax_left.plot_surface(
                    left_3d[..., 0], left_3d[..., 1], left_3d[..., 2],
                    facecolors=cmap_left(norm_left(color_left)),
                    alpha=0.8, linewidth=0, antialiased=True, shade=False
                )
                surfaces_left.append(surf_l)

                surf_r = ax_right.plot_surface(
                    right_3d[..., 0], right_3d[..., 1], right_3d[..., 2],
                    facecolors=cmap_right(norm_right(color_right)),
                    alpha=0.8, linewidth=0, antialiased=True, shade=False
                )
                surfaces_right.append(surf_r)
            except Exception as e:
                pass

        # Minimal labels - let the visuals speak
        ax_left.set_title('Dimensions 1, 2, 3', color='#666666', fontsize=10)
        ax_right.set_title('Dimensions 4, 5, 6', color='#666666', fontsize=10)

        # Hide axis labels for cleaner look
        ax_left.set_xlabel('')
        ax_left.set_ylabel('')
        ax_left.set_zlabel('')
        ax_right.set_xlabel('')
        ax_right.set_ylabel('')
        ax_right.set_zlabel('')

        return surfaces_left + surfaces_right

    plt.suptitle(f'Calabi-Yau Manifold (n={n})',
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

    parser = argparse.ArgumentParser(description="6D Fermat Quintic visualization")
    parser.add_argument("-n", type=int, default=3, help="Degree (default: 3, use 2-5)")
    parser.add_argument("--frames", type=int, default=120, help="Animation frames")
    parser.add_argument("--show", action="store_true", help="Show interactive")

    args = parser.parse_args()

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    save_path = output_dir / f"fermat_quintic_6d_n{args.n}.mp4"

    fig, anim = create_dual_animation(n=args.n, frames=args.frames, save_path=save_path)

    if args.show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()
