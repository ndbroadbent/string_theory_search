#!/usr/bin/env python3
"""
Visualize the Fermat Quintic Calabi-Yau manifold.

Shows two 3D plots side by side to cover all 6 dimensions:
- Left: (z₁, z₂) slice
- Right: (z₃, z₄) slice with z₅ as color

The Fermat quintic: z₁ⁿ + z₂ⁿ + z₃ⁿ + z₄ⁿ + z₅ⁿ = 0 in CP⁴
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from pathlib import Path


def calabi_yau_patch(n, k1, k2, resolution=50):
    """
    Generate a single patch of the CY surface z₁ⁿ + z₂ⁿ = 1.

    Args:
        n: Degree (integer >= 2)
        k1, k2: Patch indices in [0, n-1]
        resolution: Grid resolution

    Returns:
        X, Y, Z, W: 4D coordinates (W is the 4th real dimension)
    """
    x = np.linspace(0.001, np.pi/2 - 0.001, resolution)
    y = np.linspace(-np.pi/2 + 0.001, np.pi/2 - 0.001, resolution)
    X_grid, Y_grid = np.meshgrid(x, y)

    # Complex parameter
    u = X_grid + 1j * Y_grid

    # z₁ and z₂ satisfying z₁ⁿ + z₂ⁿ = 1
    phase1 = np.exp(2j * np.pi * k1 / n)
    phase2 = np.exp(2j * np.pi * k2 / n)

    z1 = phase1 * np.cos(u) ** (2/n)
    z2 = phase2 * np.sin(u) ** (2/n)

    # 4 real dimensions: Re(z1), Im(z1), Re(z2), Im(z2)
    return np.real(z1), np.imag(z1), np.real(z2), np.imag(z2)


def project_to_3d(re1, im1, re2, im2, angle=0):
    """
    Project 4D (Re(z1), Im(z1), Re(z2), Im(z2)) to 3D.

    Uses angle to mix the imaginary parts for the Z coordinate.
    """
    X = re1
    Y = re2
    Z = im1 * np.cos(angle) + im2 * np.sin(angle)
    W = im1 * np.sin(angle) - im2 * np.cos(angle)  # "hidden" 4th dimension for color
    return X, Y, Z, W


def plot_single_cy(ax, n, angle=0, cmap='plasma', alpha=0.7):
    """Plot all n² patches of the CY surface on a single axis."""

    # Collect all patches for consistent color normalization
    all_W = []
    patches = []

    for k1 in range(n):
        for k2 in range(n):
            re1, im1, re2, im2 = calabi_yau_patch(n, k1, k2, resolution=30)
            X, Y, Z, W = project_to_3d(re1, im1, re2, im2, angle)
            patches.append((X, Y, Z, W))
            all_W.append(W.flatten())

    all_W = np.concatenate(all_W)
    norm = Normalize(vmin=all_W.min(), vmax=all_W.max())

    # Plot each patch
    for X, Y, Z, W in patches:
        colors = cm.get_cmap(cmap)(norm(W))
        ax.plot_surface(X, Y, Z, facecolors=colors, alpha=alpha,
                       linewidth=0, antialiased=True, shade=False)

    return norm


def visualize_dual(n=5, save_path=None, show=True):
    """
    Create dual 3D visualization showing all 6 dimensions.

    Left: projection angle 0 (emphasizes Im(z1))
    Right: projection angle π/2 (emphasizes Im(z2))
    """
    fig = plt.figure(figsize=(16, 7))
    fig.patch.set_facecolor('#0a0a12')

    # Left plot - angle 0
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_facecolor('#0a0a12')
    norm1 = plot_single_cy(ax1, n, angle=0, cmap='plasma')
    ax1.set_title(f'Fermat Quintic (n={n})\nProjection angle = 0°',
                  color='white', fontsize=12)
    ax1.set_xlabel('Re(z₁)', color='white')
    ax1.set_ylabel('Re(z₂)', color='white')
    ax1.set_zlabel('Im(z₁)·cos + Im(z₂)·sin', color='white')

    # Right plot - angle π/2
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_facecolor('#0a0a12')
    norm2 = plot_single_cy(ax2, n, angle=np.pi/2, cmap='viridis')
    ax2.set_title(f'Fermat Quintic (n={n})\nProjection angle = 90°',
                  color='white', fontsize=12)
    ax2.set_xlabel('Re(z₁)', color='white')
    ax2.set_ylabel('Re(z₂)', color='white')
    ax2.set_zlabel('Im(z₁)·cos + Im(z₂)·sin', color='white')

    # Style both axes
    for ax in [ax1, ax2]:
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.zaxis.label.set_color('white')

    # Add colorbar
    sm = cm.ScalarMappable(norm=norm1, cmap='plasma')
    cbar = fig.colorbar(sm, ax=[ax1, ax2], shrink=0.5, pad=0.1)
    cbar.set_label('4th dimension (hidden)', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    plt.suptitle('Calabi-Yau Manifold: Two 3D Projections = Full 6D View',
                 color='white', fontsize=14, y=0.98)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(),
                    bbox_inches='tight')
        print(f"Saved: {save_path}")

    if show:
        plt.show()

    return fig


def visualize_single(n=5, angle=0, save_path=None, show=True):
    """Create a single beautiful CY visualization."""
    fig = plt.figure(figsize=(10, 8))
    fig.patch.set_facecolor('#0a0a12')

    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#0a0a12')

    norm = plot_single_cy(ax, n, angle=angle, cmap='plasma', alpha=0.8)

    ax.set_title(f'Fermat Quintic Calabi-Yau (n={n})\n{n}² = {n**2} patches',
                 color='white', fontsize=14)
    ax.set_xlabel('Re(z₁)', color='white')
    ax.set_ylabel('Re(z₂)', color='white')
    ax.set_zlabel('Z (mixed imaginary)', color='white')

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.tick_params(colors='white')

    # Colorbar
    sm = cm.ScalarMappable(norm=norm, cmap='plasma')
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label('4th dimension', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(),
                    bbox_inches='tight')
        print(f"Saved: {save_path}")

    if show:
        plt.show()

    return fig


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Visualize Fermat Quintic CY")
    parser.add_argument("-n", type=int, default=5, help="Degree (default: 5)")
    parser.add_argument("--dual", action="store_true", help="Show dual 6D view")
    parser.add_argument("--angle", type=float, default=0.3, help="Projection angle")
    parser.add_argument("--no-show", action="store_true", help="Don't show, just save")

    args = parser.parse_args()

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    if args.dual:
        save_path = output_dir / f"fermat_quintic_n{args.n}_dual.png"
        visualize_dual(n=args.n, save_path=save_path, show=not args.no_show)
    else:
        save_path = output_dir / f"fermat_quintic_n{args.n}.png"
        visualize_single(n=args.n, angle=args.angle, save_path=save_path,
                        show=not args.no_show)


if __name__ == "__main__":
    main()
