#!/usr/bin/env python3
"""
Visualize a cyrus-ga candidate vacuum end-to-end:

1. The 4D reflexive polytope (rotating projection), with points colored by
   their PHYSICS ROLE from the cyrus orientifold layer: O7-plane divisors
   (involution parity class), other prime toric divisors, and the origin.
2. The quintessence panel: the axion potential at the candidate's vacuum
   energy, the integrated w(z) from the Friedmann + Klein-Gordon equations
   (same conventions as cyrus-ga's fitness), the CPL fit, and DESI's
   measured (w0, wa) band.

Usage:
  uv run python visualizations/visualize_candidate.py \
      --verify-dir ~/code/cyrus/runs/verify-h21_4_367 \
      --name h21_4_367 --log10-v0 -121.540 --sigma 0,0,1,0 \
      --w0 1.1e-61 --g-s 0.0214 --q-flux 20.5 --k -1,5,0,0 --m 11,-6,9,-8
"""

import argparse
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial import ConvexHull

OUTPUT = Path(__file__).parent / "output"

H0_PLANCK = 1.18e-61
OMEGA_M = 0.3
DECAY_CONSTANT = 0.5  # f in Mpl, cyrus-ga default
DISPLACEMENT = 2.0  # phi_i = DISPLACEMENT * f
DESI_W0, DESI_W0_ERR = -0.45, 0.21
DESI_WA, DESI_WA_ERR = -1.8, 0.6


def rotation_matrix_4d(angle, plane):
    c, s = np.cos(angle), np.sin(angle)
    R = np.eye(4)
    planes = {"xy": (0, 1), "xz": (0, 2), "xw": (0, 3),
              "yz": (1, 2), "yw": (1, 3), "zw": (2, 3)}
    i, j = planes[plane]
    R[i, i] = c
    R[i, j] = -s
    R[j, i] = s
    R[j, j] = c
    return R


def polytope_edges(points):
    """Edges of the 4D convex hull (pairs of vertex indices)."""
    hull = ConvexHull(points)
    edges = set()
    for simplex in hull.simplices:
        for a in simplex:
            for b in simplex:
                if a < b:
                    edges.add((a, b))
    return hull, sorted(edges)


def render_polytope(points, sigma, name, animate=True):
    """Rotating 4D projection; color = physics role from the orientifold."""
    pts = np.array(points, dtype=float)
    parity = np.array(points) % 2
    is_origin = np.all(np.array(points) == 0, axis=1)
    is_o7 = np.all(parity == np.array(sigma), axis=1) & ~is_origin

    hull, edges = polytope_edges(pts)

    def frame_points(t):
        R = (rotation_matrix_4d(t, "xw")
             @ rotation_matrix_4d(0.7 * t, "yw")
             @ rotation_matrix_4d(0.4 * t, "zw"))
        return (R @ pts.T).T[:, :3]

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    def draw(t):
        ax.clear()
        ax.set_facecolor("black")
        ax.set_axis_off()
        p3 = frame_points(t)
        for a, b in edges:
            ax.plot(*zip(p3[a], p3[b]), color="#2a4a6a", lw=0.6, alpha=0.5)
        normal = ~is_o7 & ~is_origin
        ax.scatter(*p3[normal].T, c="#4fc3f7", s=28, alpha=0.9,
                   label="prime toric divisors (ED3 instantons)")
        ax.scatter(*p3[is_o7].T, c="#ff5252", s=90, marker="o",
                   edgecolors="white", linewidths=0.5,
                   label=f"O7-plane divisors (parity = {list(sigma)})")
        ax.scatter(*p3[is_origin].T, c="#ffd740", s=120, marker="*",
                   label="origin")
        ax.legend(loc="upper left", facecolor="black", labelcolor="white",
                  fontsize=9, framealpha=0.4)
        ax.set_title(
            f"{name}: reflexive polytope (4D rotation), "
            f"{is_o7.sum()} O7-planes",
            color="white", fontsize=12)
        lim = np.abs(pts).max()
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)

    draw(0.6)
    png = OUTPUT / f"candidate_{name}_polytope.png"
    fig.savefig(png, dpi=150, facecolor="black", bbox_inches="tight")
    print(f"wrote {png}")

    if animate:
        anim = animation.FuncAnimation(
            fig, lambda i: draw(2 * np.pi * i / 120), frames=120, interval=60)
        try:
            mp4 = OUTPUT / f"candidate_{name}_polytope.mp4"
            anim.save(mp4, writer="ffmpeg", fps=24,
                      savefig_kwargs={"facecolor": "black"})
            print(f"wrote {mp4}")
        except Exception as e:  # noqa: BLE001 - ffmpeg may be broken
            print(f"ffmpeg failed ({e}); falling back to GIF")
            gif = OUTPUT / f"candidate_{name}_polytope.gif"
            anim.save(gif, writer=animation.PillowWriter(fps=15),
                      savefig_kwargs={"facecolor": "black"}, dpi=80)
            print(f"wrote {gif}")
    plt.close(fig)


def integrate_quintessence(log10_v0):
    """Friedmann + Klein-Gordon, cyrus-ga conventions (H0 = 1 units)."""
    f = DECAY_CONSTANT
    height = 10.0 ** (log10_v0) / H0_PLANCK**2  # |V0| in H0^2 Mpl^2 units

    def V(phi):
        return height * (1.0 - np.cos(phi / f))

    def dV(phi):
        return height / f * np.sin(phi / f)

    # Evolve in e-folds N = ln a, from z=100 to z=0.
    def rhs(N, y):
        phi, dphi = y  # dphi = dphi/dN
        a = np.exp(N)
        rho_m = 3.0 * OMEGA_M * a**-3
        # H^2 (1 - dphi^2/6) = (rho_m + V)/3
        h2 = (rho_m + V(phi)) / (3.0 - 0.5 * dphi**2)
        h2 = max(h2, 1e-12)
        dlnH = -0.5 * (rho_m / h2 + dphi**2) / 1.0  # d ln H/dN approx
        ddphi = -(3.0 + dlnH) * dphi - dV(phi) / h2
        return [dphi, ddphi]

    N0, N1 = np.log(1 / 101.0), 0.0
    sol = solve_ivp(rhs, (N0, N1), [DISPLACEMENT * f, 0.0],
                    dense_output=True, rtol=1e-8, atol=1e-10)
    N = np.linspace(N0, N1, 600)
    phi, dphi = sol.sol(N)
    a = np.exp(N)
    z = 1 / a - 1
    rho_m = 3.0 * OMEGA_M * a**-3
    h2 = (rho_m + V(phi)) / (3.0 - 0.5 * dphi**2)
    kin = 0.5 * dphi**2 * h2
    w = (kin - V(phi)) / (kin + V(phi))
    return z, w, phi, V, f, height


def fit_cpl(z, w, zmax=3.0):
    mask = z <= zmax
    a = 1 / (1 + z[mask])
    A = np.vstack([np.ones_like(a), 1 - a]).T
    coef, *_ = np.linalg.lstsq(A, w[mask], rcond=None)
    return coef  # w0, wa


def render_physics(name, log10_v0, meta):
    z, w, phi, V, f, height = integrate_quintessence(log10_v0)
    w0_fit, wa_fit = fit_cpl(z, w)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    # Panel 1: the axion potential
    phis = np.linspace(-np.pi * f, 3 * np.pi * f, 400)
    ax1.plot(phis, V(phis) / height, color="#7e57c2", lw=2)
    ax1.axvline(DISPLACEMENT * f, color="#ff5252", ls="--", lw=1.5,
                label=r"$\phi_i = 2f$ (initial)")
    ax1.scatter([phi[-1]], [V(phi[-1]) / height], color="#ffd740", zorder=5,
                s=80, label=r"$\phi(z{=}0)$ (today)")
    ax1.set_xlabel(r"$\phi$ [$M_{pl}$]")
    ax1.set_ylabel(r"$V(\phi)\,/\,|V_0|$")
    ax1.set_title(
        rf"Thawing axion potential, $|V_0| = 10^{{{log10_v0:.2f}}}\,M_{{pl}}^4$")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Panel 2: w(z) vs DESI
    ax2.plot(z, w, color="#26a69a", lw=2, label=r"integrated $w(z)$")
    zz = np.linspace(0, 3, 100)
    aa = 1 / (1 + zz)
    ax2.plot(zz, w0_fit + wa_fit * (1 - aa), "--", color="#ef6c00", lw=1.5,
             label=rf"CPL fit: $w_0={w0_fit:.2f}$, $w_a={wa_fit:.2f}$")
    ax2.fill_between(
        zz,
        (DESI_W0 - DESI_W0_ERR) + (DESI_WA - DESI_WA_ERR) * (1 - aa),
        (DESI_W0 + DESI_W0_ERR) + (DESI_WA + DESI_WA_ERR) * (1 - aa),
        color="#90caf9", alpha=0.3, label=r"DESI $1\sigma$ band")
    ax2.axhline(-1, color="gray", ls=":", lw=1, label=r"$\Lambda$ ($w=-1$)")
    ax2.set_xlim(0, 3)
    ax2.set_ylim(-1.15, 0.2)
    ax2.set_xlabel("redshift $z$")
    ax2.set_ylabel("$w(z)$")
    ax2.set_title("Dark-energy equation of state vs DESI")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    fig.suptitle(
        f"{name}:  K={meta['k']}  M={meta['m']}  "
        rf"$q={meta['q']}$, $g_s={meta['g_s']}$, $|W_0|={meta['w0']}$",
        fontsize=11)
    fig.tight_layout()
    png = OUTPUT / f"candidate_{name}_quintessence.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"wrote {png}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--log10-v0", type=float, required=True)
    ap.add_argument("--sigma", required=True, help="involution parity, e.g. 0,0,1,0")
    ap.add_argument("--w0", default="?")
    ap.add_argument("--g-s", default="?")
    ap.add_argument("--q-flux", default="?")
    ap.add_argument("--k", default="?")
    ap.add_argument("--m", default="?")
    ap.add_argument("--no-animation", action="store_true")
    args = ap.parse_args()

    OUTPUT.mkdir(exist_ok=True)
    points = []
    for line in open(Path(args.verify_dir).expanduser() / "points.dat"):
        if line.strip():
            points.append([int(x) for x in line.split(",")])
    sigma = [int(x) for x in args.sigma.split(",")]

    render_polytope(points, sigma, args.name, animate=not args.no_animation)
    render_physics(args.name, args.log10_v0, {
        "k": args.k, "m": args.m, "q": args.q_flux,
        "g_s": args.g_s, "w0": args.w0,
    })


if __name__ == "__main__":
    main()
