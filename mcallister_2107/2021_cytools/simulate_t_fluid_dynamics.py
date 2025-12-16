#!/usr/bin/env python3
"""
Fluid/Particle Dynamics Approach to KKLT Solver.

A wild experiment: treating the h11-dimensional Kähler cone as a "fluid basin"
where particles flow and settle into minima corresponding to solutions of τ(t) = τ_target.

CONCEPTS BORROWED FROM FLUID DYNAMICS:
- Particles with positions (t values) and velocities
- Energy landscape: E(t) = |τ(t) - τ_target|² - λ·V(t)
- Gradient flow: particles flow downhill in energy
- Thermal diffusion: random kicks for exploration (Langevin dynamics)
- "Settling": particles accumulate at local minima

VISUALIZATION:
- Project h11-dim space to 2D using PCA
- Show particle positions and energy landscape
- Animate the evolution
- Use astronomix-style fluid rendering for density

This is EXPERIMENTAL and may not work, but should produce cool visualizations!
"""

import sys
from pathlib import Path
from typing import NamedTuple
import time

import jax
import jax.numpy as jnp
from jax import jit, vmap, grad
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.animation as animation
from sklearn.decomposition import PCA

# Enable 64-bit for numerical stability
jax.config.update("jax_enable_x64", True)

# Paths
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent.parent
DATA_BASE = ROOT_DIR / "resources/small_cc_2107.09064_source/anc/paper_data"
CYTOOLS_2021 = ROOT_DIR / "vendor/cytools_mcallister_2107"
OUTPUT_DIR = SCRIPT_DIR / "fluid_dynamics_output"

sys.path.insert(0, str(CYTOOLS_2021))


class ParticleState(NamedTuple):
    """State of all particles in the simulation."""
    positions: jnp.ndarray  # (n_particles, h11) - t values
    velocities: jnp.ndarray  # (n_particles, h11) - momentum
    energies: jnp.ndarray  # (n_particles,) - E(t)
    volumes: jnp.ndarray  # (n_particles,) - V(t)
    tau_errors: jnp.ndarray  # (n_particles,) - |τ - τ_target|²
    time: float


class KahlerFluidSimulator:
    """Particle-based exploration of the Kähler cone."""

    def __init__(self, kappa_tensor: np.ndarray, tau_target: np.ndarray,
                 n_particles: int = 1000, temperature: float = 1.0,
                 damping: float = 0.9, dt: float = 0.01):
        """
        Initialize the simulator.

        Args:
            kappa_tensor: (h11, h11, h11) intersection numbers
            tau_target: (h11,) target τ values
            n_particles: Number of particles to simulate
            temperature: Langevin noise strength
            damping: Velocity damping factor (0-1)
            dt: Timestep
        """
        self.h11 = kappa_tensor.shape[0]
        self.kappa = jnp.array(kappa_tensor)
        self.tau_target = jnp.array(tau_target)
        self.n_particles = n_particles
        self.temperature = temperature
        self.damping = damping
        self.dt = dt

        # Precompute for efficiency
        self._setup_jit_functions()

    def _setup_jit_functions(self):
        """JIT compile the core functions."""
        kappa = self.kappa
        tau_target = self.tau_target
        h11 = self.h11

        @jit
        def compute_tau(t):
            """τ_i = (1/2) κ_ijk t^j t^k"""
            return 0.5 * jnp.einsum('ijk,j,k->i', kappa, t, t)

        @jit
        def compute_V(t):
            """V = (1/6) κ_ijk t^i t^j t^k"""
            return jnp.einsum('ijk,i,j,k->', kappa, t, t, t) / 6.0

        @jit
        def compute_tau_error(t):
            """||τ(t) - τ_target||²"""
            tau = compute_tau(t)
            return jnp.sum((tau - tau_target) ** 2)

        @jit
        def compute_energy(t, lambda_V=0.01):
            """E(t) = ||τ(t) - τ_target||² - λ·V(t)

            We want to minimize τ error while preferring larger V.
            """
            tau_err = compute_tau_error(t)
            V = compute_V(t)
            # Penalize negative V heavily
            V_penalty = jnp.where(V < 0, 1000 * V**2, 0.0)
            return tau_err - lambda_V * V + V_penalty

        # Gradient of energy w.r.t. t
        grad_energy = jit(grad(compute_energy))

        @jit
        def step_particle(t, v, key, dt, temperature, damping):
            """Single Langevin dynamics step for one particle."""
            # Gradient force
            force = -grad_energy(t)

            # Clip force to prevent explosions
            force_norm = jnp.sqrt(jnp.sum(force**2) + 1e-10)
            force = jnp.where(force_norm > 100, force * 100 / force_norm, force)

            # Langevin noise
            noise = jax.random.normal(key, shape=(h11,)) * jnp.sqrt(2 * temperature * dt)

            # Velocity update with damping
            v_new = damping * v + dt * force + noise

            # Position update
            t_new = t + dt * v_new

            # Keep t positive (Kähler cone constraint)
            t_new = jnp.maximum(t_new, 0.01)

            return t_new, v_new

        # Vectorized over particles
        @jit
        def step_all_particles(positions, velocities, key, dt, temperature, damping):
            keys = jax.random.split(key, positions.shape[0])
            t_new, v_new = vmap(lambda t, v, k: step_particle(t, v, k, dt, temperature, damping))(
                positions, velocities, keys
            )
            return t_new, v_new

        @jit
        def compute_all_metrics(positions):
            """Compute E, V, tau_error for all particles."""
            energies = vmap(compute_energy)(positions)
            volumes = vmap(compute_V)(positions)
            tau_errors = vmap(compute_tau_error)(positions)
            return energies, volumes, tau_errors

        self.compute_tau = compute_tau
        self.compute_V = compute_V
        self.compute_energy = compute_energy
        self.grad_energy = grad_energy
        self.step_all_particles = step_all_particles
        self.compute_all_metrics = compute_all_metrics

    def initialize_particles(self, key, scale_range=(0.5, 20.0)):
        """Initialize particles with random positive t values."""
        key1, key2 = jax.random.split(key)

        # Random scales for each particle
        log_scales = jax.random.uniform(key1, (self.n_particles,),
                                        minval=jnp.log(scale_range[0]),
                                        maxval=jnp.log(scale_range[1]))
        scales = jnp.exp(log_scales)

        # Random directions (normalized, then scaled)
        directions = jnp.abs(jax.random.normal(key2, (self.n_particles, self.h11)))
        directions = directions / (jnp.linalg.norm(directions, axis=1, keepdims=True) + 1e-10)

        positions = directions * scales[:, None]

        # Scale each particle to roughly match target τ magnitude
        tau_target_mean = jnp.mean(self.tau_target)
        for i in range(self.n_particles):
            tau_mean = jnp.mean(self.compute_tau(positions[i]))
            if tau_mean > 0:
                adjust = jnp.sqrt(tau_target_mean / tau_mean)
                if 0.01 < adjust < 100:
                    positions = positions.at[i].set(positions[i] * adjust)

        velocities = jnp.zeros_like(positions)
        energies, volumes, tau_errors = self.compute_all_metrics(positions)

        return ParticleState(
            positions=positions,
            velocities=velocities,
            energies=energies,
            volumes=volumes,
            tau_errors=tau_errors,
            time=0.0
        )

    def step(self, state: ParticleState, key) -> ParticleState:
        """Advance simulation by one timestep."""
        positions, velocities = self.step_all_particles(
            state.positions, state.velocities, key,
            self.dt, self.temperature, self.damping
        )
        energies, volumes, tau_errors = self.compute_all_metrics(positions)

        return ParticleState(
            positions=positions,
            velocities=velocities,
            energies=energies,
            volumes=volumes,
            tau_errors=tau_errors,
            time=state.time + self.dt
        )

    def run(self, n_steps: int, key, callback=None) -> list[ParticleState]:
        """Run simulation for n_steps."""
        state = self.initialize_particles(key)
        history = [state]

        for i in range(n_steps):
            key, subkey = jax.random.split(key)
            state = self.step(state, subkey)
            history.append(state)

            if callback is not None:
                callback(i, state)

        return history


class AstronomixFluidRenderer:
    """Use astronomix to create fluid-like density evolution in 2D projection.

    This treats the particle density field as a diffusing fluid, using
    astronomix's finite volume solver to smooth and evolve it.
    """

    def __init__(self, grid_size: int = 100):
        self.grid_size = grid_size

    def particles_to_density(self, positions_2d: np.ndarray,
                             bounds: tuple, weights: np.ndarray = None) -> np.ndarray:
        """Convert particle positions to density field."""
        x_min, x_max, y_min, y_max = bounds

        density = np.zeros((self.grid_size, self.grid_size))
        dx = (x_max - x_min) / self.grid_size
        dy = (y_max - y_min) / self.grid_size

        if weights is None:
            weights = np.ones(len(positions_2d))

        for i, (x, y) in enumerate(positions_2d):
            ix = int((x - x_min) / dx)
            iy = int((y - y_min) / dy)
            if 0 <= ix < self.grid_size and 0 <= iy < self.grid_size:
                density[iy, ix] += weights[i]

        return density

    def evolve_density_astronomix(self, density: np.ndarray,
                                  n_steps: int = 10) -> np.ndarray:
        """Use astronomix to evolve/diffuse the density field.

        Treats density as a passive scalar in a simple diffusion problem.
        """
        try:
            from astronomix import (
                SimulationConfig, SimulationParams,
                get_registered_variables, get_helper_data,
                construct_primitive_state, finalize_config, time_integration,
                PERIODIC_BOUNDARY, HLL
            )
            import jax.numpy as jnp

            # Create a 2D astronomix simulation
            # We'll set up uniform flow with the density as initial condition
            config = SimulationConfig(
                box_size=1.0,
                num_cells=self.grid_size,
                dimensionality=2,
                riemann_solver=HLL,
                boundary_handling=PERIODIC_BOUNDARY,
                progress_bar=False,
                num_timesteps=n_steps,
            )

            params = SimulationParams(t_end=0.01)  # Short evolution = smoothing
            variables = get_registered_variables(config)

            # Normalize density to be like a fluid density
            density_norm = density / (density.max() + 1e-10) + 0.1
            density_jax = jnp.array(density_norm)

            # Zero velocity, uniform pressure
            velocity_x = jnp.zeros_like(density_jax)
            velocity_y = jnp.zeros_like(density_jax)
            pressure = jnp.ones_like(density_jax)

            initial_state = construct_primitive_state(
                config, variables, density_jax, velocity_x, velocity_y, pressure
            )

            config = finalize_config(config, initial_state.shape)
            final_state = time_integration(initial_state, config, params, variables)

            # Extract evolved density
            evolved = np.array(final_state[variables.density_index])
            return evolved

        except Exception as e:
            # Fall back to simple Gaussian smoothing
            from scipy.ndimage import gaussian_filter
            return gaussian_filter(density, sigma=2.0)


class FluidVisualizer:
    """Visualize particle dynamics as fluid-like density fields."""

    def __init__(self, simulator: KahlerFluidSimulator):
        self.sim = simulator
        self.pca = None
        self.fig = None
        self.axes = None
        self.astro_renderer = AstronomixFluidRenderer(grid_size=80)

    def fit_pca(self, history: list[ParticleState]):
        """Fit PCA on all particle positions from history."""
        all_positions = np.vstack([np.array(s.positions) for s in history])
        self.pca = PCA(n_components=2)
        self.pca.fit(all_positions)
        print(f"PCA explained variance: {self.pca.explained_variance_ratio_}")

    def project_to_2d(self, positions: np.ndarray) -> np.ndarray:
        """Project h11-dim positions to 2D."""
        return self.pca.transform(np.array(positions))

    def create_density_field(self, positions_2d: np.ndarray, volumes: np.ndarray,
                             grid_size: int = 100, sigma: float = 1.0) -> tuple:
        """Create a density field from particle positions using Gaussian smoothing."""
        from scipy.ndimage import gaussian_filter

        x = positions_2d[:, 0]
        y = positions_2d[:, 1]

        # Grid bounds
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1

        # Create grid
        x_grid = np.linspace(x_min, x_max, grid_size)
        y_grid = np.linspace(y_min, y_max, grid_size)
        dx = x_grid[1] - x_grid[0]
        dy = y_grid[1] - y_grid[0]

        # Bin particles
        density = np.zeros((grid_size, grid_size))
        V_positive_density = np.zeros((grid_size, grid_size))

        for i in range(len(x)):
            ix = int((x[i] - x_min) / dx)
            iy = int((y[i] - y_min) / dy)
            if 0 <= ix < grid_size and 0 <= iy < grid_size:
                density[iy, ix] += 1
                if volumes[i] > 0:
                    V_positive_density[iy, ix] += 1

        # Smooth
        density = gaussian_filter(density, sigma=sigma)
        V_positive_density = gaussian_filter(V_positive_density, sigma=sigma)

        return x_grid, y_grid, density, V_positive_density

    def plot_snapshot(self, state: ParticleState, ax=None, show_V_positive=True):
        """Plot a single snapshot."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))

        positions_2d = self.project_to_2d(state.positions)
        volumes = np.array(state.volumes)
        tau_errors = np.array(state.tau_errors)

        # Color by V (green = positive, red = negative)
        colors = np.where(volumes > 0, 'green', 'red')
        alphas = np.clip(1.0 / (1.0 + tau_errors / 10), 0.1, 1.0)

        ax.scatter(positions_2d[:, 0], positions_2d[:, 1],
                   c=colors, alpha=alphas, s=20)

        # Highlight best solutions (V > 0 and low tau_error)
        good_mask = (volumes > 0) & (tau_errors < 1.0)
        if good_mask.any():
            ax.scatter(positions_2d[good_mask, 0], positions_2d[good_mask, 1],
                       c='gold', s=100, marker='*', edgecolors='black', linewidths=1,
                       label=f'{good_mask.sum()} good solutions')
            ax.legend()

        ax.set_xlabel('PCA Component 1')
        ax.set_ylabel('PCA Component 2')
        ax.set_title(f't = {state.time:.3f}, V>0: {(volumes > 0).sum()}/{len(volumes)}')

        return ax

    def plot_density_evolution(self, history: list[ParticleState],
                               frames=[0, -1], output_path=None):
        """Plot density field at different times."""
        fig, axes = plt.subplots(1, len(frames), figsize=(5*len(frames), 5))
        if len(frames) == 1:
            axes = [axes]

        for ax, frame_idx in zip(axes, frames):
            state = history[frame_idx]
            positions_2d = self.project_to_2d(state.positions)
            volumes = np.array(state.volumes)

            x_grid, y_grid, density, V_pos = self.create_density_field(
                positions_2d, volumes, grid_size=80, sigma=2.0
            )

            # Plot total density as blue, V>0 as green overlay
            ax.imshow(density.T, origin='lower', cmap='Blues', alpha=0.7,
                      extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]])
            ax.imshow(V_pos.T, origin='lower', cmap='Greens', alpha=0.5,
                      extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]])

            ax.set_xlabel('PCA Component 1')
            ax.set_ylabel('PCA Component 2')
            ax.set_title(f't = {state.time:.2f}')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150)
            print(f"Saved: {output_path}")

        return fig

    def create_astronomix_density_plot(self, state: ParticleState,
                                        bounds: tuple, ax=None):
        """Create density plot using astronomix fluid evolution."""
        positions_2d = self.project_to_2d(state.positions)
        volumes = np.array(state.volumes)

        # Create raw density
        raw_density = self.astro_renderer.particles_to_density(
            positions_2d, bounds, weights=np.ones(len(positions_2d))
        )

        # Evolve with astronomix (or fallback to Gaussian)
        evolved_density = self.astro_renderer.evolve_density_astronomix(raw_density, n_steps=5)

        # Create V > 0 density
        V_positive_mask = volumes > 0
        V_pos_density = self.astro_renderer.particles_to_density(
            positions_2d[V_positive_mask], bounds
        )
        V_pos_evolved = self.astro_renderer.evolve_density_astronomix(V_pos_density, n_steps=5)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))

        x_min, x_max, y_min, y_max = bounds
        ax.imshow(evolved_density, origin='lower', cmap='viridis',
                  extent=[x_min, x_max, y_min, y_max], alpha=0.8)
        ax.contour(np.linspace(x_min, x_max, evolved_density.shape[1]),
                   np.linspace(y_min, y_max, evolved_density.shape[0]),
                   V_pos_evolved, levels=3, colors='lime', linewidths=1.5)

        # Overlay actual particles
        colors = np.where(volumes > 0, 'lime', 'red')
        ax.scatter(positions_2d[:, 0], positions_2d[:, 1],
                   c=colors, alpha=0.3, s=5, edgecolors='none')

        return ax

    def create_animation(self, history: list[ParticleState],
                         output_path: Path, fps: int = 30,
                         skip_frames: int = 1):
        """Create MP4 animation of particle evolution."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Get bounds from all data
        all_2d = np.vstack([self.project_to_2d(s.positions) for s in history])
        x_min, x_max = all_2d[:, 0].min() - 1, all_2d[:, 0].max() + 1
        y_min, y_max = all_2d[:, 1].min() - 1, all_2d[:, 1].max() + 1
        bounds = (x_min, x_max, y_min, y_max)

        def update(frame_num):
            state = history[frame_num * skip_frames]
            positions_2d = self.project_to_2d(state.positions)
            volumes = np.array(state.volumes)
            tau_errors = np.array(state.tau_errors)

            for ax in axes:
                ax.clear()

            # Left: particle scatter
            colors = np.where(volumes > 0, 'green', 'red')
            axes[0].scatter(positions_2d[:, 0], positions_2d[:, 1],
                            c=colors, alpha=0.5, s=10)
            axes[0].set_xlim(x_min, x_max)
            axes[0].set_ylim(y_min, y_max)
            axes[0].set_xlabel('PCA 1')
            axes[0].set_ylabel('PCA 2')
            axes[0].set_title('Particles (V>0: green, V<0: red)')

            # Center: Gaussian smoothed density
            x_grid, y_grid, density, V_pos = self.create_density_field(
                positions_2d, volumes, grid_size=60, sigma=1.5
            )
            axes[1].imshow(density.T, origin='lower', cmap='Blues', alpha=0.7,
                           extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]])
            axes[1].contour(x_grid, y_grid, V_pos.T, levels=[0.5], colors='lime', linewidths=2)
            axes[1].set_xlabel('PCA 1')
            axes[1].set_ylabel('PCA 2')
            axes[1].set_title('Gaussian Density')

            # Right: Astronomix fluid-evolved density
            self.create_astronomix_density_plot(state, bounds, ax=axes[2])
            axes[2].set_xlabel('PCA 1')
            axes[2].set_ylabel('PCA 2')
            axes[2].set_title('Astronomix Fluid Density')

            fig.suptitle(f't = {state.time:.3f}  |  V>0: {(volumes > 0).sum()}/{len(volumes)}  |  '
                         f'min τ_err: {tau_errors.min():.4f}', fontsize=12)

            return axes

        n_frames = len(history) // skip_frames
        anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000/fps)

        # Save as MP4
        writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
        anim.save(str(output_path), writer=writer)
        print(f"Saved animation: {output_path}")
        plt.close(fig)


# =============================================================================
# DATA LOADING
# =============================================================================


def load_example_data(example_name: str) -> dict:
    """Load data for a McAllister example."""
    from cytools import Polytope

    data_dir = DATA_BASE / example_name

    points = np.array([[int(x) for x in line.split(',')]
                       for line in (data_dir / "points.dat").read_text().strip().split('\n')])
    heights = np.array([float(x) for x in (data_dir / "corrected_heights.dat").read_text().strip().split(',')])
    basis = [int(x) for x in (data_dir / "basis.dat").read_text().strip().split(',')]
    c_i = np.array([int(x) for x in (data_dir / "target_volumes.dat").read_text().strip().split(',')])
    t_expected = np.array([float(x) for x in (data_dir / "kahler_param.dat").read_text().strip().split(',')])

    # Build CY
    poly = Polytope(points)
    tri = poly.triangulate(heights=heights)
    cy = tri.get_cy()
    cy.set_divisor_basis(basis)

    kappa_basis = cy.intersection_numbers(in_basis=True)
    h11 = len(basis)

    # Build tensor
    kappa_tensor = np.zeros((h11, h11, h11))
    if hasattr(kappa_basis, 'items'):
        for (i, j, k), val in kappa_basis.items():
            if val != 0:
                for perm in [(i, j, k), (i, k, j), (j, i, k), (j, k, i), (k, i, j), (k, j, i)]:
                    kappa_tensor[perm] = val
    else:
        for row in kappa_basis:
            i, j, k = int(row[0]), int(row[1]), int(row[2])
            val = row[3]
            for perm in [(i, j, k), (i, k, j), (j, i, k), (j, k, i), (k, i, j), (k, j, i)]:
                kappa_tensor[perm] = val

    return {
        "h11": h11,
        "kappa_tensor": kappa_tensor,
        "c_i": c_i,  # Target τ = c_i for uncorrected solve
        "t_expected": t_expected,
    }


# =============================================================================
# MAIN
# =============================================================================


def run_simulation_for_example(example_name: str = "5-81-3213",
                               n_particles: int = 500,
                               n_steps: int = 500,
                               temperature: float = 5.0,
                               create_video: bool = True):
    """
    Run fluid dynamics simulation for a McAllister example.

    Using 5-81-3213 (h11=81) as default since it's smaller than 4-214-647 (h11=214).
    """
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 70)
    print(f"FLUID DYNAMICS KKLT SOLVER - {example_name}")
    print("=" * 70)
    print()
    print(f"  n_particles: {n_particles}")
    print(f"  n_steps: {n_steps}")
    print(f"  temperature: {temperature}")
    print()

    # Load data
    print("Loading example data...")
    data = load_example_data(example_name)
    print(f"  h11 = {data['h11']}")
    print(f"  c_i range: [{data['c_i'].min()}, {data['c_i'].max()}]")
    print()

    # Create simulator
    print("Creating simulator...")
    sim = KahlerFluidSimulator(
        kappa_tensor=data["kappa_tensor"],
        tau_target=data["c_i"].astype(float),  # τ = c_i for uncorrected
        n_particles=n_particles,
        temperature=temperature,
        damping=0.95,
        dt=0.005,
    )

    # Run simulation
    print(f"Running {n_steps} steps...")
    key = jax.random.PRNGKey(42)

    start_time = time.time()
    last_print = 0

    def progress_callback(step, state):
        nonlocal last_print
        if step - last_print >= n_steps // 10:
            last_print = step
            v_pos = (np.array(state.volumes) > 0).sum()
            tau_err_min = np.array(state.tau_errors).min()
            print(f"  Step {step:5d}: V>0 = {v_pos:3d}/{n_particles}, min τ_err = {tau_err_min:.4f}")

    history = sim.run(n_steps, key, callback=progress_callback)
    elapsed = time.time() - start_time
    print(f"\nSimulation completed in {elapsed:.1f}s")

    # Analyze results
    final_state = history[-1]
    volumes = np.array(final_state.volumes)
    tau_errors = np.array(final_state.tau_errors)

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"  V > 0: {(volumes > 0).sum()}/{n_particles}")
    print(f"  min τ_error: {tau_errors.min():.6f}")

    # Find best V > 0 solutions
    good_mask = volumes > 0
    if good_mask.any():
        good_indices = np.where(good_mask)[0]
        good_tau_errors = tau_errors[good_indices]
        best_idx = good_indices[np.argmin(good_tau_errors)]

        print(f"\n  Best V > 0 solution:")
        print(f"    Index: {best_idx}")
        print(f"    V: {volumes[best_idx]:.2f}")
        print(f"    τ_error: {tau_errors[best_idx]:.6f}")

        # Compare to expected
        t_found = np.array(final_state.positions[best_idx])
        t_expected = data["t_expected"]
        corr = np.corrcoef(t_found, t_expected)[0, 1]
        print(f"    t correlation with expected: {corr:.4f}")

    # Create visualizations
    print("\nCreating visualizations...")
    viz = FluidVisualizer(sim)
    viz.fit_pca(history)

    # Static plots
    fig = viz.plot_density_evolution(
        history,
        frames=[0, len(history)//4, len(history)//2, -1],
        output_path=OUTPUT_DIR / f"{example_name}_density_evolution.png"
    )
    plt.close(fig)

    # Final snapshot
    fig, ax = plt.subplots(figsize=(10, 8))
    viz.plot_snapshot(final_state, ax=ax)
    plt.savefig(OUTPUT_DIR / f"{example_name}_final_snapshot.png", dpi=150)
    print(f"Saved: {OUTPUT_DIR / f'{example_name}_final_snapshot.png'}")
    plt.close(fig)

    # Animation (optional)
    if create_video:
        print("\nCreating animation (this may take a while)...")
        viz.create_animation(
            history,
            output_path=OUTPUT_DIR / f"{example_name}_evolution.mp4",
            fps=30,
            skip_frames=max(1, len(history) // 300)  # ~10s video
        )

    print("\nDone!")
    return history, sim, data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fluid dynamics KKLT solver")
    parser.add_argument("--example", default="5-81-3213",
                        help="Example name (default: 5-81-3213)")
    parser.add_argument("--particles", type=int, default=500,
                        help="Number of particles")
    parser.add_argument("--steps", type=int, default=500,
                        help="Number of simulation steps")
    parser.add_argument("--temperature", type=float, default=5.0,
                        help="Langevin temperature")
    parser.add_argument("--no-video", action="store_true",
                        help="Skip video generation")

    args = parser.parse_args()

    run_simulation_for_example(
        example_name=args.example,
        n_particles=args.particles,
        n_steps=args.steps,
        temperature=args.temperature,
        create_video=not args.no_video,
    )
