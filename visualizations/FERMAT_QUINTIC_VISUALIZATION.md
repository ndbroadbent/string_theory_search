# Calabi-Yau Visualization: The Fermat Quintic

## What Are We Looking At?

Calabi-Yau manifolds are 6-dimensional (3 complex dimensions).

**Key insight:** We can visualize all 6D by showing **two 3D plots side by side**:

- **Left plot:** dimensions (1, 2, 3)
- **Right plot:** dimensions (4, 5, 6)

That's 2×3 = 6 dimensions fully covered!

## 6D Rotation and the Dual View

### Compact Dimensions vs Rotation - Important Distinctions

**Compactness** is about translations/geodesics, not rotations:
- On a circle: walk straight → return to start
- On a flat torus: a generic straight line might **never close** (can be dense!)
- On a curved compact manifold: geodesics need not be periodic

**Rotation** (360° = identity for vectors) is about the group SO(n), not compactness. This works the same in any Euclidean Rⁿ.

**Global vs Local:** On a generic Calabi-Yau, there's usually **no global SO(6) symmetry** acting on the whole manifold. You can rotate the tangent space at a point (local SO(6) frame), but "rotating the manifold" globally isn't generally meaningful.

**Spinor gotcha:** Fermions pick up a sign under 360° rotation - they need 720° to return. This is the Spin(n) → SO(n) double cover, not compactness.

### What We're Actually Visualizing

When we "rotate" the CY visualization, we're rotating the **embedding coordinates** in R⁶ - essentially changing our viewing angle. This is a rotation of the ambient space, not a symmetry of the manifold itself.

### Independent Rotations in the Dual View

Here's the key insight about dual 3D visualization:

| Rotation Axis | Left Image (dims 1,2,3) | Right Image (dims 4,5,6) |
|---------------|-------------------------|--------------------------|
| Axis 1, 2, or 3 | **Animates** | Static |
| Axis 4, 5, or 6 | Static | **Animates** |

If you rotate around an axis that isn't visible in one image, **that image stays completely static** - you're rotating dimensions it doesn't show!

### Efficient 6D Animation

The most efficient way to show all 6D structure:

```
Two INDEPENDENT parallel rotations:
- Left:  compound rotation in (dim1, dim2, dim3) space
- Right: compound rotation in (dim4, dim5, dim6) space

Both animate simultaneously - no wasted frames!
```

Each 3D subspace has **3 rotation planes**:
- Left: (1-2), (1-3), (2-3) planes
- Right: (4-5), (4-6), (5-6) planes

That's **6 independent rotation axes** - fully covering SO(6), the rotation group in 6D.

### Animation Strategy

1. **Sequential:** Rotate each axis one at a time (6 phases, shows each axis clearly)
2. **Parallel:** Rotate all 6 axes simultaneously at different rates (compact, shows everything)
3. **Compound:** Rotate in both 3D subspaces with incommensurate frequencies (ergodic, eventually covers all orientations)

See `visualize_fermat_quintic_6d.py` for the implementation.

The most common single visualization is the **Fermat quintic** - the simplest CY defined by:

```
z₁ⁿ + z₂ⁿ + z₃ⁿ + z₄ⁿ + z₅ⁿ = 0   in CP⁴
```

For visualization, we take a 2D slice in C² satisfying `z₁ⁿ + z₂ⁿ = 1`.

## The Parametric Equations

The surface is parametrized by real coordinates (x, y) with x ∈ [0, π/2], y ∈ [-π/2, π/2]:

```python
import numpy as np

def calabi_yau_surface(n, k1, k2, a, x, y):
    """
    Generate a patch of the Fermat quintic CY surface.

    Args:
        n: Degree of the surface (typically 2-9)
        k1, k2: Integers in [0, n-1] selecting which patch
        a: Rotation angle for 3D projection
        x, y: Parameter arrays

    Returns:
        X, Y, Z: 3D coordinates
    """
    # Complex coordinates satisfying z1^n + z2^n = 1
    z1 = np.exp(2j * np.pi * k1 / n) * np.cos(x + 1j*y) ** (2/n)
    z2 = np.exp(2j * np.pi * k2 / n) * np.sin(x + 1j*y) ** (2/n)

    # Project to 3D
    X = np.real(z1)
    Y = np.real(z2)
    Z = np.imag(z1) * np.cos(a) + np.imag(z2) * np.sin(a)

    return X, Y, Z
```

### Why n² Patches?

For degree n, there are n choices for k1 and n choices for k2, giving **n² patches** that tile together. The patches connect smoothly because of the complex exponential phase factors.

### The Projection Angle 'a'

The parameter `a` controls how we project from 4D (Re(z1), Im(z1), Re(z2), Im(z2)) down to 3D. Different values of `a` give different views of the same surface - like rotating a 3D object to see different sides.

## Physical Interpretation

In string theory, these extra dimensions are "compactified" - curled up so small we can't see them. The shape of this compactification determines:

- Particle masses
- Force strengths
- Number of particle generations
- The cosmological constant

What we're visualizing is the **local geometry** of these hidden dimensions.

## The "Flower" Structure

The characteristic appearance comes from:
1. **n² patches** tiling together (more patches = more "petals")
2. **Branch cuts** in the complex power function creating seams
3. **Projection** squashing 4D → 3D

Higher n values create more intricate, flower-like patterns.

## Tools & Resources

### Python
- [CalabiYauViz](https://github.com/Kuo-TingKai/CalabiYauViz) - Python → STL for 3D printing
- [Calabi Yau with Python blog](https://asahidari.hatenablog.com/entry/2020/06/08/194342) - Tutorial

### Blender
- [blender-calabi-yau-viz](https://github.com/cashetland/blender-calabi-yau-viz) - High-quality renders

### Interactive
- [Observable notebook](https://observablehq.com/@sw1227/calabi-yau-manifold-3d) - WebGL visualization
- [Wolfram Demo](https://demonstrations.wolfram.com/CalabiYauSpace/) - Manipulate parameters

### 3D Models
- [Sketchfab CY models](https://sketchfab.com/3d-models/calabi-yau-manifold-86ed1582779145528543ebb0fc3cfdd6)
- [Lawrence U 3D printing](https://blogs.lawrence.edu/makerspace/2019/06/26/into-the-manifold/)

## Beyond the Fermat Quintic

The Fermat quintic is just ONE Calabi-Yau (and a very symmetric one). The Kreuzer-Skarke database has **473 million** distinct CY threefolds!

For visualizing our actual polytopes:
1. **Polytope visualization** - Plot the convex hull in 3D/4D
2. **Toric diagram** - 2D projection showing the combinatorial structure
3. **Metric heatmaps** - Use cymyc to compute the CY metric, plot curvature

See `visualize_polytope.py` for polytope visualizations.
See `visualize_fermat.py` for the classic Fermat quintic animation.

## References

- Hanson, A. "A Construction for Computer Visualization of Certain Complex Curves" (1994)
- [Andrew Hanson's CY visualizations](https://www.cs.indiana.edu/~hanson/Calabi-Yau/)
- McAllister group: https://liammcallistergroup.com/
