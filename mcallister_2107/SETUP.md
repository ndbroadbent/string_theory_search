# McAllister 2107 Setup

## Dependencies

### CYTools 2021 (June) - CGAL Triangulation Binary

The McAllister CYTools version (vendor/cytools_mcallister_2107) requires a custom CGAL binary for triangulation.

**Compile it:**
```bash
cd vendor/cytools_mcallister_2107/external/cgal

# Install CGAL if needed
brew install cgal

# Compile the triangulation binary (requires C++17 and Eigen)
c++ -O2 -std=c++17 triangulate.cpp -o cgal-triangulate-4d \
    -I/opt/homebrew/include \
    -I/opt/homebrew/include/eigen3 \
    -L/opt/homebrew/lib \
    -lgmp -lmpfr
```

**Then update the config path:**
Edit `vendor/cytools_mcallister_2107/cytools/config.py`:
```python
cgal_path = "/path/to/vendor/cytools_mcallister_2107/external/cgal/"
```

Or symlink to a standard location:
```bash
sudo ln -s $(pwd)/vendor/cytools_mcallister_2107/external/cgal/cgal-triangulate-4d /usr/local/bin/
```

### Other Dependencies

The 2021 CYTools also needs:
- `flint` - for exact rational arithmetic
- `ppl` - Parma Polyhedra Library
- `ortools` - Google OR-Tools (fallback optimizer, replaces mosek)

These should be installed via pip/uv in the main project.

## Why CYTools 2021?

McAllister **wrote CYTools**. The June 2021 version (commit bb5b550) matches their paper's data files (dated July 20, 2021).

Key difference: **Divisor basis choice changed** between 2021 and now.
- 2021: basis = `[3, 4, 5, 8]` for the dual polytope
- 2024: basis = `[5, 6, 7, 8]` for the same polytope

This breaks all physics calculations if you use the wrong version.
