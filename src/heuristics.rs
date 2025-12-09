//! Heuristics computation for polytopes
//!
//! Computes ~40 shape and statistical metrics for each polytope.
//! These are used by the Meta-GA to learn which polytope features
//! correlate with good physics outcomes.

use flate2::write::ZlibEncoder;
use flate2::Compression;
use nalgebra::{DMatrix, DVector, SymmetricEigen};
use std::collections::HashMap;
use std::io::Write;

use crate::db::HeuristicsData;

/// Compute all heuristics for a polytope's vertices
pub fn compute_heuristics(
    polytope_id: i64,
    h11: i32,
    h21: i32,
    vertices: &[i32],
) -> HeuristicsData {
    let vertex_count = vertices.len() / 4;
    if vertex_count == 0 {
        return HeuristicsData {
            h11: Some(h11),
            h21: Some(h21),
            vertex_count: Some(0),
            ..Default::default()
        };
    }

    // Reshape to Nx4 matrix
    let verts: Vec<[f64; 4]> = vertices
        .chunks(4)
        .map(|c| [c[0] as f64, c[1] as f64, c[2] as f64, c[3] as f64])
        .collect();

    let n = verts.len();

    // Compute centroid
    let centroid = compute_centroid(&verts);

    // Center vertices
    let centered: Vec<[f64; 4]> = verts
        .iter()
        .map(|v| [
            v[0] - centroid[0],
            v[1] - centroid[1],
            v[2] - centroid[2],
            v[3] - centroid[3],
        ])
        .collect();

    // Distances from centroid
    let distances: Vec<f64> = centered.iter().map(|v| norm4(v)).collect();

    // === Sphericity ===
    let mean_dist = mean(&distances);
    let std_dist = std_dev(&distances, mean_dist);
    let sphericity = if mean_dist > 0.0 {
        1.0 - std_dist / mean_dist
    } else {
        0.0
    };

    // === Inertia tensor eigenvalues ===
    let (inertia_isotropy, _eigenvalues) = compute_inertia(&centered);

    // === Chirality ===
    let chirality_optimal = compute_kabsch_rmsd(&verts);
    let chirality_x = compute_axis_chirality(&centered, 0);
    let chirality_y = compute_axis_chirality(&centered, 1);
    let chirality_z = compute_axis_chirality(&centered, 2);
    let chirality_w = compute_axis_chirality(&centered, 3);

    // Handedness determinant (sign of det of first 4 vertices)
    let handedness_det = if n >= 4 {
        compute_det4(&verts[0..4]).signum()
    } else {
        0.0
    };

    // === Symmetry (overlap after reflection) ===
    let symmetry_x = compute_symmetry(&verts, 0);
    let symmetry_y = compute_symmetry(&verts, 1);
    let symmetry_z = compute_symmetry(&verts, 2);
    let symmetry_w = compute_symmetry(&verts, 3);

    // === PCA / Flatness ===
    let (flatness_3d, flatness_2d, intrinsic_dim) = compute_pca_flatness(&verts);

    // === Spikiness ===
    let spikiness = if mean_dist > 0.0 {
        max_f64(&distances) / mean_dist
    } else {
        0.0
    };
    let max_exposure = max_f64(&distances);

    // === Concentration / Outliers ===
    let median_dist = median(&distances);
    let core_count = distances.iter().filter(|&&d| d < median_dist).count() as f64;
    let outlier_count = distances.iter().filter(|&&d| d > 2.0 * median_dist).count() as f64;
    let conformity_ratio = core_count / (outlier_count + 1.0);

    let distance_kurtosis = kurtosis(&distances);

    // Loner score (max nearest-neighbor distance / mean)
    let loner_score = compute_loner_score(&verts);

    // === Coordinate statistics ===
    let coords_flat: Vec<f64> = vertices.iter().map(|&c| c as f64).collect();
    let coord_mean = mean(&coords_flat);
    let coord_median = median(&coords_flat);
    let coord_std = std_dev(&coords_flat, coord_mean);
    let coord_skewness = skewness(&coords_flat);
    let coord_kurtosis = kurtosis(&coords_flat);

    // === Entropy ===
    let shannon_entropy = compute_entropy(&coords_flat, 20);
    let joint_entropy = compute_joint_entropy(&verts);

    // === Compression ratio ===
    let (compression_ratio, sorted_compression_ratio, sort_compression_gain) =
        compute_compression(&coords_flat);

    // === Integer patterns ===
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let phi_ratio_count = count_phi_ratios(&verts, phi);
    let fibonacci_count = count_fibonacci(&coords_flat);
    let zero_count = coords_flat.iter().filter(|&&c| c == 0.0).count() as i32;
    let one_count = coords_flat.iter().filter(|&&c| c == 1.0).count() as i32;
    let prime_count = count_primes(&coords_flat);

    HeuristicsData {
        h11: Some(h11),
        h21: Some(h21),
        vertex_count: Some(vertex_count as i32),
        sphericity: Some(sphericity),
        inertia_isotropy: Some(inertia_isotropy),
        chirality_optimal: Some(chirality_optimal),
        chirality_x: Some(chirality_x),
        chirality_y: Some(chirality_y),
        chirality_z: Some(chirality_z),
        chirality_w: Some(chirality_w),
        handedness_det: Some(handedness_det),
        symmetry_x: Some(symmetry_x),
        symmetry_y: Some(symmetry_y),
        symmetry_z: Some(symmetry_z),
        symmetry_w: Some(symmetry_w),
        flatness_3d: Some(flatness_3d),
        flatness_2d: Some(flatness_2d),
        intrinsic_dim_estimate: Some(intrinsic_dim),
        spikiness: Some(spikiness),
        max_exposure: Some(max_exposure),
        conformity_ratio: Some(conformity_ratio),
        distance_kurtosis: Some(distance_kurtosis),
        loner_score: Some(loner_score),
        coord_mean: Some(coord_mean),
        coord_median: Some(coord_median),
        coord_std: Some(coord_std),
        coord_skewness: Some(coord_skewness),
        coord_kurtosis: Some(coord_kurtosis),
        shannon_entropy: Some(shannon_entropy),
        joint_entropy: Some(joint_entropy),
        compression_ratio: Some(compression_ratio),
        sorted_compression_ratio: Some(sorted_compression_ratio),
        sort_compression_gain: Some(sort_compression_gain),
        phi_ratio_count: Some(phi_ratio_count),
        fibonacci_count: Some(fibonacci_count),
        zero_count: Some(zero_count),
        one_count: Some(one_count),
        prime_count: Some(prime_count),
        // Outlier scores computed population-level later
        outlier_score: None,
        outlier_max_zscore: None,
        outlier_max_dim: None,
        outlier_count_2sigma: None,
        outlier_count_3sigma: None,
    }
}

// =============================================================================
// Helper functions
// =============================================================================

fn norm4(v: &[f64; 4]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]).sqrt()
}

fn compute_centroid(verts: &[[f64; 4]]) -> [f64; 4] {
    let n = verts.len() as f64;
    let mut c = [0.0; 4];
    for v in verts {
        for i in 0..4 {
            c[i] += v[i];
        }
    }
    for i in 0..4 {
        c[i] /= n;
    }
    c
}

fn mean(vals: &[f64]) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    vals.iter().sum::<f64>() / vals.len() as f64
}

fn std_dev(vals: &[f64], mean: f64) -> f64 {
    if vals.len() < 2 {
        return 0.0;
    }
    let variance = vals.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / vals.len() as f64;
    variance.sqrt()
}

fn median(vals: &[f64]) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    let mut sorted = vals.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

fn max_f64(vals: &[f64]) -> f64 {
    vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
}

fn skewness(vals: &[f64]) -> f64 {
    if vals.len() < 3 {
        return 0.0;
    }
    let m = mean(vals);
    let s = std_dev(vals, m);
    if s == 0.0 {
        return 0.0;
    }
    let n = vals.len() as f64;
    vals.iter().map(|&x| ((x - m) / s).powi(3)).sum::<f64>() / n
}

fn kurtosis(vals: &[f64]) -> f64 {
    if vals.len() < 4 {
        return 0.0;
    }
    let m = mean(vals);
    let s = std_dev(vals, m);
    if s == 0.0 {
        return 0.0;
    }
    let n = vals.len() as f64;
    vals.iter().map(|&x| ((x - m) / s).powi(4)).sum::<f64>() / n - 3.0
}

fn compute_inertia(centered: &[[f64; 4]]) -> (f64, Vec<f64>) {
    // Build 4x4 covariance matrix (inertia tensor analog)
    let mut cov = [[0.0; 4]; 4];
    for v in centered {
        for i in 0..4 {
            for j in 0..4 {
                cov[i][j] += v[i] * v[j];
            }
        }
    }

    // Convert to nalgebra matrix
    let mat = DMatrix::from_fn(4, 4, |i, j| cov[i][j]);
    let eigen = SymmetricEigen::new(mat);
    let mut eigenvalues: Vec<f64> = eigen.eigenvalues.iter().cloned().collect();
    eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let isotropy = if eigenvalues[3] > 0.0 {
        eigenvalues[0] / eigenvalues[3]
    } else {
        0.0
    };

    (isotropy, eigenvalues)
}

fn compute_kabsch_rmsd(verts: &[[f64; 4]]) -> f64 {
    // Mirror along x-axis
    let mirrored: Vec<[f64; 4]> = verts.iter().map(|v| [-v[0], v[1], v[2], v[3]]).collect();

    // Center both
    let c1 = compute_centroid(verts);
    let c2 = compute_centroid(&mirrored);

    let centered1: Vec<[f64; 4]> = verts
        .iter()
        .map(|v| [v[0] - c1[0], v[1] - c1[1], v[2] - c1[2], v[3] - c1[3]])
        .collect();
    let centered2: Vec<[f64; 4]> = mirrored
        .iter()
        .map(|v| [v[0] - c2[0], v[1] - c2[1], v[2] - c2[2], v[3] - c2[3]])
        .collect();

    // Build cross-covariance matrix H = P^T * Q
    let n = verts.len();
    let mut h = [[0.0; 4]; 4];
    for i in 0..n {
        for j in 0..4 {
            for k in 0..4 {
                h[j][k] += centered1[i][j] * centered2[i][k];
            }
        }
    }

    // SVD to find optimal rotation
    let h_mat = DMatrix::from_fn(4, 4, |i, j| h[i][j]);
    let svd = h_mat.svd(true, true);

    // R = V * U^T
    let u = svd.u.unwrap();
    let vt = svd.v_t.unwrap();
    let r = vt.transpose() * u.transpose();

    // Apply rotation and compute RMSD
    let mut sum_sq = 0.0;
    for i in 0..n {
        let p = DVector::from_row_slice(&centered1[i]);
        let q_vec = DVector::from_row_slice(&centered2[i]);
        let q_rotated = &r * q_vec;
        sum_sq += (p - q_rotated).norm_squared();
    }

    (sum_sq / n as f64).sqrt()
}

fn compute_axis_chirality(centered: &[[f64; 4]], axis: usize) -> f64 {
    // Mirror along given axis
    let mirrored: Vec<[f64; 4]> = centered
        .iter()
        .map(|v| {
            let mut m = *v;
            m[axis] *= -1.0;
            m
        })
        .collect();

    // Mean of minimum distances from each point to nearest in mirrored set
    let n = centered.len();
    let mut sum_min_dist = 0.0;

    for i in 0..n {
        let mut min_dist = f64::MAX;
        for j in 0..n {
            let mut d_sq = 0.0;
            for k in 0..4 {
                d_sq += (centered[i][k] - mirrored[j][k]).powi(2);
            }
            min_dist = min_dist.min(d_sq.sqrt());
        }
        sum_min_dist += min_dist;
    }

    sum_min_dist / n as f64
}

fn compute_det4(verts: &[[f64; 4]]) -> f64 {
    // Determinant of 4x4 matrix formed by first 4 vertices
    let mat = DMatrix::from_fn(4, 4, |i, j| verts[i][j]);
    mat.determinant()
}

fn compute_symmetry(verts: &[[f64; 4]], axis: usize) -> f64 {
    // Fraction of vertices that have a match after reflection
    let tolerance = 0.1;
    let n = verts.len();

    let reflected: Vec<[f64; 4]> = verts
        .iter()
        .map(|v| {
            let mut r = *v;
            r[axis] *= -1.0;
            r
        })
        .collect();

    let mut matches = 0;
    for v in verts {
        for r in &reflected {
            let mut d_sq = 0.0;
            for k in 0..4 {
                d_sq += (v[k] - r[k]).powi(2);
            }
            if d_sq.sqrt() < tolerance {
                matches += 1;
                break;
            }
        }
    }

    matches as f64 / n as f64
}

fn compute_pca_flatness(verts: &[[f64; 4]]) -> (f64, f64, f64) {
    if verts.len() < 5 {
        return (1.0, 1.0, 4.0);
    }

    // Build covariance matrix
    let centroid = compute_centroid(verts);
    let centered: Vec<[f64; 4]> = verts
        .iter()
        .map(|v| [
            v[0] - centroid[0],
            v[1] - centroid[1],
            v[2] - centroid[2],
            v[3] - centroid[3],
        ])
        .collect();

    let mut cov = [[0.0; 4]; 4];
    for v in &centered {
        for i in 0..4 {
            for j in 0..4 {
                cov[i][j] += v[i] * v[j];
            }
        }
    }
    let n = centered.len() as f64;
    for i in 0..4 {
        for j in 0..4 {
            cov[i][j] /= n;
        }
    }

    let mat = DMatrix::from_fn(4, 4, |i, j| cov[i][j]);
    let eigen = SymmetricEigen::new(mat);
    let mut eigenvalues: Vec<f64> = eigen.eigenvalues.iter().cloned().collect();
    eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)); // descending

    let total: f64 = eigenvalues.iter().sum();
    if total <= 0.0 {
        return (1.0, 1.0, 4.0);
    }

    let variance_ratios: Vec<f64> = eigenvalues.iter().map(|&e| e / total).collect();
    let flatness_3d: f64 = variance_ratios.iter().take(3).sum();
    let flatness_2d: f64 = variance_ratios.iter().take(2).sum();

    // Intrinsic dimension: how many components for 95% variance
    let mut cumsum = 0.0;
    let mut intrinsic_dim = 4.0;
    for (i, &r) in variance_ratios.iter().enumerate() {
        cumsum += r;
        if cumsum >= 0.95 {
            intrinsic_dim = (i + 1) as f64;
            break;
        }
    }

    (flatness_3d, flatness_2d, intrinsic_dim)
}

fn compute_loner_score(verts: &[[f64; 4]]) -> f64 {
    if verts.len() < 2 {
        return 0.0;
    }

    let n = verts.len();
    let mut nn_dists = Vec::with_capacity(n);

    for i in 0..n {
        let mut min_dist = f64::MAX;
        for j in 0..n {
            if i == j {
                continue;
            }
            let mut d_sq = 0.0;
            for k in 0..4 {
                d_sq += (verts[i][k] - verts[j][k]).powi(2);
            }
            min_dist = min_dist.min(d_sq.sqrt());
        }
        nn_dists.push(min_dist);
    }

    let mean_nn = mean(&nn_dists);
    if mean_nn > 0.0 {
        max_f64(&nn_dists) / mean_nn
    } else {
        0.0
    }
}

fn compute_entropy(vals: &[f64], bins: usize) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }

    let min_val = vals.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max_val - min_val;

    if range == 0.0 {
        return 0.0;
    }

    let mut histogram = vec![0usize; bins];
    for &v in vals {
        let bin = ((v - min_val) / range * (bins - 1) as f64).floor() as usize;
        histogram[bin.min(bins - 1)] += 1;
    }

    let n = vals.len() as f64;
    let mut entropy = 0.0;
    for &count in &histogram {
        if count > 0 {
            let p = count as f64 / n;
            entropy -= p * p.ln();
        }
    }

    entropy
}

fn compute_joint_entropy(verts: &[[f64; 4]]) -> f64 {
    // Simplified: discretize each axis into bins and count joint occurrences
    let bins = 5;

    // Find ranges for each axis
    let mut mins = [f64::INFINITY; 4];
    let mut maxs = [f64::NEG_INFINITY; 4];
    for v in verts {
        for i in 0..4 {
            mins[i] = mins[i].min(v[i]);
            maxs[i] = maxs[i].max(v[i]);
        }
    }

    let mut ranges = [0.0; 4];
    for i in 0..4 {
        ranges[i] = maxs[i] - mins[i];
        if ranges[i] == 0.0 {
            ranges[i] = 1.0;
        }
    }

    // Count joint occurrences
    let mut counts: HashMap<[usize; 4], usize> = HashMap::new();
    for v in verts {
        let mut bin = [0usize; 4];
        for i in 0..4 {
            bin[i] = ((v[i] - mins[i]) / ranges[i] * (bins - 1) as f64)
                .floor()
                .max(0.0)
                .min((bins - 1) as f64) as usize;
        }
        *counts.entry(bin).or_insert(0) += 1;
    }

    let n = verts.len() as f64;
    let mut entropy = 0.0;
    for &count in counts.values() {
        if count > 0 {
            let p = count as f64 / n;
            entropy -= p * p.ln();
        }
    }

    entropy
}

fn compute_compression(coords: &[f64]) -> (f64, f64, f64) {
    if coords.is_empty() {
        return (1.0, 1.0, 1.0);
    }

    // Original compression
    let json = serde_json::to_string(coords).unwrap_or_default();
    let json_bytes = json.as_bytes();

    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::best());
    encoder.write_all(json_bytes).ok();
    let compressed = encoder.finish().unwrap_or_default();

    let compression_ratio = if json_bytes.is_empty() {
        1.0
    } else {
        compressed.len() as f64 / json_bytes.len() as f64
    };

    // Sorted compression
    let mut sorted = coords.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let sorted_json = serde_json::to_string(&sorted).unwrap_or_default();
    let sorted_bytes = sorted_json.as_bytes();

    let mut encoder2 = ZlibEncoder::new(Vec::new(), Compression::best());
    encoder2.write_all(sorted_bytes).ok();
    let sorted_compressed = encoder2.finish().unwrap_or_default();

    let sorted_compression_ratio = if sorted_bytes.is_empty() {
        1.0
    } else {
        sorted_compressed.len() as f64 / sorted_bytes.len() as f64
    };

    let sort_compression_gain = if sorted_compressed.is_empty() {
        1.0
    } else {
        compressed.len() as f64 / sorted_compressed.len() as f64
    };

    (compression_ratio, sorted_compression_ratio, sort_compression_gain)
}

fn count_phi_ratios(verts: &[[f64; 4]], phi: f64) -> i32 {
    let mut count = 0;
    for v in verts {
        for i in 0..4 {
            for j in (i + 1)..4 {
                if v[j].abs() > 0.01 {
                    let ratio = (v[i] / v[j]).abs();
                    if (ratio - phi).abs() < 0.1 || (ratio - 1.0 / phi).abs() < 0.1 {
                        count += 1;
                    }
                }
            }
        }
    }
    count
}

fn count_fibonacci(coords: &[f64]) -> i32 {
    let fib: std::collections::HashSet<i32> =
        [1, 2, 3, 5, 8, 13, 21, 34, 55, 89].iter().cloned().collect();

    coords
        .iter()
        .filter(|&&c| {
            let i = c as i32;
            c == i as f64 && fib.contains(&i.abs())
        })
        .count() as i32
}

fn count_primes(coords: &[f64]) -> i32 {
    coords
        .iter()
        .filter(|&&c| {
            let i = c as i32;
            c == i as f64 && is_prime(i.abs())
        })
        .count() as i32
}

fn is_prime(n: i32) -> bool {
    if n < 2 {
        return false;
    }
    if n == 2 {
        return true;
    }
    if n % 2 == 0 {
        return false;
    }
    let sqrt_n = (n as f64).sqrt() as i32;
    for i in (3..=sqrt_n).step_by(2) {
        if n % i == 0 {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_heuristics_basic() {
        // Simple 4-vertex polytope (tetrahedron in 4D)
        let vertices = vec![
            1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
        ];

        let h = compute_heuristics(1, 3, 6, &vertices);

        assert_eq!(h.vertex_count, Some(4));
        assert_eq!(h.h11, Some(3));
        assert_eq!(h.h21, Some(6));
        assert!(h.sphericity.unwrap() > 0.0);
    }

    #[test]
    fn test_empty_vertices() {
        let h = compute_heuristics(1, 3, 6, &[]);
        assert_eq!(h.vertex_count, Some(0));
    }

    #[test]
    fn test_is_prime() {
        assert!(!is_prime(0));
        assert!(!is_prime(1));
        assert!(is_prime(2));
        assert!(is_prime(3));
        assert!(!is_prime(4));
        assert!(is_prime(5));
        assert!(is_prime(7));
        assert!(!is_prime(9));
        assert!(is_prime(11));
    }
}
