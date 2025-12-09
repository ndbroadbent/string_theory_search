#!/usr/bin/env python3
"""
Migrate existing JSON data to SQLite database.

Migrates:
- heuristics_sample.json -> heuristics table
- results/run_*/fit*.json -> evaluations table (best results only)

Usage:
    python scripts/migrate_json_to_db.py [--db-path data/string_theory.db]
"""

import argparse
import json
import sqlite3
from pathlib import Path
from datetime import datetime


def create_tables_if_needed(conn: sqlite3.Connection):
    """Create tables if they don't exist (should already exist from Rust migrations)."""
    # The schema should be created by Rust, but we check anyway
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='heuristics'"
    )
    if not cursor.fetchone():
        print("Warning: heuristics table doesn't exist. Run the search binary first to create schema.")
        return False
    return True


def migrate_heuristics(conn: sqlite3.Connection, json_path: str):
    """Migrate heuristics_sample.json to heuristics table."""
    if not Path(json_path).exists():
        print(f"No heuristics file found at {json_path}")
        return 0

    print(f"Loading heuristics from {json_path}...")
    with open(json_path) as f:
        data = json.load(f)

    print(f"Found {len(data)} heuristics records")

    # Map JSON keys to database columns
    column_mapping = {
        'polytope_id': 'polytope_id',
        'sphericity': 'sphericity',
        'inertia_isotropy': 'inertia_isotropy',
        'chirality_optimal': 'chirality_optimal',
        'chirality_x': 'chirality_x',
        'chirality_y': 'chirality_y',
        'chirality_z': 'chirality_z',
        'chirality_w': 'chirality_w',
        'handedness_det': 'handedness_det',
        'symmetry_x': 'symmetry_x',
        'symmetry_y': 'symmetry_y',
        'symmetry_z': 'symmetry_z',
        'symmetry_w': 'symmetry_w',
        'flatness_3d': 'flatness_3d',
        'flatness_2d': 'flatness_2d',
        'intrinsic_dim_estimate': 'intrinsic_dim_estimate',
        'spikiness': 'spikiness',
        'max_exposure': 'max_exposure',
        'conformity_ratio': 'conformity_ratio',
        'distance_kurtosis': 'distance_kurtosis',
        'loner_score': 'loner_score',
        'coord_mean': 'coord_mean',
        'coord_median': 'coord_median',
        'coord_std': 'coord_std',
        'coord_skewness': 'coord_skewness',
        'coord_kurtosis': 'coord_kurtosis',
        'shannon_entropy': 'shannon_entropy',
        'joint_entropy': 'joint_entropy',
        'compression_ratio': 'compression_ratio',
        'sorted_compression_ratio': 'sorted_compression_ratio',
        'sort_compression_gain': 'sort_compression_gain',
        'phi_ratio_count': 'phi_ratio_count',
        'fibonacci_count': 'fibonacci_count',
        'zero_count': 'zero_count',
        'one_count': 'one_count',
        'prime_count': 'prime_count',
        'outlier_score': 'outlier_score',
        'outlier_max_zscore': 'outlier_max_zscore',
        'outlier_max_dim': 'outlier_max_dim',
        'outlier_count_2sigma': 'outlier_count_2sigma',
        'outlier_count_3sigma': 'outlier_count_3sigma',
    }

    inserted = 0
    for record in data:
        polytope_id = record.get('polytope_id')
        if polytope_id is None:
            continue

        # Build column and value lists
        columns = ['polytope_id']
        values = [polytope_id]

        for json_key, db_col in column_mapping.items():
            if json_key == 'polytope_id':
                continue
            if json_key in record:
                columns.append(db_col)
                values.append(record[json_key])

        # Use INSERT OR REPLACE to handle existing records
        placeholders = ','.join(['?' for _ in values])
        col_str = ','.join(columns)

        try:
            conn.execute(
                f"INSERT OR REPLACE INTO heuristics ({col_str}) VALUES ({placeholders})",
                values
            )
            inserted += 1
        except sqlite3.Error as e:
            print(f"Error inserting polytope {polytope_id}: {e}")

    conn.commit()
    print(f"Inserted {inserted} heuristics records")
    return inserted


def migrate_best_results(conn: sqlite3.Connection, results_dir: str):
    """Migrate best results from results/run_*/fit*.json to evaluations table."""
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"No results directory found at {results_dir}")
        return 0

    # Find all fit*.json files
    fit_files = list(results_path.glob("run_*/fit*.json"))
    print(f"Found {len(fit_files)} result files")

    inserted = 0
    for fit_file in fit_files:
        try:
            with open(fit_file) as f:
                data = json.load(f)

            # Extract run ID from path
            run_id = fit_file.parent.name  # e.g., "run_81"

            genome = data.get('genome', {})
            physics = data.get('physics', {})
            fitness = data.get('fitness', 0.0)

            polytope_id = genome.get('polytope_id')
            if polytope_id is None:
                continue

            # Serialize arrays to JSON
            kahler_json = json.dumps(genome.get('kahler_moduli', []))
            complex_json = json.dumps(genome.get('complex_moduli', []))
            flux_f_json = json.dumps(genome.get('flux_f', []))
            flux_h_json = json.dumps(genome.get('flux_h', []))

            conn.execute(
                """INSERT INTO evaluations (
                    polytope_id, run_id, generation,
                    g_s, kahler_moduli, complex_moduli, flux_f, flux_h,
                    fitness, alpha_em, alpha_s, sin2_theta_w, n_generations,
                    cosmological_constant, success, error, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    polytope_id,
                    run_id,
                    None,  # generation unknown for historical data
                    genome.get('g_s'),
                    kahler_json,
                    complex_json,
                    flux_f_json,
                    flux_h_json,
                    fitness,
                    physics.get('alpha_em'),
                    physics.get('alpha_s'),
                    physics.get('sin2_theta_w'),
                    physics.get('n_generations'),
                    physics.get('cosmological_constant'),
                    1 if physics.get('success', False) else 0,
                    physics.get('error'),
                    datetime.now().isoformat(),
                )
            )
            inserted += 1

        except (json.JSONDecodeError, KeyError, sqlite3.Error) as e:
            print(f"Error processing {fit_file}: {e}")

    conn.commit()
    print(f"Inserted {inserted} evaluation records from results")
    return inserted


def main():
    parser = argparse.ArgumentParser(description="Migrate JSON data to SQLite")
    parser.add_argument(
        "--db-path",
        default="data/string_theory.db",
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--heuristics",
        default="heuristics_sample.json",
        help="Path to heuristics JSON file",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Path to results directory",
    )
    args = parser.parse_args()

    # Ensure data directory exists
    db_path = Path(args.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Connecting to database at {args.db_path}...")
    conn = sqlite3.connect(args.db_path)

    if not create_tables_if_needed(conn):
        print("Tables not found. Please run the search binary first to create schema.")
        return 1

    print()
    print("=" * 60)
    print("Migrating heuristics...")
    print("=" * 60)
    migrate_heuristics(conn, args.heuristics)

    print()
    print("=" * 60)
    print("Migrating best results...")
    print("=" * 60)
    migrate_best_results(conn, args.results_dir)

    print()
    print("=" * 60)
    print("Migration complete!")
    print("=" * 60)

    # Print summary
    cursor = conn.execute("SELECT COUNT(*) FROM heuristics")
    h_count = cursor.fetchone()[0]
    cursor = conn.execute("SELECT COUNT(*) FROM evaluations")
    e_count = cursor.fetchone()[0]
    print(f"Total heuristics records: {h_count}")
    print(f"Total evaluation records: {e_count}")

    conn.close()
    return 0


if __name__ == "__main__":
    exit(main())
