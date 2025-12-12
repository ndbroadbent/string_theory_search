# GA Prototype: Lessons Learned

This document captures technical and operational lessons from the first week of GA experiments (Dec 2024).

## Database Performance

### 1. Add Indexes for Common Queries

**Problem:** `SELECT COUNT(*) FROM evaluations` took ages on 20M+ rows.

**Solution:** Add indexes and/or cached counters:

```sql
-- For fitness-based queries (already exists but verify)
CREATE INDEX IF NOT EXISTS idx_evaluations_fitness ON evaluations(fitness DESC);

-- For run-based aggregations
CREATE INDEX IF NOT EXISTS idx_evaluations_run_id ON evaluations(run_id);

-- For algorithm analysis
CREATE INDEX IF NOT EXISTS idx_runs_algorithm_id ON runs(algorithm_id);

-- Consider a stats table for cached counts
CREATE TABLE IF NOT EXISTS stats_cache (
    key TEXT PRIMARY KEY,
    value INTEGER,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

### 2. Disk Space Monitoring

**Problem:** 33GB ZFS quota filled up, causing SQLite I/O errors and apparent data loss.

**Solution:**
- Monitor disk usage proactively
- Set up alerts at 80% capacity
- The database grew to 23GB + 370MB WAL file
- Increased quota to 100GB in Proxmox

### 3. WAL File Management

**Problem:** WAL file grew to 370MB during heavy writes.

**Solution:** Periodic checkpointing:
```sql
PRAGMA wal_checkpoint(TRUNCATE);
```

### 4. Analytical Query Architecture

**Problem:** Simple analytical queries on 20M+ rows are painfully slow:
- `SELECT COUNT(*) FROM evaluations` - minutes
- `GROUP BY polytope_id` aggregations - 2+ minutes
- Any full table scan is impractical for interactive analysis

**Root Cause:** SQLite is optimized for OLTP (transactional) workloads, not OLAP (analytical). Our use case requires both:
- **OLTP**: Fast inserts during GA runs (thousands/second)
- **OLAP**: Post-hoc analysis, aggregations, reporting

**Solutions to Consider:**

#### Option A: Better SQLite Indexing + Materialized Views
```sql
-- Composite indexes for common query patterns
CREATE INDEX idx_eval_polytope_fitness ON evaluations(polytope_id, fitness DESC);
CREATE INDEX idx_eval_run_gen ON evaluations(run_id, generation);

-- Materialized summary tables (updated periodically)
CREATE TABLE polytope_stats AS
SELECT
    polytope_id,
    COUNT(*) as eval_count,
    MAX(fitness) as best_fitness,
    AVG(fitness) as avg_fitness,
    MIN(created_at) as first_eval,
    MAX(created_at) as last_eval
FROM evaluations
GROUP BY polytope_id;

-- Update via trigger or periodic job
```

#### Option B: PostgreSQL with Proper Analytics
PostgreSQL has better query planning, parallel query execution, and partitioning:
```sql
-- Partitioned by date for efficient time-range queries
CREATE TABLE evaluations (
    ...
) PARTITION BY RANGE (created_at);

-- Parallel aggregation
SET max_parallel_workers_per_gather = 4;
```

#### Option C: Hybrid Architecture
- **SQLite for hot data**: Current run, recent evaluations
- **DuckDB for analytics**: Periodic export for analysis
- **Parquet files for archival**: Compressed, columnar, fast scans

```python
# Export to Parquet for analysis
import duckdb
duckdb.sql("""
    COPY (SELECT * FROM sqlite_scan('string_theory.db', 'evaluations'))
    TO 'evaluations.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)
""")

# Fast analytics on Parquet
duckdb.sql("SELECT polytope_id, COUNT(*), MAX(fitness) FROM 'evaluations.parquet' GROUP BY 1")
```

#### Option D: Pre-computed Analytics Tables
During GA runs, maintain rolling statistics:
```sql
-- Updated after each batch of evaluations
INSERT OR REPLACE INTO run_stats (run_id, eval_count, best_fitness, ...)
SELECT run_id, COUNT(*), MAX(fitness), ... FROM evaluations WHERE run_id = ?;
```

**Recommendation:** Start with Option A (better indexes + materialized views) for immediate relief, then consider Option C (DuckDB/Parquet) for serious analytics as data grows.

## Operational Lessons

### 5. Pre-flight Checks Before Long Runs

Before starting multi-day experiments:
- [ ] Check disk space: `df -h` (ensure >50% free)
- [ ] Verify database integrity: `PRAGMA integrity_check`
- [ ] Checkpoint WAL: `PRAGMA wal_checkpoint(TRUNCATE)`
- [ ] Backup database: `cp string_theory.db string_theory.db.backup`
- [ ] Set up monitoring/alerts for disk, memory, CPU

### 6. Graceful Degradation

**Problem:** When disk filled, SQLite failed silently in some cases, causing confusing "data loss" symptoms.

**Solution:**
- Add disk space checks before critical operations
- Fail loudly with clear error messages
- Consider read-only fallback mode when disk is low

---

*More lessons to be added as analysis continues...*
