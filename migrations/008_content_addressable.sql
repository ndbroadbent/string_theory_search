-- Migration 008: Content-addressable evaluations
-- Adds SHA256 hash for caching/deduplication, source tracking, and external polytope support

-- Content hash for deduplication/caching (SHA256 of model_version + inputs)
ALTER TABLE evaluations ADD COLUMN input_hash TEXT;

-- Model version used for this evaluation (for cache invalidation)
ALTER TABLE evaluations ADD COLUMN model_version TEXT;

-- Source of evaluation: 'ga' (genetic algorithm), 'playground' (manual), 'test' (unit test)
ALTER TABLE evaluations ADD COLUMN source TEXT DEFAULT 'ga';

-- Optional human-readable label
ALTER TABLE evaluations ADD COLUMN label TEXT;

-- For external polytopes not in our DB (e.g., McAllister paper data)
-- When set, polytope_id should be -1
ALTER TABLE evaluations ADD COLUMN vertices_json TEXT;

-- Hodge numbers (denormalized for external polytopes where polytope_id = -1)
ALTER TABLE evaluations ADD COLUMN h11 INTEGER;
ALTER TABLE evaluations ADD COLUMN h21 INTEGER;

-- Unique constraint on hash for cache lookups (only one result per unique input)
CREATE UNIQUE INDEX IF NOT EXISTS idx_evaluations_hash ON evaluations(input_hash) WHERE input_hash IS NOT NULL;

-- Index for filtering by source
CREATE INDEX IF NOT EXISTS idx_evaluations_source ON evaluations(source);

-- Index for model version queries
CREATE INDEX IF NOT EXISTS idx_evaluations_model_version ON evaluations(model_version);

-- Note: schema_version is updated automatically by the migration runner
