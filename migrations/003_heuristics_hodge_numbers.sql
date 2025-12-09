-- Migration 003: Add Hodge numbers to heuristics table
-- Store h11, h21, vertex_count directly in heuristics so we don't need a join

ALTER TABLE heuristics ADD COLUMN h11 INTEGER;
ALTER TABLE heuristics ADD COLUMN h21 INTEGER;
ALTER TABLE heuristics ADD COLUMN vertex_count INTEGER;
