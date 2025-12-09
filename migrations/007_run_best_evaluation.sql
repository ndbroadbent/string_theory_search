-- Migration 007: Cache best evaluation ID on runs table

ALTER TABLE runs ADD COLUMN best_evaluation_id INTEGER REFERENCES evaluations(id);
