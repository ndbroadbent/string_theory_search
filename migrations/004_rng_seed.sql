-- Migration 004: Add RNG seed for reproducibility
-- Each algorithm gets a seed, trials derive their seed from algo_seed + trial_number

ALTER TABLE meta_algorithms ADD COLUMN rng_seed INTEGER;

-- Also add to meta_state for the master seed
ALTER TABLE meta_state ADD COLUMN master_seed INTEGER;
