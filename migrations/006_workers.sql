-- Migration 006: Workers table for process registration and heartbeats
-- Workers register on startup, send heartbeats, and are considered dead after 60s without heartbeat

CREATE TABLE workers (
    id TEXT PRIMARY KEY,              -- Unique worker ID: "{type}_{hostname}_{pid}" e.g. "search_server1_12345"
    worker_type TEXT NOT NULL,        -- "search" or "heuristics"
    hostname TEXT NOT NULL,
    pid INTEGER NOT NULL,
    started_at TEXT DEFAULT (datetime('now')),
    last_heartbeat_at TEXT DEFAULT (datetime('now')),

    -- Current work status
    status TEXT DEFAULT 'idle',       -- idle, working, stopping
    current_task TEXT,                -- e.g. "algorithm_42" or "polytope_12345"

    -- Stats
    tasks_completed INTEGER DEFAULT 0,
    errors INTEGER DEFAULT 0
);

CREATE INDEX idx_workers_type ON workers(worker_type);
CREATE INDEX idx_workers_heartbeat ON workers(last_heartbeat_at);
