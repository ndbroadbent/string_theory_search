//! String Theory Landscape Explorer
//!
//! Support crate for the cyrus-ga dashboard pipeline: SQLite schema setup and
//! geometric heuristics computed per polytope. The GA itself lives in the
//! cyrus repository (cyrus-ga); this crate only computes polytope heuristics
//! into the shared dashboard database.

pub mod db;
pub mod heuristics;
