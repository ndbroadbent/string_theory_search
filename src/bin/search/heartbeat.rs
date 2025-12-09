//! Heartbeat thread for algorithm locking

use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use string_theory::db;

/// Heartbeat interval in seconds
pub const HEARTBEAT_INTERVAL_SECS: u64 = 20;

/// Heartbeat thread handle and control
pub struct HeartbeatThread {
    handle: JoinHandle<()>,
    stop_flag: Arc<AtomicBool>,
    algo_id: Arc<AtomicI64>,
}

impl HeartbeatThread {
    /// Start a new heartbeat thread
    pub fn start(db_path: String, my_pid: i32) -> Self {
        let algo_id = Arc::new(AtomicI64::new(0));
        let stop_flag = Arc::new(AtomicBool::new(false));

        let algo_id_clone = algo_id.clone();
        let stop_flag_clone = stop_flag.clone();

        let handle = thread::spawn(move || {
            heartbeat_loop(db_path, algo_id_clone, my_pid, stop_flag_clone);
        });

        Self {
            handle,
            stop_flag,
            algo_id,
        }
    }

    /// Set the current algorithm ID being worked on
    pub fn set_algorithm(&self, id: i64) {
        self.algo_id.store(id, Ordering::Relaxed);
    }

    /// Clear the current algorithm (done working on it)
    pub fn clear_algorithm(&self) {
        self.algo_id.store(0, Ordering::Relaxed);
    }

    /// Stop the heartbeat thread and wait for it to finish
    pub fn stop(self) {
        self.stop_flag.store(true, Ordering::Relaxed);
        let _ = self.handle.join();
    }
}

fn heartbeat_loop(
    db_path: String,
    algo_id: Arc<AtomicI64>,
    my_pid: i32,
    stop_flag: Arc<AtomicBool>,
) {
    loop {
        thread::sleep(Duration::from_secs(HEARTBEAT_INTERVAL_SECS));

        if stop_flag.load(Ordering::Relaxed) {
            break;
        }

        let current_algo_id = algo_id.load(Ordering::Relaxed);
        if current_algo_id <= 0 {
            continue;
        }

        // Open a fresh connection for heartbeat (thread-safe)
        if let Ok(conn) = rusqlite::Connection::open(&db_path) {
            if let Err(e) = db::update_heartbeat(&conn, current_algo_id, my_pid) {
                eprintln!("Warning: Failed to update heartbeat: {}", e);
            }
        }
    }
}
