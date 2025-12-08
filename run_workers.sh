#!/bin/bash
# Run multiple parallel string theory search workers
# Usage: ./run_workers.sh [NUM_WORKERS]

NUM_WORKERS="${1:-12}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"

mkdir -p "$LOG_DIR"
mkdir -p "$SCRIPT_DIR/results"

echo "═══════════════════════════════════════════════════════════════"
echo "  STRING THEORY LANDSCAPE EXPLORER - PARALLEL MODE"
echo "  Starting $NUM_WORKERS workers"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Kill any existing workers
pkill -f "real_physics" 2>/dev/null && echo "Stopped existing workers" && sleep 2

# Activate venv if it exists
if [ -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
fi

# Start workers
for i in $(seq 0 $((NUM_WORKERS - 1))); do
    LOG_FILE="$LOG_DIR/worker_$(printf '%02d' $i).log"
    echo "Starting worker $i -> $LOG_FILE"
    nohup "$SCRIPT_DIR/target/release/real_physics" "$i" > "$LOG_FILE" 2>&1 &
    sleep 0.5  # Stagger starts slightly
done

echo ""
echo "All $NUM_WORKERS workers started!"
echo ""
echo "Monitor with:"
echo "  tail -f $LOG_DIR/worker_*.log"
echo ""
echo "Check status with:"
echo "  ps aux | grep real_physics"
echo ""
echo "Find best results with:"
echo "  ls -la $SCRIPT_DIR/results/run_*/fit* | sort -t/ -k6 -r | head -20"
echo ""
echo "Stop all workers with:"
echo "  pkill -f real_physics"
