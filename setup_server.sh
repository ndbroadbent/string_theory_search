#!/bin/bash
# Setup script for string_theory on remote server
# Run with: ./setup_server.sh [user@host]
# Can be stopped and resumed - all steps are idempotent

set -e

SERVER="${1:-root@10.5.7.33}"
REMOTE_DIR="/root/string_theory"
DATA_DIR="/data/polytopes"
REPO_URL="https://github.com/ndbroadbent/string_theory_compactions.git"
NUM_WORKERS=12

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "Setting up string_theory on $SERVER"

# Step 1: Install Rust if not present
log "Step 1: Checking/Installing Rust..."
ssh "$SERVER" 'command -v cargo >/dev/null 2>&1 || {
    echo "Installing Rust..."
    curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
}
source "$HOME/.cargo/env" 2>/dev/null || true'

# Step 2: Install system dependencies (skip libssl-dev if problematic)
log "Step 2: Installing system dependencies..."
ssh "$SERVER" 'apt-get update && apt-get install -y build-essential pkg-config python3-venv python3-dev git || true'

# Step 3: Clone or update repo
log "Step 3: Cloning/updating repository..."
ssh "$SERVER" "
    if [ -d '$REMOTE_DIR/.git' ]; then
        echo 'Updating existing repo...'
        cd '$REMOTE_DIR' && git pull
    else
        echo 'Cloning fresh...'
        rm -rf '$REMOTE_DIR'
        git clone '$REPO_URL' '$REMOTE_DIR'
    fi
    mkdir -p '$REMOTE_DIR/results'
"

# Step 4: Setup Python venv
log "Step 4: Setting up Python venv..."
ssh "$SERVER" "cd $REMOTE_DIR && {
    [ -d venv ] || python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install numpy jax jaxlib pyarrow requests
}"

# Step 5: Install PALP
log "Step 5: Installing PALP..."
ssh "$SERVER" "
    if [ ! -f /root/palp_source/poly.x ]; then
        echo 'Cloning and building PALP...'
        cd /root && rm -rf palp_source
        git clone https://gitlab.com/stringstuwien/PALP.git palp_source
        cd palp_source && make -j8
    else
        echo 'PALP already installed.'
    fi
"

# Step 6: Download polytopes if not present
log "Step 6: Downloading polytope data..."
ssh "$SERVER" "cd $REMOTE_DIR && source venv/bin/activate && {
    if [ ! -f $DATA_DIR/polytopes_medium.json ]; then
        echo 'Downloading polytopes (vertices 5-8)...'
        python3 download_all_polytopes.py --output-dir $DATA_DIR/parquet --min-vertices 5 --max-vertices 8 --convert --max-polytopes 2000000
        mv polytopes_full.json $DATA_DIR/polytopes_medium.json 2>/dev/null || true
    else
        echo 'Polytopes already downloaded.'
    fi
    # Symlink to project dir
    ln -sf $DATA_DIR/polytopes_medium.json polytopes_medium.json 2>/dev/null || true
}"

# Step 7: Build the project
log "Step 7: Building project..."
ssh "$SERVER" "source \$HOME/.cargo/env && cd $REMOTE_DIR && cargo build --release --bin real_physics"

log ""
log "=========================================="
log "Setup complete!"
log "=========================================="
log ""
log "To run $NUM_WORKERS parallel workers:"
log "  ssh $SERVER 'cd $REMOTE_DIR && ./run_workers.sh $NUM_WORKERS'"
log ""
log "To monitor:"
log "  ssh $SERVER 'tail -f $REMOTE_DIR/logs/worker_*.log'"
log ""
log "To check results:"
log "  ssh $SERVER 'ls -la $REMOTE_DIR/results/run_*/'"
log ""
log "To find best results:"
log "  ssh $SERVER 'ls -la $REMOTE_DIR/results/run_*/fit* | sort -t/ -k5 -r | head -20'"
