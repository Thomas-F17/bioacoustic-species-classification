#!/bin/bash
# =============================================================================
# iNatSounds Spectrogram Generation Pipeline
# =============================================================================
# 
# We created this pipeline script to produce the full dataset using an AWS EC2
# instance given that this would avoid us having any overhang from a data download 
# and transfer perspective. This is because the iNatSounds dataset is 133GBs of .wav
# audio files.
#
# This script reproduces the exact preprocessing from the Chasmai et al. linked below:
# Title: "The iNaturalist Sounds Dataset" (NeurIPS 2024)
# Paper: https://openreview.net/forum?id=QCY01LvyKm
# Code: https://github.com/cvl-umass/iNatSounds
# Data: https://github.com/visipedia/inat_sounds/tree/main/2024
#
# Usage:
# 1. Launch an EC2 c5.4xlarge Spot Instance in us-east-1 (Ubuntu 22.04+)
# 2. SCP this entire 'pipeline/' folder to the instance
# 3. Run: bash run_pipeline.sh
#
# Disk required: c.200 GB
# =============================================================================

set -euo pipefail  # Exit in case of any error

# =============================================================================
# Configuration
# =============================================================================
DATA_DIR="$HOME/data"
SPEC_DIR="$DATA_DIR/spectrograms"
NUM_WORKERS=16
S3_BUCKET="ml-inat-competition-datasets"
S3_PREFIX="sounds/2024"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Expected MD5 checksums (from the official dataset README)
declare -A CHECKSUMS=(
    ["train.tar.gz"]="181c5419220c8a9468a12b8c938469ec"
    ["val.tar.gz"]="828e4cf96e818228cb3536469cb3fb1c"
    ["test.tar.gz"]="082a2e96ca34f6c9f61767e33cbf3626"
    ["train.json.tar.gz"]="d0ae17ef52078194d450a83b5efe68a4"
    ["val.json.tar.gz"]="8e6e4b9628d43e36f1fea4621945403a"
    ["test.json.tar.gz"]="5482f112f05db4ba8576ba425d95d6a7"
)

# =============================================================================
# Helper functions
# =============================================================================
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
fail() { log "ERROR: $*"; exit 1; }

verify_checksum() {
    local file="$1"
    local expected="$2"
    log "Verifying checksum for $file..."
    local actual
    actual=$(md5sum "$file" | awk '{print $1}')
    if [ "$actual" != "$expected" ]; then
        fail "Checksum mismatch for $file: expected=$expected actual=$actual"
    fi
    log "  OK: $file checksum verified."
}

# =============================================================================
# Step 1: Install necessary dependencies
# =============================================================================
log "Step 1/6: Installing dependencies"
sudo apt-get update -qq
sudo apt-get install -y -qq python3-pip libsndfile1 awscli > /dev/null 2>&1
pip3 install librosa==0.10.2 numpy soundfile --break-system-packages -q
log "Dependencies installed."

# =============================================================================
# Step 2: Download data from S3 (original .wav data from the paper)
# =============================================================================
log "Step 2/6: Downloading data from S3"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

FILES=("train.tar.gz" "val.tar.gz" "test.tar.gz"
       "train.json.tar.gz" "val.json.tar.gz" "test.json.tar.gz")

for f in "${FILES[@]}"; do
    if [ -f "$f" ]; then
        log "  $f already exists, skipping download."
    else
        log "  Downloading $f..."
        aws s3 cp "s3://${S3_BUCKET}/${S3_PREFIX}/$f" . --no-sign-request
    fi
done
log "All downloads complete."

# =============================================================================
# Step 3: Verify checksums to make sure our data is correct
# =============================================================================
log "Step 3/6: Verifying checksums"
for f in "${FILES[@]}"; do
    verify_checksum "$f" "${CHECKSUMS[$f]}"
done
log "All checksums verified."

# =============================================================================
# Step 4: Extract archives
# =============================================================================
log "Step 4/6: Extracting archives"

for split in train val test; do
    if [ -d "$split" ]; then
        log "  $split/ already exists, skipping extraction."
    else
        log "  Extracting $split.tar.gz..."
        tar -xzf "$split.tar.gz"
    fi
done

for split in train val test; do
    if [ -f "$split.json" ]; then
        log "  $split.json already exists, skipping."
    else
        tar -xzf "$split.json.tar.gz"
    fi
done

# Free disk space after we've extracted the data
log "Removing tar archives to free space"
rm -f train.tar.gz val.tar.gz test.tar.gz
rm -f train.json.tar.gz val.json.tar.gz test.json.tar.gz
log "Extraction complete."

# =============================================================================
# Step 5: Generate spectrograms
# =============================================================================
log "Step 5/6: Generating spectrograms"

for split in train val test; do
    log "Processing $split split"
    python3 "$SCRIPT_DIR/generate_spectrograms.py" \
        --input_dir "$DATA_DIR/$split" \
        --output_dir "$SPEC_DIR/$split" \
        --num_workers "$NUM_WORKERS"
done
log "All spectrograms generated."

# =============================================================================
# Step 6: Validate output
# =============================================================================
log "Step 6/6: Validating output"
python3 "$SCRIPT_DIR/validate_spectrograms.py" \
    --spectrogram_dir "$SPEC_DIR" \
    --json_dir "$DATA_DIR"

log ""
log "=============================================="
log "  PIPELINE COMPLETE"
log "  Spectrograms: $SPEC_DIR"
log "  Annotations:  $DATA_DIR/*.json"
log "=============================================="
log ""
