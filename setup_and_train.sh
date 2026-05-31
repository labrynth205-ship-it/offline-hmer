#!/bin/bash
# ============================================================
# Setup and Training Script for offline-hmer on Vast.ai
# 
# Usage:
#   bash setup_and_train.sh          # Train both BTTR and CAN
#   bash setup_and_train.sh bttr     # Train only BTTR
#   bash setup_and_train.sh can      # Train only CAN
#
# Instructions for Vast.ai:
#   1. Upload the entire project folder (offline-hmer) to your instance
#   2. Open a terminal in the project root directory
#   3. Run: bash setup_and_train.sh
# ============================================================

set -e

MODEL=${1:-all}

echo "========================================"
echo "  offline-hmer Setup & Training Script"
echo "  Model: $MODEL"
echo "========================================"

# 1. Install dependencies
echo ""
echo "[1/4] Installing Python dependencies..."
pip install -r requirements.txt

# 2. Verify dataset
echo ""
echo "[2/4] Verifying CROHME dataset..."
for dir in CROHME/train CROHME/2014 CROHME/2016 CROHME/2019; do
    if [ -d "$dir/img" ] && [ -f "$dir/caption.txt" ]; then
        echo "  ✓ $dir"
    else
        echo "  ✗ $dir - MISSING!"
        exit 1
    fi
done

# 3. Create checkpoint directory
echo ""
echo "[3/4] Creating checkpoint directory..."
mkdir -p checkpoints

# 4. Run training
echo ""
echo "[4/4] Starting training..."

if [ "$MODEL" = "bttr" ] || [ "$MODEL" = "all" ]; then
    echo ""
    echo "========================================"
    echo "  Training BTTR..."
    echo "========================================"
    cd models/bttr
    python train_bttr.py
    cd ../..
fi

if [ "$MODEL" = "can" ] || [ "$MODEL" = "all" ]; then
    echo ""
    echo "========================================"
    echo "  Training CAN..."
    echo "========================================"
    cd models/can
    python can_trainer.py
    cd ../..
fi

echo ""
echo "========================================"
echo "  Training complete!"
echo "  Checkpoints saved in: checkpoints/"
echo "========================================"
