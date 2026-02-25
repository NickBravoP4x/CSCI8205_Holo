#!/bin/bash
# Start inference service with logging to file

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/inference_service.log"

echo "Starting inference service with logging..."
echo "Log file: $LOG_FILE"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate holoviewer_cuda

# Run service and tee output to both terminal and log file
python inference_service.py 2>&1 | tee "$LOG_FILE"
