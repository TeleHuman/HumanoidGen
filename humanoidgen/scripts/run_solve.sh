#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Script directory: $SCRIPT_DIR"

cd "$SCRIPT_DIR/.."

echo "Current directory: $(pwd)"

ENV_ID=${1:-block_handover}
SOLVE_FOLDER=${2:-block_handover}
RENDER=${3:-False}

# Run the program
echo "Running program with env_id: $ENV_ID"
python process/run_solve.py -env "$ENV_ID" -solve "$SOLVE_FOLDER" -render "$RENDER"