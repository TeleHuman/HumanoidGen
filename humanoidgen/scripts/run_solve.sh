#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Script directory: $SCRIPT_DIR"

cd "$SCRIPT_DIR/.."

echo "Current directory: $(pwd)"

# Get environment id from the first argument, default to block_handover
ENV_ID=${1:-block_handover}
SOLVE_FOLDER=${2:-block_handover}

# Run the program
echo "Running program with env_id: $ENV_ID"
python process/run_solve.py -e "$ENV_ID" -s "$SOLVE_FOLDER"