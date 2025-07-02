#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Script directory: $SCRIPT_DIR"

cd "$SCRIPT_DIR/.."

echo "Current directory: $(pwd)"

ENV_ID=${1:-blocks_stack_easy}
SOLVE_FOLDER=${2:-blocks_stack_easy}
RENDER=${3:-True}

# Run the program
echo "Running program with env_id: $ENV_ID"
python process/run_solve.py -env "$ENV_ID" -solve "$SOLVE_FOLDER" -render "$RENDER"