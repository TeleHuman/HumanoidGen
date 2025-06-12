#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Script directory: $SCRIPT_DIR"

cd "$SCRIPT_DIR/.."

echo "Current directory: $(pwd)"

# Get environment id from the first argument, default to block_handover
ENV_ID=${1:-block_handover}
RENDER=${2:-False}
# Run the program
echo "Running program with env_id: $ENV_ID"
python process/run_scene.py -env "$ENV_ID" -render "$RENDER"