#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Script directory: $SCRIPT_DIR"

cd "$SCRIPT_DIR/.."

echo "Current directory: $(pwd)"

# Get environment id from the first argument, default to open_box_easy
ENV_ID=${1:-open_box_easy}

# Run the program
echo "Running program with env_id: $ENV_ID"
python process/run_scene.py -e "$ENV_ID"