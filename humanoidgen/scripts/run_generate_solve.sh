#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Script directory: $SCRIPT_DIR"

cd "$SCRIPT_DIR/.."

echo "Current directory: $(pwd)"

# Get environment id from the first argument, default to block_handover
ENV_ID=${1:-block_handover}

# Number of parallel processes (default 5, can be set by the second argument)
N=${2:-1}

for ((i=0; i<N; i++)); do
    echo "Starting process $((i+1)) with env_id: $ENV_ID"
    python process/generate_solve.py -e "$ENV_ID" &
    sleep 2
done

wait
echo "All processes finished."