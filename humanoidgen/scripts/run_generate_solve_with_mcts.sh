#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Script directory: $SCRIPT_DIR"

cd "$SCRIPT_DIR/.."

echo "Current directory: $(pwd)"

# Get environment id from the first argument, default to blocks_stack_hard_mcts
ENV_ID=${1:-blocks_stack_hard_mcts}

# Number of parallel processes (default 5, can be set by the second argument)
N=${2:-5}

for ((i=0; i<N; i++)); do
    echo "Starting process $((i+1)) with env_id: $ENV_ID"
    python process/generate_solve_with_mcts.py -e "$ENV_ID" &
    sleep 2
done

wait
echo "All processes finished."