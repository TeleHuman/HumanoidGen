gpu_id="0"

POLICE="$1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Script directory: $SCRIPT_DIR"

cd "$SCRIPT_DIR/.."

if [ "$POLICE" = "dp" ]; then
    cd policy/Diffusion-Policy
elif [ "$POLICE" = "dp3" ]; then
    cd policy/3D-Diffusion-Policy
else
    echo "Unknown policy type: $POLICE"
    exit 1
fi

echo "Current directory: $(pwd)"

for dataset_name in blocks_stack_easy_20250703_031929_dp
do
    for expert_epis_num in 100
    do

        if [ "$POLICE" = "dp3" ]; then
            bash train.sh ${dataset_name} D435 ${expert_epis_num} 0 ${gpu_id}
        elif [ "$POLICE" = "dp" ]; then
            sh train.sh ${dataset_name} D435 ${expert_epis_num} 0 ${gpu_id}
        else
            echo "Unknown policy type: $POLICE"
            exit 1
        fi
    done
done