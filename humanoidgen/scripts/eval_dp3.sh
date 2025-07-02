SCRIPT_DIR="$(pwd)"

cd "${SCRIPT_DIR}/process"

echo "Current directory: $(pwd)"

test_num=5
checkpoint_path="${SCRIPT_DIR}/policy/3D-Diffusion-Policy/checkpoints"
save_results_path="${SCRIPT_DIR}/policy/3D-Diffusion-Policy/data/eval"
# base folder path humanoidgen/policy/3D-Diffusion-Policy/checkpoints
task_name=(
    blocks_stack_easy
)

checkpoint_folder_name=(
    blocks_stack_easy_20250703_031929_dp3
)
expert_data_num=(
    100
)

checkpoint_num=(
    100
)



if [ "${#task_name[@]}" -ne "${#checkpoint_folder_name[@]}" ] || [ "${#task_name[@]}" -ne "${#checkpoint_num[@]}" ] || [ "${#task_name[@]}" -ne "${#expert_data_num[@]}" ]; then
    echo "Error: The lengths of task_name, checkpoint_folder_name, and checkpoint_num are not the same!"
    exit 1
else
    echo "All lengths are equal."
fi

seed=(
    0
    1
    2
)

for ((i=0; i<${#task_name[@]}; i++)); do
    for ((j=0; j<${#seed[@]}; j++)); do
        echo "Index: $i, Task name: ${task_name[$i]}"
                python eval_policy_dp3.py \
                    task=dp3_task \
                    hydra.run.dir="${SCRIPT_DIR}/policy/3D-Diffusion-Policy/data/eval/${task_name}/${checkpoint_folder_name}" \
                    task_name_base=${task_name[$i]} \
                    task.name=${checkpoint_folder_name[$i]} \
                    checkpoint_num=${checkpoint_num[$i]} \
                    expert_data_num=${expert_data_num} \
                    test_seed=${seed[$j]} \
                    save_results_path=${save_results_path}\
                    checkpoint_path=${checkpoint_path} \
                    test_num=${test_num}
    done
done