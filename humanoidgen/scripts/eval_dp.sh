SCRIPT_DIR="$(pwd)"

cd "${SCRIPT_DIR}/process"

echo "Current directory: $(pwd)"

test_num=30
checkpoint_path="${SCRIPT_DIR}/policy/Diffusion-Policy/checkpoints"
save_results_path="${SCRIPT_DIR}/policy/Diffusion-Policy/data/eval"
head_camera_type="D435"

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
    150
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

config2_task=(
    close_box_easy
    close_box_hard
    open_drawer
    close_laptop_easy
)

for ((i=0; i<${#task_name[@]}; i++)); do
    for ((j=0; j<${#seed[@]}; j++)); do
        echo "Index: $i, Task name: ${task_name[$i]}"
                found=false
                for t in "${config2_task[@]}"; do
                    if [ "${task_name[$i]}" = "$t" ]; then
                        found=true
                        break
                    fi
                done

                if [ "$found" = true ]; then
                    python eval_policy_dp.py \
                        task=dp_task \
                        hydra.run.dir="${SCRIPT_DIR}/policy/Diffusion-Policy/data/eval/${task_name}/${checkpoint_folder_name}" \
                        task_name_base=${task_name[$i]} \
                        task.name=${checkpoint_folder_name[$i]} \
                        checkpoint_num=${checkpoint_num[$i]} \
                        expert_data_num=${expert_data_num[$i]} \
                        test_seed=${seed[$j]} \
                        save_results_path=${save_results_path}\
                        test_num=${test_num}\
                        head_camera_type=${head_camera_type}\
                        checkpoint_path=${checkpoint_path} \
                        horizon=28 \
                        n_obs_steps=5\
                        n_action_steps=28
                else
                    python eval_policy_dp.py \
                        task=dp_task \
                        hydra.run.dir="${SCRIPT_DIR}/policy/Diffusion-Policy/data/eval/${task_name}/${checkpoint_folder_name}" \
                        task_name_base=${task_name[$i]} \
                        task.name=${checkpoint_folder_name[$i]} \
                        checkpoint_num=${checkpoint_num[$i]} \
                        expert_data_num=${expert_data_num[$i]} \
                        test_seed=${seed[$j]} \
                        save_results_path=${save_results_path}\
                        test_num=${test_num}\
                        checkpoint_path=${checkpoint_path} \
                        head_camera_type=${head_camera_type}
                fi


    done
done