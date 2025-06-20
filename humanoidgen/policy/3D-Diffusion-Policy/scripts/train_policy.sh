echo "train_policy.sh: Start training script"

DEBUG=False
save_ckpt=True

alg_name=${1}
# task choices: See TASK.md
task_name=${2}
head_camera_type=${3}
expert_data_num=${4}
config_name=${alg_name}
addition_info=${5}
seed=${6}
exp_name=${task_name}
run_dir="3D-Diffusion-Policy/data/outputs/${exp_name}"

SCRIPT_DIR="$(pwd)"

# gpu_id=$(bash scripts/find_gpu.sh)
gpu_id=${7}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

# if [ $DEBUG = True ]; then
#     wandb_mode=offline
#     # wandb_mode=online
#     echo -e "\033[33mDebug mode!\033[0m"
#     echo -e "\033[33mDebug mode!\033[0m"
#     echo -e "\033[33mDebug mode!\033[0m"
# else
#     wandb_mode=online
#     echo -e "\033[33mTrain mode\033[0m"
# fi

cd 3D-Diffusion-Policy

export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}

python train.py --config-name=${config_name}.yaml \
                            task=whatever \
                            task.name=${task_name} \
                            task.dataset.zarr_path="${SCRIPT_DIR}/data/${task_name}.zarr" \
                            task.dataset.max_train_episodes=${expert_data_num} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            checkpoint.save_ckpt=${save_ckpt} \
                            expert_data_num=${expert_data_num} \
                            head_camera_type=${head_camera_type}