from humanoidgen import ROOT_PATH
DP_root=f'{ROOT_PATH}/policy/Diffusion-Policy/'

import sys
sys.path.append('./') 
sys.path.insert(0, DP_root) 

import torch  
import os
import numpy as np
import hydra
from pathlib import Path
from collections import deque
import traceback
import humanoidgen.envs
import yaml
from datetime import datetime
import importlib
import dill
from argparse import ArgumentParser
from diffusion_policy.workspace.robotworkspace import RobotWorkspace
from humanoidgen.envs.example.table_scene import TableSetting
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.env_runner.dp_runner import DPRunner
from dataclasses import dataclass
from typing import List, Optional, Annotated, Union
import tyro
import gymnasium as gym
import pathlib
from termcolor import cprint
from omegaconf import OmegaConf

config_file = ROOT_PATH / "config/config_eval_dp.yml"
with open(config_file, "r") as file:
    run_config = yaml.safe_load(file)

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)

def class_decorator(env_name):
    args={
        "env_id": env_name,
    }
    # param to set
    run_config.update(args)

    if run_config["render_mode"] == "auto":
        if run_config["default"]["render_scene"]:
            run_config["render_mode"]="human"
        else:
            run_config["render_mode"]="rgb_array"

    env_kwargs = dict(
        obs_mode=run_config["obs_mode"],
        reward_mode=run_config["reward_mode"],
        control_mode=run_config["control_mode"],
        render_mode=run_config["render_mode"],
        sensor_configs=dict(shader_pack=run_config["shader"]),
        human_render_camera_configs=dict(shader_pack=run_config["shader"]),
        viewer_camera_configs=dict(shader_pack=run_config["shader"]),
        num_envs=run_config["num_envs"],
        sim_backend=run_config["sim_backend"],
        enable_shadow=run_config["enable_shadow"],
        parallel_in_single_scene=run_config["parallel_in_single_scene"],
    )
    extra_kwargs = dict(
        config_file_path=config_file
    )

    env_kwargs.update(extra_kwargs)
    env = gym.make(env_name,**env_kwargs)
    env = env.unwrapped
    return env

def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, f'{ROOT_PATH}/config/config_dp_camera.yml')
    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, 'r', encoding='utf-8') as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f'camera {camera_type} is not defined'
    return args[camera_type]

def get_policy(cfg):

    # cls = hydra.utils.get_class(cfg._target_)
    cls = RobotWorkspace
    workspace = cls(cfg)
    if cfg.model_weight_format == 'ckpt':
            ckpt_file = pathlib.Path(f'{cfg.checkpoint_path}/{cfg.task_name_base}/{cfg.task.name}/{cfg.checkpoint_num}.ckpt')
    elif cfg.model_weight_format == 'pth':
        ckpt_file = pathlib.Path(f'{cfg.checkpoint_path}/{cfg.task_name_base}/{cfg.task.name}/{cfg.checkpoint_num}.pth')
    assert ckpt_file.is_file(), f"ckpt file doesn't exist, {ckpt_file}"
    
    if ckpt_file.is_file():
        if cfg.model_weight_format == 'ckpt':
            cprint(f"Resuming from checkpoint {ckpt_file}", 'magenta')
            workspace.load_checkpoint(path=ckpt_file)
        elif cfg.model_weight_format == 'pth':
            cprint(f"Loading model from {ckpt_file}", 'magenta')
            checkpoint = torch.load(ckpt_file)
            workspace.model.load_state_dict(checkpoint['model_state_dict'])
            if workspace.ema_model is not None:
                workspace.ema_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise FileNotFoundError(f"Checkpoint file {ckpt_file} does not exist.")
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(cfg.eval.device)
    policy.to(device)
    policy.eval()

    return policy

class DP:
    def __init__(self, cfg):
        self.policy = get_policy(cfg)
        self.runner = DPRunner(output_dir=None,n_obs_steps=cfg.n_obs_steps,n_action_steps=cfg.n_action_steps)

    def update_obs(self, observation):
        self.runner.update_obs(observation)
    
    def get_action(self, observation=None):
        action = self.runner.get_action(self.policy, observation)
        return action

    def get_last_obs(self):
        return self.runner.obs[-1]

def test_policy(Demo_class, dp, test_seed, test_num=20):

    # set seed
    Demo_class.seed_everything(test_seed)

    global TASK
    Demo_class.suc = 0
    Demo_class.test_num =0

    now_id = 0
    success_num=0
    
    flag_suc = 0
    flag_fail = 0
    

    while now_id < test_num:

        success_result,failure_result=Demo_class.apply_dp(dp)
        if success_result:
            success_num += 1
            flag_suc += 1
        else:
            flag_fail += 1
        
        now_id += 1
        print(f"success rate: {success_num}/{now_id}\n")

    return success_num, now_id

def record_result(cfg, suc_num, all_num):

    suc_rate = suc_num / all_num
    print("suc_num:", suc_num)
    print("all_num:", all_num)
    print(f"Success rate: {suc_rate:.2f} ({suc_num}/{all_num})")
    file_path = f"{cfg.save_results_path}/{cfg.task_name_base}/"
    os.makedirs(file_path, exist_ok=True)
    file_path = file_path + cfg.task.name + ".txt"
    print(f'Data has been saved to {file_path}')

    # Create the file if it does not exist
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            f.write('')  # Create an empty file
        print(f"File '{file_path}' has been created.")
    else:
        print(f"File '{file_path}' already exists.")

    # Record the success rate and related information
    with open(file_path, "a", encoding="utf-8") as f:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f'Timestamp: {current_time}', file=f)
        print(f"task: {cfg.task_name_base}", file=f)
        print(f"expert_data_num: {cfg.expert_data_num}", file=f)
        print(f"test_num: {cfg.test_num}", file=f)
        print(f"checkpoint_num: {cfg.checkpoint_num}", file=f)
        print(f"seed: {cfg.test_seed}", file=f)
        print(f"suc_rate: {suc_rate}", file=f)
        print("\n", file=f)

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(ROOT_PATH/"policy/Diffusion-Policy/diffusion_policy/config"),
    config_name="robot_dp"
)
def main(cfg: OmegaConf):

    head_camera_type = cfg.head_camera_type # example: D435
    head_camera_cfg = get_camera_config(head_camera_type)
    cfg.task.image_shape = [3, head_camera_cfg['h'], head_camera_cfg['w']] # [C, H, W]
    cfg.task.shape_meta.obs.head_cam.shape = [3, head_camera_cfg['h'], head_camera_cfg['w']]
    OmegaConf.resolve(cfg)

    env_name = cfg.task_name_base # e.g. block_handover
    task:TableSetting = class_decorator(env_name)
    dp = DP(cfg)
    suc_num, all_num = test_policy(task, dp, cfg.test_seed, test_num=cfg.test_num)
    record_result(cfg,suc_num, all_num)

if __name__ == "__main__":
    main()
