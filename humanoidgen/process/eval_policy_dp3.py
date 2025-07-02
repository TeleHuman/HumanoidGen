import sys
sys.path.insert(0, './policy/3D-Diffusion-Policy/3D-Diffusion-Policy')
sys.path.append('./')

import torch  
import sapien.core as sapien
import traceback
import os
import numpy as np
from humanoidgen.envs import *
import hydra
import pathlib
from dp3_policy import *
import yaml
from datetime import datetime
import importlib
import gymnasium as gym
import mani_skill.envs
import humanoidgen.envs
import tyro
from dataclasses import dataclass
from typing import List, Optional, Annotated, Union
import numpy as np
import time
from humanoidgen.motion_planning.h1_2.utils import images_to_video
from humanoidgen import ROOT_PATH
from datetime import datetime
import yaml
from humanoidgen.llm.api_util import APIUTIL
from humanoidgen.motion_planning.h1_2.solution.generated.solver_env import *
from humanoidgen.envs.example.table_scene import TableSetting
import importlib.util
import textwrap
import re

config_file = ROOT_PATH / "config/config_eval_dp3.yml"
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
    with open(ROOT_PATH/"configure/configure.yml", "r") as file:
        camera_config = yaml.safe_load(file)
    camera_config = camera_config["camera_config"][camera_type]
    return camera_config

def load_model(model_path):
    model = torch.load(model_path)
    model.eval() 
    return model

def record_result(cfg, suc_num, all_num,results):

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
        print(f"results: {results}", file=f)
        print("\n", file=f)


@hydra.main(
    version_base=None,
    config_path=str(ROOT_PATH/"policy/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/config"),
    config_name="robot_dp3"
)
def main(cfg):

    env_name = cfg.task_name_base # e.g. block_handover
    task:TableSetting = class_decorator(env_name)
    test_num = cfg.test_num
    print(f"!!!!!!! cfg.task.name={cfg.task.name}")

    checkpoint_num = cfg.checkpoint_num
    dp3 = DP3(cfg, checkpoint_num)

    suc_num, all_num,results = test_policy(task, dp3, cfg.test_seed, test_num=test_num)

    record_result(cfg, suc_num, all_num,results)
    

def test_policy(Demo_class, dp3, test_seed, test_num=20):

    # set seed
    Demo_class.seed_everything(test_seed)

    global TASK

    Demo_class.suc = 0
    Demo_class.test_num =0

    now_id = 0
    success_num=0

    flag_suc = 0
    flag_fail = 0
    
    results=[]

    while now_id < test_num:

        success_result=Demo_class.apply_dp3(dp3)
        if success_result:
            success_num += 1
            flag_suc += 1
        else:
            flag_fail += 1
        
        # Record the result
        results.append(success_result)

        now_id += 1
        print(f"success rate: {success_num}/{now_id}\n")

    return success_num, now_id,results

if __name__ == "__main__":
    main()