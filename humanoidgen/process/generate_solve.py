import gymnasium as gym
import mani_skill.envs
import humanoidgen.envs
import tyro
from dataclasses import dataclass
from typing import List, Optional, Annotated, Union
import numpy as np
import time
from enum import Enum
from mani_skill.utils.wrappers.record import RecordEpisode
from humanoidgen.motion_planning.h1_2.utils import images_to_video
from humanoidgen import ROOT_PATH
from datetime import datetime
import yaml
from humanoidgen.llm.api_util import APIUTIL
from humanoidgen.motion_planning.h1_2.solution.generated.solver_env import *
import importlib.util
import textwrap
from humanoidgen.envs.example.table_scene import TableSetting
import re
from pathlib import Path
import json
import cv2

config_file = ROOT_PATH / "config/config_generate_solve.yml"
with open(config_file, "r") as file:
    run_config = yaml.safe_load(file)

task_file = ROOT_PATH / "config/config_task.yml"
with open(task_file, "r") as file:
    task_config = yaml.safe_load(file)

@dataclass
class Args:
    env_id: Annotated[Optional[str], tyro.conf.arg(aliases=["-e"])] = run_config["env_id"]

args=tyro.cli(Args)
run_config.update(vars(args))

env_name = run_config["env_id"]
task = task_config[env_name]

num_generations = run_config["exe_num"]
img_prompt = run_config["img_prompt"]

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
env.init_task_scene()

assets_status = env.get_actors_info()
# qpose , l_hand_base_link, r_hand_base_link, l_hand_status,r_hand_status
robot_info_all,robot_status = env.get_robot_info()
print("assets_status:",assets_status)
print("robot_status:",robot_status)

ds = APIUTIL()
task_filder_list = []
images=[]
original_image = env.render_rgb_array().cpu().numpy()[0]
downsampling_img= cv2.resize(original_image, (512, 512), interpolation=cv2.INTER_LINEAR)
images.append(downsampling_img)

if img_prompt:
    prompt_img = downsampling_img[..., ::-1]
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_save_path = str(ROOT_PATH/"imgs/prompt_img"/f"{env_name}_prompt_img_{current_time}.png")
    cv2.imwrite(image_save_path, prompt_img)


visualize_img = False
if visualize_img ==True:
    prompt_img = images[0][..., ::-1]
    cv2.imshow("Head Camera Image", prompt_img)
    cv2.waitKey(0)

for i in range(num_generations):
    task_filder=ds.generate_solve(env_name=env_name, task=task, assets_status=assets_status, robot_status=robot_status,images=images)
    task_filder_list.append(task_filder)

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
generate_info_path = str(ROOT_PATH/"motion_planning/h1_2/solution/generated/run_log/"/f"{env_name}_{current_time}.json")
with open(generate_info_path, "w") as file:
    json.dump(task_filder_list, file, indent=4)
print(f"All task_filder have been saved to {generate_info_path}.")