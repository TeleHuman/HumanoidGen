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
# import datetime
from datetime import datetime
# from humanoidgen import ROOT_PATH
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


@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "PushCube-v1"
    """The environment ID of the task you want to simulate"""

    obs_mode: Annotated[str, tyro.conf.arg(aliases=["-o"])] = "none"
    """Observation mode"""

    robot_uids: Annotated[Optional[str], tyro.conf.arg(aliases=["-r"])] = "unitree_h1_2_upper_body_with_head_camera"
    """Robot UID(s) to use. Can be a comma separated list of UIDs or empty string to have no agents. If not given then defaults to the environments default robot"""

    sim_backend: Annotated[str, tyro.conf.arg(aliases=["-b"])] = "auto"
    """Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'"""

    reward_mode: Optional[str] = None
    """Reward mode"""

    num_envs: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 1
    """Number of environments to run."""

    control_mode: Annotated[Optional[str], tyro.conf.arg(aliases=["-c"])] = None
    """Control mode"""

    render_mode: str = "human"
    """Render mode: human, rgb_array, depth_array, segmentation_array"""

    shader: str = "default"
    """Change shader used for all cameras in the environment for rendering. Default is 'minimal' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer"""

    record_dir: Optional[str] = None
    """Directory to save recordings"""

    pause: Annotated[bool, tyro.conf.arg(aliases=["-p"])] = False
    """If using human render mode, auto pauses the simulation upon loading"""

    quiet: bool = False
    """Disable verbose output."""

    seed: Annotated[Optional[Union[int, List[int]]], tyro.conf.arg(aliases=["-s"])] = None
    """Seed(s) for random actions and simulator. Can be a single integer or a list of integers. Default is None (no seeds)"""

    use_render: bool = True
    """Use render for the environment"""

    step_scene: bool = True
    """Use render for the environment"""

    save_video: bool = False
    """Use render for the environment"""

# pour_cubes
# put_can_into_bowl
## Parse arguments
args=tyro.cli(Args)
with open(ROOT_PATH/"configure/configure.yml", "r") as file:
    config = yaml.safe_load(file)
env_name = config["solve_generate"]["env_name"]
if env_name in config["solve_generate"]:
    task = config["solve_generate"][env_name]
else:
    task = config["solve_generate"]["task"]

num_generations = config["solve_generate"]["num_generations"]
api_model= config["api_config"]["llm_model_name"]
use_muti_model=config["api_config"]["multi_model"]
# param to set
args.env_id= env_name #"disrupt_pyramid" #"cube_in_row", "assets", "dual_arm_apple_can_into_bowl", args.env_id= # args.env_id="RoboCasaKitchen-v1" # "UnitreeH1_2PlaceAppleInBowl-v1"
args.save_video=False
parallel_in_single_scene = args.render_mode == "human"

if args.render_mode == "human" and args.obs_mode in ["sensor_data", "rgb", "rgbd", "depth", "point_cloud"]:
    print("Disabling parallel single scene/GUI render as observation mode is a visual one. Change observation mode to state or state_dict to see a parallel env render")
    parallel_in_single_scene = False
if args.render_mode == "human" and args.num_envs == 1:
    parallel_in_single_scene = False

args.control_mode="pd_joint_pos" # "pd_joint_pos" # args.control_mode="pd_joint_delta_pos" # "pd_joint_pos"
# args.sim_backend="gpu"

env_kwargs = dict(
    obs_mode=args.obs_mode,
    reward_mode=args.reward_mode,
    control_mode=args.control_mode,
    render_mode=args.render_mode,
    sensor_configs=dict(shader_pack=args.shader),
    human_render_camera_configs=dict(shader_pack=args.shader),
    viewer_camera_configs=dict(shader_pack=args.shader),
    num_envs=args.num_envs,
    sim_backend=args.sim_backend,
    enable_shadow=True,
    parallel_in_single_scene=parallel_in_single_scene,
)

# if args.robot_uids is not None:
#     env_kwargs["robot_uids"] = tuple(args.robot_uids.split(","))
env:TableSetting = gym.make(args.env_id,**env_kwargs)
obs, _ = env.reset(seed=0)

init_scene_step =40
for i in range(init_scene_step):
    env.render()
    defalt_pose = env.agent.robot.get_qpos()[0, :38].cpu().numpy()
    obs, reward, terminated, truncated, info = env.step(defalt_pose)
    # time.sleep(0.7)

assets_status = env.get_actors_info()
# qpose , l_hand_base_link, r_hand_base_link, l_hand_status,r_hand_status
robot_info_all,robot_status = env.get_robot_info()
print("assets_status:",assets_status)
print("robot_status:",robot_status)

ds = APIUTIL(model=api_model)
task_filder_list = []
images=[]
# images.append(env.render_rgb_array())
original_image = env.render_rgb_array().numpy()[0]
downsampling_img= cv2.resize(original_image, (512, 512), interpolation=cv2.INTER_LINEAR)
images.append(downsampling_img)

if use_muti_model:
    #  保存传入prompt的图片
    prompt_img = downsampling_img[..., ::-1]
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_save_path = str(ROOT_PATH/"imgs/prompt_img"/f"{env_name}_prompt_img_{current_time}.png")
    cv2.imwrite(image_save_path, prompt_img)


visualize_img = False
if visualize_img ==True:
    # prompt_img = images[0].numpy()[..., ::-1][0]
    prompt_img = images[0][..., ::-1]
    # downsampling_img= cv2.resize(prompt_img, (512, 512), interpolation=cv2.INTER_LINEAR)
    # cv2.imshow("Head Camera Image", downsampling_img)
    cv2.imshow("Head Camera Image", prompt_img)
    cv2.waitKey(0)

for i in range(num_generations):
    task_filder=ds.generate_solve(env_name=env_name, task=task, assets_status=assets_status, robot_status=robot_status,images=images)
    task_filder_list.append(task_filder)

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
generate_info_path = str(ROOT_PATH/"motion_planning/h1_2/solution/generated"/f"{env_name}_{current_time}.json")
with open(generate_info_path, "w") as file:
    json.dump(task_filder_list, file, indent=4)
print(f"All task_filder have been saved to {generate_info_path}.")