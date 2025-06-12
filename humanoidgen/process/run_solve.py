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
from humanoidgen.motion_planning.h1_2.solution.generated.solver_env import *
import importlib.util
import textwrap
import re

config_file = ROOT_PATH / "config/config_run_solve.yml"
with open(config_file, "r") as file:
    run_config = yaml.safe_load(file)

@dataclass
class Args:
    env_id: Annotated[Optional[str], tyro.conf.arg(aliases=["-env"])] = run_config["env_id"]
    solve_folder: Annotated[Optional[str], tyro.conf.arg(aliases=["-solve"])] = run_config["solve_folder"]
    render_scene: Annotated[Optional[bool], tyro.conf.arg(aliases=["-render"])] = run_config["default"]["render_scene"]

args = tyro.cli(Args)
print(f"Running environment: {args.env_id}")
run_config["default"]["render_scene"]=args.render_scene
run_config.update(vars(args))

solve_folder = run_config["solve_folder"]
solve_folder_path= ROOT_PATH / "motion_planning" / "h1_2" / "solution" /"generated"/ solve_folder
debug_key_frame = run_config["debug_key_frame"]
show_key_points = run_config["show_key_points"]
max_episodes = run_config["max_episodes"]
exe_until_max_episodes_success = run_config["exe_until_max_episodes_success"]
max_init_scene_num = run_config["max_init_scene_num"]

if run_config["render_mode"] == "auto":
    if run_config["default"]["render_scene"]:
        run_config["render_mode"]="human"
    else:
        run_config["render_mode"]="rgb_array"

# Originally, the config file is used to set up the environment.
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
env = gym.make(args.env_id,**env_kwargs)
env=env.unwrapped

def create_planner(env):
    # initialize the scene
    start_result = False
    for i in range(max_init_scene_num):
        start_result=env.start_task()
        if start_result:
            break
    if not start_result:
        raise RuntimeError(f"Failed to start task after {max_init_scene_num} attempts.")

    # initialize the planner
    planner = HumanoidMotionPlanner(
        env,
        debug=False,
        vis=True,
        base_pose=env.agent.robot.pose,
        visualize_target_grasp_pose=True,
        print_env_info=False,
        show_key_points=show_key_points,
        debug_key_frame=debug_key_frame,
        # use_point_cloud=False,
        # use_obj_point_cloud=False,
    )
    return planner

# 4. run step
def run_step(solve_folder_path,file_name,planner):
    spec = importlib.util.spec_from_file_location(file_name.removesuffix(".py"), str(solve_folder_path/file_name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.step(planner)

def count_step_files(solve_folder_path):
    files = os.listdir(solve_folder_path)
    step_files = [f for f in files if re.match(r"step\d+\.py$", f)]
    return len(step_files)


start_time = time.time() 
execution_id = 0
while True:
    planner=create_planner(env)
    for i in range(count_step_files(solve_folder_path)):
        run_step(solve_folder_path,f"step{i}.py",planner)
    env.end_task(save_file=f"run_solve/{run_config['env_id']}")
    del planner
    execution_id+=1
    if exe_until_max_episodes_success == True:
        if env.success_count >= max_episodes:
            break
    else:
        if execution_id >= max_episodes:
            break

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total execution time: {elapsed_time:.2f} seconds")

env.end_run()