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
from humanoidgen import ROOT_PATH
import datetime
import yaml

config_file = ROOT_PATH / "config/config_run_scene.yml"
with open(config_file, "r") as file:
    run_config = yaml.safe_load(file)

@dataclass
class Args:
    env_id: Annotated[Optional[str], tyro.conf.arg(aliases=["-e"])] = run_config["env_id"]

args = tyro.cli(Args)
print(f"Running environment: {args.env_id}")
run_config.update(vars(args))

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
add_kwargs = dict(
    config_file_path=config_file
)
env_kwargs.update(add_kwargs)

env = gym.make(run_config["env_id"],**env_kwargs)
env=env.unwrapped
env.start_task()
time_step = 0
while True:
    time_step += 1
    env.render()
    default_pose = env.agent.robot.get_qpos()[0, :38].cpu().numpy()
    qpos=default_pose
    obs, reward, terminated, truncated, info = env.step(qpos)
    if time_step % 100 == 0:
        env.end_task(save_file=f"run_scene/{run_config['env_id']}")
        env.start_task()