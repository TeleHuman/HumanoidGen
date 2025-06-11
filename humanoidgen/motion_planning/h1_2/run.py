import gymnasium as gym
import mani_skill.envs
import humanoidgen.envs
import tyro
from dataclasses import dataclass
from typing import List, Optional, Annotated, Union
import numpy as np
import time
from humanoidgen.motion_planning.h1_2.solution import solvePutAppleIntoBowl,solvePutAppleIntoBowlExample
from enum import Enum
from mani_skill.utils.wrappers.record import RecordEpisode
MP_SOLUTIONS = {
    "PutAppleIntoBowl-v1": solvePutAppleIntoBowlExample,
    "PutAppleIntoBowl-latest": solvePutAppleIntoBowl,

}

class ProcessMode(Enum):
    STATIC = "static"
    ACTION = "action"
    SOLVER = "solver"
    EXAMPLE = "example"

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

args=tyro.cli(Args)
# args.env_id="RoboCasaKitchen-v1" # "UnitreeH1_2PlaceAppleInBowl-v1"
args.env_id="UnitreeH1_2PlaceAppleInBowl-v1" # "UnitreeH1_2PlaceAppleInBowl-v1"

parallel_in_single_scene = args.render_mode == "human"
if args.render_mode == "human" and args.obs_mode in ["sensor_data", "rgb", "rgbd", "depth", "point_cloud"]:
    print("Disabling parallel single scene/GUI render as observation mode is a visual one. Change observation mode to state or state_dict to see a parallel env render")
    parallel_in_single_scene = False
if args.render_mode == "human" and args.num_envs == 1:
    parallel_in_single_scene = False

args.control_mode="pd_joint_pos" # "pd_joint_pos"
# args.control_mode="pd_joint_delta_pos" # "pd_joint_pos"
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
env = gym.make(args.env_id,**env_kwargs)
# env = RecordEpisode(env, output_dir="eval_videos", save_video=True,max_steps_per_video=10000)
# env = gym.make("UnitreeH1_2PlaceAppleInBowl-v1",render_mode="human")

process_mode = ProcessMode.SOLVER
obs, _ = env.reset(seed=0) # will be called in solve()

if process_mode == ProcessMode.STATIC:
    pass
elif process_mode == ProcessMode.ACTION:
    pass
elif process_mode == ProcessMode.EXAMPLE:
    solve=MP_SOLUTIONS["PutAppleIntoBowl-v1"]

elif process_mode == ProcessMode.SOLVER:
    solve=MP_SOLUTIONS["PutAppleIntoBowl-latest"]

while True:
# for i in range(1):

    if process_mode == ProcessMode.STATIC:
        env.render()
    elif process_mode == ProcessMode.ACTION:
        # action= np.array([0]+[0]*37)
        # action= np.array([1]*1+[0]*23+[-1]*1)
        # action= np.array([0.0]*38)
        # action = env.action_space.sample() if env.action_space is not None else None
        # action = [0]*38
        action= np.array([0.0]*14+[1]*24)
        obs, reward, terminated, truncated, info = env.step(action)
        print(env.agent.robot.get_qpos())
    elif process_mode == ProcessMode.SOLVER:
        solve(env=env,seed=0,vis=True)
        env.render()
    elif process_mode == ProcessMode.EXAMPLE:
        solve(env=env,seed=0,vis=True)
        env.render()
    

    # for i in range(100):
    #     env.render()
    #     time.sleep(0.1)

    # action = env.action_space.sample() if env.action_space is not None else None
    # print("len(action):",len(action))
    # print("type:",type(action))
    # action = [0]*38
    # action= np.array([0.0]*14+[1]*24)
    # action= np.array([0]+[0]*37)
    # action= np.array([1]*1+[0]*23+[-1]*1)
    # action= np.array([0.0]*38)

    # print("action:",action)
    # obs, reward, terminated, truncated, info = env.step(action)
    # print(env.agent.robot.get_qpos())
    # print("action space:",env.action_space)
    # env.render()
    # time.sleep(0.1)
    # if args.render_mode is None or args.render_mode != "human":
    #     if (terminated | truncated).any():
    #         break