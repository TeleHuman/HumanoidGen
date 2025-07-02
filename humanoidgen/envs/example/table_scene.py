import copy
import os
from typing import Any, Dict

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat
from sapien.physx import PhysxMaterial
from humanoidgen.agents.robots.unitree_h1_2.h1_2_upper_body import (
    UnitreeH1_2UpperBodyWithHeadCamera,
)
from mani_skill.agents.robots.unitree_g1.g1_upper_body import (
    UnitreeG1UpperBodyWithHeadCamera,
)
import open3d as o3d
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.kitchen_counter import KitchenCounterSceneBuilder
from mani_skill.utils.scene_builder.robocasa.scene_builder import RoboCasaSceneBuilder
from humanoidgen.scene_builder.table.scene_builder import TableSceneBuilder

from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SceneConfig, SimConfig
# from mani_skill import ASSET_DIR
from mani_skill.utils import common, io_utils, sapien_utils
from mani_skill.utils.scene_builder.robocasa.objects.objects import MJCFObject
from humanoidgen import ROOT_PATH
from humanoidgen import ASSET_DIR, HGENSIM_ASSET_DIR,BRIDGE_DATASET_ASSET_PATH,ROBOCASA_ASSET_DIR,REPLICA_ASSET_DIR
from mani_skill.examples.motionplanning.panda.utils import get_actor_obb
from humanoidgen.tool.utils import get_pose_in_env,out_put_nvml_memory_info
import yaml
import datetime
from humanoidgen.tool.utils import images_to_video
from typing import Union
import json
from humanoidgen.agents.objects.articulated_object import ArticulatedObject
import transforms3d as t3d
from scipy.spatial.transform import Rotation
import pytorch3d.ops as torch3d_ops
import pickle
import time
import re


class StateInfo:
    assets= None
    joints = None


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_pkl(save_path,dic_file):
    ensure_dir(save_path)
    with open(save_path, 'wb') as f:
        pickle.dump(dic_file, f)

def fps(points, num_points=1024, use_cuda=True):

    K = [num_points]
    if use_cuda:
        points = torch.from_numpy(points).cuda()
        sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
        sampled_points = sampled_points.squeeze(0)
        sampled_points = sampled_points.cpu().numpy()
    else:
        points = torch.from_numpy(points)
        sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
        sampled_points = sampled_points.squeeze(0)
        sampled_points = sampled_points.numpy()

    return sampled_points, indices

# BRIDGE_DATASET_ASSET_PATH = ASSET_DIR / "tasks/bridge_v2_real2sim_dataset/"

class TableEnv(BaseEnv):
    SUPPORTED_REWARD_MODES = ["sparse", "none"]
    kitchen_scene_scale = 1.0
    # scene_scale = 0.82
    scene_scale = 1
    # control_timestep=250

    def check_failure(self):
        
        tmp_bbqvel_list = [bbqvel.tolist() for bbqvel in self.bbqvel_list]
        
        self.bbqvel_list = []
        
        result = False
        violated_qvel = 999
        violated_qvel_index = 999
        for bbqvel in tmp_bbqvel_list:
            for i in range(len(bbqvel)):
                if (bbqvel[i] > 30) or (bbqvel[i] < -30):
                    violated_qvel = bbqvel[i]
                    violated_qvel_index = i
                    result = True
                    break
                else:
                    pass
                
        # print(f"violated_qvel: {violated_qvel}, violated_qvel_index: {violated_qvel_index}")
        if result == True:
            # print("=========== check_failre ===========")
            # print("bbqvel_list:", tmp_bbqvel_list)
            # print("bbqvel:", bbqvel)
            print("violated_qvel: ", violated_qvel)
            print("violated_qvel_index: ", violated_qvel_index)
            print(f'failed you idiot!!!!!!!!!!!!!!!!!!!!!!')
        
        return result

    def __init__(self, *args, robot_init_qpos_noise=0.02, **kwargs):
        self.bbqvel_list = []
        kwargs["robot_uids"]="unitree_h1_2_upper_body_with_head_camera"


        self.init_robot_pose = copy.deepcopy(
            UnitreeH1_2UpperBodyWithHeadCamera.keyframes["standing"].pose
        )
        if "stack" in self.env_name or "block" in self.env_name or "handover" in self.env_name:
            UnitreeH1_2UpperBodyWithHeadCamera.hand_damping =100
            UnitreeH1_2UpperBodyWithHeadCamera.hand_force_limit = 0.7
        self.init_robot_pose.p = [-0.85, 0, 0]
        self.scene_name = "Table"
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.assets_info: Dict[str, Dict] = io_utils.load_json(
            ROOT_PATH / "assets/objects/assets_info.json"
        )
        self.run_cofig_file_path=kwargs["config_file_path"]
        valid_keys = [
            "num_envs", "obs_mode", "reward_mode", "control_mode", "render_mode",
            "shader_dir", "enable_shadow", "sensor_configs", "human_render_camera_configs",
            "viewer_camera_configs", "robot_uids", "sim_config", "reconfiguration_freq",
            "sim_backend", "render_backend", "parallel_in_single_scene", "enhanced_determinism"
        ]
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}

        super().__init__(*args, **filtered_kwargs)

        self.load_camera()
        self.agent.robot.set_qpos(self.agent.keyframes["standing"].qpos)
        self.agent.robot.set_pose(self.init_robot_pose)
        self.last_record_action = self.agent.robot.get_qpos()[0, :38].cpu().numpy()
        for joint in self.agent.robot.get_active_joints():
            joint.set_friction(0.9)
        
    def render(self):
        if self.render_scene:
            super().render()
        if self.save_video or self.save_when_fail:
            if self.render_step % 5==0:
                self.images.append(self.render_rgb_array())
            self.render_step+=1
        if not self.render_scene and not (self.save_video or self.save_when_fail):
            self.scene.update_render()

    def start_task(self):

        self.reset(seed=self.env_seed)

        if self.init_scene:
            init_scene_step_num = self.init_scene_step_num
            for i in range(init_scene_step_num):
                self.render()
                defalt_pose = self.agent.robot.get_qpos()[0, :38].cpu().numpy()
                obs, reward, terminated, truncated, info = self.step(defalt_pose)

        if hasattr(self, "check_init_success"):
            if not self.check_init_success():
                self.images=[] # clear the failed images
                self.start_task_flag = False
                print("The initial state is not success!")
                return False
        
        self.start_task_flag = True
        return True
    
    def end_task(self,save_file="defalt"):
        
        if self.end_scene:
            end_scene_step_num = self.end_scene_step_num
            for i in range(end_scene_step_num):
                self.render()
                defalt_pose = self.agent.robot.get_qpos()[0, :38].cpu().numpy()
                obs, reward, terminated, truncated, info = self.step(defalt_pose)
        
        solve_success=False
        if self.start_task_flag== True and self.calculate_success_rate:
            solve_success=bool(self.check_success())
            self.execution_results.append({"execution_id": self.ep_num, "success": solve_success})
            if solve_success:
                self.success_count += 1
                print(f" Execution{self.ep_num} succeeded.")
            else:
                print(f"Execution {self.ep_num} failed.")
            self.ep_num += 1
            
            # out the success rate
            success_rate = self.success_count / self.ep_num * 100
            print(f"Success rate now: {success_rate:.2f}%")

        elif self.start_task_flag== True:
            self.ep_num += 1

        self.PCD_INDEX = 0
        self.step_count = 0

        if (self.save_video and self.start_task_flag== True) or (self.save_when_fail and not solve_success and self.start_task_flag== True):
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = save_file.split('/')[-1]
            file_name_time = f"{file_name}_{current_time}"
            if not os.path.exists(ROOT_PATH / "videos"/save_file):
                os.makedirs(ROOT_PATH / "videos"/save_file)
            images_to_video(self.images,
                output_dir = ROOT_PATH / "videos"/save_file,
                video_name = file_name_time,
                fps=30)

        self.images=[]
        self.start_task_flag = False


    def end_run(self):
        if self.calculate_success_rate:
            # Calculate the success rate
            print(f"Total episodes executed: {self.ep_num}")
            print(f"Total successful episodes: {self.success_count}")
            success_rate = self.success_count / self.ep_num * 100
            print(f"Execution completed. Success rate: {success_rate:.2f}%")
            import json

            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.results_file), exist_ok=True)
            
            # Save the results to a file
            with open(self.results_file, "w") as file:
                json.dump({"success_rate": success_rate, "results": self.execution_results}, file, indent=4)
            print(f"Results saved to {self.results_file}.")


    def reset(self, seed: Union[None, int, list[int]] = None, options: Union[None, dict] = None,save_video=False,save_file="defalt"):
        obs, info = super().reset(seed=seed, options=options)
        return obs, info

    # @property
    # def _default_sim_config(self):
    #     return SimConfig(
    #         gpu_memory_config=GPUMemoryConfig(
    #             max_rigid_contact_count=2**22,
    #         )
    #     )
    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=2**22, max_rigid_patch_count=2**21
            ),
            control_freq=100,
            # TODO (stao): G1 robot may need some custom collision disabling as the dextrous fingers may often be close to each other
            # and slow down simulation. A temporary fix is to reduce contact_offset value down so that we don't check so many possible
            # collisions
            scene_config=SceneConfig(contact_offset=0.01),
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [
            # CameraConfig("base_camera", pose=pose, width=128, height=128, fov=np.pi / 2)
            CameraConfig("base_camera", pose=pose, width=512, height=512, fov=np.pi / 2)
        ]
    
    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose=pose, width=512, height=512, fov=1)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0, 0, 1]))

    def _load_scene(self, options: dict):
        # self.scene_builder = KitchenCounterSceneBuilder(self)
        if self.scene_name == "RoboCasa":
            self.scene_builder = RoboCasaSceneBuilder(self)
        elif self.scene_name == "Table":
            self.scene_builder = TableSceneBuilder(self)
        # self.kitchen_scene = self.scene_builder.build(scale=self.kitchen_scene_scale)
        self.kitchen_scene = self.scene_builder.build()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        if not hasattr(self, "save_video"):
            with open(self.run_cofig_file_path, "r") as file:
                self.run_config = yaml.safe_load(file)
                self.save_video = self.run_config["default"]["save_video"]
                self.save_when_fail = self.run_config["default"]["save_when_fail"]
                self.render_scene = self.run_config["default"]["render_scene"]
                self.random_scene = self.run_config["default"]["random_scene"]
                # initialize scene parameters
                self.init_scene = self.run_config["default"]["init_scene"]
                self.init_scene_step_num = self.run_config["default"]["init_scene_step_num"]
                self.max_init_scene_num = self.run_config["max_init_scene_num"]
                
                # end scene parameters
                self.end_scene = self.run_config["default"]["end_scene"]
                self.end_scene_step_num = self.run_config["default"]["end_scene_step_num"]

                self.images=[]
                self.render_step = 0
                # front camera
                self.front_camera_type = self.run_config["default"]["front_camera_type"]
                self.front_camera_w = self.run_config["camera_config"][self.front_camera_type]["w"]
                self.front_camera_h = self.run_config["camera_config"][self.front_camera_type]["h"]
                self.front_camera_fovy = self.run_config["camera_config"][self.front_camera_type]["fovy"]
                # head camera
                self.head_camera_type = self.run_config["default"]["head_camera_type"]
                self.head_camera_w = self.run_config["camera_config"][self.head_camera_type]["w"]
                self.head_camera_h = self.run_config["camera_config"][self.head_camera_type]["h"]
                self.head_camera_fovy = self.run_config["camera_config"][self.head_camera_type]["fovy"]

                self.pcd_crop = self.run_config["default"]["pcd_crop"]
                self.record_data = self.run_config["default"]["record_data"]
                self.pcd_down_sample_num = self.run_config["default"]["pcd_down_sample_num"]
                self.record_freq = self.run_config["default"]["record_freq"]
                
                if "env_id" in self.run_config:
                    self.record_env_name = self.run_config["env_id"]
                elif hasattr(self, "env_name"):
                    self.record_env_name = self.env_name
                else:
                    print("Warning: env_id and env_name is not set")
                    class_name = self.__class__.__name__
                    self.record_env_name=re.sub(r'(?<!^)(?=[A-Z])', '_', class_name).lower()

                self.random_once= self.run_config["default"]["random_once"]
                current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                self.results_file = str(ROOT_PATH/f"logs/{self.record_env_name}_{current_time}")  # File path to save the results
                self.save_dir=str(ROOT_PATH/f"datasets/{self.record_env_name}_{current_time}")
                self.calculate_success_rate = self.run_config["default"]["calculate_success_rate"]
                self.pcd_crop_bbox=[[-0.75, -0.7, 0.005],[0.4, 0.7, 1]]
                self.ep_num = 0
                self.PCD_INDEX = 0
                self.step_count = 0
                self.execution_results = []
                self.start_task_flag = False
                self.success_count = 0

                self.seed_everything(self.run_config["default"]["env_seed"])
                # self.dp3_data_file = self.run_config["eval_policy_dp3"]["data_file"]
        
        self.use_env_setting=False
        if hasattr(self,"env_name"):
            env_setting_path=str(ROOT_PATH/f"envs/example/env_setting/{self.env_name}.yml")
            if os.path.exists(env_setting_path):
                with open(env_setting_path, "r") as file:
                    env_setting_config = yaml.safe_load(file)
                    self.use_random_range = env_setting_config["use_random_range"]
                    self.random_range = env_setting_config["random_range"]
                    self.use_random_angle = env_setting_config["use_random_angle"]
                    self.random_angle = env_setting_config["random_angle"]
                    self.use_default_angle = env_setting_config["use_default_angle"]
                    self.default_angle = env_setting_config["default_angle"]
                    self.use_env_setting = True

        self.scene_builder.initialize(env_idx)
        self.agent.robot.set_qpos(self.agent.keyframes["standing"].qpos)
        self.agent.robot.set_pose(self.init_robot_pose)
    
    # use this function to update the run config file path (notice: before reset())
    def update_run_cofig_file_path(self, run_cofig_file_path):
        self.run_cofig_file_path=run_cofig_file_path
        if hasattr(self, "save_video"):
            delattr(self, "save_video")

    def evaluate(self):
        return {
            "success": torch.zeros(self.num_envs, device=self.device, dtype=bool),
            "fail": torch.zeros(self.num_envs, device=self.device, dtype=bool),
        }

    def _get_obs_extra(self, info: Dict):
        return dict()


class TableSetting(TableEnv):
    SUPPORTED_REWARD_MODES = ["normalized_dense", "dense", "sparse", "none"]
    obj_static_friction = 0.999
    obj_dynamic_friction = 0.999

    actor_count ={}
    total_actor_count = 0

    def qpos_to_action(self,qpos):
        qpos = np.array(qpos).copy()

        # left_hand
        # qpos[14] = action[0]
        # qpos[[24, 34, 36]] = action[1]
        # qpos[[15, 25]] = action[2]
        # qpos[[16, 26]] = action[3]
        # qpos[[17, 27]] = action[4]
        # qpos[[18, 28]] = action[5]

        # right_hand
        # qpos[19] = action[0]
        # qpos[[29, 35, 37]] = action[1]
        # qpos[[20, 30]] = action[2]
        # qpos[[21, 31]] = action[3]
        # qpos[[22, 32]] = action[4]
        # qpos[[23, 33]] = action[5]

        result = np.zeros(26)

        # left hand
        for i,index in enumerate(self.agent.left_arm_joint_indexes):
            result[i]=qpos[index] #left_arm [0:7]
        result[7]=qpos[14]
        result[8]=qpos[24]
        result[9]=qpos[15]
        result[10]=qpos[16]
        result[11]=qpos[17]
        result[12]=qpos[18] #left_hand [7:13]
        # right hand
        for i,index in enumerate(self.agent.right_arm_joint_indexes):
            result[i+13]=qpos[index] #right_arm [13:20]
        result[20]=qpos[19]
        result[21]=qpos[29]
        result[22]=qpos[20]
        result[23]=qpos[21]
        result[24]=qpos[22]
        result[25]=qpos[23] #right_hand [20:26]
        return result
    
    def action_to_qpos(self,action):
        qpos = np.zeros(38)
        # left arm
        for i,index in enumerate(self.agent.left_arm_joint_indexes):
            qpos[index]=action[i]
        # left hand
        qpos[14]=action[7]
        qpos[[24, 34, 36]]=action[8]
        qpos[[15, 25]]=action[9]
        qpos[[16, 26]]=action[10]
        qpos[[17, 27]]=action[11]
        qpos[[18, 28]]=action[12]

        # right arm
        for i,index in enumerate(self.agent.right_arm_joint_indexes):
            qpos[index]=action[i+13]
        # right hand
        qpos[19]=action[20]
        qpos[[29, 35, 37]] =action[21]
        qpos[[20, 30]]=action[22]
        qpos[[21, 31]]=action[23]
        qpos[[22, 32]]=action[24]
        qpos[[23, 33]]=action[25]

        return qpos
    
    def check_hand_open(self):
        qpos = self.qpos_to_action(self.agent.robot.get_qpos()[0, :38].cpu().numpy())
        left_hand_open = np.abs(qpos[9]) < 0.05
        right_hand_open = np.abs(qpos[22]) < 0.01
        return left_hand_open , right_hand_open
    
    ## camera and data recording
    @property
    def _default_sensor_configs(self):
        return CameraConfig(
            "base_camera",
            sapien.Pose(
                [1.279123, 0.303438, 1.34794], [0.252428, 0.396735, 0.114442, -0.875091]
            ),
            128,
            128,
            np.pi / 2,
            0.01,
            100,
        )

    @property
    def _default_human_render_camera_configs(self):
        if self.scene_name == "RoboCasa":
            return CameraConfig(
                "render_camera",
                sapien.Pose(
                    # [4.279123, -1.303438, 1.54794], [0.252428, 0.396735, 0.114442, -0.875091]
                    # [3.85513, -3.61328, 1.65987], [0.474556, -0.22859, 0.129117, 0.840162]
                    [3.15335, -3.13846, 2.09491], [0.795576, -0.171838, 0.264218, 0.517416]
                ),
                512,
                512,
                np.pi / 2,
                0.01,
                100,
            )
        
        elif self.scene_name == "Table":
            return CameraConfig(
                "render_camera",
                sapien.Pose(
                    [-0.135307, -0.710533, 0.446276], [0.54266, -0.109031, 0.0713047, 0.829788]
                ),
                1024,1024,
                # 512,512,
                np.pi / 2,
                0.01,
                100,
            )
        
    def load_camera(self):
        near, far = 0.1, 100

        # front camera
        front_cam_pos = np.array([0.44865, 0, 0.440116])
        front_cam_forward = np.array([-1,0,-1]) / np.linalg.norm(np.array([-1,0,-2]))
        front_cam_left = np.array([0,-1,0]) / np.linalg.norm(np.array([0,-1,0]))
        front_up = np.cross(front_cam_forward, front_cam_left)
        front_mat44 = np.eye(4)
        front_mat44[:3, :3] = np.stack([front_cam_forward, front_cam_left, front_up], axis=1)
        front_mat44[:3, 3] = front_cam_pos

        # head camera
        head_cam_pos = np.array([-0.732214, 0, 0.676484])
        head_cam_forward = np.array([1,0,-2]) / np.linalg.norm(np.array([1,0,-2]))
        head_cam_left =  np.array([0,1,0]) / np.linalg.norm(np.array([0,1,0]))
        head_up = np.cross(head_cam_forward, head_cam_left)
        head_mat44 = np.eye(4)
        head_mat44[:3, :3] = np.stack([head_cam_forward, head_cam_left, head_up], axis=1)
        head_mat44[:3, 3] = head_cam_pos

        self.front_camera = self.scene.add_camera(
            name="front_camera",
            pose=sapien.Pose(front_mat44),
            width=self.front_camera_w,
            height=self.front_camera_h,
            fovy=np.deg2rad(self.front_camera_fovy),
            near=near,
            far=far,
        )

        self.head_camera = self.scene.add_camera(
            name="head_camera",
            pose=sapien.Pose(head_mat44),
            width=self.head_camera_w,
            height=self.head_camera_h,
            fovy=np.deg2rad(self.head_camera_fovy),
            near=near,
            far=far,
        )
        # self.front_camera.entity.set_pose(sapien.Pose(front_mat44))
        # self.head_camera.entity.set_pose(sapien.Pose(head_mat44))

        self.scene.step()  # run a physical step
        self.scene.update_render()  # sync pose from SAPIEN to renderer

    # Get Camera PointCloud
    def _get_camera_pcd(self, camera, point_num = 0):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # rgba = np.array(camera.get_picture("Color")) # [H, W, 4] || [1, H, W, 4](list[0])
        
        rgba = camera.get_picture("Color")
        if type(rgba) == list:
            rgba = rgba[0]
        if isinstance(rgba, torch.Tensor):
            rgba = rgba.cpu().numpy()
        rgba = np.array(rgba) # [H, W, 4] || [1, H, W, 4](list[0])    

        # position =  np.array(camera.get_picture("Position"))
        position =  camera.get_picture("Position")
        if type(position) == list:
            position = position[0]
        if isinstance(position, torch.Tensor):
            position = position.cpu().numpy()
        position = np.array(position) # [H, W, 4] || [1, H, W, 4](list[0])

        model_matrix = camera.get_model_matrix()

        rgba=torch.tensor(rgba, dtype=torch.float32).to(device)   # torch.Size([1, 1, H, W, 4])
        position=torch.tensor(position, dtype=torch.float32).to(device) # torch.Size([1, 1, H, W, 4])
        model_matrix = torch.tensor(model_matrix, dtype=torch.float32).to(device) # torch.Size([1, 4, 4])

        # Extract valid three-dimensional points and corresponding color data.
        valid_mask = position[..., 3] < 1    # torch.Size([1, 1, H, W])
        points_opengl = position[..., :3][valid_mask]   # torch.Size([N, 3])
        points_color = rgba[valid_mask][:,:3]   # torch.Size([N, 3])
        # Transform into the world coordinate system.    torch.bmm(torch.Size([1, N, 3]), torch.Size([1, 3, 3]))
        points_world = torch.bmm(points_opengl.reshape(1, -1, 3),model_matrix[0, :3, :3].transpose(0,1).view(-1, 3, 3)).squeeze(1) + model_matrix[0,:3, 3]

        # Format color data.   torch.Size([N, 3])
        points_color = torch.clamp(points_color, 0, 1)
        #  torch.Size([1, N, 3])-> torch.Size([N, 3])
        points_world = points_world.squeeze(0)

        # If crop is needed
        if self.pcd_crop:
            min_bound = torch.tensor(self.pcd_crop_bbox[0], dtype=torch.float32).to(device)
            max_bound = torch.tensor(self.pcd_crop_bbox[1], dtype=torch.float32).to(device)
            inside_bounds_mask = (points_world.squeeze(0) >= min_bound).all(dim=1) & (points_world.squeeze(0)  <= max_bound).all(dim=1)
            points_world = points_world[inside_bounds_mask]
            points_color = points_color[inside_bounds_mask]
        
        # Convert the tensor back to a NumPy array for use with Open3D.
        points_world_np = points_world.cpu().numpy()
        points_color_np = points_color.cpu().numpy()

        if point_num > 0:
            points_world_np,index = fps(points_world_np,point_num)
            index = index.detach().cpu().numpy()[0]
            points_color_np = points_color_np[index,:]

        return np.hstack((points_world_np, points_color_np))

    # Get Camera RGBA
    def _get_camera_rgba(self, camera):
        # rgba = np.array(camera.get_picture("Color"))
        rgba = camera.get_picture("Color")
        if type(rgba) == list:
            rgba = rgba[0] # list(tensor)->tensor
        if isinstance(rgba, torch.Tensor):
            rgba = rgba.cpu().numpy()
        rgba = np.array(rgba) # [H, W, 4] || [1, H, W, 4](list[0])    
        rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
        return rgba_img

    # Get Camera Depth
    def _get_camera_depth(self, camera):
        # position = camera.get_picture("Position")
        position =  camera.get_picture("Position")
        if type(position) == list:
            position = position[0]
        if isinstance(position, torch.Tensor):
            position = position.cpu().numpy()
        position = np.array(position) # [H, W, 4] || [1, H, W, 4](list[0])
        depth = -position[..., 2]
        depth_image = (depth * 1000.0).astype(np.float64)
        return depth_image
    
    def get_obs_now(self):
        self.head_camera.take_picture()
        self.front_camera.take_picture()
        pkl_dic = {
            "observation":{
                "head_camera":{},   # rbg , mesh_seg , actior_seg , depth , intrinsic_cv , extrinsic_cv , cam2world_gl(model_matrix)
                "left_camera":{},
                "right_camera":{},
                "front_camera":{}
            },
            "pointcloud":[],   # conbinet pcd
            "joint_action":[],
            "joint_state":[],
            "endpose":[]
        }

        pkl_dic["joint_state"] = np.array(self.qpos_to_action(self.agent.robot.get_qpos()[0, :38].cpu().numpy()))

        # # ---------------------------------------------------------------------------- #
        # # PointCloud
        # # ---------------------------------------------------------------------------- #
        # if self.data_type.get('pointcloud', False):
        head_pcd = self._get_camera_pcd(self.head_camera, point_num=0)
        front_pcd = self._get_camera_pcd(self.front_camera, point_num=0)
        conbine_pcd = np.vstack((head_pcd, front_pcd))

        
        pcd_array,index = conbine_pcd[:,:3], np.array(range(len(conbine_pcd)))
        
        if self.pcd_down_sample_num > 0:
            if conbine_pcd.shape[0] > self.pcd_down_sample_num:
                # shape: e.g. (501422, 6)-> (pcd_down_sample_num, 6)
                pcd_array,index = fps(conbine_pcd[:,:3],self.pcd_down_sample_num)
                index = index.detach().cpu().numpy()[0]


        if conbine_pcd.shape[0] > 0:
            pkl_dic["pointcloud"] = conbine_pcd[index]
        else:
            pkl_dic["pointcloud"] = conbine_pcd

        head_rgba = self._get_camera_rgba(self.head_camera)
        pkl_dic["observation"]["head_camera"]["rgb"] = np.squeeze(head_rgba)[:,:,:3]   #[1,..,H, W, 4] -> [H, W, 3]

        return pkl_dic
    
    def seed_everything(self, seed: int):
        print(f"Setting environment seed: {seed}")
        np.random.seed(seed)
        self.env_seed = seed

    def apply_dp3(self, model):
        self.init_task_scene()
        
        print("Asset init state: ", self.get_asset_state())
        show_joint_state = False
        visualize_pcd = False
        run_sccess = False

        all_actions = []
        all_state = []
        state_old = []
        vis = None
        if visualize_pcd:
            vis = o3d.visualization.Visualizer()
            vis.create_window()

        max_play_num = 40    
        if hasattr(self, "env_name"):
            if self.env_name == "blocks_stack_hard":
                max_play_num=70
                print("blocks_stack_hard: max_play_num set to 70")
    
        # 运行模型
        for i in range(max_play_num):
            observation = self.get_obs_now() 
            obs = dict()
            # breakpoint()
            obs['point_cloud'] = observation['pointcloud']
            obs['agent_pos'] = observation['joint_state']

            if visualize_pcd:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(obs['point_cloud'][:,:3])
                pcd.colors = o3d.utility.Vector3dVector(obs['point_cloud'][:,3:])
                vis.add_geometry(pcd)
                # vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.1)
            # 
            # breakpoint()
            print("obs['point_cloud']:",obs['point_cloud'])
            # breakpoint()
            actions = model.get_action(obs)
            

            for k in range(len(actions)):
                # if k==0 :
                #     continue
                state_old.append(self.agent.robot.get_qpos()[0, :38].cpu().numpy()[0:7])
                action_target = self.action_to_qpos(actions[k])
                # all_actions.append(action_target[0:4])
                all_actions.append(action_target[0:7])

                # defalt_pose = self.agent.robot.get_qpos()[0, :38].cpu().numpy()
                # delta_action = (action_target - defalt_pose)/30.0
                for j in range(20):
                    # defalt_pose = defalt_pose + delta_action
                    obs, reward, terminated, truncated, info = self.step(action_target)
                    self.render()
                # all_state.append(self.agent.robot.get_qpos()[0, :38].cpu().numpy()[0:4])
                all_state.append(self.agent.robot.get_qpos()[0, :38].cpu().numpy()[0:7])
                
                if k != len(actions)-1:
                    observation = self.get_obs_now()

                    obs = dict()
                    obs['point_cloud'] = observation['pointcloud']
                    obs['agent_pos'] = observation['joint_state']
                    model.update_obs(obs)

            
            if self.check_success():
                run_sccess=True
                break
        self.end_task(save_file=f"eval_dp3/{self.env_name}")
        model.env_runner.reset_obs()
        if visualize_pcd:
            vis.destroy_window()
        if show_joint_state:
            import numpy as np
            all_actions = np.array(all_actions)  
            all_state = np.array(all_state)  
            state_old = np.array(state_old)  
            
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            for i in range(all_actions.shape[-1]):  
                plt.plot(all_actions[:,  i], label=f"Action Dimension {i+1}")
            for i in range(all_state.shape[-1]):  
                plt.plot(all_state[:,  i], label=f"State Dimension {i+1}")
            for i in range(state_old.shape[-1]):  
                plt.plot(state_old[:,  i], label=f"State Old Dimension {i+1}")

            plt.xlabel("Step")
            plt.ylabel("Action Value")
            plt.title("Action Values Over Time")
            plt.legend()
            plt.grid()
            plt.show()
        print("run_sccess:",run_sccess)
        return run_sccess

    

    def interpolate_actions(self, actions, current_state, interval=0.005):
        
        new_actions = []
        old_action = self.qpos_to_action(current_state)
        for i in range(len(actions)):
            new_actions = []
            target_action = actions[i]
            
            max_diff = np.max(np.abs(target_action - old_action))
            if max_diff > interval:
                # interpolate the action
                num_steps = int(max_diff / interval)
                for j in range(num_steps):
                    new_action = old_action + (target_action - old_action) * (j + 1) / num_steps
                    new_actions.append(new_action)
                    
            new_actions.append(target_action)
            old_action = target_action
            
        print(f'step number after interpolation: {len(new_actions)}')    
        
        return new_actions


    def apply_dp(self, model):
        
        self.init_task_scene()
        print("Asset init state: ", self.get_asset_state())
        show_joint_state = False
        visualize_pcd = False
        run_sccess = False
        run_failure = False

        all_actions = []
        all_state = []
        state_old = []
        vis = None



        for i in range(40):
            observation = self.get_obs_now() 
            obs = dict()
            obs['head_cam'] = observation['observation']["head_camera"]['rgb']
            obs['head_cam'] = np.moveaxis(obs['head_cam'], -1, 0)
            obs['agent_pos'] = observation['joint_state']


            actions = model.get_action(obs)

            qpos = self.agent.robot.get_qpos()[0, :38].cpu().numpy()
            
            for k in range(len(actions)):

                state_old.append(self.agent.robot.get_qpos()[0, :38].cpu().numpy()[0:4])
                action_target = self.action_to_qpos(actions[k])

                for j in range(20):
                    obs, reward, terminated, truncated, info = self.step(action_target)
                    self.render()


                all_state.append(self.agent.robot.get_qpos()[0, :38].cpu().numpy()[0:4])
                
                if k != len(actions)-1:
                    observation = self.get_obs_now()
                    obs = dict()
                    obs['head_cam'] = observation['observation']["head_camera"]['rgb']
                    obs['head_cam'] = np.moveaxis(obs['head_cam'], -1, 0)
                    obs['agent_pos'] = observation['joint_state']

                    model.update_obs(obs)
            
            if self.check_success():
                run_sccess=True
                break
            
        self.end_task(save_file=f"eval_dp/{self.env_name}")
        model.runner.reset_obs()
    

        if visualize_pcd:
            vis.destroy_window()
        if show_joint_state:
            all_actions = np.array(all_actions)  
            all_state = np.array(all_state)    
            state_old = np.array(state_old)  
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 6))
            for i in range(all_actions.shape[-1]): 
                plt.plot(all_actions[:,  i], label=f"Action Dimension {i+1}")
            for i in range(all_state.shape[-1]):  
                plt.plot(all_state[:,  i], label=f"State Dimension {i+1}")
            for i in range(state_old.shape[-1]):  
                plt.plot(state_old[:,  i], label=f"State Old Dimension {i+1}")

            plt.xlabel("Step")
            plt.ylabel("Action Value")
            plt.title("Action Values Over Time")
            plt.legend()
            plt.grid()
            plt.show()
        print("run_sccess:",run_sccess)
        return run_sccess, run_failure

    def step(self,action): # Save data        
        # get the state of the joints(position and velocity)
        bbqpos = self.agent.robot.get_qpos()[0, :38].cpu().numpy()
        bbqvel = self.agent.robot.get_qvel()[0, :38].cpu().numpy()
        
        self.bbqvel_list.append(bbqvel)
        
        
        # print("step_count:",self.step_count)
        if not self.record_data or self.step_count % self.record_freq != 0 or self.start_task_flag== False:
            self.step_count += 1
            self.last_record_action = action

            return super().step(action)
        else:
            self.step_count += 1
            if not hasattr(self,"head_camera"):
                return super().step(action)
            self.head_camera.take_picture()
            self.front_camera.take_picture()

            if self.PCD_INDEX == 0:
                self.file_path ={
                    "observer_color" : f"{self.save_dir}/episode{self.ep_num}/camera/color/observer/",

                    "l_color" : f"{self.save_dir}/episode{self.ep_num}/camera/color/left/",
                    "l_depth" : f"{self.save_dir}/episode{self.ep_num}/camera/depth/left/",
                    "l_pcd" : f"{self.save_dir}/episode{self.ep_num}/camera/pointCloud/left/",

                    "f_color" : f"{self.save_dir}/episode{self.ep_num}/camera/color/front/",
                    "f_depth" : f"{self.save_dir}/episode{self.ep_num}/camera/depth/front/",
                    "f_pcd" : f"{self.save_dir}/episode{self.ep_num}/camera/pointCloud/front/",

                    "r_color" : f"{self.save_dir}/episode{self.ep_num}/camera/color/right/",
                    "r_depth" : f"{self.save_dir}/episode{self.ep_num}/camera/depth/right/",
                    "r_pcd" : f"{self.save_dir}/episode{self.ep_num}/camera/pointCloud/right/",

                    "t_color" : f"{self.save_dir}/episode{self.ep_num}/camera/color/head/",
                    "t_depth" : f"{self.save_dir}/episode{self.ep_num}/camera/depth/head/",
                    "t_pcd" : f"{self.save_dir}/episode{self.ep_num}/camera/pointCloud/head/",

                    "f_seg_mesh" : f"{self.save_dir}/episode{self.ep_num}/camera/segmentation/front/mesh/",
                    "l_seg_mesh" : f"{self.save_dir}/episode{self.ep_num}/camera/segmentation/left/mesh/",
                    "r_seg_mesh" : f"{self.save_dir}/episode{self.ep_num}/camera/segmentation/right/mesh/",
                    "t_seg_mesh" : f"{self.save_dir}/episode{self.ep_num}/camera/segmentation/top/mesh/",

                    "f_seg_actor" : f"{self.save_dir}/episode{self.ep_num}/camera/segmentation/front/actor/",
                    "l_seg_actor" : f"{self.save_dir}/episode{self.ep_num}/camera/segmentation/left/actor/",
                    "r_seg_actor" : f"{self.save_dir}/episode{self.ep_num}/camera/segmentation/right/actor/",
                    "t_seg_actor" : f"{self.save_dir}/episode{self.ep_num}/camera/segmentation/head/actor/",

                    "f_camera" : f"{self.save_dir}/episode{self.ep_num}/camera/model_camera/front/",
                    "t_camera" : f"{self.save_dir}/episode{self.ep_num}/camera/model_camera/head/",
                    "l_camera" : f"{self.save_dir}/episode{self.ep_num}/camera/model_camera/left/",
                    "r_camera" : f"{self.save_dir}/episode{self.ep_num}/camera/model_camera/right/",

                    "ml_ep" : f"{self.save_dir}/episode{self.ep_num}/arm/endPose/masterLeft/",
                    "mr_ep" : f"{self.save_dir}/episode{self.ep_num}/arm/endPose/masterRight/",
                    "pl_ep" : f"{self.save_dir}/episode{self.ep_num}/arm/endPose/puppetLeft/",
                    "pr_ep" : f"{self.save_dir}/episode{self.ep_num}/arm/endPose/puppetRight/",
                    "pl_joint" : f"{self.save_dir}/episode{self.ep_num}/arm/jointState/puppetLeft/",
                    "pr_joint" : f"{self.save_dir}/episode{self.ep_num}/arm/jointState/puppetRight/",
                    "ml_joint" : f"{self.save_dir}/episode{self.ep_num}/arm/jointState/masterLeft/",
                    "mr_joint" : f"{self.save_dir}/episode{self.ep_num}/arm/jointState/masterRight/",
                    # "pkl" : f"{self.save_dir}_pkl/episode{self.ep_num}/",
                    "pkl" : f"{self.save_dir}_pkl/episode{self.success_count}/",
                    "conbine_pcd" : f"{self.save_dir}/episode{self.ep_num}/camera/pointCloud/conbine/",
                }

                for directory in self.file_path.values():
                    if os.path.exists(directory):
                        file_list = os.listdir(directory)
                        for file in file_list:
                            os.remove(directory + file)

            pkl_dic = {
                "observation":{
                    "head_camera":{},   # rbg , mesh_seg , actior_seg , depth , intrinsic_cv , extrinsic_cv , cam2world_gl(model_matrix)
                    "left_camera":{},
                    "right_camera":{},
                    "front_camera":{}
                },
                "pointcloud":[],   # conbinet pcd
                "joint_action":[],
                "joint_state":[],
                "endpose":[]
            }

            # # ---------------------------------------------------------------------------- #
            # # RGBA
            # # ---------------------------------------------------------------------------- #
            # if self.data_type.get('rgb', False):
            front_rgba = self._get_camera_rgba(self.front_camera)
            head_rgba = self._get_camera_rgba(self.head_camera)

            pkl_dic["observation"]["head_camera"]["rgb"] = np.squeeze(head_rgba)[:,:,:3]   #[1,..,H, W, 4] -> [H, W, 3]

            front_depth = self._get_camera_depth(self.front_camera)
            head_depth = self._get_camera_depth(self.head_camera)

            jointstate = {
                "effort" : [ 0, 0, 0, 0, 0, 0, 0 ],
                "position" : self.qpos_to_action(self.agent.robot.get_qpos()[0, :38].cpu().numpy()),
                "velocity" : [ 0, 0, 0, 0, 0, 0, 0 ]
            }


            pkl_dic["joint_state"] = np.array(jointstate["position"])
                # else:
                    # pkl_dic["joint_action"] = np.array(right_jointstate["position"])


            # # ---------------------------------------------------------------------------- #
            # # JointAction JSON
            # # ---------------------------------------------------------------------------- #
            # if self.data_type.get('qpos', False):
            jointaction = {
                "effort" : [ 0, 0, 0, 0, 0, 0, 0 ],
                "position" : self.qpos_to_action(self.last_record_action),
                "velocity" : [ 0, 0, 0, 0, 0, 0, 0 ]
            }

            pkl_dic["joint_action"] = np.array(jointaction["position"])


            # # ---------------------------------------------------------------------------- #
            # # PointCloud
            # # ---------------------------------------------------------------------------- #
            head_pcd = self._get_camera_pcd(self.head_camera, point_num=0)
            front_pcd = self._get_camera_pcd(self.front_camera, point_num=0)
            # left_pcd = self._get_camera_pcd(self.left_camera, point_num=0)
            # right_pcd = self._get_camera_pcd(self.right_camera, point_num=0) 

            # Merge pointcloud
            conbine_pcd = np.vstack((head_pcd, front_pcd))
            # else:
                # conbine_pcd = head_pcd
            
            pcd_array,index = conbine_pcd[:,:3], np.array(range(len(conbine_pcd)))
            
            if self.pcd_down_sample_num > 0:
                # shape: (501422, 6)-> (pcd_down_sample_num, 6)
                pcd_array,index = fps(conbine_pcd[:,:3],self.pcd_down_sample_num)
                index = index.detach().cpu().numpy()[0]

            pkl_dic["pointcloud"] = conbine_pcd[index]
            save_pkl(self.file_path["pkl"]+f"{self.PCD_INDEX}.pkl", pkl_dic)

            self.PCD_INDEX +=1
            return super().step(action)

    def get_object_pose(self,type_name,obj_id):
        objs=getattr(self, type_name)
        return objs[obj_id].pose
    
    def _set_object_pose(self,type_name,obj_id,pose):
        objs=getattr(self, type_name)
        objs[obj_id].set_pose(pose)
    
    def _add_object(
            self,
            type_name: str,
            type_id: int = 0,
    ):
        if not hasattr(self, type_name):
            setattr(self, type_name, [])
        objs=getattr(self, type_name)
        objs.append(self._build_actor_helper(type_name=type_name,obj_id=type_id,scale=self.scene_scale))

        if type_name not in self.actor_count:
            self.actor_count[type_name]=0
        self.actor_count[type_name]+=1
        self.total_actor_count+=1
        print(f"Added actor of type '{type_name}'. Total for this type: {self.actor_count[type_name]}. Total actors: {self.total_actor_count}.")

    def get_actors_info(self):
        actors_info = {}
        for type_name, count in self.actor_count.items():
            poses = []
            for obj_id in range(count):
                pose = get_pose_in_env(self, type_name, obj_id)
                # 检查并解包 pose.p 和 pose.q
                p_raw = pose.p.numpy().tolist()
                q_raw = pose.q.numpy().tolist()

                # 如果 p_raw 或 q_raw 是嵌套列表，解包第一个元素
                if isinstance(p_raw[0], list):
                    p_raw = p_raw[0]
                if isinstance(q_raw[0], list):
                    q_raw = q_raw[0]

                p = [round(float(coord), 3) for coord in p_raw]  # 
                q = [round(float(coord), 3) for coord in q_raw]  #
                poses.append({"p": p, "q": q})
                
            actors_info[type_name] = {
                "count": count,
                "poses": poses
            }
        return actors_info
    
    def get_robot_info(self):
        robot_info_prompt = {}
        robot_info_all = {}
        robot_info_all["qpose"]=self.agent.robot.get_qpos()[0, :38].numpy()
        robot_info_all["l_hand_base_link"]=self.agent.left_tcp.pose
        robot_info_all["r_hand_base_link"]=self.agent.right_tcp.pose
        p_raw = self.agent.left_tcp.pose.p.numpy().tolist().copy()
        q_raw = self.agent.left_tcp.pose.q.numpy().tolist().copy()
        # 如果 p_raw 或 q_raw 是嵌套列表，解包第一个元素
        if isinstance(p_raw[0], list):
            p_raw = p_raw[0]
        if isinstance(q_raw[0], list):
            q_raw = q_raw[0]

        # 提取位置向量和四元数，并限制小数点位数
        p = [round(float(coord), 3) for coord in p_raw]  # 确保每个元素是 float
        q = [round(float(coord), 3) for coord in q_raw]  # 确保每个元素是 float        
        robot_info_prompt["l_hand_base_link"]={"p": p, "q": q}

        p_raw = self.agent.right_tcp.pose.p.numpy().tolist().copy()
        q_raw = self.agent.right_tcp.pose.q.numpy().tolist().copy()
        # 如果 p_raw 或 q_raw 是嵌套列表，解包第一个元素
        if isinstance(p_raw[0], list):
            p_raw = p_raw[0]
        if isinstance(q_raw[0], list):
            q_raw = q_raw[0]

        # 提取位置向量和四元数，并限制小数点位数
        p = [round(float(coord), 3) for coord in p_raw]  # 确保每个元素是 float
        q = [round(float(coord), 3) for coord in q_raw]  # 确保每个元素是 float        
        robot_info_prompt["r_hand_base_link"]={"p": p, "q": q}

        return robot_info_all, robot_info_prompt
    


    #refer to BaseBridgeEnv
    def _build_actor_helper(
        self,
        type_name: str,
        obj_id: int = 0,
        scale: float = 1,
        # kinematic: bool = False,
        initial_pose: sapien.Pose = None,
        old_version=False
    ):
        builder = self.scene.create_actor_builder()
        asset_info=self.assets_info[type_name][obj_id]
        if old_version:
            num_id=obj_id
        else:
            num_id=len(getattr(self, type_name))
        obj_name=f"{type_name}_{num_id}"

        if "scale" in asset_info:
            if isinstance(asset_info["scale"], list):
                obj_scale=asset_info["scale"]
            else:
                obj_scale=float(asset_info["scale"])
        else:
            obj_scale=1.0
        
        print("obj_scale",obj_scale)
        if asset_info["dataset"]=="local":

            if "cube" in asset_info["name"]:
                if asset_info["name"]=="rectangular_cube":
                    physical_material = PhysxMaterial(
                        static_friction=0.1,
                        dynamic_friction=0.1,
                        restitution=0,
                    )
                    builder.add_box_collision(half_size=asset_info["bbx"],density=10,material=physical_material)
                else:
                    physical_material = PhysxMaterial(
                        static_friction=0.1,
                        dynamic_friction=0.1,
                        restitution=0,
                    )
                    builder.add_box_collision(half_size=asset_info["bbx"],density=100,material=physical_material)
                builder.add_box_visual(half_size=asset_info["bbx"], material=asset_info["material"])
                actor = builder.build(name=obj_name)
            else:
                fix_rotation_pose = sapien.Pose(q=euler2quat(np.pi / 2, 0, 0))
                # fix_rotation_pose = initial_pose
                # model_dir = os.path.dirname(__file__) + "/assets" # old version
                objects_dir = HGENSIM_ASSET_DIR / "objects"
                if asset_info["collision_type"] == "convex":
                    builder.add_multiple_convex_collisions_from_file(
                        filename=os.path.join(objects_dir, asset_info["collision_file"]),
                        pose=fix_rotation_pose,
                        scale=[scale*asset_info["scale"]] * 3,
                    )
                elif asset_info["collision_type"] == "nonconvex":
                    builder.add_nonconvex_collision_from_file(
                        filename=os.path.join(objects_dir, asset_info["collision_file"]),
                        scale=[scale*asset_info["scale"]] * 3,
                        pose=fix_rotation_pose,
                    )
                builder.add_visual_from_file(
                    filename=os.path.join(objects_dir, asset_info["visual_file"]),
                    scale=[scale*asset_info["scale"]] * 3,
                    pose=fix_rotation_pose,
                )
                builder.initial_pose = sapien.Pose(p=[0, 0, 0])

                if asset_info["kinematic"]:
                    actor = builder.build_kinematic(name=obj_name)
                else:
                    actor = builder.build(name=obj_name)

        elif asset_info["dataset"]=="robotwin":
            if isinstance(obj_scale,list):
                setting_scale=obj_scale*scale
            else:
                setting_scale=[scale*obj_scale] * 3
            if "glb" in asset_info:
                collision_file = str(
                    HGENSIM_ASSET_DIR / "objects/robotwin" / asset_info["glb"]
                )
                visual_file = str(
                    HGENSIM_ASSET_DIR / "objects/robotwin" / asset_info["glb"]
                )
                density=10
                if "nonconvex" in asset_info and asset_info["nonconvex"]:
                    builder.add_nonconvex_collision_from_file(
                        str(collision_file), scale=np.array(setting_scale)
                    )
                else:
                    builder.add_multiple_convex_collisions_from_file(
                        str(collision_file), scale=np.array(setting_scale), density=density
                    )
                builder.add_visual_from_file(str(visual_file), scale=np.array(setting_scale))
                if "kinematic" in asset_info and asset_info["kinematic"]:
                    instance = builder.build_kinematic(name=obj_name)
                else:
                    instance = builder.build(name=obj_name)
                actor = instance

        elif asset_info["dataset"]=="local_rigid":
            collision_file = str(
                HGENSIM_ASSET_DIR / "objects/rigidbody_objs/models" / asset_info["file_name"] / "collision.obj"
            )
            visual_file = str(
                HGENSIM_ASSET_DIR / "objects/rigidbody_objs/models" / asset_info["file_name"] / "textured.obj"
            )
            density=300
            print("[scale*obj_scale] * 3:",[scale*obj_scale] * 3)
            builder.add_multiple_convex_collisions_from_file(
                str(collision_file), scale=np.array([scale*obj_scale] * 3), density=density
            )
            builder.add_visual_from_file(str(visual_file), scale=np.array([scale*obj_scale] * 3))
            instance = builder.build(name=obj_name)

            actor = instance

        elif asset_info["dataset"]=="local_urdf":
            # 1. create a URDF loader
            loader: sapien.URDFLoader = self.scene.create_urdf_loader()
            # 2. set the loader parameters
            loader.load_multiple_collisions_from_file = True


            urdf_path = str(
                HGENSIM_ASSET_DIR / "objects/articulated_objs" / asset_info["file_name"] / "mobility.urdf"
            )
            keypoint_path = str(
                HGENSIM_ASSET_DIR / "objects/articulated_objs" / asset_info["file_name"]/ "info.json"
            )
            info = json.load(open(keypoint_path))

            loader.scale =obj_scale
            loader.density = 10


            instance: sapien.Articulation = loader.load(
                # urdf_path, config={"density": 100}
                urdf_path
            )

            for joint in instance.get_joints():
                if "laptop" in asset_info["file_name"]:
                    # joint.set_friction(0.1)
                    joint.set_friction(0.8)
                elif "box" in asset_info["file_name"]:
                    # joint.set_friction(0.1)
                    joint.set_friction(0.8)
                else:
                    joint.set_friction(0.001)
            joint = instance.get_active_joints()[0]
            if "joint_position_range" in info:
                joint_position_range = np.array(
                    info["joint_position_range"], dtype=np.float32
                )
                joint.set_limits(np.expand_dims(joint_position_range, axis=0))
            else:
                joint_position_range = joint.get_limits()[0]


            obj = ArticulatedObject(
                instance=instance,
                name=obj_name,
                scale =obj_scale,
                keypoint_path=keypoint_path,
                open_qpos=joint_position_range[1],
                close_qpos=joint_position_range[0],
                # tcp_link=self.tcp,
            )

            actor=obj


        elif asset_info["dataset"]=="bridge":
            # density = self.bridge_model_db[asset_info["model_id"]].get("density", 1000)
            if "scale" in asset_info:
                setting_scale = asset_info["scale"]
            else:
                setting_scale = 1.0
            physical_material = PhysxMaterial(
                static_friction=self.obj_static_friction,
                dynamic_friction=self.obj_dynamic_friction,
                restitution=0.0,
            )
            set_scale = [setting_scale*scale] * 3
            set_scale[1]=set_scale[1]*1.5
            # builder = self.scene.create_actor_builder()
            objects_dir = BRIDGE_DATASET_ASSET_PATH / "custom" / "models" / asset_info["file_name"]
            collision_file = str(objects_dir / "collision.obj")
            builder.add_multiple_convex_collisions_from_file(
                filename=collision_file,
                scale=set_scale,
                material=physical_material,
                density=100,
            )
            visual_file = str(objects_dir / "textured.obj")
            if not os.path.exists(visual_file):
                visual_file = str(objects_dir / "textured.dae")
                if not os.path.exists(visual_file):
                    visual_file = str(objects_dir / "textured.glb")
            builder.add_visual_from_file(filename=visual_file, scale=set_scale)
            # if initial_pose is not None:
            #     builder.initial_pose = initial_pose
            # else:
            builder.initial_pose = sapien.Pose(p=[0, 0, 0])

            if asset_info["kinematic"]:
                actor = builder.build_kinematic(name=obj_name)
            else:
                actor = builder.build(name=obj_name)

        elif asset_info["dataset"]=="bridge_urdf":
            objects_dir = BRIDGE_DATASET_ASSET_PATH / "custom"/ asset_info["file_name"]
            builder: sapien.URDFLoader = self.scene.create_urdf_loader()
            builder.name=obj_name
            builder.scale=obj_scale
            builder.fix_root_link = True
            builder.initial_pose = sapien.Pose(p=[0, 0, 0])
            actor: sapien.Articulation = builder.load(str(objects_dir),name=obj_name)
            # actor.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))

        elif asset_info["dataset"]=="robocasa":
            # print("obj_scale:",obj_scale)
            mjobj = MJCFObject(self.scene, name=type_name, mjcf_path=str(ROBOCASA_ASSET_DIR / asset_info["file_name"]),scale=obj_scale)
            # actor=mjobj.build([obj_id]).actor
            mjobj.actor_builder.set_scene_idxs([0])
            mjobj.actor_builder.initial_pose = sapien.Pose(p=mjobj.pos, q=mjobj.quat)
            actor = mjobj.actor_builder.build_dynamic(
                name=obj_name
            )

        elif asset_info["dataset"]=="replica_urdf":
            objects_dir = REPLICA_ASSET_DIR / "urdf"/ asset_info["file_name"]
            builder: sapien.URDFLoader = self.scene.create_urdf_loader()
            builder.name=obj_name
            builder.scale=obj_scale
            # builder.fix_root_link = True
            # builder.initial_pose = sapien.Pose(p=[0, 0, 0],q=[0.709, 0.705, 0, 0])
            actor: sapien.Articulation = builder.load(str(objects_dir),name=obj_name)
            # actor.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))

        else:
            print("")
            return -1
        return actor
       



    def evaluate(self):
        return {}

    def _get_obs_extra(self, info: Dict):
        return dict()

    def _grasp_release_reward(self):
        """a dense reward that rewards the agent for opening their hand"""
        return 1 - torch.tanh(self.agent.right_hand_dist_to_open_grasp())

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return 1

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 10
    
    def get_random_pose(self,default_pose,random_angle=None,default_angle=None,random_range=None):        
        if self.use_env_setting:
            if self.use_random_range:
                random_range=self.random_range
            if self.use_default_angle:
                default_angle=self.default_angle
                random_angle=None
            elif self.use_random_angle:
                random_angle=self.random_angle
                default_angle=None

        random_poses = []
        for pose in default_pose:
            if random_range is not None:
                # Randomize position within the specified range
                if len(random_range)==2:
                    random_x = pose.p[0] + np.random.uniform(-random_range[0], random_range[0])
                    random_y = pose.p[1] + np.random.uniform(-random_range[1], random_range[1])
                elif len(random_range)==4:
                    random_x = pose.p[0] + np.random.uniform(random_range[0], random_range[1])
                    random_y = pose.p[1] + np.random.uniform(random_range[2], random_range[3])
                random_z = pose.p[2]  # Keep z unchanged
            else:
                # Randomize position
                random_x = pose.p[0] + np.random.uniform(-0.02, 0.02)  # Randomize x within ±0.02
                random_y = pose.p[1] + np.random.uniform(-0.04, 0.04)  # Randomize y within ±0.04
                random_z = pose.p[2]  # Keep z unchanged

            # Randomize orientation (q) with a rotation around the z-axis
            if default_angle is not None:
                random_angle_set = default_angle

            elif random_angle is not None:
                random_angle_set = np.random.uniform(random_angle[0],random_angle[1])  # Random angle in degrees
            else:
                random_angle_set = np.random.uniform(-30,30)  # Random angle in degrees
            random_rotation = Rotation.from_euler('z', random_angle_set, degrees=True)  # Create rotation around z-axis
    
            # quaternion_xyzw=(random_rotation).as_quat()
            # quaternion_wxyz = np.array([quaternion_xyzw[3], quaternion_xyzw[0], quaternion_xyzw[1], quaternion_xyzw[2]])
            set_pose_q = pose.q.copy()
            set_pose_q = np.array([set_pose_q[1], set_pose_q[2], set_pose_q[3],set_pose_q[0]])

            random_q = (random_rotation*Rotation.from_quat(set_pose_q)).as_quat()  # Combine random rotation with original orientation
            # random_q = (Rotation.from_quat(quaternion_wxyz)).as_quat()  # Combine random rotation with original orientation
            random_q = np.array([random_q[3],random_q[0],random_q[1], random_q[2]])
            # Create a new Pose with randomized position and orientation
            randomized_pose = sapien.Pose(p=[random_x, random_y, random_z], q=random_q)
            random_poses.append(randomized_pose)
        return random_poses

    def rand_pose(
        self,
        xlim: np.ndarray,
        ylim: np.ndarray,
        zlim: np.ndarray,
        ylim_prop = False,
        rotate_rand = False,
        rotate_lim = [0,0,0],
        qpos = [1,0,0,0],
    ) -> sapien.Pose:  
        if (len(xlim)<2 or xlim[1]<xlim[0]):
            xlim=np.array([xlim[0],xlim[0]])
        if (len(ylim)<2 or ylim[1]<ylim[0]):
            ylim=np.array([ylim[0],ylim[0]])
        if (len(zlim)<2 or zlim[1]<zlim[0]):
            zlim=np.array([zlim[0],zlim[0]])
        
        x = np.random.uniform(xlim[0],xlim[1])
        y = np.random.uniform(ylim[0],ylim[1])

        while ylim_prop and abs(x) < 0.15 and y > 0:
            y = np.random.uniform(ylim[0],0)
            
        z = np.random.uniform(zlim[0],zlim[1])

        rotate = qpos
        if (rotate_rand):
            angles = [0,0,0]
            for i in range(3):
                angles[i] = np.random.uniform(-rotate_lim[i],rotate_lim[i])
            rotate_quat = t3d.euler.euler2quat(angles[0], angles[1], angles[2])
            rotate = t3d.quaternions.qmult(rotate, rotate_quat)

        return sapien.Pose([x, y, z],rotate)
    
    ######## mcts #########
    # 重置场景(运行一定的step使得场景稳定)，判断场景是否初始化成功
    def init_task_scene(self):
        start_result = False
        for i in range(self.max_init_scene_num):
            start_result=self.start_task()
            if start_result:
                break
        if not start_result:
            raise RuntimeError(f"Failed to start task after {self.max_init_scene_num} attempts.")        

    # 获取所有物体的位置信息
    def get_asset_state(self):
        asset_states = []
        for type_name, count in self.actor_count.items():
            for obj_id in range(count):
                pose = get_pose_in_env(self, type_name, obj_id)
                # 添加物体状态
                asset_state={
                    "type_name": type_name,
                    "obj_id": obj_id,
                    "pose": pose
                }
                asset_states.append(asset_state)
        return asset_states
    

    # 获取场景中物体的故有属性
    def get_asset_attribute(self):
        obj_info_file = os.path.join(ROOT_PATH, "assets/objects/assets_info.json")
        obj_info = json.load(open(obj_info_file))
        asset_attributes = []
        for type_name, count in self.actor_count.items():
            for obj_id in range(count):
                if type_name in obj_info:
                    if type_name == "cube":
                        bounding_box=[x * 2 for x in obj_info[type_name][0]["bbx"]]
                    else:
                        bounding_box=obj_info[type_name][0]["bbx"]
                    asset_attribute = {
                        "type_name": type_name,
                        "obj_id": obj_id,
                        "bounding box": bounding_box,
                        "status": obj_info[type_name][0]["status"],
                    }
                    asset_attributes.append(asset_attribute)    
        return asset_attributes
    
    def get_state_info_now(self):
        state_info = StateInfo()
        state_info.joints = self.agent.robot.get_qpos()[0, :38].cpu().numpy()
        state_info.assets = self.get_asset_state()
        return state_info