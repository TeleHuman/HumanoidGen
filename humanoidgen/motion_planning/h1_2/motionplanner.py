import mplib
import sapien
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.utils.structs.pose import to_sapien_pose
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.structs.actor import Actor
from enum import Enum
import numpy as np
import time
import os
import tqdm
import imageio
from typing import List, Optional
from humanoidgen.motion_planning.h1_2.utils import *
# from pydrake.all import *
import pydrake.all as drake
import pydrake
import pydrake.math
from pydrake.geometry import (
    ClippingRange,
    ColorRenderCamera,
    DepthRange,
    DepthRenderCamera,
    MakeRenderEngineVtk,
    RenderCameraCore,
    RenderEngineVtkParams,
    RenderLabel,
    Role,
    StartMeshcat,
)
from pydrake.geometry import SceneGraph
import matplotlib.pyplot as plt
from humanoidgen.motion_planning.h1_2.constraint import Constraint
from humanoidgen.motion_planning.h1_2.cost import Cost
# from mani_skill.utils.structs.pose import to_sapien_pose
from transforms3d.euler import euler2quat, quat2mat
from transforms3d.quaternions import mat2quat
import transforms3d
from humanoidgen import ROOT_PATH
import datetime
from humanoidgen.envs.example.table_scene import TableSetting , StateInfo
from humanoidgen.agents.objects.articulated_object import ArticulatedObject
class HandState(Enum):
    DEFAULT = 0
    PRE_GRASP = 1
    GRASP = 2 
    CLOSE_RIGHT = 3  # right hand close and left hand open
    PRE_PINCH=4
    PINCH=5

class HumanoidMotionPlanner:
    def __init__(
        self,
        env: TableSetting,
        debug: bool = False,
        vis: bool = True,
        base_pose: sapien.Pose = None,  # TODO mplib doesn't support robot base being anywhere but 0
        visualize_target_grasp_pose: bool = True,
        print_env_info: bool = False,
        joint_vel_limits=0.9,
        joint_acc_limits=0.9,
        show_key_points=False,
        debug_key_frame=False,
        use_point_cloud=True,
        use_obj_point_cloud=True,
    ):
        self.env = env
        self.base_env = env.unwrapped
        self.env_agent: BaseAgent = self.base_env.agent
        self.robot = self.env_agent.robot
        self.control_mode = self.base_env.control_mode # "pd_joint_pos"
        self.print_env_info = print_env_info # True or False
        self.vis = vis
        self.base_pose = to_sapien_pose(base_pose)
        self.joint_vel_limits = joint_vel_limits
        self.joint_acc_limits = joint_acc_limits
        # self.planner = self.setup_planner()
        self.r_planner = self.setup_planner(hand="right")
        self.l_planner = self.setup_planner(hand="left")
        self.elapsed_steps = 0
        self.show_key_points = show_key_points
        self.debug_key_frame = debug_key_frame

        self.use_obj_point_cloud = use_obj_point_cloud
        self.use_point_cloud = use_point_cloud
        self.use_attach=[False,False]
        self.debug = debug
        self.save_video = False
        self.images = [env.unwrapped.render_rgb_array()]
        
        self.render_pydrake = False

        self.show_collision_cloud = False
        # right hand pre pose
        self.hand_pre_pose = None
        self.hand_grasp_point = None
        # left hand pre pose
        self.hand_pre_pose_left = None
        self.hand_grasp_point_left = None
        if self.show_key_points:
            self.show_collision_cloud = True
            if "hand_pre_pose" not in self.base_env.scene.actors:
                self.hand_pre_pose,self.hand_grasp_point = self.build_premotion_visual(self.base_env.scene)
            else:
                self.hand_pre_pose = self.base_env.scene.actors["hand_pre_pose"]
                self.hand_grasp_point = self.base_env.scene.actors["hand_grasp_point"]


            if "hand_pre_pose_left" not in self.base_env.scene.actors:
                self.hand_pre_pose_left,self.hand_grasp_point_left = self.build_premotion_visual(self.base_env.scene,hand_name="left")
            else:
                self.hand_pre_pose_left = self.base_env.scene.actors["hand_pre_pose_left"]
                self.hand_grasp_point_left = self.base_env.scene.actors["hand_grasp_point_left"]

        self.gripper_state = [HandState.DEFAULT, HandState.DEFAULT] # [left_hand, right_hand]
        self.build_robot_plant()
        if self.use_point_cloud:
            self.init_all_point_cloud()

        self.left_hand_init_pose = self.env.agent.left_tcp.pose
        self.right_hand_init_pose = self.env.agent.right_tcp.pose
        self.delta_pinch_action_right = None
        self.delta_pinch_action_left = None
        self.attach_obj = [None,None]
        self.attach_obj_id = [None,None]

        self.left_hand_default_joint=np.array([0,0,0,0,0,0])
        self.right_hand_default_joint=np.array([0,0,0,0,0,0])

        # pinch
        self.left_hand_pinch_joint=np.array([1.174,0.32,0.8,0,0,0])
        self.left_hand_pre_pinch_joint=np.array([1.174,0,0.2,0,0,0])

        self.right_hand_pinch_joint=np.array([1.174,0.32,0.8,0,0,0])
        self.right_hand_pre_pinch_joint=np.array([1.174,0,0.4,0,0,0])

        self.left_hand_grasp_joint=np.array([1.174,0.32,0.8,0.8,0.8,0.8])
        self.left_hand_pre_grasp_joint=np.array([1.174,0,0,0,0,0])

        self.right_hand_grasp_joint=np.array([1.174,0.32,0.8,0.8,0.8,0.8])
        self.right_hand_pre_grasp_joint=np.array([1.174,0,0,0,0,0])

        self.left_hand_open_joint=np.array([1.174,0,0,0,0,0])
        self.right_hand_open_joint=np.array([1.174,0,0,0,0,0])
        self.execution_action_name = ["open_hand", "hand_pre_grasp", "hand_grasp", "hand_pre_pinch", "hand_pinch", "move"]
        self.execution_result = []

    def get_state_info_now(self):
        state_info = StateInfo()
        state_info.joints = self.robot.get_qpos()[0, :38].cpu().numpy()
        state_info.assets = self.env.get_asset_state()
        return state_info

    def build_premotion_visual(self, scene: ManiSkillScene,hand_name="right"):
        if "right" in hand_name:
            # build hand pre point
            builder = scene.create_actor_builder()
            hand_pre_pose_width = 0.01
            builder.add_sphere_visual(
                pose=sapien.Pose(p=[0, 0, 0.0]),
                radius=hand_pre_pose_width,
                material=sapien.render.RenderMaterial(base_color=[0.3, 0.4, 0.8, 0.7])
            )
            hand_pre_pose = builder.build_kinematic(name="hand_pre_pose")

            # build grasp point
            builder = scene.create_actor_builder()
            hand_pre_pose_width = 0.01
            builder.add_sphere_visual(
                pose=sapien.Pose(p=[0, 0, 0.0]),
                radius=hand_pre_pose_width,
                material=sapien.render.RenderMaterial(base_color=[1, 0, 0, 0.7])
            )
            hand_grasp_point= builder.build_kinematic(name="hand_grasp_point")

        elif "left" in hand_name:
            # build hand pre point
            builder = scene.create_actor_builder()
            hand_pre_pose_width = 0.01
            builder.add_sphere_visual(
                pose=sapien.Pose(p=[0, 0, 0.0]),
                radius=hand_pre_pose_width,
                material=sapien.render.RenderMaterial(base_color=[0.3, 0.4, 0.8, 0.7])
            )
            hand_pre_pose = builder.build_kinematic(name="hand_pre_pose_left")

            # build grasp point
            builder = scene.create_actor_builder()
            hand_pre_pose_width = 0.01
            builder.add_sphere_visual(
                pose=sapien.Pose(p=[0, 0, 0.0]),
                radius=hand_pre_pose_width,
                material=sapien.render.RenderMaterial(base_color=[1, 0, 0, 0.7])
            )
            hand_grasp_point= builder.build_kinematic(name="hand_grasp_point_left")

        return hand_pre_pose , hand_grasp_point
    

    def get_default_pose(self):
        default_pose = self.robot.get_qpos()[0, :38].cpu().numpy()
        if self.gripper_state[0] == HandState.GRASP:
            # if self.delta_grasp_action_left is not None:
            #     default_pose = default_pose+self.delta_grasp_action_left*2
            action = self.left_hand_grasp_joint
            default_pose = self.left_hand_action_to_pose(action,default_pose,low_joint=True)
            # else:
            #     action=[1.3, 0.931, 1.02, 1.027, 1.023,0.279, 0.797, 0.814, 0.815, 0.818, 0.27,0.278]
            #     default_pose = self.left_hand_action_to_pose(action,default_pose)
        elif self.gripper_state[0] == HandState.PINCH:
            # if self.delta_pinch_action_left is not None:
            #     default_pose = default_pose+self.delta_pinch_action_left
            action = self.left_hand_pinch_joint
            default_pose = self.left_hand_action_to_pose(action,default_pose,low_joint=True)
            
        if self.gripper_state[1] == HandState.GRASP:
            # if self.delta_grasp_action_right is not None:
                # default_pose = default_pose+self.delta_grasp_action_right*2
            # action=[1.3, 0.931, 1.02, 1.027, 1.023,0.279, 0.797, 0.814, 0.815, 0.818, 0.27,0.278]
            # default_pose = self.right_hand_action_to_pose(action,default_pose)
            action = self.right_hand_grasp_joint
            default_pose = self.right_hand_action_to_pose(action,default_pose,low_joint=True)

        elif self.gripper_state[1] == HandState.PINCH:
            # if self.delta_pinch_action_right is not None:
            #     default_pose = default_pose+self.delta_pinch_action_right*2
            action = self.right_hand_pinch_joint
            default_pose = self.right_hand_action_to_pose(action,default_pose,low_joint=True)
        return default_pose

    def setup_planner(self,hand="right"):
        if hand == "right":
            link_names = [link.get_name() for link in self.robot.get_links()]
            joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]
            planner = mplib.Planner(
                urdf=self.env_agent.urdf_path,
                srdf=self.env_agent.urdf_path.replace(".urdf", ".srdf"),
                user_link_names=link_names,
                user_joint_names=joint_names,
                move_group="R_hand_base_link",
                joint_vel_limits=np.ones(7) * self.joint_vel_limits,
                joint_acc_limits=np.ones(7) * self.joint_acc_limits,
            )
            planner.set_base_pose(np.hstack([self.base_pose.p, self.base_pose.q]))
            return planner
        elif hand == "left":
            link_names = [link.get_name() for link in self.robot.get_links()]
            joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]
            planner = mplib.Planner(
                urdf=self.env_agent.urdf_path,
                srdf=self.env_agent.urdf_path.replace(".urdf", ".srdf"),
                # srdf="/home/js/HGenSim/HGenSim/ManiSkill/mani_skill/assets/robots/h1_2/h1_2_upper_body_left.srdf",
                # /home/js/HGenSim/HGenSim/ManiSkill/mani_skill/assets/robots/h1_2/h1_2_upper_body_left.srdf
                user_link_names=link_names,
                user_joint_names=joint_names,
                move_group="L_hand_base_link",
                joint_vel_limits=np.ones(7) * self.joint_vel_limits,
                joint_acc_limits=np.ones(7) * self.joint_acc_limits,
            )
            planner.set_base_pose(np.hstack([self.base_pose.p, self.base_pose.q]))
            return planner

    def close_all_hands(self, t=6):
        print("Closing hands!!!!")
        # qpos = self.robot.get_qpos()[0, :14].cpu().numpy()
        qpos = self.robot.get_qpos()[0, :38].cpu().numpy()
        
        # print("Gripper state: ", self.robot.get_qpos()) # get robot state (38,)
        # print("control_mode: ", self.control_mode)
        # print(qpos.shape)
        # print(qpos)

        for i in range(t):
            if self.control_mode == "pd_joint_pos":
                # action = np.hstack([qpos, np.ones(24)])
                # action = np.hstack([qpos,np.ones(1)])
                action = np.hstack([qpos])
                action[0] = 0.01
            # else:
                # action = np.hstack([qpos, qpos * 0, self.gripper_state])
            print("Action: ", action)
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render()

        # self.gripper_state = HandStateState.CLOSE_ALL

        return obs, reward, terminated, truncated, info

    ###### hand base functions
    def get_hand_joint(self,hand_name="right",low_joint=False):
        qpos = self.robot.get_qpos()[0, :38].cpu().numpy()
        if "left" in hand_name:
            if not low_joint:
                return np.concatenate([qpos[14:19], qpos[24:29], [qpos[34]], [qpos[36]]])
            else:
                return np.array([qpos[14], qpos[24], qpos[15], qpos[16],qpos[17],qpos[18]])
        elif "right" in hand_name:
            if not low_joint:
                return np.concatenate([qpos[19:24], qpos[29:34], [qpos[35]], [qpos[37]]])
            else:
                return np.array([qpos[19], qpos[29], qpos[20], qpos[21],qpos[22],qpos[23]])
    
    def left_hand_action_to_pose(self, action,default_pose=None,low_joint=False):
        # input: action (12,) output: pose (38,)
        if default_pose is None:
            qpos = self.robot.get_qpos()[0, :38].cpu().numpy()
        else:
            qpos = default_pose.copy()
        # qpos = self.robot.get_qpos()[0, :38].cpu().numpy()
        if not low_joint:
            qpos[14:19]=action[0:5]
            qpos[24:29]=action[5:10]
            qpos[34]=action[10]
            qpos[36]=action[11]
        else:
            qpos[14] = action[0]
            qpos[[24, 34, 36]] = action[1]
            qpos[[15, 25]] = action[2]
            qpos[[16, 26]] = action[3]
            qpos[[17, 27]] = action[4]
            qpos[[18, 28]] = action[5]
        return qpos
    
    def right_hand_action_to_pose(self, action,default_pose=None,low_joint=False):
        if default_pose is None:
            qpos = self.robot.get_qpos()[0, :38].cpu().numpy()
        else:
            qpos = default_pose.copy()
        # input: action (12,) output: pose (38,)
        # qpos = self.robot.get_qpos()[0, :38].cpu().numpy()
        if not low_joint:
            qpos[19:24]=action[0:5]
            qpos[29:34]=action[5:10]
            qpos[35]=action[10]
            qpos[37]=action[11]
        else:
            qpos[19] = action[0]
            qpos[[29, 35, 37]] = action[1]
            qpos[[20, 30]] = action[2]
            qpos[[21, 31]] = action[3]
            qpos[[22, 32]] = action[4]
            qpos[[23, 33]] = action[5]
        return qpos
    
    def judge_object_in_hand(self,hand_name,object_name,object_id):
        # 确定hand_name
        # hand_id = 0 if "left" in hand_name else 1
        # print("judge object in hand: ",hand_name,object_name,object_id)
        point_name = ""
        if "left" in hand_name:
            if self.gripper_state[0] == HandState.GRASP:
                point_name = "grasp_point_base_left_hand"                
            elif self.gripper_state[0] == HandState.PINCH:
                point_name = "pinch_point_base_left_hand"
        elif "right" in hand_name:
            if self.gripper_state[1] == HandState.GRASP:
                point_name = "grasp_point_base_right_hand"
            elif self.gripper_state[1] == HandState.PINCH:
                point_name = "pinch_point_base_right_hand"
        hand_key_point=get_point_in_env(self.env, point_name=point_name)
        object_key_point=get_point_in_env(self.env, type_name=object_name, obj_id=object_id)
        
        # 判断关键点之间的距离
        distance = np.linalg.norm(hand_key_point - object_key_point)  # 计算欧几里得距离
        
        # 如果距离小于一定阈值，认为物体在手中
        threshold = 0.1 
        if distance < threshold:
            hand_joint=self.get_hand_joint(hand_name,low_joint=True)
            # print(hand_name,"hand_joint:",hand_joint)
            if hand_joint[1]>0.30 and hand_joint[2]>0.78:
                print("not hold the object: hand total close")
                return False
            return True
        else:
            return False

    def left_hand_pre_grasp(self, t=6):
        # default_pose = self.robot.get_qpos()[0, :38].cpu().numpy()  
        # if self.gripper_state[1] == HandState.PINCH:
        #     if self.delta_pinch_action_right is not None:
        #         default_pose = default_pose+self.delta_pinch_action_right*2
        # print("left hand pre grasp!!!!")
        # action=[1.25] + [0] * 11
        # qpos = self.left_hand_action_to_pose(action)
        # del_action = (np.array(qpos)-self.robot.get_qpos()[0, :38].cpu().numpy())/t

        # for i in range(t):
        #     # qpos = self.left_hand_action_to_pose(action,default_pose)
        #     obs, reward, terminated, truncated, info = self.env.step(self.robot.get_qpos()[0, :38].cpu().numpy()+del_action+self.delta_pinch_action_right)
        #     if self.vis:
        #         self.base_env.render()
            
        #     if self.save_video:
        #         rgb = self.env.unwrapped.render_rgb_array()
        #         self.images.append(rgb)
        # self.gripper_state[0] = HandState.PRE_GRASP
        # default_pose = self.robot.get_qpos()[0, :38].cpu().numpy()  
        # if self.gripper_state[1] == HandState.PINCH:
        #     if self.delta_pinch_action_right is not None:
        #         default_pose = default_pose+self.delta_pinch_action_right*2
        print("left hand pre grasp!!!!")
        # action=[1.174] + [0] * 11
        # qpos = self.left_hand_action_to_pose(action)
        default_pose = self.get_default_pose()
        action=self.left_hand_pre_grasp_joint
        qpos = self.left_hand_action_to_pose(action,default_pose,low_joint=True)
        for i in range(1):
            # qpos = self.left_hand_action_to_pose(action,default_pose)
            obs, reward, terminated, truncated, info = self.env.step(qpos)
            if self.vis:
                self.base_env.render()
            
            if self.save_video:
                rgb = self.env.unwrapped.render_rgb_array()
                self.images.append(rgb)
        self.gripper_state[0] = HandState.PRE_GRASP



    # def left_hand_grasp(self, t=6):
    #     if self.gripper_state[0] != HandState.PRE_GRASP:
    #         exception_str = "Left hand is not in pre-grasp state. Cannot grasp."
    #         print(exception_str)
    #         raise Exception(exception_str)
    #     print("left hand grasp!!!!")
    #     action=[1.3, 0.758, 0.752, 0.755, 0.758, 0.315, 0.815, 1.097, 1.1, 1.101,0.399,0.582]
    #     qpos = self.left_hand_action_to_pose(action)
    #     for i in range(t):
    #         obs, reward, terminated, truncated, info = self.env.step(qpos)
    #         if self.vis:
    #             self.base_env.render()
    #     self.gripper_state[0] = HandState.GRASP

    def left_hand_grasp(self, t=50):
        if self.gripper_state[0] != HandState.PRE_GRASP:
            exception_str = "right hand is not in pre-grasp state. Cannot grasp."
            print(exception_str)
            raise Exception(exception_str)
        
        default_pose = self.get_default_pose()
        # action=[1.3, 0.931, 1.02, 1.027, 1.023,0.279, 0.797, 0.814, 0.815, 0.818, 0.27,0.278]
        action = self.left_hand_grasp_joint
        qpos = self.left_hand_action_to_pose(action,default_pose,low_joint=True)
        self.delta_grasp_action_left = (np.array(qpos)-default_pose)/t
        for i in range(t):
            # if self.gripper_state[1] == HandState.PINCH and self.delta_pinch_action_right is not None:
            #     obs, reward, terminated, truncated, info = self.env.step(self.robot.get_qpos()[0, :38].cpu().numpy()+del_action+self.delta_pinch_action_right*2)
            # else:
            #     obs, reward, terminated, truncated, info = self.env.step(self.robot.get_qpos()[0, :38].cpu().numpy()+del_action)
            
            # hand_joint =self.get_hand_joint("left")
            # default_pose = self.left_hand_action_to_pose(hand_joint,default_pose)
            # obs, reward, terminated, truncated, info = self.env.step(default_pose+self.delta_grasp_action_left)
            # if self.vis:
            #     self.base_env.render()
            # if self.save_video:
            #     rgb = self.env.unwrapped.render_rgb_array()
            #     self.images.append(rgb)

            obs, reward, terminated, truncated, info = self.env.step(default_pose+self.delta_grasp_action_left*(i+1))
            if self.vis:
                self.base_env.render()
            if self.save_video:
                rgb = self.env.unwrapped.render_rgb_array()
                self.images.append(rgb)
        

        # for j in range(15):
        #     hand_joint =self.get_hand_joint("left")
        #     default_pose = self.left_hand_action_to_pose(hand_joint,default_pose)
        #     obs, reward, terminated, truncated, info = self.env.step(default_pose+self.delta_grasp_action_left)
        #     if self.vis:
        #         self.base_env.render()
        #     if self.save_video:
        #         rgb = self.env.unwrapped.render_rgb_array()
        #         self.images.append(rgb)
        # default_pose = self.get_default_pose()
        # delta_grasp2 = (np.array(qpos)-default_pose)/15
        for j in range(15):
            obs, reward, terminated, truncated, info = self.env.step(qpos)
            if self.vis:
                self.base_env.render()
            if self.save_video:
                rgb = self.env.unwrapped.render_rgb_array()
                self.images.append(rgb)
        
        self.gripper_state[0] = HandState.GRASP
    
    def right_hand_pre_grasp(self, t=6):
        # print("right hand pre grasp!!!!")
        # action=[1.25] + [0] * 11
        # qpos = self.right_hand_action_to_pose(action)
        action=self.right_hand_pre_grasp_joint
        qpos = self.right_hand_action_to_pose(action,low_joint=True)
        for i in range(t):
            obs, reward, terminated, truncated, info = self.env.step(qpos)
            if self.vis:
                self.base_env.render()

            if self.save_video:
                rgb = self.env.unwrapped.render_rgb_array()
                self.images.append(rgb)
        self.gripper_state[1] = HandState.PRE_GRASP

    def right_hand_grasp(self, t=50):
        if self.gripper_state[1] != HandState.PRE_GRASP:
            exception_str = "right hand is not in pre-grasp state. Cannot grasp."
            print(exception_str)
            raise Exception(exception_str)

        default_pose = self.get_default_pose()
        # action=[1.3, 0.931, 1.02, 1.027, 1.023,0.279, 0.797, 0.814, 0.815, 0.818, 0.27,0.278]
        action = self.right_hand_grasp_joint
        qpos = self.right_hand_action_to_pose(action,default_pose,low_joint=True)
        self.delta_grasp_action_right = (np.array(qpos)-default_pose)/t
        for i in range(t):
            # hand_joint =self.get_hand_joint("right")
            # default_pose = self.right_hand_action_to_pose(hand_joint,default_pose)
            obs, reward, terminated, truncated, info = self.env.step(default_pose+self.delta_grasp_action_right*(i+1))
            if self.vis:
                self.base_env.render()
            if self.save_video:
                rgb = self.env.unwrapped.render_rgb_array()
                self.images.append(rgb)

        for j in range(15):
            # hand_joint =self.get_hand_joint("right")
            # default_pose = self.right_hand_action_to_pose(hand_joint,default_pose)
            obs, reward, terminated, truncated, info = self.env.step(qpos)
            if self.vis:
                self.base_env.render()
            if self.save_video:
                rgb = self.env.unwrapped.render_rgb_array()
                self.images.append(rgb)
        self.gripper_state[1] = HandState.GRASP

    ### Right arm functions
    def right_arm_action_to_pose(self, action,default_pose):
        qpos = np.array(default_pose).copy()
        for i,index in enumerate(self.env_agent.right_arm_joint_indexes):
            qpos[index] = action[i]
        return qpos
    
    def left_arm_action_to_pose(self, action,default_pose):
        qpos = default_pose
        for i,index in enumerate(self.env_agent.left_arm_joint_indexes):
            qpos[index] = action[i]
        return qpos

    def render_wait(self):
        if not self.vis or not self.debug:
            return
        print("Press [c] to continue")
        viewer = self.base_env.render()
        if self.save_video:
            rgb = self.env.unwrapped.render_rgb_array()
            self.images.append(rgb)
        while True:
            if viewer.window.key_down("c"):
                break
            self.base_env.render()

    def right_follow_path(self, result, refine_steps: int = 0):
        
        n_step = result["position"].shape[0]
        default_pose =self.get_default_pose()
        # if self.gripper_state[1] == HandState.GRASP:
        #     action=[1.3, 0.931, 1.02, 1.027, 1.023,0.279, 0.797, 0.814, 0.815, 0.818, 0.27,0.278]
        #     default_pose = self.right_hand_action_to_pose(action)
        # elif self.gripper_state[1] == HandState.PINCH:
        #     if self.delta_pinch_action_right is not None:
        #         default_pose = self.robot.get_qpos()[0, :38].cpu().numpy()+self.delta_pinch_action_right*5
        #     else:
        #         action=[1.174, 1.0]+[0] * 3+[0.167, 0.664]+[0] * 3+[0.39, 0.534]
        #         default_pose = self.right_hand_action_to_pose(action)
        # else:
        #     default_pose = self.robot.get_qpos()[0, :38].cpu().numpy()
        
        path_point=np.array([[]])
        for i in range(n_step):
            qpos = result["position"][i]
            # action=self.right_arm_action_to_pose(qpos,default_pose)
            new_point=self.fk_robot(qpos).p.reshape(1,3)
            if path_point.shape[1]==0:
                path_point=new_point
            else:
                path_point=np.concatenate([path_point,new_point],axis=0)
        if self.show_key_points:
            self.show_path(path_point)

        for i in range(n_step + refine_steps):
            qpos = result["position"][int(min(i, n_step - 1))]
            
            action=self.right_arm_action_to_pose(qpos,default_pose)
            obs, reward, terminated, truncated, info = self.env.step(action)
            

            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render()
            if self.save_video:
                rgb = self.env.unwrapped.render_rgb_array()
                self.images.append(rgb)

            if i%10==0:
                max_cyclic_steps = 20
                for j in range(max_cyclic_steps):
                    array = action[:14] - self.robot.get_qpos()[0, :14].cpu().numpy()
                    tolerance = 0.1  # 设置容差
                    is_close_to_zero = np.allclose(array, np.zeros_like(array), atol=tolerance)
                    # print("is_close_to_zero:",is_close_to_zero)
                    if is_close_to_zero:
                        break
                    else:
                        print("do not follow the path, reaction!!!!!!!!!!!!")
                        # print("action:",action)
                        obs, reward, terminated, truncated, info = self.env.step(action)
                        if self.vis:
                            self.base_env.render()
        return obs, reward, terminated, truncated, info

    def get_eef_z(self):
        """Helper function for constraint"""
        ee_idx = self.r_planner.link_name_2_idx[self.r_planner.move_group]
        ee_pose = self.r_planner.robot.get_pinocchio_model().get_link_pose(ee_idx)
        mat = transforms3d.quaternions.quat2mat(ee_pose[3:7])
        return mat[:, 2]


    def make_f(self):
        """
        Create a constraint function that takes in a qpos and outputs a scalar.
        A valid constraint function should evaluates to 0 when the constraint
        is satisfied.

        See [ompl constrained planning](https://ompl.kavrakilab.org/constrainedPlanning.html)
        for more details.
        """

        # constraint function ankor
        def f(x, out):
            self.r_planner.robot.set_qpos(x)
            out[0] = (
                self.get_eef_z().dot(np.array([0, 0, -1])) - 0.966
            )  # maintain 15 degrees w.r.t. -z axis
            # all_pose=self.r_planner.robot.get_qpos().copy()
            # all_pose=self.robot.get_qpos()[0, :38].cpu().numpy().copy()
            # # move_group_pose=all_pose[self.r_planner.move_group_joint_indices]
            # all_pose[self.r_planner.move_group_joint_indices]=x
            # self.r_planner.robot.set_qpos(x)
            # self.r_planner.pinocchio_model.compute_forward_kinematics(all_pose)
            # ee_index=self.r_planner.link_name_2_idx["R_hand_base_link"]
            # ee_pose=self.r_planner.pinocchio_model.get_link_pose(ee_index)
            # pose_quat=ee_pose[3:7]
            # quat2mat(pose_quat)
            # # out=[]
            # # out.append(-quat2mat(pose_quat)[:,2].dot(np.array([0,0,1]))- 0.966)
            # out[0]=-quat2mat(pose_quat)[:,2].dot(np.array([0,0,1]))- 0.966
            # return out

        # constraint function ankor end
        return f
    def make_j(self):
        """
        Create the jacobian of the constraint function w.r.t. qpos.
        This is needed because the planner uses the jacobian to project a random sample
        to the constraint manifold.
        """

        # constraint jacobian ankor
        def j(x, out):
            full_qpos=self.robot.get_qpos()[0, :38].cpu().numpy().copy()
            full_qpos[self.r_planner.move_group_joint_indices]=x
            # full_qpos = self.r_planner.pad_move_group_qpos(x)
            jac = self.r_planner.robot.get_pinocchio_model().compute_single_link_jacobian(
                full_qpos, len(self.r_planner.move_group_joint_indices) - 1
            )
            rot_jac = jac[3:, self.r_planner.move_group_joint_indices]
            for i in range(len(self.r_planner.move_group_joint_indices)):
                out[i] = np.cross(rot_jac[:, i], self.get_eef_z()).dot(
                    np.array([0, 0, -1])
                )
            # all_pose=self.r_planner.robot.get_qpos()
            # all_pose=self.robot.get_qpos()[0, :38].cpu().numpy().copy()
            # # move_group_pose=all_pose[self.r_planner.move_group_joint_indices]
            # all_pose[self.r_planner.move_group_joint_indices]=x
            # # full_qpos = self.r_planner.pad_move_group_qpos(x)
            # ee_index=self.r_planner.link_name_2_idx["R_hand_base_link"]
            # jac = self.r_planner.robot.get_pinocchio_model().compute_single_link_jacobian(
            #     all_pose, ee_index
            # )
            # rot_jac = jac[3:, self.r_planner.move_group_joint_indices]
            # # out=[]
            # ee_pose=self.r_planner.pinocchio_model.get_link_pose(ee_index)
            # pose_quat=ee_pose[3:7]
            # eef_z=-quat2mat(pose_quat)[:,2]
            # # i=0
            # for i in range(len(self.r_planner.move_group_joint_indices)):
            #     out[i]=np.cross(rot_jac[:, i], eef_z).dot(
            #         np.array([0, 0, 1])
            #     )

        # constraint jacobian ankor end
        return j

    def right_move_to_pose_with_screw(
        self, pose: sapien.Pose, dry_run: bool = False, 
        refine_steps: int = 0,
        easy_plan=False,
        constraints=None,
    ):
        pose = to_sapien_pose(pose)
        # try screw two times before giving up
        if self.hand_pre_pose is not None:
            self.hand_pre_pose.set_pose(pose)
            self.hand_grasp_point.set_pose(sapien.Pose(p=transform_keypoint_to_base(self.env.agent.key_points["grasp_point_base_right_hand"],pose)))
        pose = sapien.Pose(p=pose.p , q=pose.q)

        # self_collision_list=self.r_planner.check_for_self_collision(qpos=self.robot.get_qpos().cpu().numpy()[0])
        # for collision in self_collision_list:
        #     print(f"\033[91mCollision between {collision.link_name1} and {collision.link_name2}\033[0m")
        pre_point = transform_keypoint_to_base(np.array([0, 0.1, 0]), pose)
        pre_pose = sapien.Pose(p=pre_point , q=pose.q)
        
        # result = self.planner.plan_qpos_to_pose(
        #     np.concatenate([pre_pose.p, pre_pose.q]),
        #     self.robot.get_qpos().cpu().numpy()[0],
        #     time_step=self.base_env.control_timestep,
        #     # time_step=1/250,
        #     use_point_cloud=self.use_point_cloud,
        #     use_attach=self.use_attach
        # )
        # if result["status"] != "Success":
        #     # return -1
        #     result = self.planner.plan_qpos_to_pose(
        #         np.concatenate([pre_pose.p, pre_pose.q]),
        #         self.robot.get_qpos().cpu().numpy()[0],
        #         # time_step=1/250,
        #         time_step=self.base_env.control_timestep,
        #         use_point_cloud=self.use_point_cloud,
        #         use_attach=self.use_attach
        #     )
        #     if result["status"] != "Success":
        #         print(result["status"])
        #         self.render_wait()
        #         return -1
        # self.render_wait()
        # self.use_attach = False

        # # self.build_robot_plant()
        # self.right_follow_path(result, refine_steps=refine_steps)

        # pose = sapien.Pose(p=pose.p-delta_p , q=pose.q)
        # result = self.r_planner.plan_qpos_to_pose(
        if easy_plan:
            result = self.r_planner.plan_screw(
                np.concatenate([pose.p, pose.q]),
                self.robot.get_qpos().cpu().numpy()[0],
                time_step=self.base_env.control_timestep,
                # time_step=1/250,
                use_point_cloud=self.use_point_cloud,
                use_attach=self.use_attach[1]
            )
        else:
            if constraints is None:
                result = self.r_planner.plan_qpos_to_pose(
                    np.concatenate([pose.p, pose.q]),
                    self.robot.get_qpos().cpu().numpy()[0],
                    time_step=self.base_env.control_timestep,
                    # time_step=1/250,
                    use_point_cloud=self.use_point_cloud,
                    use_attach=self.use_attach[1]
                )
            else:
                # def f(x, out):
                # def f():
                #     all_pose=self.r_planner.robot.get_qpos()
                #     move_group_pose=all_pose[self.r_planner.move_group_joint_indices]
                #     self.r_planner.robot.set_qpos(move_group_pose)
                #     self.r_planner.pinocchio_model.compute_forward_kinematics(all_pose)
                #     ee_index=self.r_planner.link_name_2_idx["R_hand_base_link"]
                #     ee_pose=self.r_planner.pinocchio_model.get_link_pose(ee_index)
                #     pose_quat=ee_pose[3:7]
                #     quat2mat(pose_quat)
                #     out=[]
                #     out.append(-quat2mat(pose_quat)[:,2].dot(np.array([0,0,1]))- 0.966)
                #     return out
                #     # out[0] = (
                #     #     self.get_eef_z().dot(np.array([0, 0, -1])) - 0.966
                #     # )  # maintain 15 degrees w.r.t. -z axis
                # def j():
                #     all_pose=self.r_planner.robot.get_qpos()
                #     move_group_pose=all_pose[self.r_planner.move_group_joint_indices]
                #     # full_qpos = self.r_planner.pad_move_group_qpos(x)
                #     ee_index=self.r_planner.link_name_2_idx["R_hand_base_link"]
                #     jac = self.r_planner.robot.get_pinocchio_model().compute_single_link_jacobian(
                #         all_pose, ee_index
                #     )
                #     rot_jac = jac[3:, self.r_planner.move_group_joint_indices]
                #     out=[]
                #     ee_pose=self.r_planner.pinocchio_model.get_link_pose(ee_index)
                #     pose_quat=ee_pose[3:7]
                #     eef_z=-quat2mat(pose_quat)[:,2]
                #     for index in self.r_planner.move_group_joint_indices:
                #         out.append(np.cross(rot_jac[:, index], eef_z).dot(
                #             np.array([0, 0, 1])
                #         ))
                #     return out
                result = self.r_planner.plan_qpos_to_pose(
                    np.concatenate([pose.p, pose.q]),
                    self.robot.get_qpos().cpu().numpy()[0],
                    time_step=self.base_env.control_timestep,
                    constraint_function=self.make_f(),
                    constraint_jacobian=self.make_j(),
                    constraint_tolerance=0.05,
                    # time_step=1/250,
                    use_point_cloud=self.use_point_cloud,
                    use_attach=self.use_attach[1]
                )
        if result["status"] != "Success":
            # transform_keypoint_to_base(hand_key_point,np.linalg.inv(pose.to_transformation_matrix()))
            result = self.r_planner.plan_qpos_to_pose(
                np.concatenate([pose.p, pose.q]),
                self.robot.get_qpos().cpu().numpy()[0],
                time_step=self.base_env.control_timestep,
                # time_step=1/250,
                use_point_cloud=self.use_point_cloud,
                use_attach=self.use_attach[1]
            )
            # return -1
            # result = self.planner.plan_qpos_to_pose(
            #     np.concatenate([pose.p, pose.q]),
            #     self.robot.get_qpos().cpu().numpy()[0],
            #     # time_step=1/250,
            #     time_step=self.base_env.control_timestep,
            #     use_point_cloud=self.use_point_cloud,
            #     use_attach=self.use_attach
            # )
            if result["status"] != "Success":
                print(result["status"])
                self.render_wait()
                return -1
        self.render_wait()
        if dry_run:
            return result
        self.right_follow_path(result, refine_steps=refine_steps)
        # self.build_robot_plant()
        self.use_attach[1] = False
        self.r_planner.planning_world.set_use_attach(False)
        
        return 1

    def generate_constraints(self,obj_name,obj_id,action,hand_name="right",relative_obj_name=None,relative_obj_id=None,relative_p=None,openness=None):
        from .label import labeled_conatraints
        return labeled_conatraints(self,obj_name,obj_id,action,hand_name,relative_obj_name,relative_obj_id,relative_p,openness)

    def generate_end_effector_pose(self,constraints,hand_name="right"):
        target_effector_joints = []
        target_effector_poses = []
        if isinstance(constraints[0],list):
            for i in range(len(constraints)):
                target_effector_joint, target_effector_pose =self.generate_end_effector_pose_tool(constraints[i],hand_name)
                target_effector_joints.append(target_effector_joint)
                target_effector_poses.append(target_effector_pose)
        else:
            target_effector_joint, target_effector_pose = self.generate_end_effector_pose_tool(constraints,hand_name)
            target_effector_joints.append(target_effector_joint)
            target_effector_poses.append(target_effector_pose)
        return target_effector_joints, target_effector_poses

    def compute_hand_info(self,hand_name,end_effector_pose):
        print("### start compute_hand_info() !!! ###")
        print("hand_name:",hand_name)
        if "left" in hand_name:
            base_link_name = "l_hand_base_link"

            grasp_point_name="grasp_point_base_left_hand"
            grasp_axis_name = "left_grasp_axis"
            
            ring_2_index_axis="left_ring_2_index"

            pinch_axis = "left_pinch_axis"
            pinch_point_name="pinch_point_base_left_hand"
            pinch_wrist_2_palm_axis="left_pinch_wrist_2_palm_axis"
            grasp_wrist_2_palm_axis="left_grasp_wrist_2_palm_axis"
            base_hand_name="base_left_hand"

        elif "right" in hand_name:
            base_link_name = "r_hand_base_link"
            
            grasp_point_name="grasp_point_base_right_hand"
            grasp_axis_name = "right_grasp_axis"

            ring_2_index_axis="right_ring_2_index"
            
            pinch_axis = "right_pinch_axis"
            pinch_point_name="pinch_point_base_right_hand"
            pinch_wrist_2_palm_axis="right_pinch_wrist_2_palm_axis"
            grasp_wrist_2_palm_axis="right_grasp_wrist_2_palm_axis"

            base_hand_name="base_right_hand"
        grasp_axis_base_hand=self.env.agent.key_axes[grasp_axis_name]
        grasp_axis=transform_keypoint_to_base(grasp_axis_base_hand,end_effector_pose)
        # grasp_axis=get_axis_in_env(self.env,axis_name=grasp_axis_name)
        print("grasp_axis:",grasp_axis)

        grasp_point_base_hand =self.env.agent.key_points[grasp_point_name]
        grasp_point=transform_keypoint_to_base(grasp_point_base_hand,end_effector_pose)
        # grasp_point = get_point_in_env(self.env, point_name=grasp_point_name)
        print("grasp_point:",grasp_point)

        print("### end compute_hand_info() !!! ###")

    def generate_end_effector_pose_tool(self,constraints,hand_name="right"):
        print("######################################################")
        # self.build_robot_plant()
        # print("################")
        # print("generate_end_effector_pose")
        # print("################")
        print("start generate_end_effector_pose_tool() !!!")
        # ik_context=self.robot_plant.GetMyContextFromRoot(self.fk_plant_context)
        # ik_context = self.robot_plant.CreateDefaultContext()
        # initial_guess = np.zeros(self.robot_plant.num_positions())
        # # 例如，设置为当前状态或合理猜测
        # self.robot_plant.SetPositions(self.fk_plant_context, initial_guess)
        # fk_context = self.robot_plant.CreateDefaultContext()
        ik = drake.InverseKinematics(self.robot_plant)

        # collision_model = self.robot_plant.CreateCollisionModel()
        print("constraints num: ",len(constraints))
        for constraint in constraints:
            if isinstance(constraint, Constraint):
                if constraint.type == "point2point":
                    # constraint.object_key_point[2]=constraint.object_key_point[2]+0.035
                    # hand_key_point=constraint.hand_key_point
                    # object_key_point=constraint.object_key_point
                    range_min = constraint.object_key_point-np.array([0.005,0.005,0.001])
                    range_max = constraint.object_key_point + np.array([0.005,0.005,0.001])
                    ik.AddPositionConstraint(
                        self.robot_plant.GetFrameByName(constraint.end_effector_frame),
                        constraint.hand_key_point,
                        self.robot_plant.world_frame(),
                        # constraint.object_key_point - constraint.tolerance,
                        # constraint.object_key_point + constraint.tolerance,
                        range_min,
                        range_max
                    )
                elif constraint.type == "parallel":
                    # if constraint.hand_axis=="grasp_axis":
                    ik.AddAngleBetweenVectorsConstraint(
                        self.robot_plant.GetFrameByName(constraint.end_effector_frame),
                        constraint.hand_axis.reshape(3, 1),
                        self.robot_plant.world_frame(),
                        constraint.object_axis.reshape(3, 1),
                        angle_lower=0,
                        angle_upper=0.02,
                    )
                    # ik.AddAngleBetweenVectorsCost(
                    #     self.robot_plant.GetFrameByName(constraint.end_effector_frame),
                    #     constraint.hand_axis.reshape(3, 1),
                    #     self.robot_plant.world_frame(),
                    #     constraint.object_axis.reshape(3, 1),
                    #     # angle_lower=0,
                    #     # angle_upper=0.00001,
                    #     c=10
                    # )
                elif constraint.type == "attach_obj_target_pose":
                    # constraint.attach_obj.pose.q
                    # print("constraint.attach_obj.pose.q:",constraint.attach_obj.pose.q)
                    # quat2mat(constraint.attach_obj.pose.get_q()[0])
                    # constraint.attach_obj.pose.to_transformation_matrix()
                    # quat2mat(constraint.attach_obj.pose.get_q()[0])
                    # quat2mat(constraint.attach_obj.pose.get_q()[0])
                    R_AbarA=pydrake.math.RotationMatrix(constraint.attach_obj_target_pose_base_hand)
                    R_BbarB=pydrake.math.RotationMatrix(constraint.attach_obj_target_pose)
                    ik.AddOrientationConstraint(
                        self.robot_plant.GetFrameByName(constraint.end_effector_frame),
                        R_AbarA,
                        self.robot_plant.world_frame(),
                        R_BbarB,
                        theta_bound=0.001
                    )
                    # ik.AddOrientationCost(
                    #     self.robot_plant.GetFrameByName(constraint.end_effector_frame),
                    #     R_AbarA,
                    #     self.robot_plant.world_frame(),
                    #     R_BbarB,
                    #     c=10
                    # )
            elif isinstance(constraint, Cost):
                if constraint.type == "point2point":
                    # constraint.object_key_point[2]=constraint.object_key_point[2]+0.035
                    # hand_key_point=constraint.hand_key_point
                    # object_key_point=constraint.object_key_point
                    # range_min = constraint.object_key_point-np.array([0.01,0.01,0.001])
                    # range_max = constraint.object_key_point + np.array([0.01,0.01,0.001])
                    temp = np.eye(3)
                    ik.AddPositionCost(
                        self.robot_plant.GetFrameByName(constraint.end_effector_frame),
                        constraint.hand_key_point,
                        self.robot_plant.world_frame(),
                        constraint.object_key_point,
                        # constraint.object_key_point - constraint.tolerance,
                        # constraint.object_key_point + constraint.tolerance,
                        temp
                    )
                elif constraint.type == "parallel":
                    # if constraint.hand_axis=="grasp_axis":
                    # ik.AddAngleBetweenVectorsConstraint(
                    #     self.robot_plant.GetFrameByName(constraint.end_effector_frame),
                    #     np.array([0, 0, -1]).reshape(3, 1),
                    #     self.robot_plant.world_frame(),
                    #     constraint.object_axis.reshape(3, 1),
                    #     angle_lower=0,
                    #     angle_upper=0.001,
                    # )
                    ik.AddAngleBetweenVectorsCost(
                        self.robot_plant.GetFrameByName(constraint.end_effector_frame),
                        constraint.hand_axis.reshape(3, 1),
                        self.robot_plant.world_frame(),
                        constraint.object_axis.reshape(3, 1),
                        # angle_lower=0,
                        # angle_upper=0.00001,
                        c=10
                    )
                elif constraint.type == "attach_obj_target_pose":
                    # constraint.attach_obj.pose.q
                    # print("constraint.attach_obj.pose.q:",constraint.attach_obj.pose.q)
                    # quat2mat(constraint.attach_obj.pose.get_q()[0])
                    # constraint.attach_obj.pose.to_transformation_matrix()
                    # quat2mat(constraint.attach_obj.pose.get_q()[0])
                    # quat2mat(constraint.attach_obj.pose.get_q()[0])
                    R_AbarA=pydrake.math.RotationMatrix(constraint.attach_obj_target_pose_base_hand)
                    R_BbarB=pydrake.math.RotationMatrix(constraint.attach_obj_target_pose)
                    ik.AddOrientationCost(
                        self.robot_plant.GetFrameByName(constraint.end_effector_frame),
                        R_AbarA,
                        self.robot_plant.world_frame(),
                        R_BbarB,
                        c=10
                    )
        # print("drake.math.RotationMatrix.MakeXRotation(0):",pydrake.math.RotationMatrix.MakeXRotation(0))
        # print("drake.math.RotationMatrix.MakeXRotation(0):",pydrake.math.RotationMatrix.MakeYRotation(0))
        # R_AbarA = pydrake.math.RotationMatrix.MakeXRotation(-1.57)  #
        # R_BbarB = pydrake.math.RotationMatrix.MakeYRotation(-0.9)  # 

        # ik.AddOrientationConstraint(
        #     self.robot_plant.GetFrameByName("r_hand_base_link"),
        #     R_AbarA,
        #     self.robot_plant.world_frame(),
        #     R_BbarB,
        #     theta_bound=0.001
        # )
        
        # R_AbarA = pydrake.math.RotationMatrix.MakeZRotation(-2.5)  #
        # R_BbarB = pydrake.math.RotationMatrix.MakeYRotation(1.57)  # 
        #### test
        # ik.AddOrientationCost(
        #     self.robot_plant.GetFrameByName("r_hand_base_link"),
        #     R_AbarA,
        #     self.robot_plant.world_frame(),
        #     R_BbarB,
        #     c=1
        # )
        
        ####
        if hand_name == "right":
            end_effector_frame = "r_hand_base_link"
            # let hand over the table
            ik.AddPositionConstraint(
                self.robot_plant.GetFrameByName("r_hand_base_link"),
                [0, 0, 0],
                self.robot_plant.world_frame(),
                [-100, -100, 0],
                [100, 100, 100],
            )
            ik.AddPositionConstraint(
                self.robot_plant.GetFrameByName("r_hand_base_link"),
                [0, -0.030, 0.045],
                self.robot_plant.world_frame(),
                [-100, -100, 0],
                [100, 100, 100],
            )
        elif hand_name == "left":
            end_effector_frame = "l_hand_base_link"
            # let hand over the table
            ik.AddPositionConstraint(
                self.robot_plant.GetFrameByName("l_hand_base_link"),
                [0, 0, 0],
                self.robot_plant.world_frame(),
                [-100, -100, 0],
                [100, 100, 100],
            )
            ik.AddPositionConstraint(
                self.robot_plant.GetFrameByName("l_hand_base_link"),
                [0, -0.030, 0.045],
                self.robot_plant.world_frame(),
                [-100, -100, 0],
                [100, 100, 100],
            )
        
        # ik.AddMinimumDistanceLowerBoundConstraint(0.01, influence_distance_offset=0.01)
        solver = drake.SnoptSolver()
        options = drake.SolverOptions()
        # options.SetOption(solver.solver_id(), "Major feasibility tolerance", 1e-6)
        # options.SetOption(solver.solver_id(), "Major optimality tolerance", 1e-6)
        # options.SetOption(solver.solver_id(), "Iterations limit", 1000)
        # initial_guess = np.zeros(self.robot_plant.num_positions())
        # 例如，设置为当前状态或合理猜测
        # self.robot_plant.SetPositions(self.fk_plant_context, initial_guess)
        # ik.prog().SetInitialGuess(ik.q(), initial_guess)
        # for i in range(10):
        #     result = solver.Solve(ik.prog())
        #     print("result:",result.is_success())
        #     result = solver.Solve(ik.prog())
        #     print("result:",result.is_success())

        result = solver.Solve(ik.prog())

        if not result.is_success():
            print("!!!!!!!!!!IK solve failure!!!! \nDetail:", result.get_solver_details().info)
            # return None
        self.robot_plant.SetPositions(self.fk_plant_context, result.get_x_val())
        task_goal_hand_pose = self.robot_plant.EvalBodyPoseInWorld(
            # self.fk_context, self.plant.GetBodyByName("panda_hand")
            self.fk_plant_context, self.robot_plant.GetBodyByName(end_effector_frame)
        )
        print("IK结果验证：", result.GetSolution(ik.q()))

        if self.debug_key_frame:
            # check collision
            self.check_collision(result.GetSolution(ik.q()),save_photo=False,render=True,hand_name=hand_name)

        # convert from robot frame to base frame
        target_effector_pose=pack_pose_to_sapien(task_goal_hand_pose)
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3]=quat2mat(self.env.agent.base_link.pose.get_q()[0])
        transform_matrix[:3, 3]=self.env.agent.base_link.pose.get_p().numpy()
        right_tcp_matrix=np.eye(4)
        right_tcp_matrix[:3, :3]=quat2mat(target_effector_pose.q)
        right_tcp_matrix[:3, 3]=target_effector_pose.p
        target_effector_pose=pack_pose_to_sapien(transform_matrix @ right_tcp_matrix)

        # print("##################################\nIk solve completed!!!")
        print(hand_name," target_effector_pose:\n",target_effector_pose)
        # print("check pinch pose:")
        # print(transform_keypoint_to_base(constraints[0].hand_key_point,target_effector_pose))
        self.compute_hand_info(hand_name,target_effector_pose)
        print("End generate_end_effector_pose_tool() function!")
        print("######################################################")
        return result.get_x_val(),target_effector_pose
    
    def generate_end_effector_pose_test(self,constraints,hand_name="right"):

        print("################")
        print("generate_end_effector_pose")
        print("################")
        ik = drake.InverseKinematics(self.robot_plant)
        
        for constraint in constraints[0:2]:
            if isinstance(constraint, Constraint):
                
                if constraint.type == "point2point":
                    range_min = constraint.object_key_point-np.array([0.001,0.001,0.001])
                    range_max = constraint.object_key_point + np.array([0.001,0.001,0.001])
                    ik.AddPositionConstraint(
                        self.robot_plant.GetFrameByName(constraint.end_effector_frame),
                        constraint.hand_key_point,
                        self.robot_plant.world_frame(),
                        range_min,
                        range_max
                    )
                elif constraint.type == "parallel":
                    ik.AddAngleBetweenVectorsCost(
                        self.robot_plant.GetFrameByName(constraint.end_effector_frame),
                        constraint.hand_axis.reshape(3, 1),
                        self.robot_plant.world_frame(),
                        constraint.object_axis.reshape(3, 1),
                        c=10
                    )
                elif constraint.type == "attach_obj_target_pose":
                    R_AbarA=pydrake.math.RotationMatrix(constraint.attach_obj_target_pose_base_hand)
                    R_BbarB=pydrake.math.RotationMatrix(constraint.attach_obj_target_pose)

                    ik.AddOrientationCost(
                        self.robot_plant.GetFrameByName(constraint.end_effector_frame),
                        R_AbarA,
                        self.robot_plant.world_frame(),
                        R_BbarB,
                        c=10
                    )
        
        ####
        if hand_name == "right":
            end_effector_frame = "r_hand_base_link"
            # let hand over the table
            # ik.AddPositionConstraint(
            #     self.robot_plant.GetFrameByName("r_hand_base_link"),
            #     [0, 0, 0],
            #     self.robot_plant.world_frame(),
            #     [-100, -100, 0],
            #     [100, 100, 100],
            # )
            # ik.AddPositionConstraint(
            #     self.robot_plant.GetFrameByName("r_hand_base_link"),
            #     [0, -0.030, 0.045],
            #     self.robot_plant.world_frame(),
            #     [-100, -100, 0],
            #     [100, 100, 100],
            # )
            
        elif hand_name == "left":
            end_effector_frame = "l_hand_base_link"
            # let hand over the table
            ik.AddPositionConstraint(
                self.robot_plant.GetFrameByName("l_hand_base_link"),
                [0, 0, 0],
                self.robot_plant.world_frame(),
                [-100, -100, 0],
                [100, 100, 100],
            )
            # ik.AddPositionConstraint(
            #     self.robot_plant.GetFrameByName("l_hand_base_link"),
            #     [0, -0.030, 0.045],
            #     self.robot_plant.world_frame(),
            #     [-100, -100, 0],
            #     [100, 100, 100],
            # )
        
        # ik.AddMinimumDistanceLowerBoundConstraint(0.01, influence_distance_offset=0.01)
        solver = drake.SnoptSolver()
        options = drake.SolverOptions()

        result = solver.Solve(ik.prog())
        print("result1:",result.is_success())

        if not result.is_success():
            print("!!!!!!!!!!IK solve failure!!!! \nDetail:", result.get_solver_details().info)
        
        self.robot_plant.SetPositions(self.fk_plant_context, result.get_x_val())
        task_goal_hand_pose = self.robot_plant.EvalBodyPoseInWorld(
            # self.fk_context, self.plant.GetBodyByName("panda_hand")
            self.fk_plant_context, self.robot_plant.GetBodyByName(end_effector_frame)
        )
        # print("IK结果验证:", result.GetSolution(ik.q()))

        # convert from robot frame to base frame
        target_effector_pose=pack_pose_to_sapien(task_goal_hand_pose)
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3]=quat2mat(self.env.agent.base_link.pose.get_q()[0])
        transform_matrix[:3, 3]=self.env.agent.base_link.pose.get_p().numpy()
        right_tcp_matrix=np.eye(4)
        right_tcp_matrix[:3, :3]=quat2mat(target_effector_pose.q)
        right_tcp_matrix[:3, 3]=target_effector_pose.p
        target_effector_pose=pack_pose_to_sapien(transform_matrix @ right_tcp_matrix)

        print("##################################\nIk solve completed!!!")
        print("Target_effector_pose:\n",target_effector_pose)
        print("check pinch pose:")
        print(transform_keypoint_to_base(constraints[0].hand_key_point,target_effector_pose))
        print("######################################################")

        # return None
        ik2 = drake.InverseKinematics(self.robot_plant)
        for constraint in constraints[2:4]:
            if isinstance(constraint, Constraint):
                
                if constraint.type == "point2point":
                    range_min = constraint.object_key_point-np.array([0.01,0.01,0.001])
                    range_max = constraint.object_key_point + np.array([0.01,0.01,0.001])
                    ik2.AddPositionConstraint(
                        self.robot_plant.GetFrameByName(constraint.end_effector_frame),
                        constraint.hand_key_point,
                        self.robot_plant.world_frame(),
                        range_min,
                        range_max
                    )
                elif constraint.type == "parallel":
                    ik2.AddAngleBetweenVectorsCost(
                        self.robot_plant.GetFrameByName(constraint.end_effector_frame),
                        constraint.hand_axis.reshape(3, 1),
                        self.robot_plant.world_frame(),
                        constraint.object_axis.reshape(3, 1),
                        c=10
                    )
                elif constraint.type == "attach_obj_target_pose":
                    R_AbarA=pydrake.math.RotationMatrix(constraint.attach_obj_target_pose_base_hand)
                    R_BbarB=pydrake.math.RotationMatrix(constraint.attach_obj_target_pose)

                    ik2.AddOrientationCost(
                        self.robot_plant.GetFrameByName(constraint.end_effector_frame),
                        R_AbarA,
                        self.robot_plant.world_frame(),
                        R_BbarB,
                        c=10
                    )
        
        if hand_name == "right":
            end_effector_frame = "r_hand_base_link"

        solver = drake.SnoptSolver()
        options = drake.SolverOptions()

        result2 = solver.Solve(ik2.prog())

        if not result2.is_success():
            print("!!!!!!!!!!IK solve failure!!!! \nDetail:", result2.get_solver_details().info)
        print("result2:",result2.is_success())

        return result.get_x_val(),target_effector_pose
    
    def fk_robot(self, qpos,hand="right"):
        if hand == "right":
            ik_context = self.robot_plant.CreateDefaultContext()
            t_qpos = np.zeros(38)
            t_qpos[19:26] = qpos
            self.robot_plant.SetPositions(ik_context, t_qpos)
            task_goal_hand_pose = self.robot_plant.EvalBodyPoseInWorld(
                ik_context, self.robot_plant.GetBodyByName("r_hand_base_link")
            )
            target_effector_pose=pack_pose_to_sapien(task_goal_hand_pose)
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3]=quat2mat(self.env.agent.base_link.pose.get_q()[0])
            transform_matrix[:3, 3]=self.env.agent.base_link.pose.get_p().numpy()
            right_tcp_matrix=np.eye(4)
            right_tcp_matrix[:3, :3]=quat2mat(target_effector_pose.q)
            right_tcp_matrix[:3, 3]=target_effector_pose.p
            target_effector_pose=pack_pose_to_sapien(transform_matrix @ right_tcp_matrix)
        elif hand == "left":
            ik_context = self.robot_plant.CreateDefaultContext()
            t_qpos = np.zeros(38)
            t_qpos[0:7] = qpos
            self.robot_plant.SetPositions(ik_context, t_qpos)
            task_goal_hand_pose = self.robot_plant.EvalBodyPoseInWorld(
                ik_context, self.robot_plant.GetBodyByName("l_hand_base_link")
            )
            target_effector_pose=pack_pose_to_sapien(task_goal_hand_pose)
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3]=quat2mat(self.env.agent.base_link.pose.get_q()[0])
            transform_matrix[:3, 3]=self.env.agent.base_link.pose.get_p().numpy()
            right_tcp_matrix=np.eye(4)
            right_tcp_matrix[:3, :3]=quat2mat(target_effector_pose.q)
            right_tcp_matrix[:3, 3]=target_effector_pose.p
            target_effector_pose=pack_pose_to_sapien(transform_matrix @ right_tcp_matrix)
        return target_effector_pose

    def xyz_rpy_deg(self,xyz, rpy_deg):
        """Shorthand for defining a pose."""
        rpy_deg = np.asarray(rpy_deg)
        return RigidTransform(drake.RollPitchYaw(rpy_deg * np.pi / 180), xyz)

    ### Robot Plant
    def build_robot_plant(self):
        time_step = 0.004
        multibody_plant_config =drake.MultibodyPlantConfig(
            time_step=time_step,
            discrete_contact_solver="sap",
        )
        meshcat = StartMeshcat()
        # builder = DiagramBuilder()
        # self.robot_plant, scene_graph = AddMultibodyPlant(multibody_plant_config, builder)
        
        builder = drake.DiagramBuilder()
        # scene_graph = builder.AddSystem(SceneGraph())
        self.robot_plant = builder.AddSystem(drake.MultibodyPlant(time_step=0.01))
        # self.robot_plant.RegisterAsSourceForSceneGraph(scene_graph)
            
        # self.robot_plant.RegisterAsSourceForSceneGraph(scene_graph)
        parser = drake.Parser(self.robot_plant)

        # agent_parser = parser.AddModelFromFile(self.env_agent.drake_urdf_path)
        
        agent_parser=parser.AddModels(self.env_agent.drake_urdf_path)
        agent_parser = agent_parser[0]

        self.robot_plant.WeldFrames(self.robot_plant.world_frame(), self.robot_plant.GetFrameByName("pelvis"))
        for joint_index in self.robot_plant.GetJointIndices(agent_parser):
            joint = self.robot_plant.get_mutable_joint(joint_index)
            if isinstance(joint, drake.RevoluteJoint):
                joint.set_default_angle(0)
        
        # box_shape = Box(0.5, 0.5, 0.5)  
        # box_pose = RigidTransform([1.0, 0.0, 0.25])  
        # default_model_instance = self.robot_plant.GetModelInstanceByName("DefaultModelInstance")
        # box_body = self.robot_plant.AddRigidBody("box", default_model_instance, SpatialInertia(mass=1.0, p_PScm_E=np.zeros(3), G_SP_E=UnitInertia.SolidBox(0.5, 0.5, 0.5)))
        # # box_body = self.robot_plant.AddRigidBody("box", SpatialInertia(mass=1.0, p_PScm_E=np.zeros(3), G_SP_E=UnitInertia.SolidBox(0.5, 0.5, 0.5)))
        # self.robot_plant.RegisterCollisionGeometry(box_body, box_pose, box_shape, "box_collision", CoulombFriction(0.9, 0.8))
        # self.robot_plant.RegisterVisualGeometry(box_body, box_pose, box_shape, "box_visual", [0.5, 0.5, 0.5, 1.0])

        ##### render #####
        if self.render_pydrake:
            renderer_name = "renderer"
            scene_graph.AddRenderer(renderer_name, MakeRenderEngineVtk(RenderEngineVtkParams()))
            # Add camera with same color and depth properties.
            # N.B. These properties are chosen arbitrarily.
            intrinsics = drake.CameraInfo(
                width=640,
                height=480,
                fov_y=np.pi/4,
            )
            core = RenderCameraCore(
                renderer_name,
                intrinsics,
                ClippingRange(0.01, 10.0),
                RigidTransform(),
            )
            color_camera = ColorRenderCamera(core, show_window=False)
            depth_camera = DepthRenderCamera(core, DepthRange(0.01, 10.0))
            world_id = self.robot_plant.GetBodyFrameIdOrThrow(self.robot_plant.world_body().index())
            # X_WB = self.xyz_rpy_deg([2, 0, 0.75], [-90, 0, 90])
            X_WB = self.xyz_rpy_deg([2, 0, 0], [-90, 0, 90])
            sensor = drake.RgbdSensor(
                world_id,
                X_PB=X_WB,
                color_camera=color_camera,
                depth_camera=depth_camera,
            )

            builder.AddSystem(sensor)
            builder.Connect(
                scene_graph.get_query_output_port(),
                sensor.query_object_input_port(),
            )

            # Add depth and label colorizers.
            colorize_depth = builder.AddSystem(drake.ColorizeDepthImage())
            colorize_label = builder.AddSystem(drake.ColorizeLabelImage())
            colorize_label.background_color.set([0,0,0])
            builder.Connect(sensor.GetOutputPort("depth_image_32f"),
                            colorize_depth.GetInputPort("depth_image_32f"))
            builder.Connect(sensor.GetOutputPort("label_image"),
                            colorize_label.GetInputPort("label_image"))
        


        #### end render #####
        self.robot_plant.Finalize()
        # # Connect the plant and scene graph
        # builder.Connect(
        #     self.robot_plant.get_geometry_poses_output_port(),
        #     scene_graph.get_source_pose_port(self.robot_plant.get_source_id())
        # )
        # builder.Connect(
        #     scene_graph.get_query_output_port(),
        #     self.robot_plant.get_geometry_query_input_port()
        # )

        if self.render_pydrake:
            # Add visualization.
            drake.AddDefaultVisualization(builder=builder, meshcat=meshcat)

        diagram = builder.Build()
        fk_context = diagram.CreateDefaultContext()
        self.fk_plant_context = self.robot_plant.GetMyMutableContextFromRoot(fk_context)

        if self.render_pydrake:
            drake.Simulator(diagram).Initialize()
            color = sensor.color_image_output_port().Eval(
                sensor.GetMyContextFromRoot(fk_context)).data
            depth = colorize_depth.get_output_port().Eval(
                colorize_depth.GetMyContextFromRoot(fk_context)).data
            label = colorize_label.get_output_port().Eval(
                colorize_label.GetMyContextFromRoot(fk_context)).data

            # fig, ax = plt.subplots(1, 3, figsize=(15, 10))
            # ax[0].imshow(color)
            # ax[1].imshow(depth)
            # ax[2].imshow(label)
            plt.imsave( ROOT_PATH/'imgs/color_image.png', color)
            plt.imsave( ROOT_PATH/'imgs/depth_image.png', depth, cmap='gray')  # 深度图通常使用灰度图
            plt.imsave( ROOT_PATH/'imgs/label_image.png', label)

    def show_collision_point_cloud(self,point_cloud):
        if "collision_point_cloud" in self.base_env.scene.actors:
            # self.base_env.scene.remove_actor(self.collision_point_cloud)
            # self.env.scene.actors["collision_point_cloud"].remove()
            self.base_env.scene.remove_from_state_dict_registry(self.collision_point_cloud)
            self.env.scene.actors.pop("collision_point_cloud")
            self.collision_point_cloud.remove_from_scene()
            # self.collision_point_cloud=self.base_env.scene.actors["collision_point_cloud"]
            # self.collision_point_cloud.set_pose(point_cloud)
        # else:
        builder = self.env.scene.create_actor_builder()
        hand_pre_pose_width = 0.005
        for point in point_cloud:
            builder.add_sphere_visual(
                pose=sapien.Pose(p=point),
                radius=hand_pre_pose_width,
                material=sapien.render.RenderMaterial(base_color=[0.5, 0.5, 0.5, 0.7])
            )
        builder.initial_pose = sapien.Pose(p=[0, 0, 0])
        self.collision_point_cloud = builder.build_kinematic(name="collision_point_cloud")
        # self.collision_point_cloud.set_pose(sapien.Pose([0, 0, 0]))
        # self.collision_point_cloud.hide_visual()
        # self.collision_point_cloud.show_visual()


    def init_all_point_cloud(self):
        point_cloud = []
        for type_name in self.env.actor_count:
            for i in range(self.env.actor_count[type_name]):
                point_cloud.append(getattr(self.env, type_name)[i])
        self.add_point_cloud(point_cloud)
    
    def update_point_cloud(self,except_actor,except_actor_id=0, hand_name="all"):
        if isinstance(except_actor, str):
            if "left" in hand_name:
                self.attach_obj[0] = except_actor
                self.attach_obj_id[0] = except_actor_id
            elif "right" in hand_name:
                self.attach_obj[1] = except_actor
                self.attach_obj_id[1] = except_actor_id
        elif isinstance(except_actor, list):
            if len(except_actor)==1:
                if "left" in hand_name:
                    self.attach_obj[0] = except_actor[0]
                    self.attach_obj_id[0] = except_actor_id[0]
                elif "right" in hand_name:
                    self.attach_obj[1] = except_actor[0]
                    self.attach_obj_id[1] = except_actor_id[0]
            elif len(except_actor)==2:
                self.attach_obj = except_actor
                self.attach_obj_id = except_actor_id
        # except_actors=[]
        # except_actor_ids=[]
        except_actors = self.attach_obj
        except_actor_ids= self.attach_obj_id
        # if isinstance(except_actor, str):
        #     except_actors = [except_actor]
        # elif isinstance(except_actor, list):
        #     except_actors = except_actor
        # if isinstance(except_actor_id, int):
        #     except_actor_ids = [except_actor_id]
        # elif isinstance(except_actor_id, list):
        #     except_actor_ids = except_actor_id
        point_cloud = []
        for type_name in self.env.actor_count:
            for i in range(self.env.actor_count[type_name]):
                is_found = False
                for j in range(len(except_actors)):
                    if except_actors[j] is not None and except_actors[j] == type_name and except_actor_ids[j] == i:
                        is_found= True
                        break
                if type_name =="cube_small":
                    is_found= True
                if not is_found:
                    point_cloud.append(getattr(self.env, type_name)[i])
        self.add_point_cloud(point_cloud,hand_name)

    def add_point_cloud(self, actors: list[Actor],hand_name="all"):
        import trimesh
        point_cloud =np.array([[]])
        # get all actors' collision mesh
        if self.use_obj_point_cloud:
            for actor in actors:
                # if isinstance(actor, ArticulatedObject):
                #     actor.instance.set_root_pose(np.array([0, 0, 0,1,0,0,0]))
                # 获取 Actor 的碰撞网格
                collision_meshes = actor.get_collision_meshes()
                # print("collision_meshes:",type(collision_meshes[0]))
                if actor.name == "drawer_0":
                    new_point_cloud, _ = trimesh.sample.sample_surface(collision_meshes[0], count=400000)
                else:
                    new_point_cloud, _ = trimesh.sample.sample_surface(collision_meshes[0], count=10000)
                # if isinstance(actor, ArticulatedObject):
                    # new_point_cloud-=actor.instance.get_root_pose().p.numpy()
                # new_point_cloud *= 1.1
                if point_cloud.shape[1] == 0:
                    point_cloud = new_point_cloud
                else:
                    point_cloud = np.concatenate([point_cloud, new_point_cloud], axis=0)
        
        # get table collision mesh
        # table 1
        # table = trimesh.creation.box([1.209,2.418,  0.9196429])
        # table_point_cloud,_ = trimesh.sample.sample_surface(table, count=1000)
        # table_point_cloud += [-0.12, 0, -0.9196429 / 2]
        # table 2
        table = trimesh.creation.box([1.209,2.418,  0.025])
        table_point_cloud,_ = trimesh.sample.sample_surface(table, count=30000)
        table_point_cloud += [-0.12, 0, -0.05 / 2+0.008]

        if point_cloud.shape[1] == 0:
                point_cloud = table_point_cloud
        else:
            point_cloud = np.concatenate([point_cloud, table_point_cloud], axis=0)
        if hand_name == "all":
            self.l_planner.update_point_cloud(point_cloud)
            self.r_planner.update_point_cloud(point_cloud)
        elif hand_name == "left_hand":
            self.l_planner.update_point_cloud(point_cloud)
        elif hand_name == "right_hand":
            self.r_planner.update_point_cloud(point_cloud)
        if self.show_collision_cloud:
            self.show_collision_point_cloud(point_cloud)
        return point_cloud

        # # 获取 Actor 的碰撞网格
        # collision_meshes = actor.get_collision_meshes()
        # # 存储所有点的列表
        # # all_points = []

        # # for mesh in collision_meshes:
        # #     # 获取几何体的顶点
        # #     vertices = mesh.vertices
        # #     # 将顶点转换为 numpy 数组
        # #     vertices = np.array(vertices)
        # #     # 将顶点转换为世界坐标系
        # #     vertices = actor.pose.transform_points(vertices)
        # #     # 将顶点添加到点列表中
        # #     all_points.append(vertices)

        # # 将所有点合并为一个 numpy 数组
        # # all_points = np.concatenate(all_points, axis=0)
        # # 使用 trimesh 采样表面点
        # point_cloud, _ = trimesh.sample.sample_surface(collision_meshes[0], count=100)
        # box = trimesh.creation.box([2.418 / 2, 1.209 / 2, 0.9196429 / 2])
        # point_cloud2, _ = trimesh.sample.sample_surface(box, count=100)
        # point_cloud2 += [0, 0, 0.9196429 / 2]
        # # point_cloud = np.concatenate([point_cloud, point_cloud2], axis=0)
        # self.planner.update_point_cloud(point_cloud)
        # # point_cloud -= [3.5, -2, 1]
        # # 更新点云到规划器
        # # self.planner.update_point_cloud(point_cloud)
        
    # def add_table(self):
    #     import trimesh
    #     box = trimesh.creation.box([2.418 / 2, 1.209 / 2, 0.9196429 / 2])
    #     points, _ = trimesh.sample.sample_surface(box, 1000)
    #     points += [0, 0, 0.9196429 / 2]
    #     self.planner.update_point_cloud(points)

    def show_attach_point_cloud(self,point_cloud):
        if not self.show_key_points:
            return
        if "attach_point_cloud" in self.base_env.scene.actors:
            self.base_env.scene.remove_from_state_dict_registry(self.attach_point_cloud)
            self.env.scene.actors.pop("attach_point_cloud")
            self.attach_point_cloud.remove_from_scene()
        builder = self.env.scene.create_actor_builder()
        hand_pre_pose_width = 0.005
        for point in point_cloud:
            builder.add_sphere_visual(
                pose=sapien.Pose(p=point),
                radius=hand_pre_pose_width,
                material=sapien.render.RenderMaterial(base_color=[1.0, 1.0, 0.0, 0.7])
            )
        builder.initial_pose = sapien.Pose(p=[0, 0, 0])
        self.attach_point_cloud = builder.build_kinematic(name="attach_point_cloud")

    # def add_attach_obj(self, actor: Actor,hand_name="right"):
    def add_attach_obj(self, hand_name="right",obj_name="can"):
        if "right" in hand_name :
            if isinstance(obj_name, list):
                obj_name = obj_name[1]
            if obj_name=="cube" :
                show_box =[0.04, 0.04, 0.04]
                attach_box_collision = [0.04, 0.04, 0.04]
            elif obj_name=="drawer" :
                show_box =[0.04, 0.04, 0.04]
                attach_box_collision = [0.04, 0.04, 0.04]
            elif obj_name=="rectangular_cube":
                show_box =[0.04, 0.04, 0.04]
                # attach_box_collision = [0.30, 0.20, 0.25]
                attach_box_collision = [0.30, 0.30, 0.3]
            elif obj_name=="laptop" or obj_name=="box":
                show_box =[0.04, 0.04, 0.04]
                attach_box_collision = [0.04, 0.04, 0.04]
            else:
                show_box =[0.10, 0.10, 0.10]
                # attach_box_collision = [0.30, 0.30, 0.30]
                attach_box_collision = [0.30, 0.20, 0.25]

            self.use_attach[1] = True
            attach_box = trimesh.creation.box(show_box)
            attach_box_points, _ = trimesh.sample.sample_surface(attach_box, 100)
            attach_box_points+=np.array(transform_keypoint_to_base(np.array([-0.0485, -0.13, -0.025]),self.env.agent.right_tcp.pose))
            self.show_attach_point_cloud(attach_box_points)
            # self.r_planner.update_attached_box([0.30, 0.20, 0.25], [-0.0485, -0.13, -0.025, 1, 0, 0, 0])
            self.r_planner.update_attached_box(attach_box_collision, [-0.07445, -0.145609, -0.0281028, 1, 0, 0, 0])
        
        elif "left" in hand_name:
            if isinstance(obj_name, list):
                obj_name = obj_name[0]
            if obj_name=="cube" :
                show_box =[0.04, 0.04, 0.04]
                attach_box_collision = [0.04, 0.04, 0.04]
            elif obj_name=="rectangular_cube":
                show_box =[0.04, 0.04, 0.04]
                attach_box_collision = [0.10, 0.10, 0.10]
            else:
                show_box =[0.10, 0.10, 0.10]
                attach_box_collision = [0.30, 0.20, 0.25]
            
            self.use_attach[0] = True
            attach_box = trimesh.creation.box(show_box)
            attach_box_points, _ = trimesh.sample.sample_surface(attach_box, 100)
            attach_box_points+=np.array(transform_keypoint_to_base(np.array([-0.0485, -0.13, -0.025]),self.env.agent.left_tcp.pose))
            self.show_attach_point_cloud(attach_box_points)
            # self.l_planner.update_attached_box([0.30, 0.20, 0.25], [-0.0485, -0.13, -0.025, 1, 0, 0, 0])
            self.l_planner.update_attached_box(attach_box_collision, [-0.07445, -0.145609, 0.0281028, 1, 0, 0, 0])
            
    def remove_attach_obj(self, hand_name = "right"):
        if hand_name == "right":
            self.use_attach[1] = False
            self.r_planner.planning_world.set_use_attach(False)
            if "attach_point_cloud" in self.base_env.scene.actors:
                self.base_env.scene.remove_from_state_dict_registry(self.attach_point_cloud)
                self.env.scene.actors.pop("attach_point_cloud")
                self.attach_point_cloud.remove_from_scene()
        elif hand_name == "left":
            self.use_attach[0] = False
            self.l_planner.planning_world.set_use_attach(False)
            if "attach_point_cloud" in self.base_env.scene.actors:
                self.base_env.scene.remove_from_state_dict_registry(self.attach_point_cloud)
                self.env.scene.actors.pop("attach_point_cloud")
                self.attach_point_cloud.remove_from_scene()

    def show_path(self, point_cloud):
        if ("path_point_cloud" in self.base_env.scene.actors)and hasattr(self,"path_point_cloud"):
            self.base_env.scene.remove_from_state_dict_registry(self.path_point_cloud)
            if "path_point_cloud" in self.env.scene.actors:
                self.env.scene.actors.pop("path_point_cloud")
            # self.env.scene.actors.pop("path_point_cloud")
            self.path_point_cloud.remove_from_scene()
        builder = self.env.scene.create_actor_builder()
        hand_pre_pose_width = 0.005
        for point in point_cloud:
            builder.add_sphere_visual(
                pose=sapien.Pose(p=point),
                radius=hand_pre_pose_width,
                material=sapien.render.RenderMaterial(base_color=[0.0, 1.0, 0.0, 0.7])
            )
        builder.initial_pose = sapien.Pose(p=[0, 0, 0])
        self.path_point_cloud = builder.build_kinematic(name="path_point_cloud")

    ### new functions
    def hand_pre_grasp(self, hand_name="all",t=6):

        # self.update_point_cloud(except_actor=None,hand_name=hand_name)
        if hand_name == "all":
            self.right_hand_pre_grasp(t)
            self.left_hand_pre_grasp(t)
        elif hand_name == "right":
            for i in range(10):
                self.right_hand_pre_grasp(t)
                # time.sleep(0.1)
        elif hand_name == "left":
            for i in range(10):
                self.left_hand_pre_grasp(t)
                # time.sleep(0.1)
        else:
            print("Invalid hand selected")
        self.execution_result.append([1,1,"success"])

    def hand_grasp(self, hand_name, grasp_object=None,obj_id=None,t=50):
        if "right" in hand_name :
            self.update_point_cloud(grasp_object,obj_id,hand_name="right_hand")
            self.right_hand_grasp(t)
            result_description=""
            object_in_hand =self.judge_object_in_hand("right",grasp_object,obj_id)
            if object_in_hand:
                result_description=f"success grasp {grasp_object}{obj_id}"
            else:
                result_description=f"failed grasp {grasp_object}{obj_id}"
            self.execution_result.append([2,object_in_hand,result_description])
        elif "left" in hand_name:
            self.update_point_cloud(grasp_object,obj_id,hand_name="left_hand")
            self.left_hand_grasp(t)
            result_description=""
            object_in_hand =self.judge_object_in_hand("left",grasp_object,obj_id)
            if object_in_hand:
                result_description=f"success grasp {grasp_object}{obj_id}"
            else:
                result_description=f"failed grasp {grasp_object}{obj_id}"
            self.execution_result.append([2,object_in_hand,result_description])
        else:
            print("Invalid hand selected")
            result_description="Invalid parameter hand_name. Please choose from 'right', or 'left'."
            self.execution_result.append([2,False,result_description])
        

    #### finger base manipulation control
    ## hand pre pinch
    def left_hand_pre_pinch(self, t=6):
        # print("right hand pre grasp!!!!")
        # default_pose=self.get_default_pose()
        # 
        # qpos = self.left_hand_action_to_pose(action,default_pose)
        # t=50
        # del_action = (np.array(qpos)-default_pose)/t
        t=50
        default_pose=self.get_default_pose()
        # action=[1.174, 0.5] + [0] * 10
        # qpos = self.left_hand_action_to_pose(action,np.zeros(38))
        action=self.left_hand_pre_pinch_joint   # target joint
        qpos = self.left_hand_action_to_pose(action,np.zeros(38),low_joint=True)
        hand_joint =self.get_hand_joint("left")
        base_qpos = self.left_hand_action_to_pose(hand_joint,np.zeros(38))
        del_action = (qpos-base_qpos)/t
        for i in range(t):
            obs, reward, terminated, truncated, info = self.env.step(default_pose+del_action*(i+1))
            if self.vis:
                self.base_env.render()
            if self.save_video:
                rgb = self.env.unwrapped.render_rgb_array()
                self.images.append(rgb)
        self.gripper_state[0] = HandState.PRE_PINCH

    def right_hand_pre_pinch(self, t=6):
        # print("right hand pre grasp!!!!")
        
        t=10
        default_pose=self.get_default_pose()
        # action=[1.174, 0.1] + [0] * 10
        # qpos = self.right_hand_action_to_pose(action,np.zeros(38))
        action=self.right_hand_pre_pinch_joint   # target joint
        qpos = self.right_hand_action_to_pose(action,np.zeros(38),low_joint=True)
        hand_joint =self.get_hand_joint("right")
        base_qpos = self.right_hand_action_to_pose(hand_joint,np.zeros(38))
        del_action = (qpos-base_qpos)/t
        for i in range(t):
            obs, reward, terminated, truncated, info = self.env.step(default_pose+del_action*(i+1))
            # obs, reward, terminated, truncated, info = self.env.step(qpos)
            if self.vis:
                self.base_env.render()
            if self.save_video:
                rgb = self.env.unwrapped.render_rgb_array()
                self.images.append(rgb)
        self.gripper_state[1] = HandState.PRE_PINCH
    
    def hand_pre_pinch(self, hand_name="all",t=6):
        if "all" in hand_name:
            self.right_hand_pre_pinch(t)
            self.left_hand_pre_pinch(t)
            self.execution_result.append([3,True,"sucess"])
        elif "right" in hand_name:
            # for i in range(10):
            self.right_hand_pre_pinch(t)
                # time.sleep(0.1)
            self.execution_result.append([3,True,"sucess"])
        elif "left" in hand_name:
            # for i in range(10):
            self.left_hand_pre_pinch(t)
                # time.sleep(0.1)
            self.execution_result.append([3,True,"sucess"])
        else:
            print("Invalid hand selected")
            self.execution_result.append([3,False,"Invalid parameter hand_name. Please choose from 'all', 'right', or 'left'."])
    
    #### hand pinch
    # def left_hand_pinch(self, t=50):
    #     if self.gripper_state[0] != HandState.PRE_PINCH:
    #         exception_str = "left hand is not in pre-pinch state. Cannot pinch."
    #         print(exception_str)
    #         raise Exception(exception_str)
    #     default_pose=self.get_default_pose()
    #     # print("right hand grasp!!!!")
    #     # action= [1.174, 1.0]+[0] * 3+[0.167, 0.664]+[0] * 3+[0.39, 0.534]
    #     # qpos = self.left_hand_action_to_pose(action,default_pose)
    #     action= self.left_hand_pinch_joint
    #     qpos = self.left_hand_action_to_pose(action,default_pose,low_joint=True)

    #     # del_action = (np.array(qpos)-self.robot.get_qpos()[0, :38].cpu().numpy())/t
    #     self.delta_pinch_action_left = (np.array(qpos)-default_pose)/t
    #     for i in range(t):
    #         hand_joint =self.get_hand_joint("left")
    #         default_pose = self.left_hand_action_to_pose(hand_joint,default_pose)
    #         obs, reward, terminated, truncated, info = self.env.step(default_pose+self.delta_pinch_action_left)
    #         if self.vis:
    #             self.base_env.render()
    #         if self.save_video:
    #             rgb = self.env.unwrapped.render_rgb_array()
    #             self.images.append(rgb)
    #     for j in range(15):
    #         hand_joint =self.get_hand_joint("left")
    #         default_pose = self.left_hand_action_to_pose(hand_joint,default_pose)
    #         obs, reward, terminated, truncated, info = self.env.step(qpos)
    #         if self.vis:
    #             self.base_env.render()
    #         if self.save_video:
    #             rgb = self.env.unwrapped.render_rgb_array()
    #             self.images.append(rgb)
    #     self.gripper_state[0] = HandState.PINCH

    # def right_hand_pinch(self, t=50):
    #     if self.gripper_state[1] != HandState.PRE_PINCH:
    #         exception_str = "right hand is not in pre-grasp state. Cannot grasp."
    #         print(exception_str)
    #         raise Exception(exception_str)
    #     default_pose=self.get_default_pose()
    #     # print("right hand grasp!!!!")
    #     # action=[1.174, 1.0]+[0] * 3+[0.167, 0.664]+[0] * 3+[0.39, 0.534]
    #     # qpos = self.right_hand_action_to_pose(action,default_pose)
    #     action=self.right_hand_pinch_joint
    #     qpos = self.right_hand_action_to_pose(action,default_pose,low_joint=True)
    #     # del_action = (np.array(qpos)-self.robot.get_qpos()[0, :38].cpu().numpy())/t
    #     self.delta_pinch_action_right = (np.array(qpos)-default_pose)/t
    #     for i in range(t):
    #         hand_joint =self.get_hand_joint("right")
    #         default_pose = self.right_hand_action_to_pose(hand_joint,default_pose)
    #         obs, reward, terminated, truncated, info = self.env.step(default_pose+self.delta_pinch_action_right)
    #         if self.vis:
    #             self.base_env.render()
    #         if self.save_video:
    #             rgb = self.env.unwrapped.render_rgb_array()
    #             self.images.append(rgb)
    #     for j in range(15):
    #         hand_joint =self.get_hand_joint("right")
    #         default_pose = self.right_hand_action_to_pose(hand_joint,default_pose)
    #         obs, reward, terminated, truncated, info = self.env.step(default_pose+self.delta_pinch_action_right)
    #         if self.vis:
    #             self.base_env.render()
    #         if self.save_video:
    #             rgb = self.env.unwrapped.render_rgb_array()
    #             self.images.append(rgb)
    #     self.gripper_state[1] = HandState.PINCH
    
    def left_hand_pinch(self, t=50):
        if self.gripper_state[0] != HandState.PRE_PINCH:
            exception_str = "left hand is not in pre-pinch state. Cannot pinch."
            print(exception_str)
            raise Exception(exception_str)
        default_pose=self.get_default_pose()
        # print("right hand grasp!!!!")
        # action= [1.174, 1.0]+[0] * 3+[0.167, 0.664]+[0] * 3+[0.39, 0.534]
        # qpos = self.left_hand_action_to_pose(action,default_pose)
        action= self.left_hand_pinch_joint
        qpos = self.left_hand_action_to_pose(action,default_pose,low_joint=True)

        # del_action = (np.array(qpos)-self.robot.get_qpos()[0, :38].cpu().numpy())/t
        self.delta_pinch_action_left = (np.array(qpos)-default_pose)/t
        # print("t:",t)
        for i in range(t):
            # print("i:",i)
            # hand_joint =self.get_hand_joint("left")
            # default_pose = self.left_hand_action_to_pose(hand_joint,default_pose)
            obs, reward, terminated, truncated, info = self.env.step(default_pose+self.delta_pinch_action_left*(i+1))
            if self.vis:
                self.base_env.render()
            if self.save_video:
                rgb = self.env.unwrapped.render_rgb_array()
                self.images.append(rgb)
        for j in range(15):
            # hand_joint =self.get_hand_joint("left")
            # default_pose = self.left_hand_action_to_pose(hand_joint,default_pose)
            obs, reward, terminated, truncated, info = self.env.step(qpos)
            if self.vis:
                self.base_env.render()
            if self.save_video:
                rgb = self.env.unwrapped.render_rgb_array()
                self.images.append(rgb)
        self.gripper_state[0] = HandState.PINCH

    def right_hand_pinch(self, t=50):
        if self.gripper_state[1] != HandState.PRE_PINCH:
            exception_str = "right hand is not in pre-grasp state. Cannot grasp."
            print(exception_str)
            raise Exception(exception_str)
        default_pose=self.get_default_pose()
        # print("right hand grasp!!!!")
        # action=[1.174, 1.0]+[0] * 3+[0.167, 0.664]+[0] * 3+[0.39, 0.534]
        # qpos = self.right_hand_action_to_pose(action,default_pose)
        action=self.right_hand_pinch_joint
        qpos = self.right_hand_action_to_pose(action,default_pose,low_joint=True)
        # del_action = (np.array(qpos)-self.robot.get_qpos()[0, :38].cpu().numpy())/t
        # print("t:",t)
        self.delta_pinch_action_right = (np.array(qpos)-default_pose)/t
        for i in range(t):
            # print("i:",i)
            # hand_joint =self.get_hand_joint("right")
            # default_pose = self.right_hand_action_to_pose(hand_joint,default_pose)
            obs, reward, terminated, truncated, info = self.env.step(default_pose+self.delta_pinch_action_right*(i+1))
            if self.vis:
                self.base_env.render()
            if self.save_video:
                rgb = self.env.unwrapped.render_rgb_array()
                self.images.append(rgb)
        for j in range(15):
            hand_joint =self.get_hand_joint("right")
            default_pose = self.right_hand_action_to_pose(hand_joint,default_pose)
            obs, reward, terminated, truncated, info = self.env.step(qpos)
            if self.vis:
                self.base_env.render()
            if self.save_video:
                rgb = self.env.unwrapped.render_rgb_array()
                self.images.append(rgb)
        self.gripper_state[1] = HandState.PINCH

    def hand_pinch(self, hand_name,pinch_object=None,obj_id=0, t=100):
        if "right" in hand_name:
            self.update_point_cloud(pinch_object,obj_id,hand_name="right_hand")
            self.right_hand_pinch(t)
            result_description=""
            object_in_hand =self.judge_object_in_hand("right",pinch_object,obj_id)
            if object_in_hand:
                result_description=f"success pinch {pinch_object}{obj_id}"
            else:
                result_description=f"failed pinch {pinch_object}{obj_id}"
            self.execution_result.append([4,object_in_hand,result_description])
        elif "left" in hand_name:
            self.update_point_cloud(pinch_object,obj_id,hand_name="left_hand")
            self.left_hand_pinch(t)
            result_description=""
            object_in_hand =self.judge_object_in_hand("left",pinch_object,obj_id)
            if object_in_hand:
                result_description=f"success pinch {pinch_object}{obj_id}"
            else:
                result_description=f"failed pinch {pinch_object}{obj_id}"
            self.execution_result.append([4,object_in_hand,result_description])
        else:
            print("Invalid hand selected when pinching")
            result_description="Invalid parameter hand_name. Please choose from 'right', or 'left'."
            self.execution_result.append([4,False,result_description])

    ##### open hand
    def left_hand_open(self, t=6):
        print("left hand pre grasp!!!!")
        # action=[1.25] + [0] * 11
        # qpos = self.left_hand_action_to_pose(action)
        default_pose=self.get_default_pose()
        action=self.left_hand_open_joint
        qpos = self.left_hand_action_to_pose(action,default_pose,low_joint=True)
        self.delta_open_action_left = (np.array(qpos)-default_pose)/t
        for i in range(t):
            hand_joint =self.get_hand_joint("left")
            default_pose = self.left_hand_action_to_pose(hand_joint,default_pose)
            obs, reward, terminated, truncated, info = self.env.step(default_pose+self.delta_open_action_left)
            if self.vis:
                self.base_env.render()
            
            if self.save_video:
                rgb = self.env.unwrapped.render_rgb_array()
                self.images.append(rgb)
        
        for i in range(15):
            # hand_joint =self.get_hand_joint("right")
            # default_pose = self.right_hand_action_to_pose(hand_joint,default_pose)
            obs, reward, terminated, truncated, info = self.env.step(qpos)
            if self.vis:
                self.base_env.render()

            if self.save_video:
                rgb = self.env.unwrapped.render_rgb_array()
                self.images.append(rgb)
        self.gripper_state[0] = HandState.DEFAULT
    
    def right_hand_open(self, t=6):
        # print("right hand pre grasp!!!!")
        # action=[1.25] + [0] * 11
        # qpos = self.right_hand_action_to_pose(action)
        default_pose=self.get_default_pose()
        action=self.right_hand_open_joint
        qpos = self.right_hand_action_to_pose(action,default_pose,low_joint=True)
        self.delta_open_action_right = (np.array(qpos)-default_pose)/t
        for i in range(t):
            hand_joint =self.get_hand_joint("right")
            default_pose = self.right_hand_action_to_pose(hand_joint,default_pose)
            obs, reward, terminated, truncated, info = self.env.step(default_pose+self.delta_open_action_right)
            if self.vis:
                self.base_env.render()

        for i in range(15):
            # hand_joint =self.get_hand_joint("right")
            # default_pose = self.right_hand_action_to_pose(hand_joint,default_pose)
            obs, reward, terminated, truncated, info = self.env.step(qpos)
            if self.vis:
                self.base_env.render()

            if self.save_video:
                rgb = self.env.unwrapped.render_rgb_array()
                self.images.append(rgb)
        self.gripper_state[1] = HandState.DEFAULT

    
    def open_hand(self, hand_name="all",t=6):
        # self.update_point_cloud(except_actor=None,hand_name=hand_name)
        object_in_hand = False
        if hand_name == "all":
            object_in_hand = self.judge_object_in_hand("left",self.attach_obj[0],self.attach_obj_id[0]) and self.judge_object_in_hand("right",self.attach_obj[1],self.attach_obj_id[1])
            self.attach_obj[0] = None
            self.attach_obj_id[0] = None
            self.attach_obj[1] = None
            self.attach_obj_id[1] = None
            
            self.right_hand_open(t)
            self.left_hand_open(t)

        elif hand_name == "right":
            object_in_hand = self.judge_object_in_hand("right",self.attach_obj[1],self.attach_obj_id[1])
            self.attach_obj[1] = None
            self.attach_obj_id[1] = None
            # for i in range(10):
            self.right_hand_open(t)
            # time.sleep(0.1)
            
        elif hand_name == "left":
            object_in_hand = self.judge_object_in_hand("left",self.attach_obj[0],self.attach_obj_id[0])
            self.attach_obj[0] = None
            self.attach_obj_id[0] = None
            # for i in range(10):
            self.left_hand_open(t)
                # time.sleep(0.1)
        else:
            print("Invalid hand selected")
        
        result_description = ""
        if object_in_hand:
            result_description = "Object in hand"
        else:
            result_description = "No object in hand"
        self.execution_result.append([0,object_in_hand,result_description])


    #### collision check
    def check_collision(self,pose,save_photo=False,render=False,hand_name="right"):
        old_qpos=self.robot.get_qpos()[0, :38].numpy()
        if "right" in hand_name:
            arm_pose =pose[19:26]
            pose=self.right_arm_action_to_pose(arm_pose,old_qpos.copy())
        else:
            arm_pose =pose[0:7]
            pose=self.left_arm_action_to_pose(arm_pose,old_qpos.copy())
        if render:
            self.env.render()
        self.env.agent.robot.set_qpos(pose)
        if render:
            self.env.render()
        if save_photo:
            rgb = self.env.unwrapped.render_rgb_array().squeeze(0).cpu().numpy()
            print(rgb.shape)
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.imsave( ROOT_PATH/f'imgs/collision_check_{current_time}.png', rgb)
        self.env.agent.robot.set_qpos(old_qpos)
        if render:
            self.env.render()
        return None
    
    def get_eef_z_left(self):
        """Helper function for constraint"""
        ee_idx = self.l_planner.link_name_2_idx[self.l_planner.move_group]
        ee_pose = self.l_planner.robot.get_pinocchio_model().get_link_pose(ee_idx)
        mat = transforms3d.quaternions.quat2mat(ee_pose[3:7])
        return mat[:, 2]

    def make_f_left(self):
        """
        Create a constraint function that takes in a qpos and outputs a scalar.
        A valid constraint function should evaluates to 0 when the constraint
        is satisfied.

        See [ompl constrained planning](https://ompl.kavrakilab.org/constrainedPlanning.html)
        for more details.
        """
        # constraint function ankor
        def f(x, out):
            self.l_planner.robot.set_qpos(x)
            out[0] = (
                self.get_eef_z_left().dot(np.array([0, 0, -1])) - 0.966
            )  # maintain 15 degrees w.r.t. -z axis
        return f
    
    def make_j_left(self):
        """
        Create the jacobian of the constraint function w.r.t. qpos.
        This is needed because the planner uses the jacobian to project a random sample
        to the constraint manifold.
        """

        # constraint jacobian ankor
        def j(x, out):
            full_qpos=self.robot.get_qpos()[0, :38].cpu().numpy().copy()
            full_qpos[self.l_planner.move_group_joint_indices]=x
            # full_qpos = self.r_planner.pad_move_group_qpos(x)
            jac = self.l_planner.robot.get_pinocchio_model().compute_single_link_jacobian(
                full_qpos, len(self.l_planner.move_group_joint_indices) - 1
            )
            rot_jac = jac[3:, self.l_planner.move_group_joint_indices]
            for i in range(len(self.l_planner.move_group_joint_indices)):
                out[i] = np.cross(rot_jac[:, i], self.get_eef_z_left()).dot(
                    np.array([0, 0, -1])
                )

        # constraint jacobian ankor end
        return j
    
    def left_follow_path(self, result, refine_steps: int = 0):
        n_step = result["position"].shape[0]
        default_pose =self.get_default_pose()        
        # if self.gripper_state[0] == HandState.GRASP:
        #     action=[1.3, 0.931, 1.02, 1.027, 1.023,0.279, 0.797, 0.814, 0.815, 0.818, 0.27,0.278]
        #     default_pose = self.left_hand_action_to_pose(action)
        # else:
        #     default_pose = self.robot.get_qpos()[0, :38].cpu().numpy()
        # if self.gripper_state[1] == HandState.PINCH:
        #     if self.delta_pinch_action_right is not None:
        #         default_pose = default_pose+self.delta_pinch_action_right*3

        path_point=np.array([[]])
        for i in range(n_step):
            qpos = result["position"][i]
            # action=self.right_arm_action_to_pose(qpos,default_pose)
            new_point=self.fk_robot(qpos,hand="left").p.reshape(1,3)
            if path_point.shape[1]==0:
                path_point=new_point
            else:
                path_point=np.concatenate([path_point,new_point],axis=0)
        if n_step!=0 and self.show_key_points:
            self.show_path(path_point)

        for i in range(n_step + refine_steps):
            qpos = result["position"][int(min(i, n_step - 1))]
            
            action=self.left_arm_action_to_pose(qpos,default_pose)
            obs, reward, terminated, truncated, info = self.env.step(action)
            

            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render()
            if self.save_video:
                rgb = self.env.unwrapped.render_rgb_array()
                self.images.append(rgb)

            if i%10==0:
                max_cyclic_steps = 20
                for j in range(max_cyclic_steps):
                    array = action[:14] - self.robot.get_qpos()[0, :14].cpu().numpy()
                    tolerance = 0.1  # 设置容差
                    is_close_to_zero = np.allclose(array, np.zeros_like(array), atol=tolerance)
                    # print("is_close_to_zero:",is_close_to_zero)
                    if is_close_to_zero:
                        break
                    else:
                        print("do not follow the path, reaction!!!!!!!!!!!!")
                        # print("action:",action)
                        obs, reward, terminated, truncated, info = self.env.step(action)
                        if self.vis:
                            self.base_env.render()
        return obs, reward, terminated, truncated, info
    
    def left_move_to_pose_with_screw(
        self, pose: sapien.Pose, dry_run: bool = False, 
        refine_steps: int = 0,
        easy_plan=False,
        constraints=None,
    ):
        pose = to_sapien_pose(pose)
        # try screw two times before giving up
        if self.hand_pre_pose_left is not None:
            self.hand_pre_pose_left.set_pose(pose)
            self.hand_grasp_point_left.set_pose(sapien.Pose(p=transform_keypoint_to_base(self.env.agent.key_points["grasp_point_base_left_hand"],pose)))
            print("grasp point pose:",sapien.Pose(p=transform_keypoint_to_base(self.env.agent.key_points["grasp_point_base_left_hand"],pose)))
        
        pose = sapien.Pose(p=pose.p , q=pose.q)
        # self_collision_list=self.l_planner.check_for_self_collision(qpos=self.robot.get_qpos().cpu().numpy()[0])
        # for collision in self_collision_list:
        #     print(f"\033[91mCollision between {collision.link_name1} and {collision.link_name2}\033[0m")
        pre_point = transform_keypoint_to_base(np.array([0, 0.1, 0]), pose)
        pre_pose = sapien.Pose(p=pre_point , q=pose.q)

        if easy_plan:
            result = self.l_planner.plan_screw(
                np.concatenate([pose.p, pose.q]),
                self.robot.get_qpos().cpu().numpy()[0],
                time_step=self.base_env.control_timestep,
                # time_step=1/250,
                use_point_cloud=self.use_point_cloud,
                use_attach=self.use_attach[0]
            )
        else:
            if constraints is None:
                result = self.l_planner.plan_qpos_to_pose(
                    np.concatenate([pose.p, pose.q]),
                    self.robot.get_qpos().cpu().numpy()[0],
                    # time_step=self.base_env.control_timestep,
                    time_step=1/250,
                    # use_point_cloud=self.use_point_cloud,
                    use_point_cloud=self.use_point_cloud,
                    use_attach=self.use_attach[0]
                    # planning_time=
                )
            else:
                result = self.l_planner.plan_qpos_to_pose(
                    np.concatenate([pose.p, pose.q]),
                    self.robot.get_qpos().cpu().numpy()[0],
                    time_step=self.base_env.control_timestep,
                    constraint_function=self.make_f_left(),
                    constraint_jacobian=self.make_j_left(),
                    constraint_tolerance=0.05,
                    # time_step=1/250,
                    use_point_cloud=self.use_point_cloud,
                    use_attach=self.use_attach[0]
                )
        if result["status"] != "Success":
            result = self.l_planner.plan_qpos_to_pose(
                np.concatenate([pose.p, pose.q]),
                self.robot.get_qpos().cpu().numpy()[0],
                time_step=self.base_env.control_timestep,
                # time_step=1/250,
                use_point_cloud=self.use_point_cloud,
                use_attach=self.use_attach[0]
            )
            if result["status"] != "Success":
                print(result["status"])
                self.render_wait()
                return -1
        self.render_wait()
        if dry_run:
            return result
        if result["position"].shape[0]==0:
            print("no path")
            return -1
        self.left_follow_path(result, refine_steps=refine_steps)
        # self.build_robot_plant()
        self.use_attach[0] = False
        self.l_planner.planning_world.set_use_attach(False)
        return 1    
    
    def follow_path(self, result_left,result_right, refine_steps: list[int]):
        if isinstance(result_left,int) or isinstance(result_right,int):
            print("no target pose")
            return -1
        n_step_left = result_left["position"].shape[0]
        n_step_right = result_right["position"].shape[0]
        # default_pose = self.robot.get_qpos()[0, :38].cpu().numpy()

        # if self.gripper_state[0] == HandState.GRASP:
        #     action=[1.3, 0.931, 1.02, 1.027, 1.023,0.279, 0.797, 0.814, 0.815, 0.818, 0.27,0.278]
        #     default_pose = self.left_hand_action_to_pose(action,default_pose)
        # elif self.gripper_state[0] == HandState.PINCH:
        #     if self.delta_pinch_action_left is not None:
        #         default_pose = default_pose+self.delta_pinch_action_left*3
            
        # if self.gripper_state[1] == HandState.GRASP:
        #     action=[1.3, 0.931, 1.02, 1.027, 1.023,0.279, 0.797, 0.814, 0.815, 0.818, 0.27,0.278]
        #     default_pose = self.right_hand_action_to_pose(action,default_pose)
        # elif self.gripper_state[1] == HandState.PINCH:
        #     if self.delta_pinch_action_right is not None:
        #         default_pose = default_pose+self.delta_pinch_action_right*3
        
        default_pose = self.get_default_pose()
        path_point=np.array([[]])
        for i in range(n_step_left):
            qpos = result_left["position"][i]
            # action=self.right_arm_action_to_pose(qpos,default_pose)
            new_point=self.fk_robot(qpos,hand="left").p.reshape(1,3)
            if path_point.shape[1]==0:
                path_point=new_point
            else:
                path_point=np.concatenate([path_point,new_point],axis=0)
        for i in range(n_step_right):
            qpos = result_right["position"][i]
            # action=self.right_arm_action_to_pose(qpos,default_pose)
            new_point=self.fk_robot(qpos,hand="right").p.reshape(1,3)
            if path_point.shape[1]==0:
                path_point=new_point
            else:
                path_point=np.concatenate([path_point,new_point],axis=0)
        if n_step_left!=0 and n_step_right!=0 and self.show_key_points:
            self.show_path(path_point)
        max_steps=max(n_step_left,n_step_right)

        for i in range(max_steps + refine_steps[0]):
            l_qpos = result_left["position"][int(min(i, n_step_left - 1))]
            action=self.left_arm_action_to_pose(l_qpos,default_pose)
            r_qpos = result_right["position"][int(min(i, n_step_right - 1))]
            action=self.right_arm_action_to_pose(r_qpos,action)
            obs, reward, terminated, truncated, info = self.env.step(action)

            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render()
            if self.save_video:
                rgb = self.env.unwrapped.render_rgb_array()
                self.images.append(rgb)
            
            if i%10==0:
                max_cyclic_steps = 20
                for j in range(max_cyclic_steps):
                    array = action[:14] - self.robot.get_qpos()[0, :14].cpu().numpy()
                    tolerance = 0.1  # 设置容差
                    is_close_to_zero = np.allclose(array, np.zeros_like(array), atol=tolerance)
                    # print("is_close_to_zero:",is_close_to_zero)
                    if is_close_to_zero:
                        break
                    else:
                        print("do not follow the path, reaction!!!!!!!!!!!!")
                        # print("action:",action)
                        obs, reward, terminated, truncated, info = self.env.step(action)
                        if self.vis:
                            self.base_env.render()
        return obs, reward, terminated, truncated, info
        
        
    # self.all_hand_move_to_pose_with_screw(pose, refine_steps=refine_steps,attach_obj=attach_obj,object_name=object_name,object_id=object_id,easy_plan=easy_plan)
    def all_hand_move_to_pose_with_screw(
        self, pose: list[sapien.Pose], 
        refine_steps: int = 0,
        attach_obj=False,
        object_name=None,
        object_id=0,
        easy_plan=False,
    ):
        if not isinstance(refine_steps,list):
            refine_steps = [refine_steps, refine_steps]
        if not isinstance(attach_obj,list):
            attach_obj = [attach_obj, attach_obj]
        if not isinstance(object_name,list):
            object_name = [object_name, object_name]
        if not isinstance(object_id,list):
            object_id = [object_id, object_id]
        if not isinstance(easy_plan,list):
            easy_plan = [easy_plan, easy_plan]

        if attach_obj[0]:
            if object_name[0] is not None:
                self.add_attach_obj("left",object_name[0])
                self.update_point_cloud(object_name,object_id,hand_name="left_hand")
            else:
                self.add_attach_obj("left")
        else:
            self.update_point_cloud(object_name,object_id,hand_name="left_hand")
        result_left=self.left_move_to_pose_with_screw(pose[0], dry_run=True, refine_steps=refine_steps[0],easy_plan=easy_plan[0])
        
        if attach_obj[1]:
            self.add_attach_obj("right")
            if object_name[1] is not None:
                self.update_point_cloud(object_name,object_id,hand_name="right_hand")
        else:
            self.update_point_cloud(object_name,object_id,hand_name="right_hand")
        result_right=self.right_move_to_pose_with_screw(pose[1], dry_run=True, refine_steps=refine_steps[1],easy_plan=easy_plan[1])
        
        if result_left==-1 or result_right==-1:
            print("no path")
            return -1
        
        self.follow_path(result_left,result_right, refine_steps=refine_steps)
        # self.build_robot_plant()
        self.use_attach[0] = False
        self.use_attach[1] = False
        self.l_planner.planning_world.set_use_attach(False)
        self.r_planner.planning_world.set_use_attach(False)
        return 1    
    
    def move_to_pose_with_screw(
        self, pose: list, hand_name , 
        attach_obj=False,
        object_name=None,
        object_id=0,
        easy_plan=False,
        constraints=None,
        dry_run: bool = False,
        refine_steps: int = 0,
    ):
        if hand_name=="all":
            pose[0]=pose[0][0]
            pose[1]=pose[1][0]
            result=self.move_to_pose_with_screw_tool(pose,hand_name,attach_obj,object_name,object_id,easy_plan,constraints,dry_run,refine_steps)
            if result==-1:
                self.execution_result.append([5,False,
                """Failed move to pose. Possible reasons for failure: 
    -Target pose is unreachable: Don't let one hand move beyond its own range. For example, don't let the right hand pinch or grasp an object in the area of the left hand, or let the right hand place an object in the area of the left hand.
    -Too many constraints make it impossible to find a reasonable solution.
    -The target pose will lead to a collision with the environment: Generally, there is a lack of constraints. For example, only the coincidence of points is constrained, and the randomness of the pose is caused by not constraining the parallel condition, resulting in a collision. Or the set parallel relationship of the constraints is incorrect. Please think carefully.
    -The pose of the target leads to self-collision of the two arms. For example, after the left hand places an item in the middle area, the right hand moves to the middle area to place an item without retracting the left arm to the left_hand_init_pose. After put the object into center area, must return to init_pose before move the another arm into!
    Advice:
    -If you find that an arm has not moved to the left_hand_init_pose or right_hand_init_pose after performing an open hand action to place an object in the middle area of the table, please move it(to "left_hand_init_pose"/"right_hand_init_pose position")in the next step.
    -If you find that the target position is unreachable, change your approach and move to a reachable area.
    -If you find that there are too many constraints, you can reduce the number of constraints; otherwise, you can increase them.
                            """
                ])
            else:
                self.execution_result.append([5,True,"success move to pose"])
        else:
            run_sccess=True
            for i in range(len(pose)):
                result=self.move_to_pose_with_screw_tool(pose[i],hand_name,attach_obj,object_name,object_id,easy_plan,constraints,dry_run,refine_steps)
                if result==-1:
                    run_sccess=False
            if run_sccess:
                self.execution_result.append([5,True,"success move to pose"])
            else:
                self.execution_result.append([5,False,
                """Failed move to pose. Possible reasons for failure: 
    -Target pose is unreachable: Don't let one hand move beyond its own range. For example, don't let the right hand pinch or grasp an object in the area of the left hand, or let the right hand place an object in the area of the left hand.
    -Too many constraints make it impossible to find a reasonable solution.
    -The target pose will lead to a collision with the environment: Generally, there is a lack of constraints. For example, only the coincidence of points is constrained, and the randomness of the pose is caused by not constraining the parallel condition, resulting in a collision. Or the set parallel relationship of the constraints is incorrect. Please think carefully.
    -The pose of the target leads to self-collision of the two arms. For example, after the left hand places an item in the middle area, the right hand moves to the middle area to place an item without retracting the left arm to the left_hand_init_pose. After put the object into center area, must return to init_pose before move the another arm into!
    Advice:
    -If you find that an arm has not moved to the left_hand_init_pose or right_hand_init_pose after performing an open hand action to place an object in the middle area of the table, please move it(to "left_hand_init_pose"/"right_hand_init_pose position")in the next step.
    -If you find that the target position is unreachable, change your approach and move to a reachable area.
    -If you find that there are too many constraints, you can reduce the number of constraints; otherwise, you can increase them.
                            """
                ])
                # hard code
                # self.execution_result.append([5,False,"Failed move to pose. It could be unreachable, or the target pose might cause collisions with other objects or the other arm. Please carefully consider why it cannot be reached and take reasoning action."])

    def move_to_pose_with_screw_tool(
        self, pose: sapien.Pose, hand_name , 
        attach_obj=False,
        object_name=None,
        object_id=0,
        easy_plan=False,
        constraints=None,
        dry_run: bool = False,
        refine_steps: int = 0,
    ):
        if "right" in hand_name:
            if attach_obj:
                if object_name is not None:
                    self.add_attach_obj("right",object_name)
                    self.update_point_cloud(object_name,object_id,hand_name="right_hand")
                else:
                    self.add_attach_obj("right")
            else:
                self.update_point_cloud(object_name,object_id,hand_name="right_hand")
            return self.right_move_to_pose_with_screw(pose, dry_run, refine_steps,easy_plan=easy_plan,constraints=constraints)
        elif "left" in hand_name:
            if attach_obj:
                self.add_attach_obj("left",object_name)
                if object_name is not None:
                    self.update_point_cloud(object_name,object_id,hand_name="left_hand")
            else:
                self.update_point_cloud(object_name,object_id,hand_name="left_hand")
            return self.left_move_to_pose_with_screw(pose, dry_run, refine_steps,easy_plan=easy_plan,constraints=constraints)
        if "all" in hand_name:
            if isinstance(pose,list):
                return self.all_hand_move_to_pose_with_screw(pose, refine_steps=refine_steps,attach_obj=attach_obj,object_name=object_name,object_id=object_id,easy_plan=easy_plan)
            else:
                print("Invalid hand pose")
        else:
            print("Invalid hand selected")

    def end_planner(self):
        # self.planner.end()
        # self.planner = None
        # self.env.close()
        end_scene_step=20
        defalt_pose = self.get_default_pose()
        for i in range(end_scene_step):
            self.env.render()
            obs, reward, terminated, truncated, info = self.env.step(defalt_pose)

        if "collision_point_cloud" in self.base_env.scene.actors:
            # self.base_env.scene.remove_actor(self.collision_point_cloud)
            # self.env.scene.actors["collision_point_cloud"].remove()
            self.base_env.scene.remove_from_state_dict_registry(self.collision_point_cloud)
            self.env.scene.actors.pop("collision_point_cloud")
            self.collision_point_cloud.remove_from_scene()
        if "attach_point_cloud" in self.base_env.scene.actors:
            self.base_env.scene.remove_from_state_dict_registry(self.attach_point_cloud)
            self.env.scene.actors.pop("attach_point_cloud")
            self.attach_point_cloud.remove_from_scene()
        if "path_point_cloud" in self.base_env.scene.actors:
            self.base_env.scene.remove_from_state_dict_registry(self.path_point_cloud)
            self.env.scene.actors.pop("path_point_cloud")
            self.path_point_cloud.remove_from_scene()
        


        # if self.save_video:
        #     images_to_video(
        #         self.images,
        #         output_dir="./videos",
        #         video_name=f"put_apple_into_bowl",
        #         fps=30,
        #     )
        #     del self.images