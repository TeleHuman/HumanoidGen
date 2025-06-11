import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill import PACKAGE_ASSET_DIR
from humanoidgen import HGENSIM_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common
from mani_skill.utils.structs.actor import Actor


@register_agent()
class UnitreeH1_2UpperBody(BaseAgent):
    """The H1_2 Robot with control over its torso rotation and its two arms. Legs are fixed."""


    uid = "unitree_h1_2_simplified_upper_body"
    hand_name = "inspire_hand"
    # hand_name = "xhand"
    if hand_name == "inspire_hand":
        # inspire hand 
        # urdf_path = f"{PACKAGE_ASSET_DIR}/robots/h1_2/h1_2_upper_body.urdf"
        urdf_path = f"{HGENSIM_ASSET_DIR}/robots/h1_2/h1_2_upper_body.urdf"
        drake_urdf_path = f"{HGENSIM_ASSET_DIR}/robots/h1_2_drake/h1_2_upper_body.urdf"
    elif hand_name == "xhand":
        # xhand
        urdf_path = f"{PACKAGE_ASSET_DIR}/robots/h1_2/h1_2_upper_body_with_XHAND.urdf"
        drake_urdf_path = f"{HGENSIM_ASSET_DIR}/robots/h1_2_drake/h1_2_upper_body.urdf"


    urdf_config = dict(
        _materials=dict(
            finger=dict(static_friction=4.0, dynamic_friction=4.0, restitution=0.0),
        ),
        link={
            **{
                f"L_{k}": dict(
                    material="finger", patch_radius=0.1, min_patch_radius=0.1
                )
                for k in ["thumb_proximal_base", "thumb_proximal","thumb_intermediate",
                          "thumb_distal","index_proximal","index_intermediate","middle_proximal",
                          "middle_intermediate","ring_proximal","ring_intermediate",
                          "pinky_proximal","pinky_intermediate"]
            },
            **{
                f"R_{k}": dict(
                    material="finger", patch_radius=0.1, min_patch_radius=0.1
                )
                for k in ["thumb_proximal_base", "thumb_proximal","thumb_intermediate",
                          "thumb_distal","index_proximal","index_intermediate","middle_proximal",
                          "middle_intermediate","ring_proximal","ring_intermediate",
                          "pinky_proximal","pinky_intermediate"]
            },
        },
    )
    
    fix_root_link = True
    load_multiple_collisions = False

    keyframes = dict(
        # standing=Keyframe(
        #     pose=sapien.Pose(p=[0, 0, 0.755]),
        #     qpos=np.array([0.0]*2+[1]+[-1]+[0.0] * (34)),
        # ),
        standing=Keyframe(
            pose=sapien.Pose(p=[0, 0, 0.755]),
            qpos=np.array([0.0]*2 + [1] + [-1] + [0.0]*10 + [1.174] + [0.0] * 4+ [1.174] + [0.0]*(18)),
        ),
        # standing_hand_down=Keyframe(
        #     pose=sapien.Pose(p=[0, 0, 0.755]),
        #     # qpos=np.array([1.0]*1+[0.0]*13+[1]*24),
        #     qpos=np.array([0.0, 0.0, 0.154, -0.146, 0.0, 0.0, 1.666, 1.666, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        # )
    )

    key_points = dict(
        # pinch_point_base_right_hand = np.array([-0.07445, -0.145609, -0.0221028]), #for table
        # right_pinch_axis =  np.array([-1.32008681,-0.57664681,-0.22339487])
        # [-0.68273579 -1.0233418  -0.31344128]
        # right_pinch_axis =  np.array([-0.68273579,-1.0233418,-0.31344128])
        # base point
        base_right_hand= np.array([0,0,0]),
        base_left_hand= np.array([0,0,0]),
        # grasp point
        # grasp_point_base_right_hand = np.array([-0.05045, -0.159809, -0.0281028]),  # default
        grasp_point_base_right_hand = np.array([-0.05845, -0.149809, -0.0281028]),  # default

        # grasp_point_base_left_hand = np.array([-0.05045, -0.159809, -0.0281028]),  #default
        # grasp_point_base_left_hand = np.array([-0.05045, -0.159809, 0]),  # drawer handler
        grasp_point_base_left_hand = np.array([-0.05845, -0.149809, 0]),  # 418
        # grasp_point_base_left_hand = np.array([-0.05145, -0.160809, -0.0281028]),
        # grasp_point_base_right_hand = np.array([-0.05845, -0.165609, -0.0281028]), #for robocasa v1
        # grasp_point_base_left_hand = np.array([-0.05845, -0.165609, -0.0281028]),
        # grasp_point_base_right_hand = np.array([-0.07445, -0.145609, -0.0281028]), #for robocasa v1
        # grasp_point_base_left_hand = np.array([-0.07445, -0.145609, 0.0281028]),
        # grasp_point_base_left_hand = np.array([-0.0585, -0.14, 0.025]),
        # grasp_point_base_right_hand = np.array([-0.0485, -0.13, -0.025]), #for robocasa v1
        # grasp_point_base_right_hand = np.array([-0.0585, -0.14, -0.025]), #for robocasa v1
        # right_grasp_point_table_exmple = np.array([-0.02824694, -0.14375293, -0.004]), #for table
        # right_grasp_point = np.array([-0.046824694, -0.14375293, -0.0055310726]),
        # right_grasp_point = np.array([-0.016824694, -0.14375293, 0.0205310726]), #for robocasa 2.13
        # right_grasp_point = np.array([-0.02824694, -0.14375293, -0.040]), #for robocasa
        # right_grasp_point = np.array([-0.016824694, -0.14375293, -0.019310726]), #for table 2.13
        # right_grasp_point_table_exmple = np.array([-0.016824694, -0.14375293, -0.018310726]), #for table
        # pinch point
        pinch_point_base_right_hand = np.array([-0.07445, -0.145609, -0.0281028]),  # default!!
        pinch_point_base_left_hand = np.array([-0.07445, -0.145609, 0.0281028]),
        # pinch_point_base_right_hand = np.array([-0.07445, -0.145609, -0.0221028]), #for table
        # pinch_point_base_right_hand = np.array([-0.08045, -0.12709, -0.0221028]), #for table
        # pinch_point_base_right_hand = np.array( [-0.07199096,-0.15723589,-0.02983844]), #for table
        # pinch_point_base_right_hand = np.array( [-0.07176098,-0.15764823,-0.02108007]), #for table
        # pinch_point_base_right_hand = np.array([-0.07445, -0.145609, -0.0281028]), #for table
    )

    key_axes =dict(
        right_pinch_axis =  np.array([-1,-0.8,0]),
        right_pinch_wrist_2_palm_axis =  np.array([0.8,-1.0,0]),
        right_ring_2_index =  np.array([0,0,-1]),
        right_grasp_axis =  np.array([-1,-0.8,0]),
        right_grasp_wrist_2_palm_axis =  np.array([0.8,-1.0,0]),
        left_pinch_axis =  np.array([-1,-0.8,0]),
        left_pinch_wrist_2_palm_axis =  np.array([0.8,-1.0,0]),
        left_ring_2_index =  np.array([0,0,1]),
        left_grasp_axis =  np.array([-1,-0.8,0]),
        left_grasp_wrist_2_palm_axis =  np.array([0.8,-1.0,0]),
    )

    arm_joints = [
        # "left_hip_pitch_joint",
        # "right_hip_pitch_joint",
        # "torso_joint",
        # "left_hip_roll_joint",
        # "right_hip_roll_joint",
        "left_shoulder_pitch_joint",
        "right_shoulder_pitch_joint",
        # "left_hip_yaw_joint",
        # "right_hip_yaw_joint",
        "left_shoulder_roll_joint",
        "right_shoulder_roll_joint",
        # "left_knee_joint",
        # "right_knee_joint",
        "left_shoulder_yaw_joint",
        "right_shoulder_yaw_joint",
        # "left_ankle_pitch_joint",
        # "right_ankle_pitch_joint",
        "left_elbow_pitch_joint",
        "right_elbow_pitch_joint",
        # "left_ankle_roll_joint",
        # "right_ankle_roll_joint",
        "left_elbow_roll_joint",
        "right_elbow_roll_joint",
        "left_wrist_pitch_joint",
        "right_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_wrist_yaw_joint",
    ]

    if hand_name == "xhand":
        finger_joints = [
            "L_thumb_proximal_yaw_joint",
            "L_index_proximal_joint",
            "L_middle_proximal_joint",
            "L_ring_proximal_joint",
            "L_pinky_proximal_joint",
            "L_thumb_proximal_pitch_joint",
            "L_index_intermediate_joint",
            "L_middle_intermediate_joint",
            "L_ring_intermediate_joint",
            "L_pinky_intermediate_joint",
            "L_thumb_intermediate_joint" ,
            "L_thumb_distal_joint",

            "right_hand_thumb_bend_joint",
            "right_hand_thumb_rota_joint1",
            "right_hand_thumb_rota_joint2",
            "right_hand_index_joint1",
            "right_hand_index_joint2",
            "right_hand_mid_joint1",
            "right_hand_mid_joint2",
            "right_hand_ring_joint1",
            "right_hand_ring_joint2",
            "right_hand_pinky_joint1",
            "right_hand_pinky_joint2",
        ]

    elif hand_name=="inspire_hand":
        finger_joints = [
            "L_thumb_proximal_yaw_joint",
            "L_index_proximal_joint",
            "L_middle_proximal_joint",
            "L_ring_proximal_joint",
            "L_pinky_proximal_joint",

            "R_thumb_proximal_yaw_joint",
            "R_index_proximal_joint",
            "R_middle_proximal_joint",
            "R_ring_proximal_joint",
            "R_pinky_proximal_joint",
            
            "L_thumb_proximal_pitch_joint",
            "L_index_intermediate_joint",
            "L_middle_intermediate_joint",
            "L_ring_intermediate_joint",
            "L_pinky_intermediate_joint",

            "R_thumb_proximal_pitch_joint",
            "R_index_intermediate_joint",
            "R_middle_intermediate_joint",
            "R_ring_intermediate_joint",
            "R_pinky_intermediate_joint",

            "L_thumb_intermediate_joint" ,
            "R_thumb_intermediate_joint",

            "L_thumb_distal_joint",
            "R_thumb_distal_joint",
        ]
    
    # body_stiffness = 1e3
    # body_damping = 1e2
    # body_force_limit = 100

    body_stiffness = 1e3
    body_damping = 100
    body_force_limit = 100

    hand_stiffness = 1e3
    hand_damping = 0.1
    # hand_damping = 100
    hand_force_limit = 1
    # hand_force_limit = 0.7

    @property
    def _controller_configs(self):
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joints,
            lower=None,
            upper=None,
            stiffness=self.body_stiffness,
            damping=self.body_damping,
            force_limit=self.body_force_limit,
            normalize_action=False,
        )
        finger_pd_joint_pos = PDJointPosControllerConfig(
            self.finger_joints,
            lower=None,
            upper=None,
            stiffness=self.hand_stiffness,
            damping=self.hand_damping,
            force_limit=self.hand_force_limit,
            normalize_action=False,
        )

        # body_pd_joint_delta_pos = PDJointPosControllerConfig(
        #     self.body_joints,
        #     lower=[-0.2] * 14 + [-0.5] * 24,
        #     upper=[0.2] * 14 + [0.5] * 24,
        #     stiffness=self.body_stiffness,
        #     damping=self.body_damping,
        #     force_limit=self.body_force_limit,
        #     use_delta=True,
        # )
        return dict(
            # pd_joint_delta_pos=dict(
            #     body=body_pd_joint_delta_pos, balance_passive_force=True
            # ),
            pd_joint_pos=dict(arm=arm_pd_joint_pos,finger=finger_pd_joint_pos, balance_passive_force=True),
        )

    @property
    def _sensor_configs(self):
        return []

    def _after_init(self):
        if self.hand_name == "inspire_hand":
            self.right_hand_finger_link_l_1 = self.robot.links_map["R_thumb_proximal"]
            self.right_hand_finger_link_r_1 = self.robot.links_map["R_ring_proximal"]
            self.right_hand_finger_link_r_2 = self.robot.links_map["R_pinky_proximal"]
            self.right_tcp = self.robot.links_map["R_hand_base_link"]
            self.right_finger_joints = [
                "R_thumb_proximal_yaw_joint",
                "R_thumb_proximal_pitch_joint",
                "R_thumb_intermediate_joint",
                "R_thumb_distal_joint",
                "R_index_proximal_joint",
                "R_index_intermediate_joint",
                "R_middle_proximal_joint",
                "R_middle_intermediate_joint",
                "R_ring_proximal_joint",
                "R_ring_intermediate_joint",
                "R_pinky_proximal_joint",
                "R_pinky_intermediate_joint",
            ]
            self.right_finger_joint_indexes = [
                self.robot.active_joints_map[joint].active_index[0].item()
                for joint in self.right_finger_joints
            ]

            self.left_hand_finger_link_l_1 = self.robot.links_map["L_thumb_proximal"]
            self.left_hand_finger_link_r_1 = self.robot.links_map["L_ring_proximal"]
            self.left_hand_finger_link_r_2 = self.robot.links_map["L_pinky_proximal"]
            self.left_tcp = self.robot.links_map["L_hand_base_link"]

            self.left_finger_joints = [
                "L_thumb_proximal_yaw_joint",
                "L_thumb_proximal_pitch_joint",
                "L_thumb_intermediate_joint" ,
                "L_thumb_distal_joint",
                "L_index_proximal_joint",
                "L_index_intermediate_joint",
                "L_middle_proximal_joint",
                "L_middle_intermediate_joint",
                "L_ring_proximal_joint",
                "L_ring_intermediate_joint",
                "L_pinky_proximal_joint",
                "L_pinky_intermediate_joint",
            ]
            self.left_finger_joint_indexes = [
                self.robot.active_joints_map[joint].active_index[0].item()
                for joint in self.left_finger_joints
            ]

            self.right_arm_joints = [
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_pitch_joint",
                "right_elbow_roll_joint",
                "right_wrist_pitch_joint",
                "right_wrist_yaw_joint",
            ]
            self.right_arm_joint_indexes = [
                self.robot.active_joints_map[joint].active_index[0].item()
                for joint in self.right_arm_joints
            ]
        elif self.hand_name == "xhand":
            self.right_tcp = self.robot.links_map["right_hand_link"]
            self.right_finger_joints = [
                "right_hand_thumb_bend_joint",
                "right_hand_thumb_rota_joint1",
                "right_hand_thumb_rota_joint2",
                "right_hand_index_joint1",
                "right_hand_index_joint2",
                "right_hand_mid_joint1",
                "right_hand_mid_joint2",
                "right_hand_ring_joint1",
                "right_hand_ring_joint2",
                "right_hand_pinky_joint1",
                "right_hand_pinky_joint2",
            ]
            self.right_finger_joint_indexes = [
                self.robot.active_joints_map[joint].active_index[0].item()
                for joint in self.right_finger_joints
            ]

            self.left_hand_finger_link_l_1 = self.robot.links_map["L_thumb_proximal"]
            self.left_hand_finger_link_r_1 = self.robot.links_map["L_ring_proximal"]
            self.left_hand_finger_link_r_2 = self.robot.links_map["L_pinky_proximal"]
            self.left_tcp = self.robot.links_map["L_hand_base_link"]

            self.left_finger_joints = [
                "L_thumb_proximal_yaw_joint",
                "L_thumb_proximal_pitch_joint",
                "L_thumb_intermediate_joint" ,
                "L_thumb_distal_joint",
                "L_index_proximal_joint",
                "L_index_intermediate_joint",
                "L_middle_proximal_joint",
                "L_middle_intermediate_joint",
                "L_ring_proximal_joint",
                "L_ring_intermediate_joint",
                "L_pinky_proximal_joint",
                "L_pinky_intermediate_joint",
            ]
            self.left_finger_joint_indexes = [
                self.robot.active_joints_map[joint].active_index[0].item()
                for joint in self.left_finger_joints
            ]

            self.right_arm_joints = [
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_pitch_joint",
                "right_elbow_roll_joint",
                "right_wrist_pitch_joint",
                "right_wrist_yaw_joint",
            ]
            self.right_arm_joint_indexes = [
                self.robot.active_joints_map[joint].active_index[0].item()
                for joint in self.right_arm_joints
            ]

        self.left_arm_joints = [
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_pitch_joint",
            "left_elbow_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
        ]
        self.left_arm_joint_indexes = [
            self.robot.active_joints_map[joint].active_index[0].item()
            for joint in self.left_arm_joints
        ]

        self.base_link = self.robot.links_map["pelvis"]

        

        # # disable collisions between fingers. Done in python here instead of the srdf as we can use less collision bits this way and do it more smartly
        # # note that the two link of the fingers can collide with other finger links and the palm link so its not included
        # link_names = ["thumb_proximal_base", "thumb_proximal","thumb_intermediate",
        #                   "thumb_distal","index_proximal","index_intermediate","middle_proximal",
        #                   "middle_intermediate","ring_proximal","ring_intermediate",
        #                   "pinky_proximal","pinky_intermediate"]
        # for ln in link_names:
        #     self.robot.links_map[f"L_{ln}"].set_collision_group_bit(2, 1, 1)
        #     self.robot.links_map[f"R_{ln}"].set_collision_group_bit(2, 2, 1)
        # self.robot.links_map["L_hand_base_link"].set_collision_group_bit(2, 1, 1)
        # self.robot.links_map["R_hand_base_link"].set_collision_group_bit(2, 2, 1)
        # self.robot.links_map["left_wrist_yaw_link"].set_collision_group_bit(2, 1, 1)
        # self.robot.links_map["right_wrist_yaw_link"].set_collision_group_bit(2, 2, 1)

        # # disable collisions between torso and some other links
        # self.robot.links_map["torso_link"].set_collision_group_bit(2, 3, 1)
        # self.robot.links_map["left_shoulder_roll_link"].set_collision_group_bit(2, 3, 1)
        # self.robot.links_map["right_shoulder_roll_link"].set_collision_group_bit(
        #     2, 3, 1
        # )

    # def right_hand_dist_to_open_grasp(self):
    #     """compute the distance from the current qpos to a open grasp qpos for the right hand"""
    #     return torch.mean(
    #         torch.abs(self.robot.qpos[:, self.right_finger_joint_indexes]), dim=1
    #     )

    # def left_hand_dist_to_open_grasp(self):
    #     """compute the distance from the current qpos to a open grasp qpos for the left hand"""
    #     return torch.mean(
    #         torch.abs(self.robot.qpos[:, self.left_finger_joint_indexes]), dim=1
    #     )

    # def left_hand_is_grasping(self, object: Actor, min_force=0.5, max_angle=85):
    #     """Check if the robot is grasping an object with just its left hand

    #     Args:
    #         object (Actor): The object to check if the robot is grasping
    #         min_force (float, optional): Minimum force before the robot is considered to be grasping the object in Newtons. Defaults to 0.5.
    #         max_angle (int, optional): Maximum angle of contact to consider grasping. Defaults to 85.
    #     """
    #     l_contact_forces = self.scene.get_pairwise_contact_forces(
    #         self.left_hand_finger_link_l_1, object
    #     )
    #     r_contact_forces_1 = self.scene.get_pairwise_contact_forces(
    #         self.left_hand_finger_link_r_1, object
    #     )
    #     r_contact_forces_2 = self.scene.get_pairwise_contact_forces(
    #         self.left_hand_finger_link_r_2, object
    #     )
    #     lforce = torch.linalg.norm(l_contact_forces, axis=1)
    #     rforce_1 = torch.linalg.norm(r_contact_forces_1, axis=1)
    #     rforce_2 = torch.linalg.norm(r_contact_forces_2, axis=1)

    #     # direction to open the gripper
    #     ldirection = self.left_hand_finger_link_l_1.pose.to_transformation_matrix()[
    #         ..., :3, 1
    #     ]
    #     rdirection1 = -self.left_hand_finger_link_r_1.pose.to_transformation_matrix()[
    #         ..., :3, 1
    #     ]
    #     rdirection2 = -self.left_hand_finger_link_r_2.pose.to_transformation_matrix()[
    #         ..., :3, 1
    #     ]

    #     langle = common.compute_angle_between(ldirection, l_contact_forces)
    #     rangle1 = common.compute_angle_between(rdirection1, r_contact_forces_1)
    #     rangle2 = common.compute_angle_between(rdirection2, r_contact_forces_2)
    #     lflag = torch.logical_and(
    #         lforce >= min_force, torch.rad2deg(langle) <= max_angle
    #     )
    #     rflag1 = torch.logical_and(
    #         rforce_1 >= min_force, torch.rad2deg(rangle1) <= max_angle
    #     )
    #     rflag2 = torch.logical_and(
    #         rforce_2 >= min_force, torch.rad2deg(rangle2) <= max_angle
    #     )
    #     rflag = rflag1 | rflag2
    #     return torch.logical_and(lflag, rflag)

    # def right_hand_is_grasping(self, object: Actor, min_force=0.5, max_angle=85):
    #     """Check if the robot is grasping an object with just its right hand

    #     Args:
    #         object (Actor): The object to check if the robot is grasping
    #         min_force (float, optional): Minimum force before the robot is considered to be grasping the object in Newtons. Defaults to 0.5.
    #         max_angle (int, optional): Maximum angle of contact to consider grasping. Defaults to 85.
    #     """
    #     l_contact_forces = self.scene.get_pairwise_contact_forces(
    #         self.right_hand_finger_link_l_1, object
    #     )
    #     r_contact_forces_1 = self.scene.get_pairwise_contact_forces(
    #         self.right_hand_finger_link_r_1, object
    #     )
    #     r_contact_forces_2 = self.scene.get_pairwise_contact_forces(
    #         self.right_hand_finger_link_r_2, object
    #     )
    #     lforce = torch.linalg.norm(l_contact_forces, axis=1)
    #     rforce_1 = torch.linalg.norm(r_contact_forces_1, axis=1)
    #     rforce_2 = torch.linalg.norm(r_contact_forces_2, axis=1)

    #     # direction to open the gripper
    #     ldirection = self.right_hand_finger_link_l_1.pose.to_transformation_matrix()[
    #         ..., :3, 1
    #     ]
    #     rdirection1 = -self.right_hand_finger_link_r_1.pose.to_transformation_matrix()[
    #         ..., :3, 1
    #     ]
    #     rdirection2 = -self.right_hand_finger_link_r_2.pose.to_transformation_matrix()[
    #         ..., :3, 1
    #     ]

    #     langle = common.compute_angle_between(ldirection, l_contact_forces)
    #     rangle1 = common.compute_angle_between(rdirection1, r_contact_forces_1)
    #     rangle2 = common.compute_angle_between(rdirection2, r_contact_forces_2)
    #     lflag = torch.logical_and(
    #         lforce >= min_force, torch.rad2deg(langle) <= max_angle
    #     )
    #     rflag1 = torch.logical_and(
    #         rforce_1 >= min_force, torch.rad2deg(rangle1) <= max_angle
    #     )
    #     rflag2 = torch.logical_and(
    #         rforce_2 >= min_force, torch.rad2deg(rangle2) <= max_angle
    #     )
    #     rflag = rflag1 | rflag2
    #     return torch.logical_and(lflag, rflag)
    
@register_agent()
class UnitreeH1_2UpperBodyWithHeadCamera(UnitreeH1_2UpperBody):
    uid = "unitree_h1_2_upper_body_with_head_camera"

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                "head_camera",
                pose=sapien.Pose(p=[0.05, 0, 0.46], q=euler2quat(0, np.pi / 6, 0)),
                width=128,
                height=128,
                near=0.01,
                far=100,
                fov=np.pi / 2,
                mount=self.robot.links_map["torso_link"],
            )
        ]
    
    @staticmethod
    def build_grasp_pose(approaching, closing, center):
        """Build a grasp pose (panda_hand_tcp)."""
        assert np.abs(1 - np.linalg.norm(approaching)) < 1e-3
        assert np.abs(1 - np.linalg.norm(closing)) < 1e-3
        assert np.abs(approaching @ closing) <= 1e-3
        ortho = np.cross(closing, approaching)
        T = np.eye(4)
        T[:3, :3] = np.stack([ortho, closing, approaching], axis=1)
        T[:3, 3] = center
        return sapien.Pose(T)
    