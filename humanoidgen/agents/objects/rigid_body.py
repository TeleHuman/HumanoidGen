import numpy as np
import sapien.core as sapien
from humanoidgen.agents.objects.base_object import HGensimBaseObject
# from humanoidgen.motion_planning.h1_2.utils import se3_inverse
import json


class RigidBody(HGensimBaseObject):
    def __init__(self, instance, name, keypoint_path=None):
        self.instance: sapien.Actor = instance
        self.name = name
        # self.tcp_link = tcp_link
        if keypoint_path is not None:
            keypoint_json = json.load(open(keypoint_path))
            self.keypoint_dict = keypoint_json["keypoints"]
            self.scale = 1
            self.axis = None
            if "scale" in keypoint_json:
                self.scale = keypoint_json["scale"]
            if "axis" in keypoint_json:
                self.axis = keypoint_json["axis"]

    def set_pose(self, pose):
        # pose = sapien.Pose(p=pose[:3], q=pose[3:])
        return self.instance.set_pose(pose)

    def get_pose(self):
        return self.instance.pose.copy()

    def get_keypoints(self):
        """return head, tail, or side keypoint based on the given name"""
        keypoints_inworld = {}
        pose = self.get_pose()
        for name, kp in self.keypoint_dict.items():
            scaled_kp = np.array(kp) * self.scale
            kp_loc = sapien.Pose(scaled_kp)
            transformed_kp = pose.transform(kp_loc).p
            keypoints_inworld[name] = transformed_kp

        return keypoints_inworld

    def get_keypoint_T_tcp(self, keypoint_name):
        # k = self.get_keypoints()[keypoint_name]
        # T_tcp = self.tcp_link.pose.to_transformation_matrix()
        # tcp_T = se3_inverse(T_tcp)
        # keypoint_T_tcp = tcp_T.dot(np.array([k[0], k[1], k[2], 1]))[:3]
        # return keypoint_T_tcp
        return None

    @property
    def pose(self):
        return self.instance.pose

    def get_collision_meshes(self):
        return self.instance.get_collision_meshes()

    def get_pose_wrt_tcp(self):
        return self.tcp_link.pose.inv() * self.instance.pose

    def get_pos_wrt_tcp(self):
        return self.get_pose_wrt_tcp().p

    def get_orn_wrt_tcp(self):
        return self.get_pose_wrt_tcp().q

    def get_pos(self):
        return self.get_pose().p

    def get_orn(self):
        return self.get_pose().q

    @property
    def pos(self):
        return self.pose.p

    @property
    def quat(self):
        return self.pose.q
