import numpy as np


class HGensimBaseObject(object):
    def __init__(self):
        pass

    def set_openness(self, openness):
        raise NotImplementedError

    def get_openness(self):
        raise NotImplementedError

    def set_pose(self, pose: np.ndarray):
        raise NotImplementedError

    @property
    def pose(self):
        raise NotImplementedError

    def get_pose_wrt_tcp(self):
        raise NotImplementedError

    def get_pos_wrt_tcp(self):
        raise NotImplementedError

    def get_orn_wrt_tcp(self):
        raise NotImplementedError

    @property
    def pos(self):
        raise NotImplementedError

    @property
    def quat(self):
        raise NotImplementedError

    def get_instance(self):
        raise NotImplementedError

    def get_keypoint(self):
        raise NotImplementedError

    def get_keypoint_vector(self, from_name, to_name):
        raise NotImplementedError

    def get_qpos(self):
        raise NotImplementedError

    def set_qpos(self, qpos):
        raise NotImplementedError
