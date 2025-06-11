import numpy as np
import sapien.physx as physx
import trimesh
import sapien
from mani_skill.utils import common
from mani_skill.utils.structs import Actor
from mani_skill.utils.geometry.trimesh_utils import get_component_mesh
from transforms3d.quaternions import *
from transforms3d.euler import *
from transforms3d.axangles import *
from pydrake.all import RigidTransform
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.structs.pose import Pose as maniskill_Pose
from scipy.spatial.transform import Rotation as R
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

def get_actor_obb(actor: Actor, to_world_frame=True, vis=False):
    mesh = get_component_mesh(
        actor._objs[0].find_component_by_type(physx.PhysxRigidDynamicComponent),
        to_world_frame=to_world_frame,
    )
    assert mesh is not None, "can not get actor mesh for {}".format(actor)

    obb: trimesh.primitives.Box = mesh.bounding_box_oriented

    if vis:
        obb.visual.vertex_colors = (255, 0, 0, 10)
        trimesh.Scene([mesh, obb]).show()

    return obb


def compute_grasp_info_by_obb(
    obb: trimesh.primitives.Box,
    approaching=(0, 0, -1),
    target_closing=None,
    depth=0.0,
    ortho=True,
):
    """Compute grasp info given an oriented bounding box.
    The grasp info includes axes to define grasp frame, namely approaching, closing, orthogonal directions and center.

    Args:
        obb: oriented bounding box to grasp
        approaching: direction to approach the object
        target_closing: target closing direction, used to select one of multiple solutions
        depth: displacement from hand to tcp along the approaching vector. Usually finger length.
        ortho: whether to orthogonalize closing  w.r.t. approaching.
    """
    # NOTE(jigu): DO NOT USE `x.extents`, which is inconsistent with `x.primitive.transform`!
    extents = np.array(obb.primitive.extents)
    T = np.array(obb.primitive.transform)

    # Assume normalized
    approaching = np.array(approaching)

    # Find the axis closest to approaching vector
    angles = approaching @ T[:3, :3]  # [3]
    inds0 = np.argsort(np.abs(angles))
    ind0 = inds0[-1]

    # Find the shorter axis as closing vector
    inds1 = np.argsort(extents[inds0[0:-1]])
    ind1 = inds0[0:-1][inds1[0]]
    ind2 = inds0[0:-1][inds1[1]]

    # If sizes are close, choose the one closest to the target closing
    if target_closing is not None and 0.99 < (extents[ind1] / extents[ind2]) < 1.01:
        vec1 = T[:3, ind1]
        vec2 = T[:3, ind2]
        if np.abs(target_closing @ vec1) < np.abs(target_closing @ vec2):
            ind1 = inds0[0:-1][inds1[1]]
            ind2 = inds0[0:-1][inds1[0]]
    closing = T[:3, ind1]

    # Flip if far from target
    if target_closing is not None and target_closing @ closing < 0:
        closing = -closing

    # Reorder extents
    extents = extents[[ind0, ind1, ind2]]

    # Find the origin on the surface
    center = T[:3, 3].copy()
    half_size = extents[0] * 0.5
    center = center + approaching * (-half_size + min(depth, half_size))

    if ortho:
        closing = closing - (approaching @ closing) * approaching
        closing = common.np_normalize_vector(closing)

    grasp_info = dict(
        approaching=approaching, closing=closing, center=center, extents=extents
    )
    return grasp_info

def transform_keypoint_to_base(keypoint, transformation_matrix):
    """
    Convert a keypoint from its local coordinate system to the base coordinate system.

    Args:
        keypoint (np.ndarray): The keypoint coordinates in the local coordinate system (shape: [3] or [4]).
        transformation_matrix (np.ndarray): The transformation matrix from the local coordinate system to the base coordinate system (shape: [4, 4]).

    Returns:
        np.ndarray: The keypoint coordinates in the base coordinate system (shape: [3]).
    """
    if not isinstance(keypoint, np.ndarray):
        keypoint = np.array(keypoint)
    
    # print("keypoint:",keypoint)
    # print("type:",type(keypoint))
    # Ensure the keypoint is in homogeneous coordinates (shape: [4])
    if keypoint.shape == (3,):
        keypoint_homogeneous = np.append(keypoint, 1)
    elif keypoint.shape == (4,):
        keypoint_homogeneous = keypoint
    elif keypoint.shape == (1, 3):
        keypoint_homogeneous = np.append(keypoint[0], 1)
    else:
        raise ValueError("Keypoint must be a 3D or 4D vector.")
    
    if isinstance(transformation_matrix,sapien.Pose):
        transformation_matrix=transformation_matrix.to_transformation_matrix()
    elif isinstance(transformation_matrix,maniskill_Pose):
        transformation_matrix=transformation_matrix.to_transformation_matrix()
    elif len(transformation_matrix.shape)==3:
        transformation_matrix=transformation_matrix[0]

    if transformation_matrix.shape == (1, 4, 4):
        transformation_matrix = transformation_matrix[0]
        # raise ValueError("Transformation matrix must be a 4x4 matrix.")

    # Apply the transformation matrix to the keypoint
    keypoint_base_homogeneous = transformation_matrix @ keypoint_homogeneous

    # Convert back to 3D coordinates by dividing by the homogeneous coordinate
    keypoint_base = keypoint_base_homogeneous[:3] / keypoint_base_homogeneous[3]

    return keypoint_base

def get_keypoint(actor, keypoint_name):
    """
    Get the coordinates of a keypoint on an actor.

    Args:
        actor (Actor): The actor to get the keypoint from.
        keypoint_name (str): The name of the keypoint to get.

    Returns:
        np.ndarray: The coordinates of the keypoint (shape: [3]).
    """
    # Get the keypoint from the actor
    keypoint = actor.key_points[keypoint_name]

    return keypoint


def pack_pose_to_sapien(pose):  # for action
    result = sapien.Pose()
    if isinstance(pose, RigidTransform):
        result.q = mat2quat(pose.GetAsMatrix4()[:3,:3])
        result.p = pose.translation()
    elif isinstance(pose, np.ndarray):
        if pose.shape == (4,4):
            result.q = mat2quat(pose[:3,:3])
            result.p = pose[:3,3]
    # rot_index = 4 if rot_type == "quat" else 3
    # if rot_type == "quat":
    #     rot_func = mat2quat
    # elif rot_type == "euler":
    #     rot_func = mat2euler
    # # elif rot_type == "axangle":
    # #     rot_func = mat2axangle_
    # packed = np.zeros(3 + rot_index)
    # packed[3 : 3 + rot_index] = rot_func(pose[:3, :3])
    # packed[:3] = pose[:3, 3]
    return result

def get_point_in_env(env : BaseEnv, point_name=None,type_name=None, obj_id=0,related_point=np.array([0,0,0]),openness=None):
    """
    Get the coordinates of a point in the environment.

    Args:
        env (BaseEnv): The environment to get the point from.
        point_name (str): The name of the point to get.

    Returns:
        np.ndarray: The coordinates of the point (shape: [3]).
    """

    # Get the point from the environment
    # point = env.points[point_name]
    if openness is not None:
        base_obj = getattr(env, type_name)
        openness_now=base_obj[obj_id].get_openness()[0]
        base_obj[obj_id].set_openness(openness)
        keypoints=base_obj[obj_id].get_keypoints()
        point=keypoints[point_name]
        base_obj[obj_id].set_openness(openness_now)
        
    elif point_name:
        if "right_hand" in point_name or "left_hand" in point_name:
            point = env.agent.key_points[point_name]
            if "right_hand" in point_name:
                point=transform_keypoint_to_base(point,env.agent.right_tcp.pose)
            elif "left_hand" in point_name:
                point=transform_keypoint_to_base(point,env.agent.left_tcp.pose)
        else:
            base_obj = getattr(env, type_name)
            keypoints=base_obj[obj_id].get_keypoints()
            point=keypoints[point_name]
    else:
        base_obj = getattr(env, type_name)
        # point = base_obj[obj_id].pose.p.numpy()[0]
        point =transform_keypoint_to_base(related_point,base_obj[obj_id].pose)
    return np.array(point)


def get_axis_in_env(env : BaseEnv, axis_name, obj_type=None, obj_id=0):
    """
    Get the coordinates of a point in the environment.

    Args:
        env (BaseEnv): The environment to get the point from.
        point_name (str): The name of the point to get.

    Returns:
        np.ndarray: The coordinates of the point (shape: [3]).
    """
    # Get the point from the environment
    # point = env.points[point_name]
    if axis_name in env.agent.key_axes:
        axis = env.agent.key_axes[axis_name]
        if "right" in axis_name:
            axis=transform_keypoint_to_base(axis,env.agent.right_tcp.pose)
        elif "left" in axis_name:
            axis=transform_keypoint_to_base(axis,env.agent.left_tcp.pose)
        return np.array(axis)
    elif axis_name in ["x","y","z"]:
        type = getattr(env, obj_type)
        pose = type[obj_id].pose.to_transformation_matrix()
        if axis_name == "x":
            axis = pose[0,:3, 0]
        elif axis_name == "y":
            axis = pose[0,:3, 1]
        elif axis_name == "z":
            axis = pose[0,:3, 2]
        return np.array(axis)


def get_pose_in_env(env : BaseEnv,type_name:str,obj_id:int=0):

    obj_type = getattr(env, type_name)
    pose = obj_type[obj_id].pose
    return pose

# [0.707, 0, 0, 0.707] y:"-90"
# [0.707,0.707, 0, 0 ] x:"-90"
# [0. 0. 1. 0.] x:"180"
def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

from typing import List, Optional
import os
import imageio
import tqdm
def images_to_video(
    images: List[np.ndarray],
    output_dir: str,
    video_name: str,
    fps: int = 10,
    quality: Optional[float] = 5,
    verbose: bool = True,
    **kwargs,
):
    r"""Calls imageio to run FFMPEG on a list of images. For more info on
    parameters, see https://imageio.readthedocs.io/en/stable/format_ffmpeg.html
    Args:
        images: The list of images. Images should be HxWx3 in RGB order.
        output_dir: The folder to put the video in.
        video_name: The name for the video.
        fps: Frames per second for the video. Not all values work with FFMPEG,
            use at your own risk.
        quality: Default is 5. Uses variable bit rate. Highest quality is 10,
            lowest is 0.  Set to None to prevent variable bitrate flags to
            FFMPEG so you can manually specify them using output_params
            instead. Specifying a fixed bitrate using ‘bitrate’ disables
            this parameter.
    References:
        https://github.com/facebookresearch/habitat-lab/blob/main/habitat/utils/visualizations/utils.py
    """
    assert 0 <= quality <= 10
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    video_name = video_name.replace(" ", "_").replace("\n", "_") + ".mp4"
    output_path = os.path.join(output_dir, video_name)
    writer = imageio.get_writer(output_path, fps=fps, quality=quality, **kwargs)
    if verbose:
        print(f"Video created: {output_path}")
        images_iter = tqdm.tqdm(images)
    else:
        images_iter = images
    for im in images_iter:
        if not isinstance(im, np.ndarray):
            im = np.array(im[0])  # 将im转换为ndarray
        writer.append_data(im)
    writer.close()

def se3_inverse(RT):
    RT = RT.reshape(4, 4)
    R = RT[:3, :3]
    T = RT[:3, 3].reshape((3, 1))
    RT_new = np.eye(4, dtype=np.float32)
    RT_new[:3, :3] = R.transpose()
    RT_new[:3, 3] = -1 * np.dot(R.transpose(), T).reshape((3))
    return RT_new

def out_put_nvml_memory_info():
    # 初始化 NVML
    nvmlInit()

    # 获取 GPU 设备句柄（假设使用第 0 个 GPU）
    handle = nvmlDeviceGetHandleByIndex(0)

    # 获取显存信息
    memory_info = nvmlDeviceGetMemoryInfo(handle)

    # 计算显存占用率
    used_memory = memory_info.used / (1024 ** 2)  # 转换为 MB
    total_memory = memory_info.total / (1024 ** 2)  # 转换为 MB
    memory_utilization = (used_memory / total_memory) * 100

    # 输出显存占用率
    print(f"GPU Memory Usage: {used_memory:.2f} MB / {total_memory:.2f} MB ({memory_utilization:.2f}%)")