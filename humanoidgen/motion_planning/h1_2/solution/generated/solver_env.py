import numpy as np
import sapien
import time
from mani_skill.envs.sapien_env import BaseEnv
from humanoidgen.motion_planning.h1_2.motionplanner import HumanoidMotionPlanner
from humanoidgen.motion_planning.h1_2.utils import (
    compute_grasp_info_by_obb, get_actor_obb)
from mani_skill.utils.structs.pose import to_sapien_pose
from transforms3d.euler import euler2quat, quat2mat
from transforms3d.quaternions import mat2quat
from humanoidgen.motion_planning.h1_2.constraint import Constraint
from humanoidgen.motion_planning.h1_2.cost import Cost
from humanoidgen.motion_planning.h1_2.utils import *
import numpy as np
import torch