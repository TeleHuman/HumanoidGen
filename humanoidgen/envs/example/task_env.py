from humanoidgen.envs.example.table_scene import TableSetting
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
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.kitchen_counter import KitchenCounterSceneBuilder
from mani_skill.utils.scene_builder.robocasa.scene_builder import RoboCasaSceneBuilder
from mani_skill.utils.scene_builder.table.scene_builder import TableSceneBuilder
from humanoidgen.tool.utils import *

from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SceneConfig, SimConfig
# from mani_skill import ASSET_DIR
from mani_skill.utils import common, io_utils, sapien_utils
from humanoidgen import ROOT_PATH
from humanoidgen import ASSET_DIR
import yaml
BRIDGE_DATASET_ASSET_PATH = ASSET_DIR / "tasks/bridge_v2_real2sim_dataset/"