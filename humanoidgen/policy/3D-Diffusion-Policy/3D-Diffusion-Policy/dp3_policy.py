if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import sys
import pathlib

THIS_DIR = pathlib.Path(__file__).parent.resolve()
DP3_DIR = str(THIS_DIR)
sys.path.insert(0, DP3_DIR)

import hydra
import torch
import dill
from omegaconf import OmegaConf

from train import TrainDP3Workspace 
import pdb

OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy_3d', 'config'))
)
def main(cfg):
    workspace = TrainDP3Workspace(cfg)
    workspace.eval()

class DP3:
    def __init__(self, cfg, checkpoint_num) -> None:
        self.policy, self.env_runner = self.get_policy_and_runner(cfg, checkpoint_num)
        
    def update_obs(self, observation):
        self.env_runner.update_obs(observation)
    
    def get_action(self, observation):
        action = self.env_runner.get_action(self.policy, observation)
        return action    

    def get_policy_and_runner(self, cfg, checkpoint_num):
        workspace = TrainDP3Workspace(cfg)
        policy, env_runner = workspace.get_policy_and_runner(cfg, checkpoint_num)
        return policy, env_runner

if __name__ == "__main__":
    main()
