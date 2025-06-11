from humanoidgen.envs.example.task_env import *
@register_env("close_laptop", max_episode_steps=200)
class CloseLaptopEnv(TableSetting):
    
    def _load_scene(self, options: Dict):
        super()._load_scene(options)
        self._add_object(type_name="laptop", type_id=0)  # Source cup
    
    def _initialize_episode(self, env_idx: torch.Tensor, options: Dict):
        super()._initialize_episode(env_idx, options)
        self._set_object_pose(
            type_name="laptop", 
            obj_id=0, 
            pose=sapien.Pose(
                # p=[-0.4, 0, 0.05],    # Z = half of bowl's height (0.1/2)
                # p=[-0.38, -0.08, 0.05],    # Z = half of bowl's height (0.1/2)
                # p=[0, 0, 0.1],    # Z = half of bowl's height (0.1/2)
                # q=[0.925, 0, 0, 0.381]          # Upright orientation

                p=[-0.38, -0.23, 0.05],    # Z = half of bowl's height (0.1/2)
                q=[1, 0, 0, 0]          # Upright orientation
            )
        )
        # self.laptop[0].get_openness()
        self.laptop[0].set_openness(0.8)