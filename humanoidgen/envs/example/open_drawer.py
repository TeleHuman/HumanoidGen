from humanoidgen.envs.example.task_env import *
@register_env("open_drawer", max_episode_steps=200)
class OpenDrawerEnv(TableSetting):
    env_name= "open_drawer"
    def _load_scene(self, options: Dict):
        super()._load_scene(options)
        self._add_object(type_name="drawer", type_id=0)  # Source cup
    
    def _initialize_episode(self, env_idx: torch.Tensor, options: Dict):
        super()._initialize_episode(env_idx, options)
        
        default_pose=[sapien.Pose(p=[-0.03, 0.24, 0.20],q=[1, 0, 0, 0])]
        if self.random_scene:
            default_pose=self.get_random_pose(default_pose=default_pose,default_angle=0)
        print("default_pose:",default_pose)
        self._set_object_pose(
            type_name="drawer", 
            obj_id=0, 
            pose=default_pose[0]
        )
        self.drawer[0].set_openness(0)

    def check_success(self):
        print("=========== check_success ===========")
        # p0 = self.drawer[0].pose.p.numpy()[0]
        print("drawer openness:", self.drawer[0].get_openness())
        success_result = self.drawer[0].get_openness()[0] > 0.7
        print("success_result:", success_result)
        return success_result