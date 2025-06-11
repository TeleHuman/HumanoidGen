from humanoidgen.envs.example.task_env import *
@register_env("open_box_hard", max_episode_steps=200)
class OpenBoxHardEnv(TableSetting):
    env_name= "open_box_hard"
    def _load_scene(self, options: Dict):
        super()._load_scene(options)
        self._add_object(type_name="box", type_id=0)  # Source cup
    
    def _initialize_episode(self, env_idx: torch.Tensor, options: Dict):
        super()._initialize_episode(env_idx, options)
        
        default_pose=[sapien.Pose(p=[-0.33, -0.2, 0.08],q=[0.965926, 0, 0, 0.258819])]
        if self.random_scene:
            default_pose=[sapien.Pose(p=[-0.33, -0.2, 0.08],q=[1, 0, 0, 0])]
            default_pose=self.get_random_pose(default_pose=default_pose,default_angle=30)
        # print("default_pose:",default_pose)
        self._set_object_pose(
            type_name="box", 
            obj_id=0, 
            pose=default_pose[0]
        )
        # self.laptop[0].get_openness()
        self.box[0].set_openness(0.4)
    
    def check_success(self):
        print("=========== check_success ===========")
        print("box openness:", self.box[0].get_openness())
        success_result = self.box[0].get_openness()[0] > 0.7
        print("success_result:", success_result)
        return success_result