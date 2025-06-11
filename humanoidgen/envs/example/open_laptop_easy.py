from humanoidgen.envs.example.task_env import *
@register_env("open_laptop_easy", max_episode_steps=200)
class OpenLaptopEasyEnv(TableSetting):
    env_name= "open_laptop_easy"
    def _load_scene(self, options: Dict):
        super()._load_scene(options)
        self._add_object(type_name="laptop", type_id=0)  # Source cup
    
    def _initialize_episode(self, env_idx: torch.Tensor, options: Dict):
        super()._initialize_episode(env_idx, options)
        
        random_range = np.array([0.01,0.04])
        random_angle = np.array([-30,10])
        default_pose = [
            sapien.Pose(p=[-0.32, -0.28, 0.05], q=[1,0, 0, 0]),
        ]

        if self.random_scene:
            default_pose=self.get_random_pose(default_pose=default_pose,random_angle=random_angle,random_range=random_range)

        self._set_object_pose(
            type_name="laptop", 
            obj_id=0, 
            pose=default_pose[0]
        )
        # self.laptop[0].get_openness()
        self.laptop[0].set_openness(0.4)

    def check_success(self):
        print("=========== check_success ===========")
        # p0 = self.laptop[0].pose.p.numpy()[0]
        print("laptop openness:", self.laptop[0].get_openness())
        success_result = self.laptop[0].get_openness()[0] > 0.52
        print("success_result:", success_result)
        return success_result