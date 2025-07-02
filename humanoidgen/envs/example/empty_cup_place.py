from humanoidgen.envs.example.task_env import *

@register_env("empty_cup_place", max_episode_steps=200)
class EmptyCupPlaceEnv(TableSetting):
    env_name= "empty_cup_place"

    def _load_scene(self, options: Dict):
        super()._load_scene(options)
        self._add_object(type_name="plate", type_id=0)  # obj_i
        self._add_object(type_name="cup", type_id=0)  # obj_id=0 for can
        
    def _initialize_episode(self, env_idx: torch.Tensor, options: Dict):
        super()._initialize_episode(env_idx, options)

        # Set can pose in right arm's workspace
        default_pose = [
            # sapien.Pose(p=[-0.31, 0.38, 0.05], q=[0.701,0.701, 0, 0]),
            sapien.Pose(p=[-0.32, 0, 0.008], q=[0.701,0.701, 0, 0]),
            # sapien.Pose(p=[-0.34, -0.28, 0.05], q=[1, 0, 0, 0]),
            sapien.Pose(p=[-0.32, -0.32, 0.05], q=[0.701, 0.701, 0, 0]),
            # sapien.Pose(p=[-0.32, -0.32, 0.05], q=[1, 0, 0, 0]),
        ]
        if self.random_scene:
            default_pose=self.get_random_pose(default_pose=default_pose)
        # random_pose
        self._set_object_pose(
            type_name="plate", 
            obj_id=0, 
            pose=default_pose[0]
        )

        self._set_object_pose(
            type_name="cup", 
            obj_id=0, 
            pose=default_pose[1]
        )
    
    def check_success(self):
        print("=========== check_success ===========")

        obj_axis=self.cup[0].pose.to_transformation_matrix()[0][:,1][:3]
        cos_theta=np.dot(obj_axis, np.array([0, 0, 1]))
        angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        angle_deg = np.degrees(angle)
        success_0 = angle_deg < 30
        p0 = self.cup[0].pose.p.numpy()[0]
        p1 = self.plate[0].pose.p.numpy()[0]
        eps = [0.04, 0.04, 0.1]
        success_1=all(abs(p0[i] - p1[i]) <= eps[i] for i in range(3))
        print("success_0:", success_0)
        print("success_1:", success_1)
        return success_0 and success_1