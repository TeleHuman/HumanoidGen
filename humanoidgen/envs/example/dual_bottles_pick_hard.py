from humanoidgen.envs.example.task_env import *

@register_env("dual_bottles_pick_hard", max_episode_steps=200)
class DualBottlesPickHardEnv(TableSetting):
    env_name= "dual_bottles_pick_hard"

    def _load_scene(self, options: Dict):
        super()._load_scene(options)
        self._add_object(type_name="bottle", type_id=0)  # obj_i
        self._add_object(type_name="bottle", type_id=1)  # obj_id=0 for can
        
    def _initialize_episode(self, env_idx: torch.Tensor, options: Dict):
        super()._initialize_episode(env_idx, options)
        # Set can pose in right arm's workspace
        default_pose = [
            # sapien.Pose(p=[-0.31, 0.38, 0.05], q=[0.701,0.701, 0, 0]),
            # sapien.Pose(p=[-0.32, 0.32, 0.05], q=[0.701,0.701, 0, 0]), # record
            sapien.Pose(p=[-0.42, 0.18, 0.05], q=[0.701,0.701, 0, 0]),
            # sapien.Pose(p=[-0.34, -0.28, 0.05], q=[1, 0, 0, 0]),
            sapien.Pose(p=[-0.32, -0.32, 0.05], q=[1, 0, 0, 0]),
        ]
        if self.random_scene:
            default_pose=self.get_random_pose(default_pose=default_pose)
        # random_pose
        self._set_object_pose(
            type_name="bottle", 
            obj_id=0, 
            pose=default_pose[0]
        )

        self._set_object_pose(
            type_name="bottle", 
            obj_id=1, 
            pose=default_pose[1]
        )
    
    def check_success(self):
        print("=========== check_success ===========")
        p0 = self.bottle[0].pose.p.numpy()[0]
        p1 = self.bottle[1].pose.p.numpy()[0]
        eps = [0.03, 0.03, 0.03]
        target_position_0 = [-0.3, 0.08, 0.20]
        target_position_1 = [-0.3, -0.08, 0.20]
        print("bottle[0] position:", p0)
        print("bottle[0] target position:", target_position_0)
        print("bottle[1] position:", p1)
        print("bottle[1] target position:", target_position_1)
        success_0 = all(abs(p0[i] - target_position_0[i]) <= eps[i] for i in range(3))
        success_1 = all(abs(p1[i] - target_position_1[i]) <= eps[i] for i in range(3))
        print("success_0:", success_0)
        print("success_1:", success_1)
        return success_0 and success_1


