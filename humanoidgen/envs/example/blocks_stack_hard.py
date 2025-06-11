from humanoidgen.envs.example.task_env import *

@register_env("blocks_stack_hard", max_episode_steps=200)
class BlocksStackHardEnv(TableSetting):
    env_name= "blocks_stack_hard"

    def _load_scene(self, options: Dict):
        super()._load_scene(options)
        self._add_object(type_name="cube", type_id=0)  # obj_i
        self._add_object(type_name="cube", type_id=1)  # obj_id=0 for can
        self._add_object(type_name="cube", type_id=2)  # obj_id=0 for can
        
    def _initialize_episode(self, env_idx: torch.Tensor, options: Dict):
        super()._initialize_episode(env_idx, options)
        # Set can pose in right arm's workspace
        default_pose = [
            # sapien.Pose(p=[-0.31, 0.38, 0.05], q=[0.701,0.701, 0, 0]),
            sapien.Pose(p=[-0.32, 0.32, 0.05], q=[1,0, 0, 0]),
            # sapien.Pose(p=[-0.34, -0.28, 0.05], q=[1, 0, 0, 0]),
            sapien.Pose(p=[-0.32, -0.32, 0.05], q=[1, 0, 0, 0]),
            sapien.Pose(p=[-0.45, -0.32, 0.05], q=[1, 0, 0, 0]),
        ]
        if self.random_scene:
            default_pose=self.get_random_pose(default_pose=default_pose)
        # random_pose
        self._set_object_pose(
            type_name="cube", 
            obj_id=0, 
            pose=default_pose[0]
        )

        self._set_object_pose(
            type_name="cube", 
            obj_id=1, 
            pose=default_pose[1]
        )

        self._set_object_pose(
            type_name="cube", 
            obj_id=2, 
            pose=default_pose[2]
        )
    
    def check_success(self):
        print("=========== check_success ===========")
        p0 = self.cube[0].pose.p.numpy()[0]
        p1 = self.cube[1].pose.p.numpy()[0]
        p2 = self.cube[2].pose.p.numpy()[0]
        eps = [0.03, 0.03, 0.01]
        target_position_0 = [-0.3, 0, 0.02]
        target_position_1 = [-0.3, 0, 0.06]
        target_position_2 = [-0.3, 0, 0.10]
        print("cube[0] position:", p0)
        print("cube[0] target position:", target_position_0)
        print("cube[1] position:", p1)
        print("cube[1] target position:", target_position_1)
        print("cube[1] position:", p2)
        print("cube[1] target position:", target_position_2)
        success_0 = all(abs(p0[i] - target_position_0[i]) <= eps[i] for i in range(3))
        success_1 = all(abs(p1[i] - target_position_1[i]) <= eps[i] for i in range(3))
        success_2 = all(abs(p2[i] - target_position_2[i]) <= eps[i] for i in range(3))
        print("success_0:", success_0)
        print("success_1:", success_1)
        print("success_2:", success_2)
        return success_0 and success_1 and success_2