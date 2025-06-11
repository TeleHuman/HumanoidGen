from humanoidgen.envs.example.task_env import *

@register_env("blocks_stack_single", max_episode_steps=200)
class BlocksStackSingleEnv(TableSetting):
    env_name= "blocks_stack_single"

    def _load_scene(self, options: Dict):
        super()._load_scene(options)
        self._add_object(type_name="cube", type_id=0)  # obj_i
        self._add_object(type_name="cube", type_id=1)  # obj_id=0 for can
        
    def _initialize_episode(self, env_idx: torch.Tensor, options: Dict):
        
        super()._initialize_episode(env_idx, options)

        # Set can pose in right arm's workspace
        default_pose = [
            # sapien.Pose(p=[-0.31, 0.38, 0.05], q=[0.701,0.701, 0, 0]),
            sapien.Pose(p=[-0.32, 0, 0.05], q=[1,0, 0, 0]),
            # sapien.Pose(p=[-0.34, -0.28, 0.05], q=[1, 0, 0, 0]),
            sapien.Pose(p=[-0.32, -0.32, 0.05], q=[1, 0, 0, 0]),
        ]
        self.random_pose = default_pose.copy()
        if self.random_scene:
            random_pose=self.get_random_pose(default_pose=default_pose)
            if not self.random_once or (self.random_once and not hasattr(self, "random_pose")):
                self.random_pose = random_pose
            
        # random_pose
        self._set_object_pose(
            type_name="cube", 
            obj_id=0, 
            pose=default_pose[0]
        )

        self._set_object_pose(
            type_name="cube", 
            obj_id=1, 
            pose=self.random_pose[1]
        )
    
    def check_success(self):
        print("=========== check_success ===========")
        p0 = self.cube[0].pose.p.numpy().squeeze()
        p1 = self.cube[1].pose.p.numpy().squeeze()
        eps_xy = 0.03  # x 和 y 的误差范围
        eps_z = 0.01   # z 的误差范围
        target_position_1_z = 0.06  # cube1 的目标 z 值

        print("cube[0] position:", p0)
        print("cube[1] position:", p1)

        # 检查 p0 和 p1 的 x 和 y 是否在误差范围内
        success_xy = all(abs(p0[i] - p1[i]) <= eps_xy for i in range(2))

        # 检查 cube1 的 z 值是否在目标范围内
        success_z = abs(p1[2] - target_position_1_z) <= eps_z

        print("success_xy (x, y match):", success_xy)
        print("success_z (z match):", success_z)

        left_hand_open,right_hand_open=self.check_hand_open()
        print("left_hand_open:", left_hand_open)
        # print("right_hand_open:", right_hand_open)

        return success_xy and success_z and left_hand_open