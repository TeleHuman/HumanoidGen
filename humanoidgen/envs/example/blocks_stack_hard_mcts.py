from humanoidgen.envs.example.task_env import *

@register_env("blocks_stack_hard_mcts", max_episode_steps=200)
class BlocksStackHardMctsEnv(TableSetting):
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
            if not self.random_once or (self.random_once and not hasattr(self, "random_pose")):
                self.random_pose = default_pose
        # random_pose
        self._set_object_pose(
            type_name="cube", 
            obj_id=0, 
            pose=self.random_pose[0]
        )

        self._set_object_pose(
            type_name="cube", 
            obj_id=1, 
            pose=self.random_pose[1]
        )

        self._set_object_pose(
            type_name="cube", 
            obj_id=2, 
            pose=self.random_pose[2]
        )
    
    # def check_success(self):
    #     print("=========== check_success ===========")
    #     p0 = self.cube[0].pose.p.numpy()[0]
    #     p1 = self.cube[1].pose.p.numpy()[0]
    #     p2 = self.cube[2].pose.p.numpy()[0]
        
    #     eps = [0.03, 0.03, 0.01]
    #     target_position_0 = [-0.3, 0, 0.02]
    #     target_position_1 = [-0.3, 0, 0.06]
    #     target_position_2 = [-0.3, 0, 0.10]
    #     print("cube[0] position:", p0)
    #     print("cube[0] target position:", target_position_0)
    #     print("cube[1] position:", p1)
    #     print("cube[1] target position:", target_position_1)
    #     print("cube[1] position:", p2)
    #     print("cube[1] target position:", target_position_2)

    #     # success_0 = all(abs(p0[i] - target_position_0[i]) <= eps[i] for i in range(3))
    #     # success_1 = all(abs(p1[i] - target_position_1[i]) <= eps[i] for i in range(3))
    #     # success_2 = all(abs(p2[i] - target_position_2[i]) <= eps[i] for i in range(3))
    #     success_0= abs(p0[2]-p1[2]-0.04)<0.005
    #     success_1= abs(p2[2]-p0[2]-0.04)<0.005

    #     print("success_0:", success_0)
    #     print("success_1:", success_1)
    #     # print("success_2:", success_2)
    #     return success_0 and success_1


    def check_success(self):
        print("=========== check_success ===========")
        # 获取三个方块的位置信息
        positions = [
            self.cube[0].pose.p.numpy()[0],
            self.cube[1].pose.p.numpy()[0],
            self.cube[2].pose.p.numpy()[0]
        ]

        # 打印方块的位置
        for i, pos in enumerate(positions):
            print(f"cube[{i}] position:", pos)

        # 按 z 坐标排序，确保最低的方块在最前面
        positions.sort(key=lambda p: p[2])

        # 允许的误差范围
        eps_xy = 0.03  # x 和 y 坐标的误差
        eps_z = 0.005  # z 坐标的误差
        height_diff = 0.04  # 方块之间的高度差

        # 判断 x 和 y 坐标是否接近一致
        xy_aligned = (
            abs(positions[0][0] - positions[1][0]) <= eps_xy and abs(positions[0][1] - positions[1][1]) <= eps_xy and
            abs(positions[1][0] - positions[2][0]) <= eps_xy and abs(positions[1][1] - positions[2][1]) <= eps_xy
        )

        # 判断 z 坐标是否按预期高度递增
        z_aligned = (
            abs(positions[1][2] - positions[0][2] - height_diff) <= eps_z and
            abs(positions[2][2] - positions[1][2] - height_diff) <= eps_z
        )

        # 打印判断结果
        print("xy_aligned:", xy_aligned)
        print("z_aligned:", z_aligned)

        # 返回最终结果
        return xy_aligned and z_aligned

