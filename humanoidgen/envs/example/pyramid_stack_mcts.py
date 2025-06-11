from humanoidgen.envs.example.task_env import *

@register_env("pyramid_stack_mcts", max_episode_steps=200)
class PyramidStackMctsEnv(TableSetting):
    env_name= "pyramid_stack"

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
    
    # def check_success(self):
    #     print("=========== check_success ===========")
    #     p0 = self.cube[0].pose.p.numpy()[0]
    #     p1 = self.cube[1].pose.p.numpy()[0]
    #     p2 = self.cube[2].pose.p.numpy()[0]
    #     target_position_0=[-0.3, 0.02,0.02]
    #     target_position_1=[-0.3, -0.02,0.02]
    #     eps = [0.03, 0.03, 0.01]
    #     success_0 = all(abs(p0[i] - target_position_0[i]) <= eps[i] for i in range(3))
    #     success_1 = all(abs(p1[i] - target_position_1[i]) <= eps[i] for i in range(3))
    #     success_2 = abs(p2[2] - 0.06)<0.005 and p2[1]>p1[1] and p2[1]<p0[1]
    #     print("cube[0] position:", p0)
    #     print("cube[0] target position:", target_position_0)
    #     print("cube[1] position:", p1)
    #     print("cube[1] target position:", target_position_1)
    #     print("cube[2] position:", p2)
    #     print("success_0:", success_0)
    #     print("success_1:", success_1)
    #     print("success_2:", success_2)
    #     return success_0 and success_1 and success_2

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

        # 允许的误差范围
        eps_xy = 0.06  # x 和 y 坐标的误差
        eps_z = 0.01  # z 坐标的误差

        # 找出两个基座方块（z 坐标接近）
        base_cubes = []
        top_cube = None
        for i in range(3):
            for j in range(i + 1, 3):
                if abs(positions[i][2] - positions[j][2]) <= eps_z:  # 判断 z 坐标是否接近
                    base_cubes = [positions[i], positions[j]]
                    top_cube = [pos for pos in positions if not any(np.array_equal(pos, base) for base in base_cubes)][0]
                    break
            if base_cubes:
                break
        print("base_cubes:", base_cubes)
        print("top_cube:", top_cube)
        # 判断基座方块是否在同一水平面上（x 和 y 坐标接近）
        if base_cubes and abs(base_cubes[0][0] - base_cubes[1][0]) <= eps_xy and abs(base_cubes[0][1] - base_cubes[1][1]) <= eps_xy:
            # 判断顶部方块是否在基座方块的上方
            if abs(top_cube[2]-0.06) < eps_z:
                print("Success: The cubes are stacked correctly.")
                return True

        print("Failure: The cubes are not stacked correctly.")
        return False

