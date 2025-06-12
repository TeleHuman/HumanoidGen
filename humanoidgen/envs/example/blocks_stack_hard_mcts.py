from humanoidgen.envs.example.task_env import *

@register_env("blocks_stack_hard_mcts", max_episode_steps=200)
class BlocksStackHardMctsEnv(TableSetting):
    env_name= "blocks_stack_hard"

    def _load_scene(self, options: Dict):
        super()._load_scene(options)
        self._add_object(type_name="cube", type_id=0)  # 
        self._add_object(type_name="cube", type_id=1)  # 
        self._add_object(type_name="cube", type_id=2)  # 
        
    def _initialize_episode(self, env_idx: torch.Tensor, options: Dict):
        super()._initialize_episode(env_idx, options)
        default_pose = [
            sapien.Pose(p=[-0.32, 0.32, 0.05], q=[1,0, 0, 0]),
            sapien.Pose(p=[-0.32, -0.32, 0.05], q=[1, 0, 0, 0]),
            sapien.Pose(p=[-0.45, -0.32, 0.05], q=[1, 0, 0, 0]),
        ]
        pose_setting = default_pose
        if self.random_scene:
            random_pose=self.get_random_pose(default_pose=default_pose)
            if not self.random_once or (self.random_once and not hasattr(self, "random_pose")):
                pose_setting = random_pose
        # random_pose
        self._set_object_pose(
            type_name="cube", 
            obj_id=0, 
            pose=pose_setting[0]
        )

        self._set_object_pose(
            type_name="cube", 
            obj_id=1, 
            pose=pose_setting[1]
        )

        self._set_object_pose(
            type_name="cube", 
            obj_id=2, 
            pose=pose_setting[2]
        )
    
    def check_success(self):
        print("=========== check_success ===========")
        positions = [
            self.cube[0].pose.p.numpy()[0],
            self.cube[1].pose.p.numpy()[0],
            self.cube[2].pose.p.numpy()[0]
        ]

        for i, pos in enumerate(positions):
            print(f"cube[{i}] position:", pos)

        positions.sort(key=lambda p: p[2])

        eps_xy = 0.03
        eps_z = 0.005
        height_diff = 0.04 

        xy_aligned = (
            abs(positions[0][0] - positions[1][0]) <= eps_xy and abs(positions[0][1] - positions[1][1]) <= eps_xy and
            abs(positions[1][0] - positions[2][0]) <= eps_xy and abs(positions[1][1] - positions[2][1]) <= eps_xy
        )
        z_aligned = (
            abs(positions[1][2] - positions[0][2] - height_diff) <= eps_z and
            abs(positions[2][2] - positions[1][2] - height_diff) <= eps_z
        )
        print("xy_aligned:", xy_aligned)
        print("z_aligned:", z_aligned)
        return xy_aligned and z_aligned

