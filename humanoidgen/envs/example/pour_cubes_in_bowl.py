from humanoidgen.envs.example.task_env import *
@register_env("pour_cubes_in_bowl", max_episode_steps=200)
class PourCubesInBowlEnv(TableSetting):
    
    def _load_scene(self, options: Dict):
        super()._load_scene(options)
        # Add cups - type_id 0=yellow, 1=green
        self._add_object(type_name="cup", type_id=1)  # Source cup
        self._add_object(type_name="bowl", type_id=0)  # Source cup
        # Add 20 small cubes (green)
        for _ in range(8):
            self._add_object(type_name="cube_small", type_id=0)
    
    def _initialize_episode(self, env_idx: torch.Tensor, options: Dict):
        super()._initialize_episode(env_idx, options)

        default_pose_0 = [sapien.Pose(p=[-0.4, 0, 0.05], q=[1,0, 0, 0])]
        default_pose_1 = [sapien.Pose(p=[-0.4, -0.3, 0.06], q=[1, 0, 0, 0])]
        
        if self.random_scene:
            default_pose_1=self.get_random_pose(default_pose=default_pose_1)


        self._set_object_pose(
            type_name="bowl", 
            obj_id=0, 
            pose=default_pose_0[0]
        )

        self._set_object_pose("cup", 0, default_pose_1[0])

        cup_x = default_pose_1[0].p[0]
        cup_y = default_pose_1[0].p[1]

        # Set cube_small poses inside source cup
        for i in range(2):
            # self._set_object_pose("cube_small", i*5, sapien.Pose(
            #     p=[-0.3 , 0.2, 0.05 + 0.02 * i],  # Arrange in a grid
            #     q=[1, 0, 0, 0]
            # ))
            self._set_object_pose("cube_small", i*4+0, sapien.Pose(
                p=[cup_x + 0.015, cup_y - 0.015, 0.01 + 0.02 * i ],  # Arrange in a grid
                q=[1, 0, 0, 0]
            ))
            self._set_object_pose("cube_small", i*4+1, sapien.Pose(
                p=[cup_x - 0.015 , cup_y - 0.015 , 0.01 + 0.02 * i // 2],  # Arrange in a grid
                q=[1, 0, 0, 0]
            ))
            self._set_object_pose("cube_small", i*4+2, sapien.Pose(
                p=[cup_x + 0.015 , cup_y + 0.015 , 0.01 + 0.02 * i ],  # Arrange in a grid
                q=[1, 0, 0, 0]
            ))
            self._set_object_pose("cube_small", i*4+3, sapien.Pose(
                p=[cup_x - 0.015 , cup_y + 0.015 , 0.01 + 0.02 * i ],  # Arrange in a grid
                q=[1, 0, 0, 0]
            ))