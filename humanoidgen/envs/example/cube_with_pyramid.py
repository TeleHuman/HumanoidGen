from humanoidgen.envs.example.task_env import * # Import necessary libraries
@register_env("disrupt_pyramid", max_episode_steps=200) # Register the environment 
class DisruptPyramidEnv(TableSetting): # Must inherit from TableSetting 
    def init(self, *args, **kwargs): # Necessary 
        super().init(*args, **kwargs)
    def _load_scene(self, options: Dict):  # Load objects
        super()._load_scene(options)
        # Add cubes for the pyramid
        for i in range(55):  # Total cubes needed for a 10-layer pyramid
            self._add_object(type_name="cube", type_id=i % 2)  # Alternate between green and yellow cubes

    def _initialize_episode(self, env_idx: torch.Tensor, options: Dict):  # Set object positions
        super()._initialize_episode(env_idx, options)
        # Construct the pyramid
        base_x, base_y, base_z = -0.3, 0.0, 0.015  # Base position of the pyramid
        cube_size = 0.03  # Cube side length
        a=0
        for layer in range(10):  # 10 layers
            print(f"Layer {layer}")
            num_cubes = 10 - layer
            start_y = base_y - (num_cubes / 2) * cube_size
            for i in range(num_cubes):
                print(f"  Cube {i}")
                print("p::",[base_x, start_y + i * cube_size, base_z + layer * cube_size])
                self._set_object_pose(type_name="cube", obj_id=a, pose=sapien.Pose(
                    p=[base_x, start_y + i * cube_size, base_z + layer * cube_size],
                    q=[1, 0, 0, 0]
                ))
                a=a+1
