from humanoidgen.envs.example.task_env import * #

@register_env("cube_in_row", max_episode_steps=200)
class CubeInRowEnv(TableSetting):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def _load_scene(self, options: Dict):
        super()._load_scene(options)
        self._add_object(type_name="cube",type_id=0) # name="cube", obj_id=0
        self._add_object(type_name="cube",type_id=0) # name="cube", obj_id=1
        self._add_object(type_name="cube",type_id=0) # name="cube", obj_id=2
        # self.cube = []
        # self.cube.append(self._build_actor_helper(type_name="cube", obj_id=0,  scale=self.scene_scale))
        # self.cube.append(self._build_actor_helper(type_name="cube", obj_id=0,  scale=self.scene_scale))
        # self.cube.append(self._build_actor_helper(type_name="cube", obj_id=0,  scale=self.scene_scale))
       
        
        # self.apple = self._build_actor_helper(type_name="apple",obj_id=0, scale=self.scene_scale)
        # self.bowl = self._build_actor_helper(type_name="bowl",obj_id=0, scale=self.scene_scale)
        # self.can = self._build_actor_helper(type_name="can",obj_id=0, scale=self.scene_scale)

    def _initialize_episode(self, env_idx: torch.Tensor, options: Dict):
        super()._initialize_episode(env_idx, options)

        self._set_object_pose(type_name="cube",obj_id=0,pose=sapien.Pose(
            p=[-0.38, 0.39, 0.05], 
            q=[1, 0, 0, 0]
        ))
        self._set_object_pose(type_name="cube",obj_id=1,pose=sapien.Pose(
            p=[-0.38, 0.35, 0.05], 
            q=[1, 0, 0, 0]
        ))
        self._set_object_pose(type_name="cube",obj_id=2,pose=sapien.Pose(
            p=[-0.38, 0.33, 0.05], 
            q=[1, 0, 0, 0]
        ))

        # Initialize starting position and step size
        # start_position = [0.4, 1.05, 0.05]  # Starting position
        # step_x, step_y = -0.26, -0.26  # Step size for each move
        # min_x, min_y = -2, -1.1  # Minimum boundaries
        # default_orientation = [1, 0, 0, 0]  # Default quaternion orientation

        # # Define a position generator
        # def position_generator(start, step_x, step_y, min_x, min_y):
        #     x, y, z = start
        #     while x >= min_x:
        #         while y >= min_y:
        #             yield [x, y, z]
        #             y += step_y
        #         y = start[1]  # Reset y
        #         x += step_x

        # # Create an instance of the position generator
        # position_iter = position_generator(start_position, step_x, step_y, min_x, min_y)

        # # Iterate through self.objects and set positions
        # for row in self.objects:
        #     for obj in row:
        #         if obj is not None:  # Ensure the object exists
        #             try:
        #                 position = next(position_iter)
        #                 if hasattr(obj, "set_pose") and callable(getattr(obj, "set_pose")):
                            
        #                     # 如果对象有 set_pose 方法并且是可调用的，则执行
        #                     obj.set_pose(sapien.Pose(p=position, q=default_orientation))
        #                 else:
        #                     # obj.set_pos(position)
        #                     # 如果没有 set_pose 方法，输出提示语
        #                     print(f"Object {obj} does not have a callable set_pose method.")
        #             except StopIteration:
        #                 print("Position generator exhausted. Not enough space for all objects.")


    # def _initialize_episode(self, env_idx: torch.Tensor, options: Dict):
    #     super()._initialize_episode(env_idx, options)
    #     self.agent.robot.set_qpos(self.agent.keyframes["standing"].qpos)
    #     self.agent.robot.set_pose(self.init_robot_pose)
        # # Position calculations based on specified areas (midpoints)
        # self.apple.set_pose(sapien.Pose(
        #     p=[-0.38, 0.39, 0.05],  # Left area (x:-0.43~-0.33, y:0.3~0.48)
        #     q=[1, 0, 0, 0]
        # ))
        # self.bowl.set_pose(sapien.Pose(
        #     p=[-0.38, 0.01, 0.05],  # Center area (y:-0.14~0.16)
        #     q=[1, 0, 0, 0]
        # ))
        # self.can.set_pose(sapien.Pose(
        #     p=[-0.38, -0.41, 0.05],  # Right area (y:-0.5~-0.32)
        #     q=[0.5, 0.5, 0.5, 0.5]
        # ))