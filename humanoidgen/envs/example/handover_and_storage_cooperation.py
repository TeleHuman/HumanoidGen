from humanoidgen.envs.example.task_env import *

@register_env("handover_and_storage_cooperation", max_episode_steps=200)
class HandoverAndStorageCooperationEnv(TableSetting):
    env_name= "handover_and_storage_cooperation"

    def _load_scene(self, options: Dict):
        super()._load_scene(options)
        self._add_object(type_name="rectangular_cube", type_id=0)  # obj_i
        self._add_object(type_name="drawer", type_id=0)  # Source cup
    
    def _initialize_episode(self, env_idx: torch.Tensor, options: Dict):
        super()._initialize_episode(env_idx, options)
        # default_pose=[sapien.Pose(p=[-0.03, 0.24, 0.20],q=[1, 0, 0, 0])]
        # Set can pose in right arm's workspace
        default_pose_0 = [
            sapien.Pose(p=[-0.035, 0.24, 0.20],q=[1, 0, 0, 0]),
        ]
        default_pose_1 = [
            sapien.Pose(p=[-0.32, -0.32, 0.05], q=[1, 0, 0, 0]),
        ]

        if self.random_scene:
            # default_angle = 0
            # random_range=[0.02,0]
            # default_pose_0=self.get_random_pose(default_pose=default_pose_0,default_angle=default_angle,random_range=random_range)
            # default_pose_1=self.get_random_pose(default_pose=default_pose_1)

            default_angle = 0
            random_range=[0,0]
            default_pose_0=self.get_random_pose(default_pose=default_pose_0,default_angle=default_angle,random_range=random_range)
            default_pose_1=self.get_random_pose(default_pose=default_pose_1,default_angle=20,random_range=random_range)

        # random_pose
        self._set_object_pose(
            type_name="drawer", 
            obj_id=0,
            pose=default_pose_0[0]
        )

        self._set_object_pose(
            type_name="rectangular_cube", 
            obj_id=0,
            pose=default_pose_1[0]
        )
        self.drawer[0].set_openness(0)

    def check_success(self):
        print("=========== check_success ===========")
        p0 = self.drawer[0].pose.p.numpy()[0]
        p1 = self.rectangular_cube[0].pose.p.numpy()[0]
        success_0= self.drawer[0].get_openness()[0] < 0.2
        success_1 = p1[2] > 0.2 and p1[2] < 0.35
        eps = [0.17, 0.17]
        success_2 = all(abs(p0[i] - p1[i]) <= eps[i] for i in range(2))
        
        print("drawer[0] position:", p0)
        print("rectangular_cube[1] position:", p1)
        print("success_0:", success_0)
        print("success_1:", success_1)
        print("success_2:", success_2)
        return success_0 and success_1 and success_2