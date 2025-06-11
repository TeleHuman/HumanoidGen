from humanoidgen.envs.example.task_env import *

@register_env("block_handover", max_episode_steps=200)
class BlockHandoverEnv(TableSetting):
    env_name= "block_handover"

    def _load_scene(self, options: Dict):
        super()._load_scene(options)
        self._add_object(type_name="rectangular_cube", type_id=0)
        self._add_object(type_name="target_cube", type_id=0)
        
    def _initialize_episode(self, env_idx: torch.Tensor, options: Dict):
        super()._initialize_episode(env_idx, options)
        default_pose = [
            sapien.Pose(p=[-0.32, 0.32, 0.05], q=[1,0, 0, 0]),
            sapien.Pose(p=[-0.32, -0.32, 0.05], q=[1, 0, 0, 0]),
        ]
        if self.random_scene:
            default_pose=self.get_random_pose(default_pose=default_pose)
        self._set_object_pose(
            type_name="target_cube", 
            obj_id=0,
            pose=default_pose[0]
        )

        self._set_object_pose(
            type_name="rectangular_cube", 
            obj_id=0,
            pose=default_pose[1]
        )
    
    def check_success(self):
        print("=========== check_success ===========")
        p0 = self.rectangular_cube[0].pose.p.numpy()[0]
        p1 = self.target_cube[0].pose.p.numpy()[0]
        print("rectangular_cube[0] position:", p0)
        print("target_cube[0] position:", p1)
        eps = 0.02
        success_0= abs(p0[0] - p1[0]) < eps and abs(p0[1] - p1[1]) < eps and abs(p0[2] - 0.036)< 0.002
        print("success_0:", success_0)
        # print("success_1:", success_1)
        #  and abs(box_pos[2] - 0.85) < 0.0015
        return success_0 