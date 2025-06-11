from humanoidgen.envs.example.task_env import *
@register_env("close_box_easy", max_episode_steps=200)
class CloseBoxEasyEnv(TableSetting):
    env_name= "close_box_easy"
    def _load_scene(self, options: Dict):
        super()._load_scene(options)
        self._add_object(type_name="box", type_id=0)  # Source cup
    
    def _initialize_episode(self, env_idx: torch.Tensor, options: Dict):
        super()._initialize_episode(env_idx, options)
        
        default_pose=[sapien.Pose(p=[-0.33, -0.2, 0.08],q=[0.965926, 0, 0, 0.258819])]
        if self.random_scene:
            default_pose=[sapien.Pose(p=[-0.33, -0.2, 0.08],q=[1, 0, 0, 0])]
            default_pose=self.get_random_pose(default_pose=default_pose,default_angle=30)
        # print("default_pose:",default_pose)
        self._set_object_pose(
            type_name="box", 
            obj_id=0, 
            pose=default_pose[0]
        )
        # self.box[0].get_openness()
        self.box[0].set_openness(0.7)
        # self.box[0].set_openness(0.5)
    
    def check_success(self):
        print("=========== check_success ===========")
        # p0 = self.box[0].pose.p.numpy()[0]
        print("box openness:", self.box[0].get_openness())
        success_result = self.box[0].get_openness()[0] < 0.4
        print("success_result:", success_result)
        
        # import json
        # # save self.bbqvel_list to a json file
        # with open("bbqvel_list.json", "w") as f:
        #     # self.bbqvel_list is a list of numpy arrays
        #     # convert numpy arrays to lists
        #     self.bbqvel_list = [bbqvel.tolist() for bbqvel in self.bbqvel_list]
        #     json.dump(self.bbqvel_list, f)
        
        return success_result
    
    
    def check_failure(self):
        
        tmp_bbqvel_list = [bbqvel.tolist() for bbqvel in self.bbqvel_list]
        
        self.bbqvel_list = []
        
        result = False
        violated_qvel = 999
        violated_qvel_index = 999
        for bbqvel in tmp_bbqvel_list:
            for i in range(len(bbqvel)):
                if (bbqvel[i] > 20) and (bbqvel[i] < -20):
                    violated_qvel = bbqvel[i]
                    violated_qvel_index = i
                    result = True
                    break
                else:
                    pass
                
        print(f"violated_qvel: {violated_qvel}, violated_qvel_index: {violated_qvel_index}")
        if result == True:
            print("=========== check_failre ===========")
            print("bbqvel_list:", tmp_bbqvel_list)
            print("bbqvel:", bbqvel)
            print("violated_qvel: ", violated_qvel)
            print("violated_qvel_index: ", violated_qvel_index)
            print(f'failed you idiot!!!!!!!!!!!!!!!!!!!!!!')
        
        return result