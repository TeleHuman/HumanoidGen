from humanoidgen.motion_planning.h1_2.utils import *
class Cost:
    def __init__(self, type, **kwargs):
        self.type = type
        if type == "point2point":
            env=kwargs.get("env",None)
            end_effector_frame = kwargs.get("end_effector_frame", None)
            hand_key_point = kwargs.get("hand_key_point", None)
            if end_effector_frame == "r_hand_base_link":
                self.hand_key_point = transform_keypoint_to_base(hand_key_point,np.linalg.inv(env.agent.right_tcp.pose.to_transformation_matrix()))
            elif end_effector_frame == "l_hand_base_link":
                self.hand_key_point = transform_keypoint_to_base(hand_key_point,np.linalg.inv(env.agent.left_tcp.pose.to_transformation_matrix()))
            self.end_effector_frame = kwargs.get("end_effector_frame", None)
            # self.hand_key_point = kwargs.get("hand_key_point", None)
            object_key_point = kwargs.get("object_key_point", None)
            self.object_key_point = transform_keypoint_to_base(object_key_point,np.linalg.inv(env.agent.base_link.pose.to_transformation_matrix()))
            # self.tolerance = kwargs.get("tolerance", None)
        elif type == "parallel":
            env=kwargs.get("env",None)
            end_effector_frame = kwargs.get("end_effector_frame", None)
            hand_axis=kwargs.get("hand_axis", None)
            if end_effector_frame == "r_hand_base_link":
                self.hand_axis = transform_keypoint_to_base(hand_axis,np.linalg.inv(env.agent.right_tcp.pose.to_transformation_matrix()))
            elif end_effector_frame == "l_hand_base_link":
                self.hand_axis = transform_keypoint_to_base(hand_axis,np.linalg.inv(env.agent.left_tcp.pose.to_transformation_matrix()))
            self.end_effector_frame = kwargs.get("end_effector_frame", None)
            self.object_axis = kwargs.get("object_axis", None) # robot quaternion frame is the same as env quaternion frame
        elif type == "attach_obj_target_pose":
            env=kwargs.get("env",None)
            end_effector_frame = kwargs.get("end_effector_frame", None)
            attach_obj_target_pose_quat = kwargs.get("attach_obj_target_pose", None)
            self.attach_obj_target_pose = quat2mat(attach_obj_target_pose_quat)
            attach_obj=kwargs.get("attach_obj",None)
            attach_obj_pose_base_env = quat2mat(attach_obj.pose.get_q()[0])
            if end_effector_frame == "r_hand_base_link":
                self.attach_obj_target_pose_base_hand = np.linalg.inv(env.agent.right_tcp.pose.to_transformation_matrix())[0,:3,:3] @ attach_obj_pose_base_env
                # self.attach_obj_target_pose_base_hand = transform_keypoint_to_base(attach_obj_target_pose_base_env,np.linalg.inv(env.agent.right_tcp.pose.to_transformation_matrix()))
            elif end_effector_frame == "l_hand_base_link":
                self.attach_obj_target_pose_base_hand = np.linalg.inv(env.agent.left_tcp.pose.to_transformation_matrix())[0,:3,:3] @ attach_obj_pose_base_env
                # self.attach_obj_target_pose_base_hand = transform_keypoint_to_base(attach_obj_target_pose_base_env,np.linalg.inv(env.agent.left_tcp.pose.to_transformation_matrix()))
            self.end_effector_frame = kwargs.get("end_effector_frame", None)
            


        else:
            raise ValueError(f"Unsupported constraint type: {type}")