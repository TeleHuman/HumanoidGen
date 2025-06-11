
from humanoidgen.motion_planning.h1_2.solution.generated.solver_env import *
def step(planner:HumanoidMotionPlanner):
    target_effector_pose=[]
    target_effector_pose.append(planner.left_hand_init_pose)
    target_effector_pose.append(planner.right_hand_init_pose)
    object_name=["bottle","bottle"]
    object_id=[0,1]
    planner.move_to_pose_with_screw(pose=target_effector_pose, hand_name="all", attach_obj=True,object_name=object_name,object_id=object_id)