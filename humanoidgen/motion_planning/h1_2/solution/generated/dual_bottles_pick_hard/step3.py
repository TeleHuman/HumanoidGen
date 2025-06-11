from humanoidgen.motion_planning.h1_2.solution.generated.solver_env import *
def step(planner:HumanoidMotionPlanner):
    target_effector_pose=[]
    constraint_l=planner.generate_constraints(obj_name="bottle", obj_id=0, action="target", hand_name="left")
    _, target_effector_pose_l = planner.generate_end_effector_pose(constraint_l,hand_name="left")
    constraint_r=planner.generate_constraints(obj_name="bottle", obj_id=1, action="target", hand_name="right")
    _, target_effector_pose_r = planner.generate_end_effector_pose(constraint_r,hand_name="right")
    target_effector_pose.append(target_effector_pose_l)
    target_effector_pose.append(target_effector_pose_r)
    object_name=["bottle","bottle"]
    object_id=[0,1]
    planner.move_to_pose_with_screw(target_effector_pose,hand_name="all", attach_obj=True,object_name=object_name,object_id=object_id)