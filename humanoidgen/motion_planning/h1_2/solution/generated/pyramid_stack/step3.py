from humanoidgen.motion_planning.h1_2.solution.generated.solver_env import *
def step(planner:HumanoidMotionPlanner):
    constraint_l=planner.generate_constraints(obj_name="cube", obj_id=0, action="target", hand_name="left")
    _, target_effector_pose_l = planner.generate_end_effector_pose(constraint_l,hand_name="left")
    object_name=["cube","cube"]
    object_id=[0,1]
    planner.move_to_pose_with_screw(target_effector_pose_l,hand_name="left", attach_obj=True,object_name=object_name,object_id=object_id)