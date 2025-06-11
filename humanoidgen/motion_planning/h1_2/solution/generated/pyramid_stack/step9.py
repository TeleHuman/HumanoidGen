from humanoidgen.motion_planning.h1_2.solution.generated.solver_env import *
def step(planner:HumanoidMotionPlanner):
    constraint_r=planner.generate_constraints(obj_name="cube", obj_id=2, action="pinch")
    _, target_effector_pose_r = planner.generate_end_effector_pose(constraint_r,hand_name="right")
    planner.move_to_pose_with_screw(target_effector_pose_r,hand_name="right", attach_obj=False)