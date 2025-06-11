from humanoidgen.motion_planning.h1_2.solution.generated.solver_env import *
def step(planner:HumanoidMotionPlanner):
    constraint_l = planner.generate_constraints(obj_name="rectangular_cube", obj_id=0, action="pinch", hand_name="left")
    _, target_effector_pose_l = planner.generate_end_effector_pose(constraint_l, "left")
    planner.move_to_pose_with_screw(target_effector_pose_l, "left", attach_obj=False)