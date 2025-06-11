from humanoidgen.motion_planning.h1_2.solution.generated.solver_env import *
def step(planner:HumanoidMotionPlanner):

    constraint_r_target = planner.generate_constraints(obj_name="rectangular_cube", obj_id=0, action="target", hand_name="right")
    _, target_effector_pose_r_target = planner.generate_end_effector_pose(constraint_r_target, "right")
    planner.move_to_pose_with_screw(target_effector_pose_r_target, "right", attach_obj=True, object_name="rectangular_cube", object_id=0)