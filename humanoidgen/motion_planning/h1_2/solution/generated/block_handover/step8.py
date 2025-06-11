from humanoidgen.motion_planning.h1_2.solution.generated.solver_env import *
def step(planner:HumanoidMotionPlanner):

    import numpy as np
    constraint_l_move = planner.generate_constraints(obj_name="rectangular_cube", obj_id=0, action="move", hand_name="left", relative_obj_name="target_cube", relative_obj_id=0, relative_p=np.array([0, 0, 0.08]))
    _, target_effector_pose_l_move = planner.generate_end_effector_pose(constraint_l_move, "left")
    planner.move_to_pose_with_screw(target_effector_pose_l_move, "left", attach_obj=True, object_name="rectangular_cube", object_id=0)