from humanoidgen.motion_planning.h1_2.solution.generated.solver_env import *
def step(planner:HumanoidMotionPlanner):

    # Step 1: Move right hand to cup's grasp pose
    constraint_r = planner.generate_constraints(obj_name="cup", obj_id=0, action="grasp", hand_name="right")
    _, target_effector_pose = planner.generate_end_effector_pose(constraint_r, hand_name="right")
    planner.move_to_pose_with_screw(target_effector_pose, "right", attach_obj=False)