from humanoidgen.motion_planning.h1_2.solution.generated.solver_env import *
def step(planner:HumanoidMotionPlanner):

    # Step 5: Return right hand to initial pose
    planner.move_to_pose_with_screw(planner.right_hand_init_pose, "right", attach_obj=False)
