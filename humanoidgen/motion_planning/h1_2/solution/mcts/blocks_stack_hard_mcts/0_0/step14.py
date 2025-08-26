from humanoidgen.motion_planning.h1_2.solution.generated.solver_env import *
def step(planner:HumanoidMotionPlanner):

    # Step 9: Return left hand to initial pose
    planner.move_to_pose_with_screw(planner.left_hand_init_pose, "left", attach_obj=False)
