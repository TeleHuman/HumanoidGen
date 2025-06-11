from humanoidgen.motion_planning.h1_2.solution.generated.solver_env import *
def step(planner:HumanoidMotionPlanner):
    planner.move_to_pose_with_screw(planner.left_hand_init_pose,hand_name="left")