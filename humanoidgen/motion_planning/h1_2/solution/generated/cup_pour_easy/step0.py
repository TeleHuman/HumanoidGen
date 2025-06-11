from humanoidgen.motion_planning.h1_2.solution.generated.solver_env import *
def step(planner:HumanoidMotionPlanner):
    # Step 0: Prepare right hand to grasp cup
    planner.hand_pre_grasp("right")