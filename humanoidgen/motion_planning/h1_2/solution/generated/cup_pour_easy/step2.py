from humanoidgen.motion_planning.h1_2.solution.generated.solver_env import *
def step(planner:HumanoidMotionPlanner):

    # Step 2: Close right hand to grasp cup
    planner.hand_grasp("right", grasp_object="cup", obj_id=0)