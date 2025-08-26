from humanoidgen.motion_planning.h1_2.solution.generated.solver_env import *
def step(planner:HumanoidMotionPlanner):

    # Step 10: Pick cube2 (right hand) at initial position
    planner.hand_pre_pinch("right")
