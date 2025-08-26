from humanoidgen.motion_planning.h1_2.solution.generated.solver_env import *
def step(planner:HumanoidMotionPlanner):

    # Step 6: Pick cube0 (left hand) at initial position
    planner.hand_pre_pinch("left")
