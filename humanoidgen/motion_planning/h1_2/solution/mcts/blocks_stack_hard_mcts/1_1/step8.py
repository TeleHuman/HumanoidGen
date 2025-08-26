from humanoidgen.motion_planning.h1_2.solution.generated.solver_env import *
def step(planner:HumanoidMotionPlanner):

    # Step 5: Pick cube0 (obj_id=0) with left hand
    planner.hand_pre_pinch("left")
