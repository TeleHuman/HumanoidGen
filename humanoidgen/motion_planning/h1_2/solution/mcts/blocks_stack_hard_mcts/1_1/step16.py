from humanoidgen.motion_planning.h1_2.solution.generated.solver_env import *
def step(planner:HumanoidMotionPlanner):

    # Step 10: Pick cube2 (obj_id=2) with right hand
    planner.hand_pre_pinch("right")
