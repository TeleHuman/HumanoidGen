from humanoidgen.motion_planning.h1_2.solution.generated.solver_env import *
def step(planner:HumanoidMotionPlanner):
    planner.hand_pinch("left",pinch_object="cube",obj_id=0)
    planner.hand_pinch("right",pinch_object="cube",obj_id=1)