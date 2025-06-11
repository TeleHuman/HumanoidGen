from humanoidgen.motion_planning.h1_2.solution.generated.solver_env import *
def step(planner:HumanoidMotionPlanner):
    planner.hand_pinch("left", pinch_object="rectangular_cube", obj_id=0)