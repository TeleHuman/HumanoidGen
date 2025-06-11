from humanoidgen.motion_planning.h1_2.solution.generated.solver_env import *
def step(planner:HumanoidMotionPlanner):
    planner.hand_grasp("left",grasp_object="can",obj_id=0)