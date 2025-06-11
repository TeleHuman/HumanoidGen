from humanoidgen.motion_planning.h1_2.solution.generated.solver_env import *
def step(planner):
    planner.hand_grasp(hand_name="right_hand",grasp_object="cup",obj_id=1)