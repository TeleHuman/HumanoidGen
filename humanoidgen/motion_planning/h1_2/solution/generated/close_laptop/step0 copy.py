from humanoidgen.motion_planning.h1_2.solution.generated.solver_env import *
def step(planner:HumanoidMotionPlanner):
    print("keypoints:",planner.env.laptop[0].get_keypoints())