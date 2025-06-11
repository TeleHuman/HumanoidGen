from humanoidgen.motion_planning.h1_2.solution.generated.solver_env import *
def step(planner:HumanoidMotionPlanner):
    constraint_r=planner.generate_constraints(obj_name="cube", obj_id=1, action="move", hand_name="right",relative_obj_name="cube",relative_obj_id=0,relative_p=np.array([0,-0.04,0]))
    _, target_effector_pose_r = planner.generate_end_effector_pose(constraint_r,hand_name="right")
    object_name="cube"
    object_id=1
    planner.move_to_pose_with_screw(target_effector_pose_r,hand_name="right", attach_obj=True,object_name=object_name,object_id=object_id)