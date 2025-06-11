from humanoidgen.motion_planning.h1_2.solution.generated.solver_env import *
def step(planner:HumanoidMotionPlanner):
    
    target_effector_pose=[]
    constraint_open = planner.generate_constraints(
        obj_name="drawer",
        obj_id=0,
        action="grasp",
        hand_name="left",
        openness=0.9
    )
    
    _, target_open_pose = planner.generate_end_effector_pose(constraint_open, "left")
    
    constraint_r_target = planner.generate_constraints(obj_name="rectangular_cube", obj_id=0, action="target", hand_name="right")
    _, target_effector_pose_r_target = planner.generate_end_effector_pose(constraint_r_target, "right")

    target_effector_pose.append(target_open_pose)
    target_effector_pose.append(target_effector_pose_r_target)
    object_name=["drawer","rectangular_cube"]
    object_id=[0,0]
    planner.move_to_pose_with_screw(target_effector_pose,hand_name="all", attach_obj=True,object_name=object_name,object_id=object_id)