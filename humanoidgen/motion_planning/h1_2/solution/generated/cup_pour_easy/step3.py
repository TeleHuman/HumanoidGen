from humanoidgen.motion_planning.h1_2.solution.generated.solver_env import *
def step(planner:HumanoidMotionPlanner):

    # Step 3: Calculate placement pose above plate
    constraint_place = planner.generate_constraints(
        obj_name="cup", 
        obj_id=0, 
        action="move", 
        hand_name="right",
        relative_obj_name="bowl",
        relative_obj_id=0,
        relative_p=np.array([0,-0.04,0.2]))
    _, target_place_pose = planner.generate_end_effector_pose(constraint_place, hand_name="right")
    planner.move_to_pose_with_screw(
        target_place_pose, 
        "right", 
        attach_obj=True, 
        object_name="cup", 
        object_id=0)