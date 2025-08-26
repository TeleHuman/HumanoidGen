from humanoidgen.motion_planning.h1_2.solution.generated.solver_env import *
def step(planner:HumanoidMotionPlanner):

    # Step 8: Place cube0 on cube1 and release
    constraints = []
    constraints.append(
        Constraint(
            env=planner.env,
            type="point2point",
            end_effector_frame="l_hand_base_link",
            hand_key_point=get_point_in_env(planner.env, type_name="cube", obj_id=0),
            object_key_point=np.array([-0.3, 0, 0.06])
        )
    )
    _, target_effector_pose = planner.generate_end_effector_pose(constraints, hand_name="left")
    planner.move_to_pose_with_screw(target_effector_pose, "left", attach_obj=True, object_name="cube", object_id=0)
