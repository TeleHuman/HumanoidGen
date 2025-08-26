from humanoidgen.motion_planning.h1_2.solution.generated.solver_env import *
def step(planner:HumanoidMotionPlanner):

    # Step 7: Move cube0 to stacking position above cube1 (-0.3, 0, 0.11)
    constraints = []
    constraints.append(
        Constraint(
            env=planner.env,
            type="point2point",
            end_effector_frame="l_hand_base_link",
            hand_key_point=get_point_in_env(planner.env, type_name="cube", obj_id=0),
            object_key_point=np.array([-0.3, 0, 0.11])
        )
    )
    constraints.append(
        Constraint(
            env=planner.env,
            type="parallel",
            end_effector_frame="l_hand_base_link",
            hand_axis=get_axis_in_env(planner.env, axis_name="left_pinch_axis"),
            object_axis=np.array([0, 0, -1])
        )
    )
    _, target_effector_pose = planner.generate_end_effector_pose(constraints, hand_name="left")
    planner.move_to_pose_with_screw(target_effector_pose, "left", attach_obj=True, object_name="cube", object_id=0)
