from humanoidgen.motion_planning.h1_2.solution.generated.solver_env import *
def step(planner:HumanoidMotionPlanner):

    # Step 1: Pick cube1 (right hand) at initial position
    planner.hand_pre_pinch("right")

    constraints = []
    constraints.append(
        Constraint(
            env=planner.env,
            type="point2point",
            end_effector_frame="r_hand_base_link",
            hand_key_point=get_point_in_env(planner.env, point_name="pinch_point_base_right_hand"),
            object_key_point=get_point_in_env(planner.env, type_name="cube", obj_id=1)
        )
    )
    constraints.append(
        Constraint(
            env=planner.env,
            type="parallel",
            end_effector_frame="r_hand_base_link",
            hand_axis=get_axis_in_env(planner.env, axis_name="right_pinch_axis"),
            object_axis=np.array([0, 0, -1])
        )
    )
    _, target_effector_pose = planner.generate_end_effector_pose(constraints, hand_name="right")
    planner.move_to_pose_with_screw(target_effector_pose, "right", attach_obj=False)

    planner.hand_pinch("right", pinch_object="cube", obj_id=1)

    # Step 1: Lift cube1 (obj_id=1) with right hand to safe height (0.07m) with palm down
    constraints = []
    constraints.append(
        Constraint(
            env=planner.env,
            type="point2point",
            end_effector_frame="r_hand_base_link",
            hand_key_point=get_point_in_env(planner.env, type_name="cube", obj_id=1),
            object_key_point=get_point_in_env(planner.env, type_name="cube", obj_id=1) + np.array([0, 0, 0.05])
        )
    )
    constraints.append(
        Constraint(
            env=planner.env,
            type="parallel",
            end_effector_frame="r_hand_base_link",
            hand_axis=get_axis_in_env(planner.env, axis_name="right_pinch_axis"),
            object_axis=np.array([0, 0, -1])
        )
    )
    _, target_effector_pose = planner.generate_end_effector_pose(constraints, hand_name="right")
    planner.move_to_pose_with_screw(target_effector_pose, "right", attach_obj=True, object_name="cube", object_id=1)

    # Step 2: Move cube1 to base stacking position (-0.3, 0, 0.07) with right hand
    constraints = []
    constraints.append(
        Constraint(
            env=planner.env,
            type="point2point",
            end_effector_frame="r_hand_base_link",
            hand_key_point=get_point_in_env(planner.env, type_name="cube", obj_id=1),
            object_key_point=np.array([-0.3, 0, 0.07])
        )
    )
    constraints.append(
        Constraint(
            env=planner.env,
            type="parallel",
            end_effector_frame="r_hand_base_link",
            hand_axis=get_axis_in_env(planner.env, axis_name="right_pinch_axis"),
            object_axis=np.array([0, 0, -1])
        )
    )
    _, target_effector_pose = planner.generate_end_effector_pose(constraints, hand_name="right")
    planner.move_to_pose_with_screw(target_effector_pose, "right", attach_obj=True, object_name="cube", object_id=1)

    # Step 3: Place cube1 at base position (-0.3, 0, 0.02) and release
    constraints = []
    constraints.append(
        Constraint(
            env=planner.env,
            type="point2point",
            end_effector_frame="r_hand_base_link",
            hand_key_point=get_point_in_env(planner.env, type_name="cube", obj_id=1),
            object_key_point=np.array([-0.3, 0, 0.02])
        )
    )
    constraints.append(
        Constraint(
            env=planner.env,
            type="parallel",
            end_effector_frame="r_hand_base_link",
            hand_axis=get_axis_in_env(planner.env, axis_name="right_pinch_axis"),
            object_axis=np.array([0, 0, -1])
        )
    )
    _, target_effector_pose = planner.generate_end_effector_pose(constraints, hand_name="right")
    planner.move_to_pose_with_screw(target_effector_pose, "right", attach_obj=True, object_name="cube", object_id=1)

    planner.open_hand("right")

    # Step 4: Retract right hand to init pose
    planner.move_to_pose_with_screw(planner.right_hand_init_pose, "right", attach_obj=False)

    # Step 5: Pick cube0 (obj_id=0) with left hand
    planner.hand_pre_pinch("left")

    constraints = []
    constraints.append(
        Constraint(
            env=planner.env,
            type="point2point",
            end_effector_frame="l_hand_base_link",
            hand_key_point=get_point_in_env(planner.env, point_name="pinch_point_base_left_hand"),
            object_key_point=get_point_in_env(planner.env, type_name="cube", obj_id=0)
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
    planner.move_to_pose_with_screw(target_effector_pose, "left", attach_obj=False)

    planner.hand_pinch("left", pinch_object="cube", obj_id=0)

    # Step 6: Lift cube0 to safe height (0.07m) with left hand
    constraints = []
    constraints.append(
        Constraint(
            env=planner.env,
            type="point2point",
            end_effector_frame="l_hand_base_link",
            hand_key_point=get_point_in_env(planner.env, type_name="cube", obj_id=0),
            object_key_point=get_point_in_env(planner.env, type_name="cube", obj_id=0) + np.array([0, 0, 0.05])
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

    # Step 8: Place cube0 on cube1 (-0.3, 0, 0.06) and release
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

    planner.open_hand("left")

    # Step 9: Retract left hand to init pose
    planner.move_to_pose_with_screw(planner.left_hand_init_pose, "left", attach_obj=False)

    # Step 10: Pick cube2 (obj_id=2) with right hand
    planner.hand_pre_pinch("right")

    constraints = []
    constraints.append(
        Constraint(
            env=planner.env,
            type="point2point",
            end_effector_frame="r_hand_base_link",
            hand_key_point=get_point_in_env(planner.env, point_name="pinch_point_base_right_hand"),
            object_key_point=get_point_in_env(planner.env, type_name="cube", obj_id=2)
        )
    )
    constraints.append(
        Constraint(
            env=planner.env,
            type="parallel",
            end_effector_frame="r_hand_base_link",
            hand_axis=get_axis_in_env(planner.env, axis_name="right_pinch_axis"),
            object_axis=np.array([0, 0, -1])
        )
    )
    _, target_effector_pose = planner.generate_end_effector_pose(constraints, hand_name="right")
    planner.move_to_pose_with_screw(target_effector_pose, "right", attach_obj=False)

    planner.hand_pinch("right", pinch_object="cube", obj_id=2)

    # Step 11: Lift cube2 to safe height (0.07m) with right hand
    constraints = []
    constraints.append(
        Constraint(
            env=planner.env,
            type="point2point",
            end_effector_frame="r_hand_base_link",
            hand_key_point=get_point_in_env(planner.env, type_name="cube", obj_id=2),
            object_key_point=get_point_in_env(planner.env, type_name="cube", obj_id=2) + np.array([0, 0, 0.05])
        )
    )
    constraints.append(
        Constraint(
            env=planner.env,
            type="parallel",
            end_effector_frame="r_hand_base_link",
            hand_axis=get_axis_in_env(planner.env, axis_name="right_pinch_axis"),
            object_axis=np.array([0, 0, -1])
        )
    )
    _, target_effector_pose = planner.generate_end_effector_pose(constraints, hand_name="right")
    planner.move_to_pose_with_screw(target_effector_pose, "right", attach_obj=True, object_name="cube", object_id=2)

    # Step 12: Move cube2 to top stacking position (-0.3, 0, 0.15)
    constraints = []
    constraints.append(
        Constraint(
            env=planner.env,
            type="point2point",
            end_effector_frame="r_hand_base_link",
            hand_key_point=get_point_in_env(planner.env, type_name="cube", obj_id=2),
            object_key_point=np.array([-0.3, 0, 0.15])
        )
    )
    constraints.append(
        Constraint(
            env=planner.env,
            type="parallel",
            end_effector_frame="r_hand_base_link",
            hand_axis=get_axis_in_env(planner.env, axis_name="right_pinch_axis"),
            object_axis=np.array([0, 0, -1])
        )
    )
    _, target_effector_pose = planner.generate_end_effector_pose(constraints, hand_name="right")
    planner.move_to_pose_with_screw(target_effector_pose, "right", attach_obj=True, object_name="cube", object_id=2)

    # Step 13: Place cube2 on stack (-0.3, 0, 0.10) and release
    constraints = []
    constraints.append(
        Constraint(
            env=planner.env,
            type="point2point",
            end_effector_frame="r_hand_base_link",
            hand_key_point=get_point_in_env(planner.env, type_name="cube", obj_id=2),
            object_key_point=np.array([-0.3, 0, 0.10])
        )
    )
    constraints.append(
        Constraint(
            env=planner.env,
            type="parallel",
            end_effector_frame="r_hand_base_link",
            hand_axis=get_axis_in_env(planner.env, axis_name="right_pinch_axis"),
            object_axis=np.array([0, 0, -1])
        )
    )
    _, target_effector_pose = planner.generate_end_effector_pose(constraints, hand_name="right")
    planner.move_to_pose_with_screw(target_effector_pose, "right", attach_obj=True, object_name="cube", object_id=2)
