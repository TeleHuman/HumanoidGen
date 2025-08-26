from humanoidgen.motion_planning.h1_2.solution.generated.solver_env import *
def step(planner:HumanoidMotionPlanner):

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
