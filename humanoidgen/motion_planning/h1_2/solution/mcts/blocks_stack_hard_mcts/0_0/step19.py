from humanoidgen.motion_planning.h1_2.solution.generated.solver_env import *
def step(planner:HumanoidMotionPlanner):

    # Step 12: Place cube2 on top and release
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
    _, target_effector_pose = planner.generate_end_effector_pose(constraints, hand_name="right")
    planner.move_to_pose_with_screw(target_effector_pose, "right", attach_obj=True, object_name="cube", object_id=2)
