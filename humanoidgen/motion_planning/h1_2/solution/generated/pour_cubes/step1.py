from humanoidgen.motion_planning.h1_2.solution.generated.solver_env import *
def step(planner):
    constraints=[]
    constraints.append(
        Constraint(
            env=planner.env,
            type="point2point",
            end_effector_frame="r_hand_base_link",
            hand_key_point=get_point_in_env(planner.env,point_name="grasp_point_base_right_hand"),
            object_key_point=get_point_in_env(planner.env,type_name="cup",obj_id=1)+np.array([0,0,0.015]),
        )
    )
    constraints.append(
        Constraint(
            env=planner.env,
            type="parallel",
            end_effector_frame="r_hand_base_link",
            hand_axis=get_axis_in_env(planner.env,axis_name="right_grasp_axis"),
            object_axis=np.array([0,1,0]),
        )
    )
    constraints.append(
        Constraint(
            env=planner.env,
            type="parallel",
            end_effector_frame="r_hand_base_link",
            hand_axis=get_axis_in_env(planner.env,axis_name="right_grasp_wrist_2_palm_axis"),
            object_axis=np.array([1,0,0]),
        )
    )
    _, target_effector_pose = planner.generate_end_effector_pose(constraints,hand_name="right")
    planner.move_to_pose_with_screw(target_effector_pose,"right")