from humanoidgen.motion_planning.h1_2.solution.generated.solver_env import *
def step(planner):
    constraints=[]
    constraints.append(
        Constraint(
            env=planner.env,
            type="point2point",
            end_effector_frame="l_hand_base_link",
            hand_key_point=get_point_in_env(planner.env,point_name="grasp_point_base_light_hand"),
            object_key_point=get_point_in_env(planner.env,type_name="cup",obj_id=0)+np.array([0,0,0.01]),
        )
    )
    constraints.append(
        Constraint(
            env=planner.env,
            type="parallel",
            end_effector_frame="l_hand_base_link",
            hand_axis=get_axis_in_env(planner.env,axis_name="light_grasp_axis"),
            object_axis=np.array([0,1,0]),
        )
    )
    constraints.append(
        Constraint(
            env=planner.env,
            type="parallel",
            end_effector_frame="l_hand_base_link",
            hand_axis=get_axis_in_env(planner.env,axis_name="light_grasp_wrist_2_palm_axis"),
            object_axis=np.array([1,0,0]),
        )
    )
    _, target_effector_pose = planner.generate_end_effector_pose(constraints,hand_name="left")
    planner.move_to_pose_with_screw(target_effector_pose,"left")