from humanoidgen.motion_planning.h1_2.solution.generated.solver_env import *
def step(planner:HumanoidMotionPlanner):
    constraints=[]
    point1=get_point_in_env(planner.env,point_name="articulated_object_head",type_name="laptop",obj_id=0,openness=0.35)
    point2=get_point_in_env(planner.env,point_name="articulated_object_tail",type_name="laptop",obj_id=0)
    vector=point2-point1
    grasp_point=point1+vector*0.1
    constraints.append(
        Constraint(
            env=planner.env,
            type="point2point",
            end_effector_frame="r_hand_base_link",
            hand_key_point=get_point_in_env(planner.env,point_name="pinch_point_base_right_hand"),
            object_key_point=grasp_point,
        )
    )
    constraints.append(
        Constraint(
            env=planner.env,
            type="parallel",
            end_effector_frame="r_hand_base_link",
            hand_axis=get_axis_in_env(planner.env,axis_name="right_pinch_axis"),
            object_axis=vector,
        )
    )
    constraints.append(
        Cost(
            env=planner.env,
            type="parallel",
            end_effector_frame="r_hand_base_link",
            hand_axis=get_axis_in_env(planner.env,axis_name="right_ring_2_index"),
            object_axis=np.array([0,1,0]),
        )
    )
    _, target_effector_pose = planner.generate_end_effector_pose(constraints,hand_name="right")
    planner.move_to_pose_with_screw(target_effector_pose,"right",attach_obj=True)