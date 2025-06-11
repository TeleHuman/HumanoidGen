from humanoidgen.motion_planning.h1_2.solution.generated.solver_env import *
def step(planner:HumanoidMotionPlanner):
    constraints=[]
    
    constraints.append(
        Constraint(
            env=planner.env,
            type="point2point",
            end_effector_frame="r_hand_base_link",
            hand_key_point=get_point_in_env(planner.env,point_name="articulated_object_tail",type_name="laptop",obj_id=0),
            # object_key_point=get_point_in_env(planner.env,type_name="bowl",obj_id=0,related_point=np.array([0,0,0.25])),
            object_key_point=get_point_in_env(planner.env,point_name="articulated_object_tail",type_name="laptop",obj_id=0),
        )
    )
    # constraints.append(
    #     Constraint(
    #         env=planner.env,
    #         type="parallel",
    #         end_effector_frame="r_hand_base_link",
    #         hand_axis=np.array([0,1,0]),
    #         object_axis=np.array([0,1,0]),
    #     )
    # )
    constraints.append(
        Cost(
            env=planner.env,
            type="parallel",
            end_effector_frame="r_hand_base_link",
            hand_axis=get_point_in_env(planner.env,point_name="articulated_object_tail",type_name="laptop",obj_id=0)-get_point_in_env(planner.env,point_name="articulated_object_head",type_name="laptop",obj_id=0),
            object_axis=np.array([1,1,-1]),
        )
    )
    _, target_effector_pose = planner.generate_end_effector_pose(constraints,hand_name="right")
    planner.move_to_pose_with_screw(target_effector_pose,"right",attach_obj=True,easy_plan=True)
    # planner.move_to_pose_with_screw(target_effector_pose,"right",attach_obj=True)