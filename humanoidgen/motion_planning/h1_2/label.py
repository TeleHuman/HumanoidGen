from humanoidgen.motion_planning.h1_2.solution.generated.solver_env import *
def labeled_conatraints(planner:HumanoidMotionPlanner,obj_name,obj_id,action,hand_name,relative_obj_name=None,relative_obj_id=None,relative_p=None,openness=None):
    if "left" in hand_name:
        base_link_name = "l_hand_base_link"

        grasp_point_name="grasp_point_base_left_hand"
        grasp_axis = "left_grasp_axis"
        
        ring_2_index_axis="left_ring_2_index"

        pinch_axis = "left_pinch_axis"
        pinch_point_name="pinch_point_base_left_hand"
        pinch_wrist_2_palm_axis="left_pinch_wrist_2_palm_axis"
        grasp_wrist_2_palm_axis="left_grasp_wrist_2_palm_axis"
        base_hand_name="base_left_hand"

    elif "right" in hand_name:
        base_link_name = "r_hand_base_link"
        
        grasp_point_name="grasp_point_base_right_hand"
        grasp_axis = "right_grasp_axis"

        ring_2_index_axis="right_ring_2_index"
        
        pinch_axis = "right_pinch_axis"
        pinch_point_name="pinch_point_base_right_hand"
        pinch_wrist_2_palm_axis="right_pinch_wrist_2_palm_axis"
        grasp_wrist_2_palm_axis="right_grasp_wrist_2_palm_axis"

        base_hand_name="base_right_hand"


    constraints=[]
    if obj_name == "bottle" and action == "grasp":
        constraints=[]
        point1=get_point_in_env(planner.env,type_name=obj_name,obj_id=obj_id)
        obj_axis=get_axis_in_env(planner.env, "y", obj_type=obj_name, obj_id=obj_id)
        if "left" in hand_name:
            base_link_name = "l_hand_base_link"
            point_name="grasp_point_base_left_hand"
            ring_2_index_axis="left_ring_2_index"
            grasp_axis = "left_grasp_axis"

        elif "right" in hand_name:
            base_link_name = "r_hand_base_link"
            point_name="grasp_point_base_right_hand"
            ring_2_index_axis="right_ring_2_index"
            grasp_axis = "right_grasp_axis"

        # get_axis_in_env(planner.env, , obj_type=None, obj_id=0):
        constraints.append(
            Constraint(
                env=planner.env,
                type="point2point",
                end_effector_frame=base_link_name,
                hand_key_point=get_point_in_env(planner.env,point_name=point_name,related_point=np.array([0,-0.03,0])),
                object_key_point=point1,
            )
        )
        constraints.append(
            Constraint(
                env=planner.env,
                type="parallel",
                end_effector_frame=base_link_name,
                hand_axis=get_axis_in_env(planner.env,axis_name=ring_2_index_axis),
                object_axis=obj_axis,
            )
        )
        cos_theta=np.dot(obj_axis, np.array([0, 0, 1]))
        angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        angle_deg = np.degrees(angle)
        if angle_deg > 30:
            constraints.append(
                Constraint(
                    env=planner.env,
                    type="parallel",
                    end_effector_frame=base_link_name,
                    hand_axis=get_axis_in_env(planner.env,axis_name=grasp_axis),
                    object_axis=np.array([0, 0, -1]),
                )
            )
        else:
            constraints.append(
                Cost(
                    env=planner.env,
                    type="parallel",
                    end_effector_frame=base_link_name,
                    hand_axis=get_axis_in_env(planner.env,axis_name=grasp_wrist_2_palm_axis),
                    object_axis=np.array([1, 0, 0]),
                )
            )
            # if "left" in hand_name:
            #     constraints.append(
            #         Cost(
            #             env=planner.env,
            #             type="parallel",
            #             end_effector_frame=base_link_name,
            #             hand_axis=get_axis_in_env(planner.env,axis_name=grasp_wrist_2_palm_axis),
            #             object_axis=np.array([1, -1, 0]),
            #         )
            #     )

            # elif "right" in hand_name:
            #     constraints.append(
            #         Cost(
            #             env=planner.env,
            #             type="parallel",
            #             end_effector_frame=base_link_name,
            #             hand_axis=get_axis_in_env(planner.env,axis_name=grasp_wrist_2_palm_axis),
            #             object_axis=np.array([1, 1, 0]),
            #         )
            #     )
        return constraints
    
    elif action == "target" and "dual_bottles_pick" in planner.env.env_name and obj_name == "bottle":
        constraints=[]
        if "left" in hand_name:
            base_link_name = "l_hand_base_link"
            target_point=np.array([-0.3, 0.08, 0.20])
            ring_2_index_axis="left_ring_2_index"
        elif "right" in hand_name:
            base_link_name = "r_hand_base_link"
            target_point=np.array([-0.3, -0.08, 0.20])
            ring_2_index_axis="right_ring_2_index"
        constraints.append(
            Constraint(
                env=planner.env,
                type="point2point",
                end_effector_frame=base_link_name,
                hand_key_point=get_point_in_env(planner.env,type_name=obj_name,obj_id=obj_id),
                object_key_point=target_point,
            )
        )
        constraints.append(
            Constraint(
                env=planner.env,
                type="parallel",
                end_effector_frame=base_link_name,
                hand_axis=get_axis_in_env(planner.env,ring_2_index_axis),
                object_axis=np.array([0,0,1]),
            )
        )
        return constraints

    elif ("laptop" in obj_name or "box" in obj_name) and "grasp" in action:
        point1=get_point_in_env(planner.env,point_name="articulated_object_head",type_name=obj_name,obj_id=obj_id)
        point2=get_point_in_env(planner.env,point_name="articulated_object_tail",type_name=obj_name,obj_id=obj_id)
        obj_axis=get_axis_in_env(planner.env, "y", obj_type=obj_name, obj_id=obj_id)
        vector=point2-point1
        
        if openness is None:
            point1=get_point_in_env(planner.env,point_name="articulated_object_head",type_name=obj_name,obj_id=obj_id)
            constraints.append(
                Constraint(
                    env=planner.env,
                    type="parallel",
                    end_effector_frame="r_hand_base_link",
                    hand_axis=get_axis_in_env(planner.env,axis_name=grasp_axis),
                    object_axis=vector,
                )
            )
        
        elif openness is not None:
            point1=get_point_in_env(planner.env,point_name="articulated_object_head",type_name=obj_name,obj_id=obj_id,openness=openness)
            vector=point2-point1
            constraints.append(
                Constraint(
                    env=planner.env,
                    type="parallel",
                    end_effector_frame="r_hand_base_link",
                    hand_axis=get_axis_in_env(planner.env,axis_name=grasp_axis),
                    object_axis=vector,
                )
            )

        grasp_point=point1+vector*0.05
        constraints.append(
            Constraint(
                env=planner.env,
                type="point2point",
                end_effector_frame="r_hand_base_link",
                hand_key_point=get_point_in_env(planner.env,point_name=grasp_point_name),
                object_key_point=grasp_point,
            )
        )

        constraints.append(
            Cost(
                env=planner.env,
                type="parallel",
                end_effector_frame="r_hand_base_link",
                hand_axis=get_axis_in_env(planner.env,axis_name=ring_2_index_axis),
                object_axis=obj_axis,
            )
        )

    if "drawer" in obj_name:

        if "grasp" in action:
            if openness is None:
                target_point=get_point_in_env(planner.env,point_name="articulated_object_head",type_name=obj_name,obj_id=obj_id)
            elif openness is not None:
                target_point=get_point_in_env(planner.env,point_name="articulated_object_head",type_name=obj_name,obj_id=obj_id,openness=openness)
            target_point=target_point+np.array([-0.012,0,0])
            constraints.append(
                Constraint(
                    env=planner.env,
                    type="point2point",
                    end_effector_frame=base_link_name,
                    hand_key_point=get_point_in_env(planner.env,point_name=grasp_point_name),
                    object_key_point=target_point,
                )
            )
            constraints.append(
                Constraint(
                    env=planner.env,
                    type="parallel",
                    end_effector_frame=base_link_name,
                    hand_axis=get_axis_in_env(planner.env,axis_name=grasp_axis),
                    object_axis=np.array([1,0,0]),
                )
            )
            constraints.append(
                Constraint(
                    env=planner.env,
                    type="parallel",
                    end_effector_frame=base_link_name,
                    hand_axis=get_axis_in_env(planner.env,axis_name=grasp_wrist_2_palm_axis),
                    object_axis=np.array([0,0,1]),
                )
            )

    elif "target" in action  and ("open_laptop" in planner.env.env_name or "open_box" in planner.env.env_name):
        if "target1" in action:
            constraints=[]
            target_point=get_point_in_env(planner.env,point_name="articulated_object_head",type_name=obj_name,obj_id=obj_id)
            if "open_box" in planner.env.env_name:
                point1=get_point_in_env(planner.env,point_name="articulated_object_head",type_name=obj_name,obj_id=obj_id)
                point2=get_point_in_env(planner.env,point_name="articulated_object_tail",type_name=obj_name,obj_id=obj_id)
                vector=point2-point1
                target_point=point1+vector*0.06
            obj_axis=get_axis_in_env(planner.env, "x", obj_type=obj_name, obj_id=obj_id)
            obj_axis2=get_axis_in_env(planner.env, "y", obj_type=obj_name, obj_id=obj_id)
            constraints.append(
                Constraint(
                    env=planner.env,
                    type="point2point",
                    end_effector_frame=base_link_name,
                    hand_key_point=get_point_in_env(planner.env,point_name=pinch_point_name),
                    object_key_point=target_point,
                )
            )

            constraints.append(
                Constraint(
                    env=planner.env,
                    type="parallel",
                    end_effector_frame=base_link_name,
                    hand_axis=get_axis_in_env(planner.env,pinch_axis),
                    object_axis=obj_axis,
                )
            )
            constraints.append(
                Constraint(
                    env=planner.env,
                    type="parallel",
                    end_effector_frame=base_link_name,
                    hand_axis=get_axis_in_env(planner.env,ring_2_index_axis),
                    object_axis=obj_axis2,
                )
            )

        elif "target2" in action:
            constraints=[]
            if "open_box" in planner.env.env_name:
                point1=get_point_in_env(planner.env,point_name="articulated_object_head",type_name=obj_name,obj_id=obj_id,openness=1)
                point2=get_point_in_env(planner.env,point_name="articulated_object_tail",type_name=obj_name,obj_id=obj_id)
                vector=point2-point1
                target_point=point1+vector*0.06
                # target_point=get_point_in_env(planner.env,point_name="articulated_object_head",type_name=obj_name,obj_id=obj_id,openness=0.7)
            else:
                target_point=get_point_in_env(planner.env,point_name="articulated_object_head",type_name=obj_name,obj_id=obj_id,openness=0.7)
            obj_axis=get_axis_in_env(planner.env, "x", obj_type=obj_name, obj_id=obj_id)
            obj_axis2=get_axis_in_env(planner.env, "y", obj_type=obj_name, obj_id=obj_id)

            constraints.append(
                Constraint(
                    env=planner.env,
                    type="point2point",
                    end_effector_frame=base_link_name,
                    hand_key_point=get_point_in_env(planner.env,point_name=pinch_point_name),
                    object_key_point=target_point,
                )
            )

            constraints.append(
                Constraint(
                    env=planner.env,
                    type="parallel",
                    end_effector_frame=base_link_name,
                    hand_axis=get_axis_in_env(planner.env,pinch_axis),
                    object_axis=obj_axis,
                )
            )
            constraints.append(
                Constraint(
                    env=planner.env,
                    type="parallel",
                    end_effector_frame=base_link_name,
                    hand_axis=get_axis_in_env(planner.env,ring_2_index_axis),
                    object_axis=obj_axis2,
                )
            )

    elif "target" in action  and ("close_laptop" in planner.env.env_name or "close_box" in planner.env.env_name):
        if "target1" in action:
            constraints=[]
            target_point=get_point_in_env(planner.env,point_name="articulated_object_head",type_name=obj_name,obj_id=obj_id)
            obj_axis=get_axis_in_env(planner.env, "z", obj_type=obj_name, obj_id=obj_id)
            obj_axis2=get_axis_in_env(planner.env, "y", obj_type=obj_name, obj_id=obj_id)
            constraints.append(
                Constraint(
                    env=planner.env,
                    type="point2point",
                    end_effector_frame=base_link_name,
                    hand_key_point=get_point_in_env(planner.env,point_name=pinch_point_name),
                    object_key_point=target_point,
                )
            )

            constraints.append(
                Constraint(
                    env=planner.env,
                    type="parallel",
                    end_effector_frame=base_link_name,
                    hand_axis=get_axis_in_env(planner.env,pinch_axis),
                    object_axis=-obj_axis,
                )
            )
            constraints.append(
                Constraint(
                    env=planner.env,
                    type="parallel",
                    end_effector_frame=base_link_name,
                    hand_axis=get_axis_in_env(planner.env,ring_2_index_axis),
                    object_axis=obj_axis2,
                )
            )

        elif "target2" in action:
            constraints=[]
            target_point=get_point_in_env(planner.env,point_name="articulated_object_head",type_name=obj_name,obj_id=obj_id,openness=0)
            obj_axis=get_axis_in_env(planner.env, "z", obj_type=obj_name, obj_id=obj_id)
            obj_axis2=get_axis_in_env(planner.env, "y", obj_type=obj_name, obj_id=obj_id)
            constraints.append(
                Constraint(
                    env=planner.env,
                    type="point2point",
                    end_effector_frame=base_link_name,
                    hand_key_point=get_point_in_env(planner.env,point_name=pinch_point_name),
                    object_key_point=target_point,
                )
            )

            constraints.append(
                Constraint(
                    env=planner.env,
                    type="parallel",
                    end_effector_frame=base_link_name,
                    hand_axis=get_axis_in_env(planner.env,pinch_axis),
                    object_axis=-obj_axis,
                )
            )
            constraints.append(
                Constraint(
                    env=planner.env,
                    type="parallel",
                    end_effector_frame=base_link_name,
                    hand_axis=get_axis_in_env(planner.env,ring_2_index_axis),
                    object_axis=obj_axis2,
                )
            )

    elif obj_name == "cube":
        constraints.append(
            Constraint(
                env=planner.env,
                type="parallel",
                end_effector_frame=base_link_name,
                hand_axis=get_axis_in_env(planner.env,axis_name=grasp_axis),
                object_axis=np.array([0, 0, -1]),
            )
        )
        if action == "pinch":
            point1=get_point_in_env(planner.env,type_name=obj_name,obj_id=obj_id)
            obj_axis=get_axis_in_env(planner.env, "x", obj_type=obj_name, obj_id=obj_id)
            constraints.append(
                Constraint(
                    env=planner.env,
                    type="point2point",
                    end_effector_frame=base_link_name,
                    hand_key_point=get_point_in_env(planner.env,point_name=pinch_point_name),
                    object_key_point=point1,
                )
            )
            constraints.append(
                Constraint(
                    env=planner.env,
                    type="parallel",
                    end_effector_frame=base_link_name,
                    hand_axis=get_axis_in_env(planner.env,axis_name=pinch_wrist_2_palm_axis),
                    object_axis=obj_axis,
                )
            )

        elif action == "target": 
            if "blocks_stack" in planner.env.env_name:
                target_point=np.array([-0.3, 0, 0.04])
                constraints.append(
                    Constraint(
                        env=planner.env,
                        type="point2point",
                        end_effector_frame=base_link_name,
                        hand_key_point=get_point_in_env(planner.env,type_name=obj_name,obj_id=obj_id),
                        object_key_point=target_point,
                    )
                )
            elif "pyramid_stack" in planner.env.env_name:
                target_point=np.array([-0.3, 0.02, 0.04])
                constraints.append(
                    Constraint(
                        env=planner.env,
                        type="point2point",
                        end_effector_frame=base_link_name,
                        hand_key_point=get_point_in_env(planner.env,type_name=obj_name,obj_id=obj_id),
                        object_key_point=target_point,
                    )
                )
                constraints.append(
                    Constraint(
                        env=planner.env,
                        type="parallel",
                        end_effector_frame=base_link_name,
                        hand_axis=get_axis_in_env(planner.env,axis_name=pinch_wrist_2_palm_axis),
                        object_axis=np.array([1, 0, 0]),
                    )
                )
        elif action == "move":
            constraints.append(
                Constraint(
                    env=planner.env,
                    type="parallel",
                    end_effector_frame=base_link_name,
                    hand_axis=get_axis_in_env(planner.env,axis_name=pinch_wrist_2_palm_axis),
                    object_axis=np.array([1, 0, 0]),
                )
            )
            
    elif obj_name == "rectangular_cube":
        if "pinch" in action:
            obj_p=get_point_in_env(planner.env,type_name=obj_name,obj_id=obj_id)
            if "right" in hand_name:
                obj_axis=get_axis_in_env(planner.env, "x", obj_type=obj_name, obj_id=obj_id)
                constraints.append(
                    Constraint(
                        env=planner.env,
                        type="point2point",
                        end_effector_frame=base_link_name,
                        hand_key_point=get_point_in_env(planner.env,point_name=pinch_point_name),
                        object_key_point=obj_p+np.array([0,0,0.01]),
                    )
                )
                constraints.append(
                    Constraint(
                        env=planner.env,
                        type="parallel",
                        end_effector_frame=base_link_name,
                        hand_axis=get_axis_in_env(planner.env,axis_name=grasp_axis),
                        object_axis=np.array([0, 0, -1]),
                    )
                )
                constraints.append(
                    Constraint(
                        env=planner.env,
                        type="parallel",
                        end_effector_frame=base_link_name,
                        hand_axis=get_axis_in_env(planner.env,axis_name=pinch_wrist_2_palm_axis),
                        object_axis=obj_axis,
                    )
                )
                # constraints.append(
                #     Constraint(
                #         env=planner.env,
                #         type="parallel",
                #         end_effector_frame=base_link_name,
                #         hand_axis=get_axis_in_env(planner.env,axis_name=grasp_axis),
                #         object_axis=np.array([0, 1, 0]),
                #     )
                # )
                # constraints.append(
                #     Constraint(
                #         env=planner.env,
                #         type="parallel",
                #         end_effector_frame=base_link_name,
                #         hand_axis=get_axis_in_env(planner.env,axis_name=pinch_wrist_2_palm_axis),
                #         object_axis=obj_axis,
                #     )
                # )
            
            elif "left" in hand_name:
                obj_axis=get_axis_in_env(planner.env, "z", obj_type=obj_name, obj_id=obj_id)
                constraints1=[]
                # pre-pinch
                obj_p=get_point_in_env(planner.env,type_name=obj_name,obj_id=obj_id,related_point=np.array([0,0,-0.06]))
                constraints1.append(
                    Constraint(
                        env=planner.env,
                        type="point2point",
                        end_effector_frame=base_link_name,
                        hand_key_point=get_point_in_env(planner.env,point_name=pinch_point_name),
                        object_key_point=obj_p,
                    )
                )
                constraints1.append(
                    Constraint(
                        env=planner.env,
                        type="parallel",
                        end_effector_frame=base_link_name,
                        hand_axis=get_axis_in_env(planner.env,axis_name=pinch_axis),
                        object_axis=obj_axis,
                    )
                )
                constraints1.append(
                    Constraint(
                        env=planner.env,
                        type="parallel",
                        end_effector_frame=base_link_name,
                        hand_axis=get_axis_in_env(planner.env,axis_name=ring_2_index_axis),
                        object_axis=np.array([0, 0, 1]),
                    )
                )
                constraints2=[]
                obj_p=get_point_in_env(planner.env,type_name=obj_name,obj_id=obj_id,related_point=np.array([0,0,-0.015]))
                constraints2.append(
                    Constraint(
                        env=planner.env,
                        type="point2point",
                        end_effector_frame=base_link_name,
                        hand_key_point=get_point_in_env(planner.env,point_name=pinch_point_name),
                        object_key_point=obj_p,
                    )
                )
                constraints2.append(
                    Constraint(
                        env=planner.env,
                        type="parallel",
                        end_effector_frame=base_link_name,
                        hand_axis=get_axis_in_env(planner.env,axis_name=pinch_axis),
                        object_axis=obj_axis,
                    )
                )
                constraints2.append(
                    Constraint(
                        env=planner.env,
                        type="parallel",
                        end_effector_frame=base_link_name,
                        hand_axis=get_axis_in_env(planner.env,axis_name=ring_2_index_axis),
                        object_axis=np.array([0, 0, 1]),
                    )
                )
                constraints.append(constraints1)
                constraints.append(constraints2)
                return constraints
        
        elif "target" in action:

            target_point=np.array([-0.4, 0, 0.2])
            constraints.append(
                Constraint(
                    env=planner.env,
                    type="point2point",
                    end_effector_frame=base_link_name,
                    hand_key_point=get_point_in_env(planner.env,type_name=obj_name,obj_id=obj_id),
                    object_key_point=target_point,
                )
            )
            constraints.append(
                Constraint(
                    env=planner.env,
                    type="parallel",
                    end_effector_frame=base_link_name,
                    hand_axis=get_axis_in_env(planner.env,axis_name=pinch_axis),
                    object_axis=np.array([0, 1, 0]),
                )
            )
            constraints.append(
                Constraint(
                    env=planner.env,
                    type="parallel",
                    end_effector_frame=base_link_name,
                    hand_axis=get_axis_in_env(planner.env,axis_name=ring_2_index_axis),
                    object_axis=np.array([0, 0, 1]),
                )
            )
        
        elif "move" in action:
            obj_axis=get_axis_in_env(planner.env, "z", obj_type=obj_name, obj_id=obj_id)
            print("obj_axis:",obj_axis)
            print("pinch_axis:",get_axis_in_env(planner.env,axis_name=pinch_axis))
            if "handover_and_storage" in planner.env.env_name:
                constraints.append(
                    Constraint(
                        env=planner.env,
                        type="parallel",
                        end_effector_frame=base_link_name,
                        hand_axis=get_axis_in_env(planner.env,axis_name=pinch_axis),
                        object_axis=np.array([1, 0, -1]),
                    )
                )
            else:
                constraints.append(
                    Constraint(
                        env=planner.env,
                        type="parallel",
                        end_effector_frame=base_link_name,
                        hand_axis=get_axis_in_env(planner.env,axis_name=pinch_axis),
                        object_axis=np.array([0, 0, -1]),
                    )
                )
            

    elif obj_name=="cup":
        constraints.append(
            Constraint(
                env=planner.env,
                type="parallel",
                end_effector_frame=base_link_name,
                hand_axis=get_axis_in_env(planner.env,axis_name=ring_2_index_axis),
                object_axis=np.array([0, 0, 1]),
            )
        )
        if action == "grasp":
            obj_p=get_point_in_env(planner.env,type_name=obj_name,obj_id=obj_id)
            constraints.append(
                Constraint(
                    env=planner.env,
                    type="point2point",
                    end_effector_frame=base_link_name,
                    hand_key_point=get_point_in_env(planner.env,point_name=grasp_point_name),
                    object_key_point=obj_p+np.array([0,0,0.015]),
                )
            )
            # if "cup_pour_easy" in planner.env.env_name:
            #     constraints.append(
            #         Constraint(
            #             env=planner.env,
            #             type="parallel",
            #             end_effector_frame=base_link_name,
            #             hand_axis=get_axis_in_env(planner.env,axis_name=grasp_wrist_2_palm_axis),
            #             object_axis=np.array([1, 0, 0]),
            #         )
            #     )

        # elif action == "move":
        #     constraints.append(
        #         Cost(
        #             env=planner.env,
        #             type="parallel",
        #             end_effector_frame=base_link_name,
        #             hand_axis=get_axis_in_env(planner.env, "y", obj_type=obj_name, obj_id=obj_id),
        #             object_axis=np.array([0, 0, 1]),
        #         )
        #     )
        elif action == "pour":
            
            constraints=[]

            constraints.append(
                Constraint(
                    env=planner.env,
                    type="point2point",
                    end_effector_frame=base_link_name,
                    hand_key_point=get_point_in_env(planner.env,base_hand_name),
                    object_key_point=get_point_in_env(planner.env,base_hand_name),
                )
            )
            constraints.append(
                Cost(
                    env=planner.env,
                    type="point2point",
                    end_effector_frame="r_hand_base_link",
                    hand_key_point=get_point_in_env(planner.env,type_name="cup",obj_id=0,related_point=np.array([0,0,0.01])),
                    # object_key_point=get_point_in_env(planner.env,type_name="bowl",obj_id=0,related_point=np.array([0,0,0.25])),
                    object_key_point=get_point_in_env(planner.env,type_name="bowl",obj_id=0,related_point=np.array([0,0,0.2])),
                )
            )
            # print("cup_z::",get_axis_in_env(planner.env,obj_type="cup",axis_name="z"))
            # print("hand_z::",get_axis_in_env(planner.env,axis_name="right_ring_2_index"))
            constraints.append(
                Cost(
                    env=planner.env,
                    type="parallel",
                    end_effector_frame=base_link_name,
                    hand_axis=get_axis_in_env(planner.env,axis_name="right_ring_2_index"),
                    # hand_axis=get_axis_in_env(planner.env,obj_type="cup",axis_name="z"),
                    object_axis=np.array([0,0,-1]),
                )
            )
        elif action == "move":
            constraints=[]
            constraint1=[]
            constraint1.append(
                Constraint(
                    env=planner.env,
                    type="parallel",
                    end_effector_frame=base_link_name,
                    # hand_axis=get_axis_in_env(planner.env, "y", obj_type=obj_name, obj_id=obj_id),
                    hand_axis=get_axis_in_env(planner.env,axis_name=ring_2_index_axis),
                    object_axis=np.array([0, 0, 1]),
                )
            )
            relative_obj_p=get_point_in_env(planner.env,type_name=relative_obj_name,obj_id=relative_obj_id)
            target_point=relative_obj_p+relative_p+np.array([0,0,0.05])
            constraint1.append(
                Constraint(
                    env=planner.env,
                    type="point2point",
                    end_effector_frame=base_link_name,
                    hand_key_point=get_point_in_env(planner.env,type_name=obj_name,obj_id=obj_id),
                    object_key_point=target_point,
                )
            )
            constraint2=[]
            constraint2.append(
                Constraint(
                    env=planner.env,
                    type="parallel",
                    end_effector_frame=base_link_name,
                    hand_axis=get_axis_in_env(planner.env,axis_name=ring_2_index_axis),
                    # hand_axis=get_axis_in_env(planner.env, "y", obj_type=obj_name, obj_id=obj_id),
                    object_axis=np.array([0, 0, 1]),
                )
            )
            relative_obj_p=get_point_in_env(planner.env,type_name=relative_obj_name,obj_id=relative_obj_id)
            target_point=relative_obj_p+relative_p
            constraint2.append(
                Constraint(
                    env=planner.env,
                    type="point2point",
                    end_effector_frame=base_link_name,
                    hand_key_point=get_point_in_env(planner.env,type_name=obj_name,obj_id=obj_id),
                    object_key_point=target_point,
                )
            )
            constraints.append(constraint1)
            constraints.append(constraint2)
            return constraints

    if action=="move":
        if relative_obj_name is "drawer":
            relative_obj_p=get_point_in_env(planner.env,point_name="articulated_object_head",type_name=relative_obj_name,obj_id=relative_obj_id)
        else:
            relative_obj_p=get_point_in_env(planner.env,type_name=relative_obj_name,obj_id=relative_obj_id)
        target_point=relative_obj_p+relative_p
        constraints.append(
            Constraint(
                env=planner.env,
                type="point2point",
                end_effector_frame=base_link_name,
                hand_key_point=get_point_in_env(planner.env,type_name=obj_name,obj_id=obj_id),
                object_key_point=target_point,
            )
        )

    return constraints
