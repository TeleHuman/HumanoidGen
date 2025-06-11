from humanoidgen.motion_planning.h1_2.solution.generated.solver_env import *
def step(planner:HumanoidMotionPlanner):
    target_effector_pose = []
    constraint_l = planner.generate_constraints(
            obj_name="drawer",
            obj_id=0,
            action="grasp",
            hand_name="left"
        )
    _, target_effector_pose_l = planner.generate_end_effector_pose(constraint_l, "left")
    constraint_r = planner.generate_constraints(obj_name="rectangular_cube", obj_id=0, action="pinch", hand_name="right")
    _, target_effector_pose_r = planner.generate_end_effector_pose(constraint_r, "right")
    target_effector_pose.append(target_effector_pose_l)
    target_effector_pose.append(target_effector_pose_r)
    planner.move_to_pose_with_screw(target_effector_pose,"all",attach_obj=False)