from humanoidgen.motion_planning.h1_2.solution.generated.solver_env import *
def step(planner:HumanoidMotionPlanner):
    constraint_r = planner.generate_constraints(
            obj_name="drawer",
            obj_id=0,
            action="grasp",
            hand_name="left"
        )
    _, target_effector_pose = planner.generate_end_effector_pose(constraint_r, "left")
    planner.move_to_pose_with_screw(
        target_effector_pose,
        "left",
        attach_obj=False
    )