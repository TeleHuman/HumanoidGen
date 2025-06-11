from humanoidgen.motion_planning.h1_2.solution.generated.solver_env import *
def step(planner:HumanoidMotionPlanner):
    constraint_r = planner.generate_constraints(
            obj_name="box",
            obj_id=0,
            action="grasp",
            hand_name="right"
        )
    _, target_effector_pose = planner.generate_end_effector_pose(constraint_r, "right")
    planner.move_to_pose_with_screw(
        target_effector_pose,
        "right",
        attach_obj=False
    )