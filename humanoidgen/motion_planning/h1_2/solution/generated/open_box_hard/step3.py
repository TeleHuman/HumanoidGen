from humanoidgen.motion_planning.h1_2.solution.generated.solver_env import *
def step(planner:HumanoidMotionPlanner):

    constraint_open = planner.generate_constraints(
        obj_name="box",
        obj_id=0,
        action="grasp",
        hand_name="right",
        openness=0.8
    )
    _, target_open_pose = planner.generate_end_effector_pose(constraint_open, "right")
    planner.move_to_pose_with_screw(
        target_open_pose,
        "right",
        attach_obj=True,
        object_name="box",
        object_id=0
    )