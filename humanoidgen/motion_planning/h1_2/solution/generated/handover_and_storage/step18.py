from humanoidgen.motion_planning.h1_2.solution.generated.solver_env import *
def step(planner:HumanoidMotionPlanner):

    constraint_open = planner.generate_constraints(
        obj_name="drawer",
        obj_id=0,
        action="grasp",
        hand_name="left",
        openness=0
    )
    _, target_open_pose = planner.generate_end_effector_pose(constraint_open, "left")
    planner.move_to_pose_with_screw(
        target_open_pose,
        "left",
        attach_obj=True,
        object_name=["drawer","rectangular_cube"],
        object_id=[0,0]
    )