from tulip.robots.movo_pblt import MOVO


def test_cartesian_plan(movo, side):
    tip_pos, tip_quat = movo.get_tip_pose(side)
    tip_pos[0] += 0.3
    tip_pos[1] -= 0.2
    tip_pos[2] -= 0.3
    path = movo.plan_arm_cartesian_motion(
        side, (tip_pos, tip_quat), execute=True
    )
    assert len(path) > 0, "No path planned"


def test_joint_plan(movo):
    path = movo.plan_joint_motion(movo.home_pos, execute=True)
    path = movo.plan_joint_motion(movo.tuck_pos, execute=True)
    path = movo.plan_joint_motion(movo.before_grasp_pos, execute=True)
    assert len(path) > 0, "No path planned"


if __name__ == "__main__":
    urdf_file = "/home/zyuwei/tulip/data/urdf/movo/simplified_movo.urdf"
    srdf_file = "/home/zyuwei/tulip/data/urdf/movo/simplified_movo.srdf"
    movo = MOVO(urdf_file, srdf_file)

    test_cartesian_plan(movo, "left")
    input("enter to continue")
    test_cartesian_plan(movo, "right")
    input("enter to continue")
    test_joint_plan(movo, movo.tuck_pos)
    input("enter to continue")
