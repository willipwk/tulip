import glob

import numpy as np
from scipy.spatial.transform import Rotation as R

from tulip.robots.movo_pblt import MOVO
from tulip.utils.pblt_utils import vis_frame


def localframe2quat(localframe):
    x_dir, y_dir, z_dir = localframe
    local_rot = np.array([x_dir, y_dir, z_dir]).T
    local_quat = R.from_matrix(local_rot).as_quat()
    return local_quat


def local2ee_frame(localframe):
    quat_candidates = []
    x_dir, y_dir, z_dir = localframe
    for x_sign in [-1]:  # , 1]:
        for z_sign in [-1]:  # , 1]:
            ee_x_dir = x_sign * np.array(x_dir)
            ee_z_dir = z_sign * np.array(z_dir)
            ee_y_dir = np.cross(ee_z_dir, ee_x_dir)
            ee_rot = np.array([ee_x_dir, ee_y_dir, ee_z_dir]).T
            ee_quat = R.from_matrix(ee_rot).as_quat()
            quat_candidates.append(ee_quat)
    return quat_candidates


def rotate_pitch_by_pi(ee_quat):
    ee_orn = R.from_quat(ee_quat).as_euler("xyz")
    ee_orn[1] += np.pi
    return R.from_euler("xyz", ee_orn).as_quat()


def execute_control_pos(side, pos, localframe, movo):
    ee_quat = getattr(movo, f"{side}_tip_pose")[1]
    vis_frame(pos, ee_quat, movo.sim_cid, length=0.1, duration=15)
    ik_q = movo.arm_ik((pos, ee_quat), side, interpolate=False)
    assert ik_q is not None, "No valid IK solution"
    if ik_q is not None:
        path = movo.plan_joint_motion(ik_q, movo.ik_joints[side], execute=True)
        assert path is not None, "No collision-free feasible path planned"
        vis_frame(
            getattr(movo, f"{side}_tip_pose")[0],
            getattr(movo, f"{side}_tip_pose")[1],
            movo.sim_cid,
            length=0.2,
            duration=15,
        )


def execute_control_pose(side, pos, localframe, movo):
    ee_quats = local2ee_frame(localframe)
    feasible_ik = []
    for ee_quat in ee_quats:
        vis_frame(pos, ee_quat, movo.sim_cid, length=0.1, duration=70)
        ik_q = movo.arm_ik((pos, ee_quat), side)
        if ik_q is not None:

            feasible_ik.append(ik_q)
    assert len(feasible_ik) > 0, "No valid IK solution"
    if len(feasible_ik) > 0:
        ik_q = feasible_ik[0]
        path = movo.plan_joint_motion(ik_q, movo.ik_joints[side], execute=True)
        assert path is not None, "No collision-free feasible path planned"
        vis_frame(
            getattr(movo, f"{side}_tip_pose")[0],
            getattr(movo, f"{side}_tip_pose")[1],
            movo.sim_cid,
            length=0.2,
            duration=15,
        )


def test_dual_arm_actions(left_pose_seq, right_pose_seq):
    for i, (lp, rp) in enumerate(zip(left_pose_seq, right_pose_seq)):
        try:
            execute_control_pose("left", lp[0], lp[1], movo)
            print("Done for left arm")
            left_success = True
        except:
            print("Infeasible pose for the left arm!")
            left_success = False
        try:
            execute_control_pose("right", rp[0], rp[1], movo)
            print("Done for right arm")
            right_success = True
        except:
            print("Infeasible pose for the right arm!")
            right_success = False
        if (not right_success) or (not left_success):
            input("enter to continue")


def test_limit(
    movo,
    side,
    mode,
    start_x=0.35,
    start_y=-0.75,
    end_x=1.15,
    end_y=0,
    delta_x=0.05,
    delta_y=0.05,
    continue_x=None,
    continue_y=None,
):
    if side == "left":
        if mode == "h":
            quat = [0.71499251, -0.69844223, 0.02806634, 0.01328292]
        else:
            quat = [-0.50334611, 0.34731909, 0.39082314, -0.68794579]
        x_sign = 1
        y_sign = 1
    if side == "right":
        if mode == "v":
            quat = [-0.50334611, 0.34731909, 0.39082314, -0.68794579]
            z = 0.468
        else:
            quat = [0.75726237, -0.63944748, -0.06846320, -0.11390090]
        x_sign = 1
        y_sign = -1
    x = start_x
    y = start_y
    z = 0.44
    while x <= end_x:
        while y <= end_y:
            pos_x = x * x_sign
            pos_y = y * y_sign
            print(side, mode, pos_x, pos_y, z)

            movo.go_to_positions(movo.home_pos)
            pos = [pos_x, pos_y, z]
            path = movo.plan_arm_cartesian_motion(
                side, (pos, quat), execute=True
            )
            print(path)
            has_solution = int(path is not None)
            print("    ==>:", has_solution)
            with open("workspace.txt", "a") as fp:
                fp.write(f"{side} {mode} {pos_x} {pos_y} {z} {has_solution}\n")
            y += delta_y
        x += delta_x
        y = start_y


def read_vec(line):
    vec = [float(element) for element in line.strip().split()]
    return vec


def parse_single_step_data(data_dir, hand_fixed_quats=None, fixed_seq=None):
    with open(f"{data_dir}/start_pos.xyz", "r") as fp:
        lines = fp.readlines()
        start_pos0 = read_vec(lines[0])
        start_pos1 = read_vec(lines[1])
    with open(f"{data_dir}/start_frame.txt", "r") as fp:
        lines = fp.readlines()
        start_rot0 = [
            read_vec(lines[0]),
            read_vec(lines[1]),
            read_vec(lines[2]),
        ]
        start_rot1 = [
            read_vec(lines[3]),
            read_vec(lines[4]),
            read_vec(lines[5]),
        ]
    with open(f"{data_dir}/end_pos.xyz", "r") as fp:
        lines = fp.readlines()
        end_pos0 = read_vec(lines[0])
        end_pos1 = read_vec(lines[1])
    with open(f"{data_dir}/end_frame.txt", "r") as fp:
        lines = fp.readlines()
        end_rot0 = [read_vec(lines[0]), read_vec(lines[1]), read_vec(lines[2])]
        end_rot1 = [read_vec(lines[3]), read_vec(lines[4]), read_vec(lines[5])]
    if hand_fixed_quats is not None:
        assert ("left" in hand_fixed_quats) and (
            "right" in hand_fixed_quats
        ), "Wrong fixed_quat"
        if False:  # start_pos0[1] >= start_pos1[1]:
            print("right fixed")
            fixed_quat = hand_fixed_quats["right"]
        else:
            print("left fixed")
            fixed_quat = hand_fixed_quats["left"]
        fixed_ee_rot = R.from_quat(fixed_quat).as_matrix()
        fixed_lf_x_dir = (-1) * fixed_ee_rot[:, 0].T
        fixed_lf_z_dir = (-1) * fixed_ee_rot[:, 2].T
        fixed_lf_y_dir = np.cross(fixed_lf_z_dir, fixed_lf_x_dir)
        fixed_rot = [fixed_lf_x_dir, fixed_lf_y_dir, fixed_lf_z_dir]
        start_rot1 = fixed_rot
        end_rot1 = fixed_rot
    # if start_pos0[1] >= start_pos1[1]:
    #    return ((start_pos0, start_rot0), (end_pos0, end_rot0)), (
    #        (start_pos1, start_rot1),
    #        (end_pos1, end_rot1),
    #    )
    # else:
    return (
        ((start_pos1, start_rot1), (end_pos1, end_rot1)),
        ((start_pos0, start_rot0), (end_pos0, end_rot0)),
    )


if __name__ == "__main__":
    urdf_file = "/home/zyuwei/tulip/data/urdf/movo/simplified_movo.urdf"
    srdf_file = "/home/zyuwei/tulip/data/urdf/movo/simplified_movo.srdf"
    movo = MOVO(urdf_file, srdf_file)
    # reset_movo(movo)
    hand_fixed_quats = {
        "left": movo.left_tip_pose[1],
        "right": movo.right_tip_pose[1],
    }

    step_data = glob.glob(
        "/Users/zyuwei/Projects/movo_stack/urdf/simplified_urdf/steps/*"
    )
    for i in range(len(step_data)):
        if i < 13:
            continue
        data_dir = f"/Users/zyuwei/tulip/examples/robot/steps/step{i}/"
        print(f"==> Step {i}: {data_dir}")
        left_pose_seq, right_pose_seq = parse_single_step_data(
            data_dir, hand_fixed_quats
        )
        test_dual_arm_actions(left_pose_seq, right_pose_seq)
