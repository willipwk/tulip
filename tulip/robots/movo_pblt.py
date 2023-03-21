import sys
import time

import ipdb
import numpy as np
import pybullet as p
from pybullet_planning import (
    get_movable_joints,
    inverse_kinematics,
    plan_cartesian_motion,
    plan_joint_motion,
)
from pybullet_planning.interfaces.task_modeling.path_interpolation import (
    interpolate_poses,
)
from pybullet_planning.utils import set_client
from pybullet_tools.ikfast.ikfast import (
    either_inverse_kinematics,
    get_ik_joints,
)
from pybullet_tools.ikfast.utils import IKFastInfo
from scipy.spatial.transform import Rotation as R
from tulip.robots.franka_panda_pblt import FrankaPanda
from tulip.utils.pblt_utils import init_sim, step_sim, vis_frame, vis_points


class MOVO(object):
    def __init__(self, urdf_file, srdf_file, mode="GUI"):
        self.mode = mode
        self.sim_cid = init_sim(mode=self.mode)

        self.robot = p.loadURDF(
            urdf_file, physicsClientId=self.sim_cid, useFixedBase=True
        )

        self.parse_joint_info()
        self.init_arm_ik()
        self.parse_srdf_file(srdf_file)

        self.home_pos = [
            -1.5,
            -0.2,
            -0.15,
            -2.0,
            2.0,
            -1.24,
            -1.1,
            1.5,
            0.2,
            0.15,
            2.0,
            -2.0,
            1.24,
            1.1,
            0.35,
        ]
        self.tuck_pos = [
            -1.6,
            -1.4,
            0.4,
            -2.7,
            0.0,
            0.5,
            -1.7,
            1.6,
            1.4,
            -0.4,
            2.7,
            0.0,
            -0.5,
            1.7,
            0.04,
        ]

        self.before_grasp_pos = [
            -1.23603711,
            -1.83125774,
            -0.0774231,
            -2.3439333,
            -0.47240137,
            -0.36272867,
            -1.63187499,
            1.23603711,
            1.83125774,
            0.0774231,
            2.3439333,
            0.47240137,
            0.36272867,
            1.63187499,
            0.028,
        ]

        self.handsup_pos = [
            -2.02393944e00,
            -7.26992644e-01,
            8.04274117e-01,
            -2.32280777e00,
            1.11521742e00,
            1.34938165e00,
            9.94467494e-01,
            2.02393944e00,
            7.26992644e-01,
            -8.04274117e-01,
            2.32280777e00,
            -1.11521742e00,
            -1.34938165e00,
            -9.94467494e-01,
            -1.00551056e00,
        ]

        # self.go_to_positions(self.tuck_pos)
        self.go_to_positions(self.before_grasp_pos)
        step_sim(20)

    def parse_joint_info(self):
        self.num_joints = p.getNumJoints(
            self.robot, physicsClientId=self.sim_cid
        )
        self.idx2jn = {}
        self.jn2idx = {}
        self.ln2idx = {}
        self.ll = []
        self.ul = []
        self.vel_limit = []
        self.f_limit = []
        for idx in range(self.num_joints):
            info = p.getJointInfo(self.robot, idx, physicsClientId=self.sim_cid)
            idx = info[0]
            jn = info[1].decode(encoding="utf-8")
            ln = info[12].decode(encoding="utf-8")
            self.idx2jn[idx] = jn
            self.jn2idx[jn] = idx
            self.ln2idx[ln] = idx
            if info[8] < info[9]:
                self.ll.append(info[8])
                self.ul.append(info[9])
                self.vel_limit.append(info[10])
                self.f_limit.append(info[11])
            else:
                self.ll.append(0)
                self.ul.append(0)
                self.vel_limit.append(0)
                self.f_limit.append(0)
            # link_state = p.getLinkState(
            #    self.robot, idx, physicsClientId=self.sim_cid
            # )
            # print(
            #    idx, ln, link_state[0], p.getEulerFromQuaternion(link_state[1])
            # )
        self.custom_limits = {
            self.jn2idx[jn]: (
                self.ll[self.jn2idx[jn]] - 1e4,
                self.ul[self.jn2idx[jn]] + 1e4,
            )
            for jn in self.jn2idx
        }
        self.upper_body_joints = [
            "right_shoulder_pan_joint",
            "right_shoulder_lift_joint",
            "right_arm_half_joint",
            "right_elbow_joint",
            "right_wrist_spherical_1_joint",
            "right_wrist_spherical_2_joint",
            "right_wrist_3_joint",
            "left_shoulder_pan_joint",
            "left_shoulder_lift_joint",
            "left_arm_half_joint",
            "left_elbow_joint",
            "left_wrist_spherical_1_joint",
            "left_wrist_spherical_2_joint",
            "left_wrist_3_joint",
            "linear_joint",
        ]
        self.upper_body_indices = [
            self.jn2idx[jn] for jn in self.upper_body_joints
        ]
        self.movable_joint_indices = get_movable_joints(self.robot)

    def init_arm_ik(self):
        """
        self.left_arm_joints = [
            "left_kinova_joint",
            "left_shoulder_pan_joint",
            "left_shoulder_lift_joint",
            "left_arm_half_joint",
            "left_elbow_joint",
            "left_wrist_spherical_1_joint",
            "left_wrist_spherical_2_joint",
            "left_wrist_3_joint",
            "left_ee_fixed_joint",
            "left_tip_fixed_joint",
        ]
        self.left_arm_indices = [self.jn2idx[n] for n in self.left_arm_joints]
        self.right_arm_joints = [
            "right_kinova_joint",
            "right_shoulder_pan_joint",
            "right_shoulder_lift_joint",
            "right_arm_half_joint",
            "right_elbow_joint",
            "right_wrist_spherical_1_joint",
            "right_wrist_spherical_2_joint",
            "right_wrist_3_joint",
            "right_ee_fixed_joint",
            "right_tip_fixed_joint",
        ]
        self.right_arm_indices = [self.jn2idx[n] for n in self.right_arm_joints]
        """
        self.ikfast_infos = {
            side: IKFastInfo(
                module_name=f"movo.movo_{side}_arm_ik",
                base_link="base_link",
                ee_link=f"{side}_ee_link",
                free_joints=["linear_joint", f"{side}_arm_half_joint"],
            )
            for side in ["left", "right"]
        }
        self.ik_joints = {
            side: get_ik_joints(
                self.robot,
                self.ikfast_infos[side],
                self.ln2idx[f"{side}_tip_link"],
            )
            for side in ["left", "right"]
        }
        self.ik_fixed_joints = {
            side: self.ik_joints[side][:1] for side in ["left", "right"]
        }

    def parse_srdf_file(self, srdf_file):
        self.disable_collision_link_pairs = []
        with open(srdf_file, "r") as fp:
            for line in fp.readlines():
                if "disable_collisions" in line:
                    link1 = (
                        line.strip()
                        .split()[1]
                        .replace('"', "")
                        .replace("link1=", "")
                    )
                    link2 = (
                        line.strip()
                        .split()[2]
                        .replace('"', "")
                        .replace("link2=", "")
                    )
                    if (link1 in self.ln2idx) and (link2 in self.ln2idx):
                        self.disable_collision_link_pairs.append(
                            (self.ln2idx[link1], self.ln2idx[link2])
                        )
                        self.disable_collision_link_pairs.append(
                            (self.ln2idx[link2], self.ln2idx[link1])
                        )
        self.disable_collision_link_pairs = set(
            self.disable_collision_link_pairs
        )

    @property
    def left_tip_pose(self):
        return self.get_tip_pose("left")

    @property
    def right_tip_pose(self):
        return self.get_tip_pose("right")

    def get_tip_pose(self, side):
        assert side in ["left", "right"], "Wrong arm side input."
        link_info = p.getLinkState(
            self.robot,
            self.ln2idx[f"{side}_tip_link"],
            physicsClientId=self.sim_cid,
        )
        return list(link_info[0]), list(link_info[1])

    def get_joint_positions(self, joint_names=None, joint_indices=None):
        if joint_indices is None:
            joint_indices = self.upper_body_indices
        joint_states = p.getJointStates(
            self.robot,
            jointIndices=joint_indices,
            physicsClientId=self.sim_cid,
        )
        return [js[1] for js in joint_states]

    # TODO to add parameters into arguments
    def arm_ik(self, final_pose, side, interpolate=True):
        stateId = p.saveState()
        if interpolate:
            init_pose = getattr(self, f"{side}_tip_pose")
            pose_path = list(
                interpolate_poses(
                    init_pose,
                    final_pose,
                    pos_step_size=0.1,
                    ori_step_size=np.pi / 6,
                )
            )
        else:
            pose_path = [final_pose]
        for i, pose in enumerate(pose_path):
            try:
                ik_q = next(
                    either_inverse_kinematics(
                        self.robot,
                        self.ikfast_infos[side],
                        self.ln2idx[f"{side}_tip_link"],
                        pose,
                        # fixed_joints=[],
                        fixed_joints=self.ik_fixed_joints[side],
                        max_time=3,
                        max_candidates=10000,
                        # max_attempts=1000,
                        verbose=False,
                    )
                )
            except:
                """
                res = input("enter to continue")
                if res == 'd':
                    ipdb; ipdb.set_trace()
                """
                ik_q = None
            if ik_q is None:
                break
            else:
                self.go_to_positions(ik_q, self.ik_joints[side])
                step_sim(20)
        p.restoreState(stateId)
        time.sleep(0.2)
        return ik_q

    def follow_waypoints(self, path, joint_indices):
        for wp in path:
            self.go_to_positions(wp, joint_indices)

    def go_to_positions(self, q, joint_indices=None):
        if joint_indices is None:
            joint_indices = self.upper_body_indices
        assert len(q) == len(joint_indices), "Wrong joint positions given"
        p.setJointMotorControlArray(
            self.robot,
            jointIndices=joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=q,
        )
        step_sim(20)

    def plan_arm_cartesian_motion(
        self,
        side,
        final_pose,
        init_pose=None,
        execute=False,
        use_ik=True,
    ):
        assert side in ["left", "right"], "Wrong arm side input."

        if init_pose is None:
            init_pose = getattr(self, f"{side}_tip_pose")
        if use_ik:
            final_q = self.arm_ik(final_pose, side, interpolate=True)
        else:
            stateId = p.saveState()
            pose_gen = interpolate_poses(
                init_pose,
                final_pose,
                pos_step_size=0.2,
                ori_step_size=np.pi / 6,
            )
            done = False
            cart_path = []
            while not done:
                try:
                    pose = next(pose_gen)
                    tmp_path = plan_cartesian_motion(
                        self.robot,
                        self.jn2idx[getattr(self, f"{side}_arm_joints")[0]],
                        self.ln2idx[f"{side}_tip_link"],
                        [pose],
                        # max_iterations=5000,
                        custom_limits=self.custom_limits,
                    )
                    cart_path += tmp_path
                except:
                    done = True
            if done:
                final_q = cart_path[-1]
            else:
                final_q = None
            p.restoreState(stateId)
            time.sleep(0.1)
        # plan a smooth traj from the beginning to the end
        stateId = p.saveState()
        if final_q is None:
            path = None
        else:
            path = self.plan_joint_motion(
                final_q,
                joint_indices=self.ik_joints[side],
                execute=execute,
            )
        if not execute:
            p.restoreState(stateId)
            time.sleep(0.2)
        return path

    def plan_joint_motion(self, q, joint_indices=None, execute=False):
        if joint_indices is None:
            joint_indices = self.upper_body_indices
        assert len(q) == len(joint_indices), "Wrong joint positions given"
        stateId = p.saveState()
        path = plan_joint_motion(
            self.robot,
            joint_indices,
            q,
            self_collisions=True,
            disabled_collisions=self.disable_collision_link_pairs,
            custom_limits=self.custom_limits,
            diagnosis=False,
        )
        if not execute:
            p.restoreState(stateId)
            time.sleep(0.2)
        return path


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


def localframe2quat(localframe):
    x_dir, y_dir, z_dir = localframe
    local_rot = np.array([x_dir, y_dir, z_dir]).T
    local_quat = R.from_matrix(local_rot).as_quat()
    return local_quat


def local2ee_frame(localframe):
    x_dir, y_dir, z_dir = localframe
    ee_x_dir = -np.array(x_dir)
    ee_z_dir = -np.array(z_dir)
    ee_y_dir = np.cross(ee_z_dir, ee_x_dir)
    ee_rot = np.array([ee_x_dir, ee_y_dir, ee_z_dir]).T
    ee_quat = R.from_matrix(ee_rot).as_quat()
    return ee_quat


def rotate_pitch_by_pi(ee_quat):
    ee_orn = R.from_quat(ee_quat).as_euler("xyz")
    ee_orn[1] += np.pi
    return R.from_euler("xyz", ee_orn).as_quat()


def execute_control_pos(side, pos, localframe, movo):
    ee_quat = getattr(movo, f"{side}_tip_pose")[1]
    vis_frame(
        pos,
        ee_quat,
        movo.sim_cid,
        length=0.1,
        duration=15,
    )
    movo.plan_arm_cartesian_motion(side, (pos, ee_quat), execute=True)
    vis_frame(
        getattr(movo, f"{side}_tip_pose")[0],
        getattr(movo, f"{side}_tip_pose")[1],
        movo.sim_cid,
        length=0.2,
        duration=15,
    )


def execute_control_pose(side, pos, localframe, movo):
    vis_frame(
        pos,
        localframe2quat(localframe),
        movo.sim_cid,
        length=0.1,
        duration=15,
    )
    ee_quat = local2ee_frame(localframe)
    ik_q = movo.arm_ik((pos, ee_quat), side)
    if ik_q is None:
        ee_quat = rotate_pitch_by_pi(ee_quat)
        vis_frame(
            pos,
            ee_quat,
            movo.sim_cid,
            length=0.1,
            duration=15,
        )
        ik_q = movo.arm_ik((pos, ee_quat), side)
    assert ik_q is not None, "No valid IK solution"
    if ik_q is not None:
        movo.plan_joint_motion(ik_q, movo.ik_joints[side], execute=True)
        vis_frame(
            getattr(movo, f"{side}_tip_pose")[0],
            getattr(movo, f"{side}_tip_pose")[1],
            movo.sim_cid,
            length=0.2,
            duration=15,
        )


def test_dual_arm_actions(left_pose_seq, right_pose_seq):
    for lp, rp in zip(left_pose_seq, right_pose_seq):
        try:
            execute_control_pose("right", rp[0], rp[1], movo)
            print("Done for right arm")
        except:
            print("Infeasible pose for the right arm!")
        try:
            execute_control_pose("left", lp[0], lp[1], movo)
            print("Done for left arm")
        except:
            print("Infeasible pose for the left arm!")
        input("enter to continue")


def test_limit(
    movo,
    side,
    mode,
    start_x=0.35,
    start_y=0,
    end_x=1.3,
    end_y=1,
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
    z = 0.4
    while x <= end_x:
        while y <= end_x:
            pos_x = x * x_sign
            pos_y = y * y_sign
            if (continue_x is not None) and (continue_y is not None):
                if pos_x < continue_x:
                    y += delta_y
                    continue
                elif pos_x == continue_x:
                    if pos_y <= continue_y:
                        y += delta_y
                        continue
            print(side, mode, pos_x, pos_y, z)

            movo.go_to_positions(movo.before_grasp_pos)
            pos = [pos_x, pos_y, z]
            path = movo.plan_arm_cartesian_motion(
                side, (pos, quat), execute=True
            )
            has_solution = int(path is not None)
            print("    ==>:", has_solution)
            with open("workspace.txt", "a") as fp:
                fp.write(f"{side} {mode} {pos_x} {pos_y} {z} {has_solution}\n")
            y += delta_y
        x += delta_x
        y = start_y


if __name__ == "__main__":

    urdf_file = "/home/zyuwei/tulip/data/urdf/movo/simplified_movo.urdf"
    srdf_file = "/home/zyuwei/tulip/data/urdf/movo/simplified_movo.srdf"
    movo = MOVO(urdf_file, srdf_file)
    """
    test_cartesian_plan(movo, "left")
    input("enter to continue")
    test_cartesian_plan(movo, "right")
    input("enter to continue")
    # test_joint_plan(movo, movo.tuck_pos)
    # input("enter to continue")
    """
    # reset_movo(movo)

    # from diffcloth_data import left_pose_seq, right_pose_seq
    # test_dual_arm_actions(left_pose_seq, right_pose_seq)
    test_limit(movo, "left", "h", continue_x=0.45, continue_y=0.05)
    test_limit(movo, "left", "v")
    test_limit(movo, "right", "h")
    test_limit(movo, "right", "v")
