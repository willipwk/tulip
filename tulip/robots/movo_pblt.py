import numpy as np

import pybullet as p
from pybullet_planning import (
    get_collision_fn,
    get_joint_positions,
    get_movable_joints,
    interpolate_poses,
    inverse_kinematics,
    plan_cartesian_motion,
    plan_cartesian_motion_lg,
    plan_joint_motion,
    sample_tool_ik,
)
from pybullet_tools.ikfast.ikfast import (
    either_inverse_kinematics,
    get_ik_joints,
    import_ikfast,
)
from pybullet_tools.ikfast.utils import IKFastInfo
from tulip.utils.pblt_utils import (
    init_sim,
    restore_states,
    save_states,
    step_sim,
    vis_frame,
    vis_points,
)


def get_sample_ik_fn(robot, ik_fn, robot_base_link, ik_joints, fixed_joints):
    def sample_ik_fn(world_from_tcp):
        return sample_tool_ik(
            ik_fn=ik_fn,
            robot=robot,
            ik_joints=ik_joints,
            world_from_tcp=world_from_tcp,
            base_link=robot_base_link,
            sampled=[0, np.random.uniform(-np.pi, np.pi)],
            get_all=True,
        )

    return sample_ik_fn


class MOVO(object):
    def __init__(self, urdf_file, srdf_file, mode="GUI"):
        self.mode = mode
        self.sim_cid = init_sim(mode=self.mode)

        self.robot = p.loadURDF(
            urdf_file, physicsClientId=self.sim_cid, useFixedBase=True
        )
        self.table = p.loadURDF(
            "cube/cube.urdf", basePosition=(0.65, 0, 0.2), useFixedBase=True
        )
        """
        self.tie = p.loadURDF("tie/tie.urdf", useFixedBase=True)
        with open("tie/tie_segment_world.xyz", "r") as fp:
            for line in fp:
                pcd.append([float(element) for element in line.strip().split()])
        vis_points(
            np.array(pcd),
            self.sim_cid,
            sample_size=5000,
            color=[0, 1, 0],
            duration=50,
        )
        pcd = []
        with open("tie/tie.xyz", "r") as fp:
            for line in fp:
                pcd.append([float(element) for element in line.strip().split()])
        vis_points(
            np.array(pcd),
            self.sim_cid,
            sample_size=5000,
            color=[0, 0, 1],
            duration=50,
        )
        """

        self.parse_joint_info()
        self.init_arm_ik()
        self.parse_srdf_file(srdf_file)
        self.state = {}

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
            self.robot, jointIndices=joint_indices, physicsClientId=self.sim_cid
        )
        return [js[0] for js in joint_states]

    def get_joint_velocities(self, joint_names=None, joint_indices=None):
        if joint_indices is None:
            joint_indices = self.upper_body_indices
        joint_states = p.getJointStates(
            self.robot, jointIndices=joint_indices, physicsClientId=self.sim_cid
        )
        return [js[1] for js in joint_states]

    def reset_to_states(self, q, dq=None, joint_indices=None):
        if joint_indices is None:
            joint_indices = self.upper_body_indices
        assert len(q) == len(joint_indices), "Wrong joint positions given"
        if dq is not None:
            assert len(dq) == len(joint_indices), "Wrong joint positions given"
        else:
            dq = [0 for idx in joint_indices]
        for joint_idx, joint_pos, joint_vel in zip(joint_indices, q, dq):
            p.resetJointState(
                self.robot,
                jointIndex=joint_idx,
                targetValue=joint_pos,
                targetVelocity=joint_vel,
                physicsClientId=self.sim_cid,
            )
        step_sim(1)

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

    # TODO to add parameters into arguments
    def arm_ik(self, final_pose, side, interpolate=True):
        state_id = save_states(self)
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

        ikfast = import_ikfast(self.ikfast_infos[side])
        ik_fn = ikfast.get_ik
        ik_joints = get_ik_joints(
            self.robot, self.ikfast_infos[side], self.ln2idx[f"{side}_tip_link"]
        )
        collision_fn = get_collision_fn(
            self.robot,
            ik_joints,
            obstacles=[],
            attachments=[],
            self_collision=True,
            disabled_collisions=self.disable_collision_link_pairs,
            custom_limits=self.custom_limits,
        )
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
                        max_time=5,
                        max_candidates=500000,
                        # max_attempts=1000,
                        verbose=False,
                        collision_fn=collision_fn,
                    )
                )
            except:
                ik_q = None
            if ik_q is None:
                break
            else:
                self.go_to_positions(ik_q, self.ik_joints[side])
                step_sim(20)
        restore_states(self, state_id)
        return ik_q

    def follow_waypoints(self, path, joint_indices):
        for wp in path:
            self.go_to_positions(wp, joint_indices)

    def plan_arm_cartesian_motion(
        self, side, final_pose, init_pose=None, execute=False
    ):

        state_id = save_states(self)

        ikfast = import_ikfast(self.ikfast_infos[side])
        ik_fn = ikfast.get_ik
        ik_joints = get_ik_joints(
            self.robot, self.ikfast_infos[side], self.ln2idx[f"{side}_tip_link"]
        )
        sample_ik_fn = get_sample_ik_fn(
            self.robot,
            ik_fn,
            -1,  # self.ln2idx[self.ikfast_infos[side].base_link],
            ik_joints,
            self.ik_fixed_joints[side],
        )

        collision_fn = get_collision_fn(
            self.robot,
            ik_joints,
            obstacles=[],
            attachments=[],
            self_collision=True,
            disabled_collisions=self.disable_collision_link_pairs,
            custom_limits=self.custom_limits,
        )
        path, cost = plan_cartesian_motion_lg(
            robot=self.robot,
            joints=ik_joints,
            waypoint_poses=[final_pose],
            sample_ik_fn=sample_ik_fn,
            collision_fn=collision_fn,
        )
        if not execute:
            restore_states(self, state_id)
        return path

    def plan_arm_cartesian_motion_bkup(
        self, side, final_pose, init_pose=None, execute=False, use_ik=True
    ):
        assert side in ["left", "right"], "Wrong arm side input."

        if init_pose is None:
            init_pose = getattr(self, f"{side}_tip_pose")
        if use_ik:
            final_q = self.arm_ik(final_pose, side, interpolate=True)
        else:
            state_id = save_states(self)
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
            restore_states(self, state_id)
        # plan a smooth traj from the beginning to the end
        state_id = save_states(self)
        if final_q is None:
            path = None
        else:
            path = self.plan_joint_motion(
                final_q, joint_indices=self.ik_joints[side], execute=execute
            )
        if not execute:
            restore_states(self, state_id)
        return path

    def plan_joint_motion(self, q, joint_indices=None, execute=False):
        if joint_indices is None:
            joint_indices = self.upper_body_indices
        assert len(q) == len(joint_indices), "Wrong joint positions given"
        state_id = save_states(self)
        path = plan_joint_motion(
            self.robot,
            joint_indices,
            q,
            self_collisions=False,  # True,
            disabled_collisions=self.disable_collision_link_pairs,
            custom_limits=self.custom_limits,
            diagnosis=False,
        )
        if not execute:
            restore_states(self, state_id)
        return path
