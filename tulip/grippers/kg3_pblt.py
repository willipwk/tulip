"""Franka Panda interface for PyBullet."""
import logging
import math
import os
from typing import Tuple

import numpy as np
import pybullet as p
from tulip.robots.base_robot import BaseRobot
from tulip.utils.pblt_utils import (
    enable_torque_sensor,
    restore_states,
    save_states,
    step_sim,
)


class KG3:
    """Kinova Gripper Gen3 interface ."""

    def __init__(
        self,
        sim_cid: int,
        urdf_file: str,
        home_pos: np.ndarray = np.array([0.0, 0.0, 0.0]),
        base_pos: np.ndarray = np.array([0.0, 0.0, 0.0]),
        base_orn: np.ndarray = np.array([0.0, 0.0, 0.0, 1.0]),
        ee_idx: int = 2,  # ee_link: 0,  tip_link: 2
    ):
        super().__init__()

        assert os.path.isfile(urdf_file), f"Urdf file {urdf_file} not found!"
        self._urdf_file = urdf_file
        self._ee_idx = ee_idx
        self._base_pos = base_pos
        self._base_orn = base_orn
        self._sim_cid = sim_cid

        self._pid = p.loadURDF(
            fileName=self._urdf_file,
            useFixedBase=False,
            physicsClientId=self._sim_cid,
        )

        self._constraint = p.createConstraint(
            parentBodyUniqueId=self._pid,
            parentLinkIndex=-1,
            childBodyUniqueId=-1,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0.0, 0.0, 0.0],
            childFrameOrientation=[0.0, 0.0, 0.0],
            physicsClientId=self._sim_cid,
        )
        p.changeDynamics(
            bodyUniqueId=self._pid,
            linkIndex=-1,
            linearDamping=10.0,
            angularDamping=10.0,
            physicsClientId=self._sim_cid,
        )

        self._num_joints = p.getNumJoints(
            bodyUniqueId=self.id, physicsClientId=self._sim_cid
        )
        self.link_names = ()
        self.joint_names = ()
        self.joint_limits = ()
        self.joint_force_limits = ()  # not in used
        self.joint_velocity_limits = ()  # not in used
        self.controllable_joint_indices = ()
        for joint_idx in range(self._num_joints):
            joint_info = p.getJointInfo(
                bodyUniqueId=self.id,
                jointIndex=joint_idx,
                physicsClientId=self._sim_cid,
            )
            self.link_names += (joint_info[12],)
            self.joint_names += (joint_info[1],)
            if joint_info[2] in [0, 1]:  # JOINT_REVOLUTE=0, JOINT_PRISMATIC=1
                self.joint_limits += ((joint_info[8], joint_info[9]),)
                self.joint_force_limits += (joint_info[10],)
                self.joint_velocity_limits += (joint_info[11],)
                self.controllable_joint_indices += (joint_idx,)
        self.state = {}

        enable_torque_sensor(self.id, self.controllable_joint_indices, sim_cid)

        self._closed_pos = np.array(
            [0.8907163779515354, 0.8907163779515354, 0.8907163779515354]
        )
        self._open_pos = np.array([0.0, 0.0, 0.0])
        self._home_pos = home_pos

        # logging.info("Setting gripper to home position:", self._home_pos)
        # self.control_base_pose(self._base_pos, self._base_orn)
        # self.set_joint_positions(self._home_pos)
        # step_sim(50)

    @property
    def id(self) -> int:
        return self._pid

    @property
    def num_joints(self) -> int:
        return self._num_joints

    @property
    def finger_links(self) -> list:
        return [4, 5, 7, 8, 10, 11]

    @property
    def tip_links(self) -> list:
        return [6, 9, 12]

    @property
    def base_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Base Cartesian pose.

        Returns:
            Base pose as (position, quaternion)."""
        return self.get_base_pose()

    @property
    def tip_pose(self) -> tuple:
        assert len(self.tip_links) == 3, "Wrong number of finger tips."
        tip_poses = ()
        for link_idx in self.tip_links:
            tip_pos, tip_quat = self.get_link_pose(link_idx)
            tip_poses += ((tip_pos, tip_quat),)
        return tip_poses

    @property
    def ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """End effector Cartesian pose.

        Returns:
            End effector pose as (position, quaternion)."""
        return self.get_ee_pose()

    @property
    def q(self) -> np.ndarray:
        """Joint positions.

        Returns:
            A fixed-length array of current joint positions."""
        return self.get_joint_positions()

    @property
    def dq(self) -> np.ndarray:
        """Joint velocities.

        Returns:
            A fixed-length array of current joint velocities."""
        return self.get_joint_velocities()

    @property
    def ddq(self) -> np.ndarray:
        """Joint accelerations.

        Returns:
            A fixed-length array of current joint accelerations."""
        return self.get_joint_accelerations

    @property
    def tau(self) -> np.ndarray:
        """Current joint torques.

        Returns:
            A fixed-length array of current joint torques."""
        return self.get_joint_torques()

    def get_ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieve the current end effector pose.

        Returns:
            End effector pose as (position, quaternion)."""
        return self.get_link_pose(self._ee_idx)

    def get_joint_positions(self, joint_indices: list = None) -> np.ndarray:
        """Retrieve the current joint positions.

        Args:
            joint_indices: joint_indices to query
        Returns:
            A fixed-length array of current joint positions."""
        if joint_indices is None:
            joint_indices = self.controllable_joint_indices
        joint_states = p.getJointStates(
            bodyUniqueId=self.id,
            jointIndices=joint_indices,
            physicsClientId=self._sim_cid,
        )
        joint_pos = np.array([js[0] for js in joint_states])
        return joint_pos

    def get_joint_velocities(self, joint_indices: list = None) -> np.ndarray:
        """Retrieve the current joint velocities.

        Args:
            joint_indices: joint_indices to query
        Returns:
            A fixed-length array of current joint velocities."""
        if joint_indices is None:
            joint_indices = self.controllable_joint_indices
        joint_states = p.getJointStates(
            bodyUniqueId=self.id,
            jointIndices=joint_indices,
            physicsClientId=self._sim_cid,
        )
        joint_vel = np.array([js[1] for js in joint_states])
        return joint_vel

    def get_joint_accelerations(self, joint_indices: list = None) -> np.ndarray:
        """Retrieve the current joint accelerations.

        Args:
            joint_indices: joint_indices to query
        Returns:
            A fixed-length array of current joint accelerations."""
        raise NotImplementedError(
            "PyBullet doesn't support direct acceleration query. You may \
            consider to calculate via (v_(t) - v_(t-1)) / dt."
        )

    # todo(zyuwei) parsing the dof for force/torque values
    def get_joint_torques(self, joint_indices: list = None) -> np.ndarray:
        """Retrieve the current joint torques.

        Args:
            joint_indices: joint_indices to query
        Returns:
            A fixed-length array of current joint torques."""
        if joint_indices is None:
            joint_indices = self.controllable_joint_indices
        joint_states = p.getJointStates(
            bodyUniqueId=self.id,
            jointIndices=joint_indices,
            physicsClientId=self._sim_cid,
        )
        joint_torques = np.array([js[2] for js in joint_states])
        return joint_torques

    def get_link_pose(self, link_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        link_info = p.getLinkState(
            self._pid,
            link_idx,
            # computeForwardKinematics=True,
            physicsClientId=self._sim_cid,
        )
        pos = np.array(link_info[0])
        quat = np.array(link_info[1])
        return pos, quat

    def get_base_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        pos, quat = p.getBasePositionAndOrientation(
            self._pid,
            physicsClientId=self._sim_cid,
        )
        return np.array(pos), np.array(quat)

    def get_joint_limits(self) -> Tuple[Tuple[float, float]]:
        """Retrieve the joint limits as per the URDF.

        Returns:
            A tuple of joint limit tuples as (lower_limit, upper_limit)."""
        return self.joint_limits

    def fk(self, q: np.ndarray) -> float:
        """Forward kinematics given input joint positions.

        Args:
            q: input joint positions.
        Returns:
            float: open distance between two fingers in meter
        """
        l1 = 30.9476 - 87.0932 * math.sin(q[0] - 0.627445866)
        l2 = 30.9476 - 87.0932 * math.sin(q[1] - 0.627445866)
        dist = l1 + l2
        if dist < (2 * 30.9476):
            dist -= 17.0
        else:
            dist += 1.08

        return dist * 0.001

    def ik(self, dist: float) -> np.ndarray:
        """Inverse kinematics given target open distance.

        Args:
            dist: open distance between two fingers in meter
        Returns:
            joint angles for three fingers in radian."""
        dist_mm = dist * 1000.0
        tmp = (0.5 * dist_mm - 30.9476) / -87.0932
        q = math.asin(tmp) + 0.627445866
        if 0.5 * dist > 30.9476:
            q += 0.00599
        else:
            q -= 0.1

        return np.array([q, q, q])

    # todo(zyuwei) to check if how gains are used in different control modes
    def set_joint_positions(
        self,
        q: np.ndarray,
        position_gain: float = 0.03,
        velocity_gain: float = 1.0,
    ) -> None:
        """Control joints to the target positions

        Args:
            q: target joint positions.
            position_gain: position gain for controller
            velocity_gain: velocity_gain for controller"""
        assert q.size == len(
            self.controllable_joint_indices
        ), "Input size does not match with the number of controllable joints"
        p.setJointMotorControlArray(
            bodyUniqueId=self._pid,
            jointIndices=self.controllable_joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=q,
            forces=len(self.controllable_joint_indices) * [2.0],
            physicsClientId=self._sim_cid,
        )
        return

    def set_delta_joint_positions(self, d_q: np.ndarray) -> None:
        """Control joints to increment by the target delta positions

        Args:
            d_q: target delta joint positions."""
        curr_q = self.q
        tgt_q = curr_q + d_q
        self.set_joint_positions(tgt_q)
        return

    def set_ee_pose(self, pos, orn) -> None:
        """Control end effector to the target pose.

        Args:
            pos: target Cartesian position
            orn: target Cartesian orientation in quaternion"""
        raise NotImplementedError("Method is not defined!")

    def set_delta_ee_pose(self, d_pos, d_orn) -> None:
        """Control end effector to increment by the target pose.

        Args:
            d_pos: target delta Cartesian position.
            d_orn: target delta Cartesian orientation in quaternion."""
        raise NotImplementedError("Method is not defined!")

    def reset_to_states(self, q, dq=None, joint_indices=None):
        if joint_indices is None:
            joint_indices = self._controllable_joint_indices
        assert len(q) == len(joint_indices), "Wrong joint positions given"
        if dq is not None:
            assert len(dq) == len(joint_indices), "Wrong joint positions given"
        else:
            dq = [0 for idx in joint_indices]
        for joint_idx, joint_pos, joint_vel in zip(joint_indices, q, dq):
            p.resetJointState(
                self._pid,
                jointIndex=joint_idx,
                targetValue=joint_pos,
                targetVelocity=joint_vel,
                physicsClientId=self._sim_cid,
            )
        step_sim(1)

    def reset_base_pose(
        self, base_pos, base_orn, keep_local_states=False
    ) -> None:
        """Reset gripper base position and orientation.

        Args:
            base_pos: gripper base position.
            base_orn: gripper base orientation in quaternion.
            keep_local_states: restore joint position and velocity states"""
        if keep_local_states:
            state_id = save_states(self)
        p.resetBasePositionAndOrientation(
            self._pid, base_pos, base_orn, physicsClientId=self._sim_cid
        )
        if keep_local_states:
            restore_states(self, state_id)

    def control_base_pose(self, base_pos, base_orn) -> None:
        """Control gripper base to position and orientation.

        Args:
            base_pos: gripper base position.
            base_orn: gripper base orientation in quaternion.
            keep_local_states: restore joint position and velocity states"""
        p.changeConstraint(
            self._constraint,
            jointChildPivot=base_pos,
            jointChildFrameOrientation=base_orn,
            maxForce=1000.0,
            physicsClientId=self._sim_cid,
        )

    def close(self) -> None:
        """Fully close the gripper."""
        return self.control_gripper(mode="close")

    def open(self) -> None:
        """Fully open the gripper."""
        return self.control_gripper(mode="open")

    def control_gripper(self, mode: str) -> None:
        """Control gripper open or close.

        Args:
            mode: choices in ['open', 'close']
        """
        assert mode in ["open", "close"], "Wrong mode for gripper control."
        if mode == "open":
            self.set_joint_positions(self._open_pos)
        else:
            self.set_joint_positions(self._closed_pos)
