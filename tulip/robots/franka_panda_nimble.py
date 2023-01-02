"""A base meta class for robot manipulator."""

import logging
import os
from typing import Tuple

import nimblephysics as n
import numpy as np
from tulip.robots.base_robot import BaseRobot
from tulip.utils.constants import FRANKA_URDF_FILE


class FrankaPanda(BaseRobot):
    """Franka Panda PyBullet robot interface ."""

    def __init__(
        self,
        world: n.simulation.World,
        urdf_file: str = FRANKA_URDF_FILE,
        home_pos: np.ndarray = np.array(
            [0, -0.785, 0, -2.356, 0, 1.57, 0.785, 0.04, 0.04]
        ),
        base_pos: np.ndarray = np.array([0.0, 0.0, 0.0]),
        base_orn: np.ndarray = np.array([0.0, 0.0, 0.0, 1.0]),
        # ee_idx: int = -1,
    ):
        super().__init__()

        assert os.path.isfile(urdf_file), f"Urdf file {urdf_file} not found!"
        self._urdf_file = urdf_file
        self._home_pos = home_pos
        self._base_pos = base_pos
        self._base_orn = base_orn
        self.world = world
        self.robot = self.world.loadSkeleton(os.path.abspath(self._urdf_file))

        self._num_joints = self.robot.getNumDofs()
        self._joint_handlers = self.robot.getDofs()
        # self.link_names = ()
        self.joint_names = ()
        self.joint_limits = ()
        self.joint_force_limits = ()  # not in used
        self.joint_velocity_limits = ()  # not in used
        self.controllable_joint_indices = ()
        self._joint_handles = self.robot.getDofs()
        for joint_idx, handler in enumerate(self._joint_handlers):
            # self.link_names += (joint_info[12],)
            self.joint_names += (handler.getName(),)
            self.joint_limits += (handler.getPositionLimits(),)
            self.joint_force_limits += (handler.getControlForceLimits(),)
            self.joint_velocity_limits += (handler.getVelocityLimits(),)
            self.controllable_joint_indices += (handler.getIndexInSkeleton(),)

        logging.info("Setting robot to home position:", self._home_pos)
        self.set_joint_positions(self._home_pos)

    @property
    def num_joints(self) -> int:
        return self._num_joints

    @property
    def ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """End effector Cartesian pose.

        Returns:
            End effector pose as (position, quaternion)."""
        raise NotImplementedError("Method is not defined!")

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
        raise NotImplementedError("Method is not defined!")

    def get_joint_positions(self) -> np.ndarray:
        """Retrieve the current joint positions.

        Returns:
            A fixed-length array of current joint positions."""
        joint_pos = [handler.getPosition() for handler in self._joint_handlers]
        return np.array(joint_pos)

    def get_joint_velocities(self) -> np.ndarray:
        """Retrieve the current joint velocities.

        Returns:
            A fixed-length array of current joint velocities."""
        joint_vel = [handler.getVelocity() for handler in self._joint_handlers]
        return np.array(joint_vel)

    def get_joint_accelerations(self) -> np.ndarray:
        """Retrieve the current joint accelerations.

        Returns:
            A fixed-length array of current joint accelerations."""
        joint_acc = [
            handler.getAcceleration() for handler in self._joint_handlers
        ]
        return np.array(joint_acc)

    # todo(zyuwei) parsing the dof for force/torque values
    def get_joint_torques(self) -> np.ndarray:
        """Retrieve the current joint torques.

        Returns:
            A fixed-length array of current joint torques."""
        raise NotImplementedError("Method is not defined!")

    def get_joint_limits(self) -> Tuple[Tuple[float, float]]:
        """Retrieve the joint limits as per the URDF.

        Returns:
            A tuple of joint limit tuples as (lower_limit, upper_limit)."""
        return self.joint_limits

    def fk(self, q: np.ndarray) -> Tuple[np.ndarray]:
        """Forward kinematics given input joint positions.

        Args:
            q: input joint positions.
        Returns:
            end effector pose as (position, quaternion)."""
        raise NotImplementedError("Method is not defined!")

    def ik(
        self,
        pos: np.ndarray,
        orn: np.ndarray,
        q_init: np.ndarray,
        n_samples: int,
        tolerance: float,
    ) -> np.ndarray:
        """Inverse kinematics given target end effector pose.

        Args:
            pos: Cartesian position.
            orn: Cartesian orientation in quaternion.
            q_init: joint configuration seed.
            n_samples: number of samples to sample solution.
            tolerance: minimum position distance error for solution.
        Returns:
            joint configurations that satisfies the target Cartesian pose. None
            will be return if there is no solution satisfies the distance
            tolerance after n_samples being sampled."""
        raise NotImplementedError("Method is not defined!")

    # todo(zyuwei) to check if how gains are used in different control modes
    def set_joint_positions(
        self,
        q: np.ndarray,
    ) -> None:
        """Control joints to the target positions

        Args:
            q: target joint positions.
            position_gain: position gain for controller
            velocity_gain: velocity_gain for controller"""
        assert q.size == len(
            self.controllable_joint_indices
        ), "Input size does not match with the number of controllable joints"
        joint_pos = self.robot.getPositions()
        for q_idx, q in enumerate(q):
            joint_idx = self.controllable_joint_indices[q_idx]
            joint_pos[joint_idx] = q
        self.robot.setPositions(joint_pos)
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
            d_pos: target delta Cartesian position
            d_orn: target delta Cartesian orientation in quaternion"""
        raise NotImplementedError("Method is not defined!")
