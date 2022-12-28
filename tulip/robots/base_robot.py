"""A base meta class for robot manipulator."""

from abc import ABC

import numpy as np


class BaseRobot(ABC):
    """Meta class for robot manipulator."""

    @property
    def ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """End effector Cartesian pose.

        Returns:
            End effector pose as (position, quaternion)."""
        raise NotImplementedError("Method is not defined!")

    @property
    def q(self) -> np.ndarray:
        """Joint positions.

        Returns:
            A fixed-length array of current joint positions."""
        raise NotImplementedError("Method is not defined!")

    @property
    def dq(self) -> np.ndarray:
        """Joint velocities.

        Returns:
            A fixed-length array of current joint velocities."""
        raise NotImplementedError("Method is not defined!")

    @property
    def ddq(self) -> np.ndarray:
        """Joint accelerations.

        Returns:
            A fixed-length array of current joint accelerations."""
        raise NotImplementedError("Method is not defined!")

    @property
    def tau(self) -> np.ndarray:
        """Current joint torques.

        Returns:
            A fixed-length array of current joint torques."""
        raise NotImplementedError("Method is not defined!")

    def get_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Retrieve the current end effector pose.

        Returns:
            End effector pose as (position, quaternion)."""
        raise NotImplementedError("Method is not defined!")

    def get_joint_positions(self) -> np.ndarray:
        """Retrieve the current joint positions.

        Returns:
            A fixed-length array of current joint positions."""
        raise NotImplementedError("Method is not defined!")

    def get_joint_velocities(self) -> np.ndarray:
        """Retrieve the current joint velocities.

        Returns:
            A fixed-length array of current joint velocities."""
        raise NotImplementedError("Method is not defined!")

    def get_joint_accelerations(self) -> np.ndarray:
        """Retrieve the current joint accelerations.

        Returns:
            A fixed-length array of current joint accelerations."""
        raise NotImplementedError("Method is not defined!")

    def get_joint_torques(self) -> np.ndarray:
        """Retrieve the current joint torques.

        Returns:
            A fixed-length array of current joint torques."""
        raise NotImplementedError("Method is not defined!")

    def get_joint_limits(self) -> tuple[tuple[float, float]]:
        """Retrieve the joint limits as per the URDF.

        Returns:
            A tuple of joint limit tuples as (lower_limit, upper_limit)."""
        raise NotImplementedError("Method is not defined!")

    def fk(self, q: np.ndarray) -> tuple[np.ndarray]:
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

    def set_joint_positions(self, q: np.ndarray) -> None:
        """Control joints to the target positions

        Args:
            q: target joint positions."""
        raise NotImplementedError("Method is not defined!")

    def set_delta_joint_positions(self, d_q: np.ndarray) -> None:
        """Control joints to increment by the target delta positions

        Args:
            d_q: target delta joint positions."""
        raise NotImplementedError("Method is not defined!")

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
