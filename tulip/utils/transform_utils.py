from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R


def pose2trans(pose: np.ndarray) -> np.ndarray:
    """Construct pose matrix from position and quaternion.

    Args:
        pose: 4x4 pose matrix.
    Returns:
        4x4 transformation matrix.
    """
    return np.linalg.inv(pose)


def trans2pose(trans: np.ndarray) -> np.ndarray:
    """Construct pose matrix from position and quaternion.

    Args:
        trans: 4x4 transformation matrix.
    Returns:
        4x4 pose matrix.
    """
    return np.linalg.inv(trans)


def pose_matrix2pos_quat(pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Construct pose matrix from position and quaternion.

    Args:
        4x4 pose matrix.
    Returns:
        pos: frame position.
        quat: frame quaternion.
    """
    pos = pose[:3, 3]
    quat = R.from_matrix(pose[:3, :3]).as_quat()
    return pos, quat


def pos_quat2pose_matrix(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """Construct pose matrix from position and quaternion.

    Args:
        pos: frame position.
        quat: frame quaternion.
    Returns:
        4x4 pose matrix.
    """
    pose = np.eye(4)
    pose[:3, 3] = pos
    pose[:3, :3] = R.from_quat(quat).as_matrix()
    return pose


def pos_quat2trans_matrix(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """Construct transformation matrix from position and quaternion.

    Args:
        pos: frame position.
        quat: frame quaternion.
    Returns:
        4x4 transformation matrix.
    """
    pose = pos_quat2pose_matrix(pos, quat)
    return pose2trans(pose)


def homogeneous_transform(
    c10_pos: np.ndarray,
    c10_quat: np.ndarray,
    c21_pos: np.ndarray,
    c21_quat: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Homogeneous tranformation.

    Args:
        c10_pos: frame 1 position wrt c0 coordinate.
        c10_quat: frame 1 quaternion wrt c0 coordinate.
        c21_pos: frame 2 position wrt c1 coordinate.
        c21_quat: frame 2 quaternion wrt c1 coordinate.
    Returns:
        frame2 position, quaternion wrt c0.
    """
    # pose of the frame to visualize
    t_10 = pos_quat2trans_matrix(c10_pos, c10_quat)
    t_21 = pos_quat2trans_matrix(c21_pos, c21_quat)
    t_20 = np.matmul(t_21, t_10)
    p_20 = trans2pose(t_20)
    return pose_matrix2pos_quat(p_20)


def relative_pose(
    c10_pos: np.ndarray,
    c10_quat: np.ndarray,
    c20_pos: np.ndarray,
    c20_quat: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Get relative pose of p2 wrt p1.

    Args:
        c10_pos: frame 1 position wrt c0 coordinate.
        c10_quat: frame 1 quaternion wrt c0 coordinate.
        c20_pos: frame 2 position wrt c0 coordinate.
        c20_quat: frame 2 quaternion wrt c0 coordinate.
    Returns:
        frame2 position, quaternion wrt c1.
    """
    # pose of the frame to visualize
    t_01 = pos_quat2pose_matrix(c10_pos, c10_quat)
    t_20 = pos_quat2trans_matrix(c20_pos, c20_quat)
    t_21 = np.matmul(t_20, t_01)
    p_21 = trans2pose(t_21)
    return pose_matrix2pos_quat(p_21)
