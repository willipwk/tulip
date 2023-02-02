"""A collection of utility functions for PyBullet simulation."""
import time
from typing import List, Tuple

import numpy as np
import pybullet as p
from tulip.utils.gl_utils import read_vertices, zbuffer_to_depth
from tulip.utils.transform_utils import (
    homogeneous_transform,
    pos_quat2pose_matrix,
)

PBLT_TIMESTEP = 1 / 240.0


def init_sim(**kwargs) -> int:
    """Initialise a PyBullet simulation client.

    Args:
        kwargs: PyBullet simulation setting in flexible kwargs input.
    Returns:
        physics client id"""
    mode = kwargs.get("mode", "GUI").upper()
    assert mode in [
        "GUI",
        "DIRECT",
        "SHARED_MEMORY",
        "UDP",
        "TCP",
    ], "Invalid input simulation mode!"

    if mode == "GUI":
        sim_cid = p.connect(p.GUI)
    elif mode == "DIRECT":
        sim_cid = p.connect(p.DIRECT)
    elif mode == "SHARED_MEMORY":
        # todo(zyuwei) remove default port num
        port_num = kwargs.get("port_num", 1234)
        sim_cid = p.connect(p.SHARED_MOMORY, port_num)
    else:
        host_ip = kwargs.get("host_ip", "localhost")
        port_num = kwargs.get("port_num", 1234)
        if mode == "UDP":
            sim_cid = p.connect(p.UDP, host_ip, port_num)
        else:
            sim_cid = p.connect(p.TCP, host_ip, port_num)
    assert sim_cid >= 0, "Failed to create a pybullet client"

    timestep = kwargs.get("timestep", PBLT_TIMESTEP)
    p.setTimeStep(timestep)

    gravity_coef = kwargs.get("gravity_coef", -9.8)
    p.setGravity(0, 0, gravity_coef)

    return sim_cid


def use_pybullet_data() -> None:
    """Extend data searching path to use pybullet_data."""
    import pybullet_data

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    return


def step_sim(
    num_steps: int = 1, mode: str = "GUI", timestep: float = PBLT_TIMESTEP
) -> None:
    """Running step simulation.

    Args:
        num_steps: number of steps to simulate
        mode: simulation connection mode
        timestep: simulation step time
    """
    for _ in range(num_steps):
        p.stepSimulation()
        if mode.upper == "GUI":
            time.sleep(PBLT_TIMESTEP)
    return


def enable_torque_sensor(
    robot_id: int,
    joint_indices: List[int],
    sim_cid: int,
) -> None:
    """Enable force/torque sensor on target joint(s).

    Args:
        robot_id: robot PyBullet uniqueBodyId.
        joint_indices: target joint indexes to enable the sensor.
        sim_cid: PyBullet physicsClientId"""
    for joint_idx in joint_indices:
        p.enableJointForceTorqueSensor(
            bodyUniqueId=robot_id,
            jointIndex=joint_idx,
            enableSensor=True,
            physicsClientId=sim_cid,
        )
    return


def disable_torque_sensor(
    robot_id: int,
    joint_indices: List[int],
    sim_cid: int,
) -> None:
    """Disable force/torque sensor on target joint(s).

    Args:
        robot_id: robot PyBullet uniqueBodyId.
        joint_indices: target joint indexes to disable the sensor.
        sim_cid: PyBullet physicsClientId"""
    for joint_idx in joint_indices:
        p.enableJointForceTorqueSensor(
            bodyUniqueId=robot_id,
            jointIndex=joint_idx,
            enableSensor=False,
            physicsClientId=sim_cid,
        )
    return


def vis_frame(
    pos: np.ndarray,
    quat: np.ndarray,
    sim_cid: int,
    length: float = 0.5,
    duration: float = 15,
) -> None:
    """Visualize target pose frame.

    Args:
        pos: frame position.
        quat: frame orientation.
        length: visualization axis length.
        duration: visualization duration.
        sim_cid: PyBullet physicsClientId."""
    x_axis_end_pos, _ = homogeneous_transform(
        pos, quat, [length, 0, 0], [0, 0, 0, 1]
    )
    y_axis_end_pos, _ = homogeneous_transform(
        pos, quat, [0, length, 0], [0, 0, 0, 1]
    )
    z_axis_end_pos, _ = homogeneous_transform(
        pos, quat, [0, 0, length], [0, 0, 0, 1]
    )
    p.addUserDebugLine(
        pos,
        x_axis_end_pos,
        lineColorRGB=[1, 0, 0],
        lifeTime=duration,
        lineWidth=length,
    )
    p.addUserDebugLine(
        pos,
        y_axis_end_pos,
        lineColorRGB=[0, 1, 0],
        lifeTime=duration,
        lineWidth=length,
    )
    p.addUserDebugLine(
        pos,
        z_axis_end_pos,
        lineColorRGB=[0, 0, 1],
        lifeTime=duration,
        lineWidth=length,
    )
    return


def vis_points(
    points: np.ndarray,
    sim_cid: int,
    sample_size: int = 5000,
    duration: float = 1,
    color: list = [1, 0, 0],
) -> None:
    """Visualize a list/array of points xyz in space.

    Args:
        points: input list of points to visualize.
        sim_cid: PyBullet physics simulation id.
        sample_size: number of points to downsample to if exceeds the number.
        duration: visualization time in second.
        color: rgb color in range [0, 1].
    """
    sample_size = min(sample_size, len(points))
    scale = int(len(points) / sample_size)
    vis_points = [points[i * scale] for i in range(sample_size)]
    colors = [color for i in range(sample_size)]
    p.addUserDebugPoints(
        vis_points,
        pointColorsRGB=colors,
        lifeTime=duration,
        physicsClientId=sim_cid,
    )


# def build_lookup_matrix_pblt(
#     camera_pos: np.ndarray,
#     camera_quat: np.ndarray,
#     lookat_axis: str,
#     up_axis: str,
#     sim_cid: int,
# ) -> np.ndarray:
#     assert lookat_axis in ["x", "y", "z", "-x", "-y", "-z"], "Wrong lookat
#     axis"
#     assert up_axis in ["x", "y", "z", "-x", "-y", "-z"], "Wrong up axis"
#     axis_offset = (
#         {
#             "x": [1, 0, 0],
#             "y": [0, 1, 0],
#             "z": [0, 0, 1],
#             "-x": [-1, 0, 0],
#             "-y": [0, -1, 0],
#             "-z": [0, 0, -1],
#         },
#     )
#     target_pos = homogeneous_transform(
#         camera_pos, camera_quat, axis_offset[lookat_axis], [0, 0, 0, 1]
#     )
#     up_pos = homogeneous_transform(
#         camera_pos, camera_quat, axis_offset[up_axis], [0, 0, 0, 1]
#     )
#     up_vec = up_pos - camera_pos
#     view_matrix = p.computeViewMatrix(camera_pos, target_pos, up_vec, sim_cid)
#     view_matrix = np.array(view_matrix).reshape(4, 4)
#     view_matrix[:, 0] *= -1  # TODO(zyuwei) to investigate the decomposition
#     return view_matrix.flatten()


def build_view_matrix_pblt(
    camera_pos: np.ndarray,
    camera_quat: np.ndarray,
    sim_cid: int,
    vis: bool = False,
) -> np.ndarray:
    """Call PyBullet corresponding function to calculate view matrix.

    Args:
        camera_pos: camera position.
        camera_quat: camera orientation.
        sim_cid: PyBullet physicsClientId.
        vis: to visualize the lookat_position and up_vec.
    Returns:
        flattened view_matrix
    """
    camera_pose = pos_quat2pose_matrix(camera_pos, camera_quat)
    camera_extrinsic = np.linalg.inv(camera_pose)
    camera_rot = camera_extrinsic[:3, :3]
    s_vec = camera_rot[0, :]
    up_prime_vec = camera_rot[1, :]
    lookat_vec = camera_rot[2, :]
    s_vec = np.cross(lookat_vec, up_prime_vec)
    up_vec = np.cross(s_vec, lookat_vec)
    target_pos = np.array(camera_pos) + lookat_vec
    print(target_pos)
    print(up_vec)
    if vis:
        vis_frame(
            target_pos,
            camera_quat,
            sim_cid,
        )
        vis_frame(
            up_vec + np.array(camera_pos),
            camera_quat,
            sim_cid,
        )

    view_matrix = p.computeViewMatrix(camera_pos, target_pos, up_vec, sim_cid)
    view_matrix = np.array(view_matrix).reshape(4, 4)
    view_matrix[:, 0] *= -1  # TODO(zyuwei) to investigate the decomposition
    return view_matrix.flatten()


def render(
    width: int,
    height: int,
    view_matrix: np.ndarray,
    proj_matrix: np.ndarray,
    near: float,
    far: float,
    sim_cid: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render in simulation.

    Args:
        width: width for rendered image
        height: width for rendered height
        proj_matrix: flattened projection matrix
        view_matrix: flattened view matrix
        near: distance to the nearer depth clipping plane.
        far: distance to the farther depth clipping plane.
        sim_cid: PyBullet physicsClientId.
    Returns:
        rendered rgb, depth and segmentation images."""
    width, height, rgb, z_buffer, seg = p.getCameraImage(
        width,
        height,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_TINY_RENDERER,
        physicsClientId=sim_cid,
    )
    # rgb postprocessing
    rgb = rgb.reshape([height, width, 4])[:, :, :3]
    rgb = rgb.astype(np.uint8)
    # depth postprocessing
    z_buffer = z_buffer.reshape([height, width])
    depth = zbuffer_to_depth(z_buffer, near, far)
    # segmentation indices postprocessing
    seg = seg.reshape([height, width])
    return rgb, depth, seg


# TODO(zyuwei) [BUG] it does not handle with baselink offset currently
def get_vertices_pos(
    obj_id: int,
    sim_cid: int,
    v_local_pos: np.ndarray = None,
    scale: list = np.array([1, 1, 1]),
) -> list:
    """Query the current vertices position.

    Args:
        obj_id: PyBullet object id.
        sim_cid: PyBullet Physics client id.
        v_local_pos: vertices local position in the objec coordinate.
        scale: object load scale in x, y, z
    Returns:
        A list of vertices position in world coordinate.
    """
    print(
        "[Warning]: The vertices query dos not handle and will be inaccurate",
        "when baselink has offset to the origin point in the urdf.",
    )
    obj_pos, obj_quat = p.getBasePositionAndOrientation(obj_id, sim_cid)
    if v_local_pos is None:
        obj_vs_data = p.getVisualShapeData(obj_id, sim_cid)
        assert len(obj_vs_data) == 1, "Does not support muti-body urdf yet"
        obj_fn = obj_vs_data[0][4].decode("utf-8")
        scale = np.array(obj_vs_data[0][3])
        v_local_pos = read_vertices(obj_fn)
    v_pos = []
    for pos in v_local_pos:
        v_pos.append(
            homogeneous_transform(obj_pos, obj_quat, pos * scale, [0, 0, 0, 1])[
                0
            ]
        )
    return v_pos
