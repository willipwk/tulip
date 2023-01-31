from typing import Tuple

import numpy as np
from PIL import Image
from tulip.utils.transform_utils import pos_quat2pose_matrix


# TODO(zyuwei) to convert into camera class to avoid passing params around.
def uvz2xyz(
    u: int,
    v: int,
    z: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    cam_pose: np.ndarray,
) -> np.ndarray:
    """convert from uv depth to xyz.

    Args:
        u: x coordinate index.
        v: y coordinate index.
        z: depth value at the pixel.
        fx: focal length in x axis.
        fy: focal length in k axis.
        cx: principle point at x axis.
        cy: principle point at y axis.
        cam_pose: 4x4 camera pose matrix wrt world coordinate.
    Return:
        xyz numpy array for pixel point.
    """
    pos_vec = np.array([(u - cx) * z / fx, (v - cy) * z / fy, z, 1])
    xyz = np.matmul(cam_pose, pos_vec.transpose())[:3]
    return xyz


def xyz2uvz(
    x: float,
    y: float,
    z: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    cam_extrinsic: np.ndarray,
) -> Tuple[int, int, float]:
    """convert from xyz to uv coordinate index.

    Args:
        x: x position in world coordinate.
        y: y position in world coordinate.
        z: z position in world coordinate.
        fx: focal length in x axis.
        fy: focal length in k axis.
        cx: principle point at x axis.
        cy: principle point at y axis.
        cam_extrinsic: 4x4 extrinsic matrix as the inverse of camera world pose.
    Return:
        u, v, z coordiante
    """
    xyz_vec = np.array([x, y, z, 1])
    uvz_vec = np.matmul(cam_extrinsic, xyz_vec.transpose())[:3]
    z = uvz_vec[2]
    u = int(uvz_vec[0] * fx / z + cx)
    v = int(uvz_vec[1] * fy / z + cy)
    return u, v, z


def depth2xyz(
    depth: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    cam_pos: np.ndarray,
    cam_quat: np.ndarray,
    return_pcd: bool = False,
) -> np.ndarray:
    """Convert depth image to point array or xyz image.

    Args:
        depth: (h, w) depth image array.
        fx: focal length in x axis.
        fy: focal length in k axis.
        cx: principle point at x axis.
        cy: principle point at y axis.
        cam_pos: camera position in world coordinate.
        cam_quat: camera quaternion in world coordinate.
        return_pcd: return as a list of pcd point rather than xyz image.
    Return:
        xyz image or pointcloud list.
    """
    cam_pose = pos_quat2pose_matrix(cam_pos, cam_quat)
    h, w = depth.shape
    pcd = []
    xyz_image = np.zeros((depth.shape[0], depth.shape[1], 3), np.float32)
    for h_i in range(h):
        for w_i in range(w):
            xyz_image[h_i, w_i] = uvz2xyz(
                w_i, h_i, depth[h_i, w_i], fx, fy, cx, fy, cam_pose
            )
            pcd.append(xyz_image[h_i, w_i])
    if return_pcd:
        return np.array(pcd)
    else:
        return xyz_image


def pcd2xyz(
    pcd: np.ndarray,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    cam_pos: np.ndarray,
    cam_quat: np.ndarray,
) -> np.ndarray:
    """Convert a list of point cloud into xyz image.

    Args:
        pcd: list of 3-d (xyz) pointcloud wrt world coordinate.
        width: camera image width.
        height: camera image height.
        fx: focal length in x axis.
        fy: focal length in k axis.
        cx: principle point at x axis.
        cy: principle point at y axis.
        cam_pos: camera position in world coordinate.
        cam_quat: camera quaternion in world coordinate.
    Return:
        (w, h, 3) xyz image
    """
    cam_pose = pos_quat2pose_matrix(cam_pos, cam_quat)
    cam_extrinsic = np.linalg.inv(cam_pose)
    xyz_image = np.zeros((height, width, 3), np.float32)
    for point in pcd:
        w_i, h_i, z = xyz2uvz(
            point[0], point[1], point[2], fx, fy, cx, cy, cam_extrinsic
        )
        xyz_image[h_i, w_i] = np.array(
            [point[0], point[1], point[2]], np.float32
        )
    return xyz_image


def vis_rgb(rgb: np.ndarray, max: int) -> None:
    """Visualize a rgb array.

    Args:
        rgb: (h, w, 3) rgb image array
        max: max value in range. 1 for [0, 1] or 255 for [0, 255] array.
    """
    vis = rgb.copy()
    vis = (vis / float(max)) * 255.0
    vis = vis.astype(np.uint8)
    vis_image = Image.fromarray(vis)
    vis_image.show()


def vis_depth(depth: np.ndarray, near: float = None, far: float = None) -> None:
    """Visualize a depth array. Value of each is the true depth value.

    Args:
        depth: (h, w) depth image array
        max: max value in range. 1 for [0, 1] or 255 for [0, 255] array.
    """
    if near is None:
        near = depth.min()
    if far is None:
        far = depth.max()
    assert far > near, "Invalid near and far input to visualize depth!"
    vis = depth.copy()
    vis = (vis - near) / (far - near)
    vis *= 255.0
    vis = vis.astype(np.uint8)
    vis_image = Image.fromarray(vis)
    vis_image.show()


def vis_seg_indices(seg: np.ndarray) -> None:
    """Visualize a segmentation indices array. Value of each is the integer
    segmentation index.

    Args:
        seg: (h, w) segmentation indices array
    """
    min_idx = seg.min()
    max_idx = seg.max()
    if max_idx == min_idx:
        vis = np.zeros(seg.shape)
    else:
        vis = seg.copy()
        vis = (vis - min_idx) / (max_idx - min_idx)
    vis *= 255.0
    vis = vis.astype(np.uint8)
    vis_image = Image.fromarray(vis)
    vis_image.show()
