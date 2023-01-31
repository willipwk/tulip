import time

import numpy as np
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation as R
from tulip.utils.gl_utils import build_projection_matrix, build_view_matrix
from tulip.utils.image_utils import vis_depth, vis_rgb, vis_seg_indices
from tulip.utils.pblt_utils import (
    build_view_matrix_pblt,
    init_sim,
    render,
    vis_frame,
    vis_points,
)
from tulip.utils.transform_utils import pos_quat2pose_matrix


def depth2xyz(depth, fx, fy, cx, cy, cam_pos, cam_quat):
    print("==> Converting depth to xyz")
    cam_pose = pos_quat2pose_matrix(cam_pos, cam_quat)
    h, w = depth.shape
    pcd = []
    xyz_image = np.zeros((depth.shape[0], depth.shape[1], 3), np.float32)
    for h_i in range(h):
        for w_i in range(w):
            z = depth[h_i, w_i]
            pos_vec = np.array([(w_i - cx) * z / fx, (h_i - cy) * z / fy, z, 1])
            xyz_image[h_i, w_i] = np.matmul(cam_pose, pos_vec.transpose())[:3]
            pcd.append(xyz_image[h_i, w_i])

    return xyz_image, pcd


def pcd2xyz(pcd, width, height, fx, fy, cx, cy, cam_pos, cam_quat):
    print("==> Converting pcd to xyz image")
    cam_pose = pos_quat2pose_matrix(cam_pos, cam_quat)
    cam_extrinsic = np.linalg.inv(cam_pose)
    xyz_image = np.zeros((height, width, 3), np.float32)
    for point in pcd:
        pos_vec = np.array([point[0], point[1], point[2], 1])
        uvz_vec = np.matmul(cam_extrinsic, pos_vec.transpose())[:3]
        z = uvz_vec[2]
        w_i = int(uvz_vec[0] * fx / z + cx)
        h_i = int(uvz_vec[1] * fy / z + cy)
        xyz_image[h_i, w_i] = pos_vec[:3]
    return xyz_image


if __name__ == "__main__":
    # initialize simulation
    mode = "GUI"
    sim_cid = init_sim(mode=mode)
    p.loadURDF(
        f"{pybullet_data.getDataPath()}/plane.urdf", physicsClientId=sim_cid
    )
    """
    p.loadURDF(
        "tie.urdf",
        basePosition=[0, 0, 0.05],
        baseOrientation=p.getQuaternionFromEuler([np.pi / 2, 0, 0]),
        useFixedBase=True,
        physicsClientId=sim_cid,
    )
    """
    p.loadURDF(
        f"{pybullet_data.getDataPath()}/duck_vhacd.urdf",
        basePosition=[0, 0, 0.1],
        baseOrientation=p.getQuaternionFromEuler([np.pi / 2, 0, 0]),
        useFixedBase=True,
        globalScaling=3,
        physicsClientId=sim_cid,
    )

    # extrinsic related
    camera_pos = [-0.58451435, -0.09919992, 0.61609813]
    camera_quat = [
        -0.6733081448310533,
        0.6659691939501913,
        -0.22584407782434218,
        0.22833227394560413,
    ]
    vis_frame(camera_pos, camera_quat, sim_cid, length=0.2, duration=15)
    gl_view_matrix = build_view_matrix(camera_pos, camera_quat)
    pblt_view_matrix = build_view_matrix_pblt(
        camera_pos, camera_quat, sim_cid, vis=True
    )
    print(np.array(gl_view_matrix).reshape(4, 4))
    print(np.array(pblt_view_matrix).reshape(4, 4))

    # intrinsic related
    width = 1920
    height = 1080
    fx = 1074.9383544900666
    fy = 1078.6895323593005
    cx = 954.0125249569526
    cy = 542.8760188199577
    # two render parameters that does not exist in a real camera
    far = 10
    near = 0.01
    proj_matrix = build_projection_matrix(
        width, height, fx, fy, cx, cy, near, far
    )

    # render and visualize
    # for view_matrix in [gl_view_matrix, pblt_view_matrix]:
    for view_matrix in [pblt_view_matrix]:
        rgb, depth, seg = render(
            width, height, view_matrix, proj_matrix, near, far, sim_cid
        )
        vis_rgb(rgb, 255)
        vis_depth(depth)
        vis_seg_indices(seg)

        xyz_image, pcd = depth2xyz(
            depth, fx, fy, cx, cy, camera_pos, camera_quat
        )
        vis_points(pcd, sim_cid)

        xyz_image = pcd2xyz(
            pcd, width, height, fx, fy, cx, cy, camera_pos, camera_quat
        )
        vis_depth(xyz_image[:, :, 2])
        time.sleep(5)
        input("enter to continue")
