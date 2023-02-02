import numpy as np
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation as R
from tulip.utils.gl_utils import build_projection_matrix, build_view_matrix
from tulip.utils.image_utils import (
    depth2xyz,
    pcd2xyz,
    vis_depth,
    vis_rgb,
    vis_seg_indices,
)
from tulip.utils.pblt_utils import (
    build_view_matrix_pblt,
    init_sim,
    render,
    vis_frame,
    vis_points,
)
from tulip.utils.transform_utils import (
    pos_quat2pose_matrix,
    pos_quat2trans_matrix,
    pose_matrix2pos_quat,
)


def perturb_camera_pose(
    camera_pos, camera_quat, d_xyz_limit=0.1, d_rpy_limit=0.2
):
    d_camera_pos = np.random.uniform(-d_xyz_limit, d_xyz_limit, 3)
    new_cam_pos = np.array(camera_pos) + d_camera_pos
    d_camera_rpy = np.random.uniform(-d_rpy_limit, d_rpy_limit, 3)
    camera_rpy = R.from_quat(camera_quat).as_euler("xyz") + d_camera_rpy
    new_cam_quat = R.from_euler("xyz", camera_rpy).as_quat()
    return new_cam_pos, new_cam_quat


def perturb_camera_pose_around_target(
    camera_pos,
    camera_quat,
    target_pos=None,
    target_quat=None,
    d_xyz_limit=0.1,
    d_rpy_limit=0.2,
):
    # retrieve target pose if not given
    if target_pos is None:
        camera_extrinsic = np.linalg.inv(
            pos_quat2pose_matrix(camera_pos, camera_quat)
        )
        lookat_vec = camera_extrinsic[2, :3]
        target_pos = np.array(camera_pos) + lookat_vec
        target_quat = camera_quat.copy()
    if target_quat is None:
        target_quat = camera_quat.copy()
    # calculate t_camera_to_target
    t_camera_to_world = pos_quat2trans_matrix(camera_pos, camera_quat)
    target_pose = pos_quat2pose_matrix(target_pos, target_quat)
    t_camera_to_target = np.matmul(target_pose, t_camera_to_world)
    # perturb target coordinate orientation which changes the sphere
    # orientation, therefore the pose of viewpoint which falls on the surface of
    # the sphere
    perturbed_target_rpy = R.from_quat(target_quat).as_euler("xyz")
    perturbed_target_rpy += np.random.uniform(-d_rpy_limit, d_rpy_limit, 3)
    perturbed_target_quat = R.from_euler("xyz", perturbed_target_rpy).as_quat()
    t_target_to_world = pos_quat2trans_matrix(target_pos, perturbed_target_quat)
    # perturbe distance to camera which changes the sphere radius
    t_perturbed_camera_to_camera = np.eye(4)
    t_perturbed_camera_to_camera[2, 2] += np.random.uniform(
        -d_xyz_limit, d_xyz_limit
    )
    # chained transformation
    t_perturbed_camera = np.matmul(
        np.matmul(t_target_to_world, t_camera_to_target),
        t_perturbed_camera_to_camera,
    )
    perturbed_camera_pose = np.linalg.inv(t_perturbed_camera)
    return pose_matrix2pos_quat(perturbed_camera_pose)


if __name__ == "__main__":
    mode = "GUI"
    sim_cid = init_sim(mode=mode)
    p.loadURDF(
        f"{pybullet_data.getDataPath()}/plane.urdf", physicsClientId=sim_cid
    )
    # tie_id = p.loadURDF(
    #    "tie.urdf",
    #    basePosition=[0, 0, 0.05],
    #    baseOrientation=p.getQuaternionFromEuler([np.pi / 2, 0, 0]),
    #    useFixedBase=True,
    #    physicsClientId=sim_cid,
    # )
    duck_id = p.loadURDF(
        f"{pybullet_data.getDataPath()}/duck_vhacd.urdf",
        basePosition=[0, 0, 0.1],
        baseOrientation=p.getQuaternionFromEuler([np.pi / 2, 0, 0]),
        useFixedBase=True,
        globalScaling=3,
        physicsClientId=sim_cid,
    )

    camera_pos = [-0.58451435, -0.09919992, 0.61609813]
    camera_quat = [
        -0.6733081448310533,
        0.6659691939501913,
        -0.22584407782434218,
        0.22833227394560413,
    ]

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

    # randomize, render and visualize
    fixed_target = True
    num_randomization = 5
    for _ in range(num_randomization):
        if fixed_target:
            (
                perturbed_cam_pos,
                perturbed_cam_quat,
            ) = perturb_camera_pose_around_target(camera_pos, camera_quat)

        else:
            perturbed_cam_pos, perturbed_cam_quat = perturb_camera_pose(
                camera_pos, camera_quat
            )
        vis_frame(
            perturbed_cam_pos,
            perturbed_cam_quat,
            sim_cid,
        )
        view_matrix = build_view_matrix(perturbed_cam_pos, perturbed_cam_quat)
        rgb, depth, seg = render(
            width, height, view_matrix, proj_matrix, near, far, sim_cid
        )
        input("enter to continue")
        continue
        vis_rgb(rgb, 255)
        vis_depth(depth)
        vis_seg_indices(seg)

        pcd = depth2xyz(
            depth,
            fx,
            fy,
            cx,
            cy,
            perturbed_cam_pos,
            perturbed_cam_quat,
            return_pcd=True,
        )
        vis_points(pcd, sim_cid)

        xyz_image = pcd2xyz(
            pcd,
            width,
            height,
            fx,
            fy,
            cx,
            cy,
            perturbed_cam_pos,
            perturbed_cam_quat,
        )
        vis_depth(xyz_image[:, :, 2])
