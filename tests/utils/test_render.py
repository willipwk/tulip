import time

import numpy as np
import pybullet as p
import pybullet_data
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
    get_vertices_pos,
    init_sim,
    render,
    vis_frame,
    vis_points,
)
from tulip.utils.transform_utils import homogeneous_transform

if __name__ == "__main__":
    mode = "GUI"
    sim_cid = init_sim(mode=mode)
    p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD, physicsClientId=sim_cid)
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

    # visualize tie vertices
    v_pos = get_vertices_pos(duck_id, sim_cid)
    vis_points(v_pos, sim_cid, color=[0, 1, 0])

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

        pcd = depth2xyz(
            depth, fx, fy, cx, cy, camera_pos, camera_quat, return_pcd=True
        )
        vis_points(pcd, sim_cid)

        xyz_image = pcd2xyz(
            pcd, width, height, fx, fy, cx, cy, camera_pos, camera_quat
        )
        vis_depth(xyz_image[:, :, 2])
        input("enter to continue")
