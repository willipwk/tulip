import time

import numpy as np
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation as R
from tulip.utils.pblt_utils import init_sim, vis_frame


def test_basis_coord(sim_cid):
    quat = p.getQuaternionFromEuler([np.pi, 0, 0])
    vis_frame([0, 0, 0], quat, sim_cid, length=0.2, duration=10)
    time.sleep(10)
    input("enter to continue")
    quat = p.getQuaternionFromEuler([0, np.pi, 0])
    vis_frame([0, 0, 0], quat, sim_cid, length=0.2, duration=10)
    time.sleep(10)
    input("enter to continue")
    quat = p.getQuaternionFromEuler([0, 0, np.pi])
    vis_frame([0, 0, 0], quat, sim_cid, length=0.2, duration=10)
    time.sleep(10)
    input("enter to continue")


def test_obj_rotation(sim_cid, n_test, duration=2):
    pos = [1.0, 0, 1.0]
    # canocical pose
    quat = [0, 0, 0, 1]
    duck_id = p.loadURDF(
        f"{pybullet_data.getDataPath()}/duck_vhacd.urdf",
        pos,
        quat,
        useFixedBase=True,
        globalScaling=3,
        physicsClientId=sim_cid,
    )
    vis_frame(pos, quat, sim_cid, length=0.2, duration=duration)
    time.sleep(duration)

    input("enter to start randomizing the duck pose.")
    for _ in range(n_test):
        quat = p.getQuaternionFromEuler(
            [
                np.random.uniform(-np.pi, np.pi),
                np.random.uniform(-np.pi, np.pi),
                np.random.uniform(-np.pi, np.pi),
            ]
        )
        p.resetBasePositionAndOrientation(duck_id, pos, quat, sim_cid)
        vis_frame(pos, quat, sim_cid, length=0.2, duration=duration)
        time.sleep(duration)


if __name__ == "__main__":
    # initialize simulation
    mode = "GUI"
    sim_cid = init_sim(mode=mode)
    p.loadURDF(f"{pybullet_data.getDataPath()}/plane.urdf")

    test_basis_coord(sim_cid)
    test_obj_rotation(sim_cid, 10)
