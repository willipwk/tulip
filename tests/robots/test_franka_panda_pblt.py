import ipdb
import numpy as np
import pybullet as p
from tulip.robots.franka_panda_pyblt import FrankaPanda
from tulip.utils.pblt_utils import init_sim, step_sim


class TestFrankaPandaPblt:
    def __init__(self, sim_cid, urdf_file):
        self.robot = FrankaPanda(sim_cid=sim_cid, urdf_file=urdf_file)
        ipdb.set_trace()
        self.robot.set_delta_joint_positions(np.array([0.2] * 7 + [-0.04] * 2))
        step_sim()


if __name__ == "__main__":
    sim_cid = init_sim()
    urdf_file = "/Users/zyuwei/Projects/tulip/data/urdf/franka_panda/panda.urdf"
    tester = TestFrankaPandaPblt(sim_cid=sim_cid, urdf_file=urdf_file)
