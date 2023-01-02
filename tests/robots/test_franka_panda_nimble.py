import os

import ipdb
import numpy as np
from tulip.robots.franka_panda_nimble import FrankaPanda
from tulip.utils.nimble_utils import init_sim, init_vis


class TestFrankaPandaNimble:
    def __init__(self, world, urdf_file):
        self.robot = FrankaPanda(world, urdf_file=urdf_file)
        self.robot.set_delta_joint_positions(np.array([0.2] * 7 + [-0.04] * 2))

        visualizer = init_vis(world)
        # visualizer.blockWhileServing()
        ipdb.set_trace()


if __name__ == "__main__":
    world = init_sim()
    urdf_file = os.path.join(
        os.path.expanduser("~"),
        "Projects/tulip/data/urdf/franka_panda/panda_nimble.urdf",
    )
    tester = TestFrankaPandaNimble(world, urdf_file=urdf_file)
