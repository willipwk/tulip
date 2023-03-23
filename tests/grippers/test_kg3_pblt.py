import os
import time

import numpy as np
import pybullet as p
from tulip.grippers.kg3_pblt import KG3
from tulip.utils.pblt_utils import init_sim, step_sim, vis_frame


class TestKG3Pblt:
    def __init__(self, sim_cid, urdf_file):
        self.sim_cid = sim_cid
        self.urdf_file = urdf_file

        self.gripper = KG3(sim_cid, urdf_file)

    def test_control_gripper(self):
        for i in range(10000):
            if i % 100 == 0:
                self.gripper.close()
            elif i % 50 == 0:
                self.gripper.open()
            step_sim()

    def test_reset_base_pose(self):
        self.gripper.close()
        for i in range(10000):
            if i % 1000 == 0:
                base_pos = [
                    np.random.uniform(-1, 1),
                    np.random.uniform(-1, 1),
                    np.random.uniform(0.5, 1.5),
                ]
                base_orn = [
                    np.random.uniform(-np.pi, np.pi),
                    np.random.uniform(-np.pi, np.pi),
                    np.random.uniform(-np.pi, np.pi),
                ]
                base_quat = p.getQuaternionFromEuler(base_orn)
                self.gripper.reset_base_pose(base_pos, base_quat)
                ee_pos, ee_quat = self.gripper.ee_pose
                vis_frame(ee_pos, ee_quat, self.sim_cid, length=0.2, duration=2)
                base_pos, base_quat = self.gripper.base_pose
                vis_frame(
                    base_pos, base_quat, self.sim_cid, length=0.2, duration=2
                )
                time.sleep(1)
            step_sim()


if __name__ == "__main__":
    sim_cid = init_sim()
    urdf_file = os.path.expanduser("~/tulip/data/urdf/movo/right_kg3.urdf")
    tester = TestKG3Pblt(sim_cid=sim_cid, urdf_file=urdf_file)
    tester.test_control_gripper()
    tester.test_reset_base_pose()
