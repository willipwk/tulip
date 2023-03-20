import ipdb
import numpy as np
import pybullet as p
from tqdm import tqdm
from tulip.grippers.mano_pblt import ManoHand
from tulip.utils.pblt_utils import init_sim, step_sim


class TestManoPblt:
    def __init__(self, sim_cid, side, urdf_file, scale):
        self.hand = ManoHand(sim_cid, side, urdf_file, scale=scale)
        rand_quat = np.random.randn(4)
        rand_quat /= np.linalg.norm(rand_quat)
        self.hand.set_base_pose([0, 0, 0.5], rand_quat, reset_targets=True)
        for i in tqdm(range(5000)):
            self.hand.track_targets()
            step_sim()
            if i % 100 == 0:
                targets = [
                    tgt
                    + np.array(
                        [
                            np.random.uniform(-0.05, 0.05),
                            np.random.uniform(-0.05, 0.05),
                            np.random.uniform(-0.05, 0.05),
                        ]
                    )
                    for tgt in self.hand.targets
                ]
                self.hand.set_targets(targets)


if __name__ == "__main__":
    sim_cid = init_sim()
    scale = 5
    side = "right"
    urdf_file = f"/home/zyuwei/tulip/data/urdf/mano/mano_hand_{side}.urdf"
    tester = TestManoPblt(
        sim_cid=sim_cid, side=side, urdf_file=urdf_file, scale=scale
    )
