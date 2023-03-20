import json
import os
import time

import numpy as np
import pybullet_data
from tqdm import tqdm

import pybullet as p
from tulip.grippers.mano_ik_pblt import ManoIKHand
from tulip.grippers.mano_pblt import HandBody, HandModel20, HandModel45
from tulip.utils.pblt_utils import init_sim, step_sim


def test_mano_ik_pblt(sim_cid, side, urdf_file, scale):
    hand = ManoIKHand(sim_cid, side, urdf_file, scale=scale)
    rand_quat = np.random.randn(4)
    rand_quat /= np.linalg.norm(rand_quat)
    hand.set_base_pose([0, 0, 0.5], rand_quat, reset_targets=True)
    for i in tqdm(range(5000)):
        hand.track_targets()
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
                for tgt in hand.targets
            ]
            hand.set_targets(targets)


def test_hand45_pblt(sim_cid, side, models_dir):

    left_hand = side == "left"
    hand_model = HandModel45(left_hand, models_dir)
    flags = HandBody.FLAG_DEFAULT  # | HandBody.FLAG_USE_SELF_COLLISION
    hand = HandBody(sim_cid, hand_model, flags=flags)

    for _ in range(1000):
        pos = [1, 0, 1]
        rand_quat = np.random.randn(4)
        rand_quat /= np.linalg.norm(rand_quat)
        rand_quat = [0, 0, 0, 1]
        dofs = np.random.uniform(*hand_model.dofs_limits)
        dofs = [0.35] + [0.0] * 19
        hand.reset(pos, rand_quat, dofs)
        step_sim()
        input("enter to continue")

        trans, pose = hand.get_mano_state()
        hand.reset_from_mano(trans, pose)
        trans2, pose2 = hand.get_mano_state()
        print(trans - trans2, pose - pose2)


def test_replay_hand20_pblt(sim_cid, side, models_dir, json_fn):
    _ = p.loadURDF(
        f"{pybullet_data.getDataPath()}/plane.urdf", physicsClientId=sim_cid
    )
    _ = p.loadURDF(
        f"{pybullet_data.getDataPath()}/duck_vhacd.urdf",
        basePosition=[0, 0, 0.1],
        baseOrientation=p.getQuaternionFromEuler([np.pi / 2, 0, 0]),
        physicsClientId=sim_cid,
    )

    left_hand = side == "left"
    hand_model = HandModel20(left_hand, models_dir)
    flags = HandBody.FLAG_DEFAULT  # | HandBody.FLAG_USE_SELF_COLLISION
    hand = HandBody(sim_cid, hand_model, flags=flags)

    assert os.path.isfile(json_fn), f"{json_fn} does not exist!"
    with open(json_fn, "r") as fp:
        action_data = json.load(fp)["actions"][0]
    for base_pos, base_quat, angles in zip(*action_data):
        hand.set_target(base_pos, base_quat, angles)
        step_sim(4)
        time.sleep(0.01)


if __name__ == "__main__":

    sim_cid = init_sim()

    side = "right"

    scale = 5
    urdf_file = f"/home/zyuwei/tulip/data/urdf/mano/mano_hand_{side}.urdf"
    test_mano_ik_pblt(
        sim_cid=sim_cid, side=side, urdf_file=urdf_file, scale=scale
    )

    models_dir = "/home/zyuwei/tulip/data/mano_v1_2/models/"
    test_hand45_pblt(sim_cid, side, models_dir)

    models_dir = "/home/zyuwei/tulip/data/mano_v1_2/models/"
    json_fn = "/home/zyuwei/tulip/tests/grippers/lift_duck.json"
    test_replay_hand20_pblt(sim_cid, side, models_dir, json_fn)
