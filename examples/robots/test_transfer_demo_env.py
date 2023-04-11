import argparse
import os
import time

import numpy as np

from transfer_grab_demo import TransferDemoEnv
from tulip.utils.pblt_utils import init_sim

parser = argparse.ArgumentParser(description="RL for demo transferring")
parser.add_argument( "--mano_models_dir",
    dest="models_dir",
    type=str,
    default="/home/zyuwei/Projects/tulip/data/mano_v1_2/models/",
    help="mano model files from https://mano.is.tue.mpg.de",
)
parser.add_argument(
    "--grab_data_dir",
    dest="grab_dir",
    type=str,
    default="/home/zyuwei/Projects/tulip/grab_data",
    help="grab data from https://grab.is.tue.mpg.de/",
)
parser.add_argument(
    "--demo_fn",
    dest="demo_npz_fn",
    type=str,
    default="/home/zyuwei/Projects/tulip/grab_data/grab/s1/waterbottle_drink_1.npz",
    help="demo npz file from GRAB dataset",
)
parser.add_argument(
    "--start_idx", type=int, default=45, help="trajectory start index"
)
parser.add_argument(
    "--end_idx", type=int, default=175, help="trajectory end index"
)
parser.add_argument(
    "--every_n_frame",
    type=int,
    default=3,
    help="use 1 out of every n steps trajectory",
)
parser.add_argument(
    "--ghost_hand",
    action="store_true",
    default=False,
    help="Replay hand while transfering using gripper to act",
)
parser.add_argument(
    "--disable_left",
    action="store_true",
    default=True,
    help="Replay hand while transfering using gripper to act",
)
parser.add_argument(
    "--disable_right",
    action="store_true",
    default=False,
    help="Replay hand while transfering using gripper to act",
)
parser.add_argument(
    "--sim_mode",
    type=str,
    choices=["direct", "gui", "DIRECT", "GUI"],
    default="GUI",
    help="pybullet simulation mode",
)
args = parser.parse_args()


def init_env(args):
    env = TransferDemoEnv(
        sim_cid,
        args.models_dir,
        args.grab_dir,
        args.demo_npz_fn,
        disable_left=args.disable_left,
        disable_right=args.disable_right,
    )
    env.init_rl_env(args.start_idx,
                    args.end_idx,
                    args.every_n_frame,
                    args.ghost_hand)
    return env


class RandomPolicy:

    def __init__(self, action_space):
        self.action_space = action_space

    def __call__(self, obs):
        action = self.action_space.sample()
        obj2ee_dist = np.linalg.norm(obs[-8:-5])
        if obj2ee_dist <= 0.04:
            action[-1] = abs(action[-1])
        if obj2ee_dist >= 0.04:
            action[-1] = -abs(action[-1])
        return action


if __name__ == "__main__":

    sim_cid = init_sim(mode=args.sim_mode)

    assert os.path.isdir(args.models_dir), f"{args.models_dir} does not exist."
    assert os.path.isdir(args.grab_dir), f"{args.grab_dir} does not exist."
    assert os.path.isdir(
        f"{args.grab_dir}/tools/object_meshes"
    ), f"object meshes does not exist in {args.grab_dir}."
    assert os.path.isfile(args.demo_npz_fn), f"{args.grab_dir} does not exist."

    env = init_env(args)
    random_policy = RandomPolicy(env.action_space)

    done = False
    next_obs = env.reset()
    for _ in range(1000):
        obs = next_obs
        action = random_policy(obs)
        next_obs, reward, done, _ = env.step(action)
        time.sleep(0.2)
        if done:
            next_obs = env.reset()
