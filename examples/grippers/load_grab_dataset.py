import os
import sys
import time

import numpy as np
import pybullet_data
from tqdm import tqdm

import pybullet as p
from tulip.grippers.mano_pblt import HandBody, HandModel45
from tulip.utils.pblt_utils import (
    create_urdf_from_mesh,
    disable_collisions,
    init_sim,
    step_sim,
)


class GrabDemo(object):
    def __init__(self, sim_cid, models_dir, grab_dir, demo_npz_fn, flags=None):

        self.sim_cid = sim_cid
        self.grab_dir = grab_dir

        _ = p.loadURDF(
            f"{pybullet_data.getDataPath()}/plane.urdf", physicsClientId=sim_cid
        )
        if flags is None:
            self.flags = (
                HandBody.FLAG_DEFAULT | HandBody.FLAG_USE_SELF_COLLISION
            )
        else:
            self.flags = flags
        self.lhand_model = HandModel45(True, models_dir)
        self.rhand_model = HandModel45(False, models_dir)
        self.lhand = HandBody(self.sim_cid, self.lhand_model, flags=self.flags)
        self.rhand = HandBody(self.sim_cid, self.rhand_model, flags=self.flags)
        self.lhand.reset([0.5, 0, 1.0], [0, 0, 0, 1], [0.35] + [0] * 19)
        self.rhand.reset([-0.5, 0, 1.0], [0, 0, 0, 1], [0.35] + [0] * 19)
        step_sim(20)

        self.demo_data = self.parse_demo(demo_npz_fn)
        self.demo_len = self.demo_data["object"]["params"]["transl"].shape[0]

    def parse_demo(self, demo_npz_fn):
        demo_data = np.load(demo_npz_fn, allow_pickle=True)
        demo_data = {k: demo_data[k].item() for k in demo_data.files}
        return demo_data

    def replay(
        self, start_idx=None, end_idx=None, every_n_frame=1, replay_object=True
    ):
        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = self.demo_len

        if replay_object:
            disable_collisions(self.lhand._pid, self.sim_cid)
            disable_collisions(self.rhand._pid, self.sim_cid)
        for step_id in tqdm(range(start_idx, end_idx, every_n_frame)):
            # set both hands
            for side in ["left", "right"]:
                trans = self.demo_data[f"{side[0]}hand"]["params"]["transl"][
                    step_id
                ]
                global_orn = self.demo_data[f"{side[0]}hand"]["params"][
                    "global_orient"
                ][step_id]
                full_pose = self.demo_data[f"{side[0]}hand"]["params"][
                    "fullpose"
                ][step_id]
                mano_pose = np.concatenate([global_orn, full_pose], -1)
                getattr(self, f"{side[0]}hand").set_target_from_mano(
                    trans, mano_pose
                )
                step_sim(4)

            # set object pose
            if step_id == start_idx:
                create_urdf_from_mesh(
                    f'{self.grab_dir}/{self.demo_data["table"]["table_mesh"]}',
                    "table.urdf",
                )
                table_pos = self.demo_data["table"]["params"]["transl"][step_id]
                table_orn = self.demo_data["table"]["params"]["global_orient"][
                    step_id
                ]
                table_orn = [np.pi / 2, 0, 0]
                self.table_id = p.loadURDF(
                    "table.urdf",
                    table_pos,
                    p.getQuaternionFromEuler(table_orn),
                    physicsClientId=self.sim_cid,
                    useFixedBase=True,
                )
                step_sim()

                create_urdf_from_mesh(
                    f'{self.grab_dir}/{self.demo_data["object"]["object_mesh"]}',
                    "obj.urdf",
                    mass=0.02,
                    scale=[1.0, 1.0, 1.0],
                    rgba=[1, 1, 0, 1],
                )
                obj_pos = self.demo_data["object"]["params"]["transl"][step_id]
                obj_orn = self.demo_data["object"]["params"]["global_orient"][
                    step_id
                ]
                self.obj_id = p.loadURDF(
                    "obj.urdf",
                    obj_pos,
                    p.getQuaternionFromEuler(obj_orn),
                    useFixedBase=False,
                    # globalScaling=0.8,
                    physicsClientId=sim_cid,
                )
                step_sim()
            elif replay_object:
                obj_pos = self.demo_data["object"]["params"]["transl"][step_id]
                obj_orn = self.demo_data["object"]["params"]["global_orient"][
                    step_id
                ]
                p.resetBasePositionAndOrientation(
                    self.obj_id,
                    obj_pos,
                    p.getQuaternionFromEuler(obj_orn),
                    physicsClientId=sim_cid,
                )
                step_sim()

            time.sleep(0.01)

        os.system("rm table.urdf")
        os.system("rm obj.urdf")


if __name__ == "__main__":

    sim_cid = init_sim()

    models_dir = "/home/zyuwei/tulip/data/mano_v1_2/models/"
    grab_dir = "/home/zyuwei/tulip/grab_data"
    demo_npz_fn = sys.argv[1]
    grab_demo = GrabDemo(sim_cid, models_dir, grab_dir, demo_npz_fn)
    grab_demo.replay(replay_object=False)
