import argparse
import inspect
import os
import sys
import time

import numpy as np
import pybullet as p
import pybullet_data
import tulip
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from tulip.grippers.kg3_pblt import KG3
from tulip.grippers.mano_pblt import HandBody, HandModel45
from tulip.utils.mesh_utils import create_urdf_from_mesh
from tulip.utils.pblt_utils import (
    disable_collisions,
    disable_collisions_between_objects,
    init_sim,
    step_sim,
    vis_frame,
)
from tulip.utils.transform_utils import homogeneous_transform, relative_pose

parser = argparse.ArgumentParser(description="Grab demo  transfer argparsing")
parser.add_argument(
    "--mano_models_dir",
    dest="models_dir",
    type=str,
    default="/home/zyuwei/tulip/data/mano_v1_2/models/",
    help="mano model files from https://mano.is.tue.mpg.de",
)
parser.add_argument(
    "--grab_data_dir",
    dest="grab_dir",
    type=str,
    default="/home/zyuwei/tulip/grab_data",
    help="grab data from https://grab.is.tue.mpg.de/",
)
parser.add_argument(
    "--demo_fn",
    dest="demo_npz_fn",
    type=str,
    default="/home/zyuwei/data/grab_data/grab/s1/waterbottle_drink_1.npz",
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
    default=1,
    help="use 1 out of every n steps trajectory",
)
parser.add_argument(
    "--replay_hand",
    action="store_true",
    default=False,
    help="Replay hand trajectory",
)
parser.add_argument(
    "--replay_object",
    action="store_true",
    default=False,
    help="Replay object pose rather than phyiscs simulation",
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
    default=False,
    help="Replay hand while transfering using gripper to act",
)
parser.add_argument(
    "--disable_right",
    action="store_true",
    default=False,
    help="Replay hand while transfering using gripper to act",
)
args = parser.parse_args()


def heuristic_mapping(
    side, hand_base_pos, hand_base_orn, tip_local_poses, hand_q
):
    if side == "left":
        coef = -1
    else:
        coef = 1
    g2h_pos = np.array([0, 0, 0])
    g2h_orn = R.from_euler("xyz", [0, 0, coef * np.pi / 5]).as_quat()
    gripper_base_pos, gripper_base_orn = homogeneous_transform(
        hand_base_pos, hand_base_orn, g2h_pos, g2h_orn
    )

    # 0: index -> 1: middle -> 2: ring --> 3: pinky -> 4: thumb
    index_pos = tip_local_poses[0][0]
    thumb_pos = tip_local_poses[-1][0]
    dist_thumb2idx = np.linalg.norm(index_pos - thumb_pos)
    if dist_thumb2idx <= 0.07:
        gripper_q = np.array([0.891, 0.891, 0.891])
    else:
        gripper_q = np.array([0, 0, 0])
    return gripper_base_pos, gripper_base_orn, gripper_q


class TransferDemo(object):
    def __init__(
        self,
        sim_cid: int,
        models_dir: str,
        grab_dir: str,
        demo_npz_fn: str,
        gripper_type: str = "kg3",
        disable_left: bool = False,
        disable_right: bool = False,
    ):
        """Initialize a pybullet simulation containing two hands for grab demo.

        Args:
            sim_cid: PyBullet simulation client id.
            models_dir: path to MANO models. Download from:
                        https://mano.is.tue.mpg.de/
            grab_dir: path to GRAB dataset which contains contact object meshes.
                      Downloaded from https://grab.is.tue.mpg.de/.
                      Folder structure as "grab_dir/tools/contact_meshes/*.ply"
            demo_npz_fn: path to demo npz file from GRAB dataset.
            flags: PyBullet simulation flag(s)."""

        self._sim_cid = sim_cid

        self.grab_dir = grab_dir
        self.demo_data = self.parse_demo(demo_npz_fn)
        self.demo_len = self.demo_data["object"]["params"]["transl"].shape[0]

        self.init_hands(models_dir)
        self.init_grippers(gripper=gripper_type)
        self.vis_pose(lh=True, rh=True, lg=True, rg=True)

        disable_collisions_between_objects(
            self.lhand._pid, self.lgripper._pid, self._sim_cid
        )
        disable_collisions_between_objects(
            self.lhand._pid, self.rgripper._pid, self._sim_cid
        )
        disable_collisions_between_objects(
            self.rhand._pid, self.lgripper._pid, self._sim_cid
        )
        disable_collisions_between_objects(
            self.rhand._pid, self.rgripper._pid, self._sim_cid
        )

        self.controllable_sides = []
        if not disable_left:
            self.controllable_sides.append("left")
        if not disable_right:
            self.controllable_sides.append("right")

    def init_hands(self, models_dir: str) -> None:
        self.lhand_model = HandModel45(True, models_dir)
        self.lhand = HandBody(self._sim_cid, self.lhand_model)
        self.lhand.set_target([0.5, 0, 1.0], [0, 0, 0, 1])
        step_sim(500)
        self.rhand_model = HandModel45(False, models_dir)
        self.rhand = HandBody(self._sim_cid, self.rhand_model)
        self.rhand.set_target([-0.5, 0, 1.0], [0, 0, 0, 1])
        step_sim(500)

    def init_grippers(self, gripper: str) -> None:
        if gripper == "kg3":
            urdf_dir = inspect.getfile(tulip).replace(
                "tulip/__init__.py", "data/urdf/movo"
            )
            self.lgripper = KG3(
                self._sim_cid,
                urdf_file=f"{urdf_dir}/left_kg3.urdf",
                base_pos=[0.15, 0, 1.0],
                base_orn=[0, 0, 0, 1],
            )
            self.rgripper = KG3(
                self._sim_cid,
                urdf_file=f"{urdf_dir}/right_kg3.urdf",
                base_pos=[-0.15, 0, 1.0],
                base_orn=[0, 0, 0, 1],
            )
        else:
            print("Gripper type not supported.")

    def vis_pose(
        self,
        lh: bool = False,
        rh: bool = False,
        lg: bool = False,
        rg: bool = False,
        duration: float = 10,
    ) -> None:
        if lh:
            lh_pos, lh_orn = self.lhand.base_pose
            vis_frame(lh_pos, lh_orn, self._sim_cid, duration=duration)
        if rh:
            rh_pos, rh_orn = self.rhand.base_pose
            vis_frame(rh_pos, rh_orn, self._sim_cid, duration=duration)
        if lg:
            lg_pos, lg_orn = self.lgripper.base_pose
            vis_frame(lg_pos, lg_orn, self._sim_cid, duration=duration)
        if rg:
            rg_pos, rg_orn = self.rgripper.base_pose
            vis_frame(rg_pos, rg_orn, self._sim_cid, duration=duration)

    def parse_demo(self, demo_npz_fn: str) -> dict:
        """Parse demo data from npz file into a dictionary data.

        Args:
            demo_npz_fn: input demo npz filename.
        Returns:
            demo data in dict type.
        """
        demo_data = np.load(demo_npz_fn, allow_pickle=True)
        demo_data = {k: demo_data[k].item() for k in demo_data.files}
        return demo_data

    def init_scene(self, step_id: int = None) -> None:
        if step_id is None:
            step_id = 0

        _ = p.loadURDF(
            f"{pybullet_data.getDataPath()}/plane.urdf",
            physicsClientId=self._sim_cid,
        )
        table_collision_fn = (
            ".".join(
                f'{self.grab_dir}/{self.demo_data["table"]["table_mesh"]}'.split(
                    "."
                )[
                    :-1
                ]
            )
            + "_coacd.obj"
        )
        create_urdf_from_mesh(
            f'{self.grab_dir}/{self.demo_data["table"]["table_mesh"]}',
            "table.urdf",
            collision_fn=table_collision_fn,
        )
        table_pos = self.demo_data["table"]["params"]["transl"][step_id]
        table_orn = self.demo_data["table"]["params"]["global_orient"][step_id]
        table_orn = [np.pi / 2, 0, 0]
        self.table_id = p.loadURDF(
            "table.urdf",
            table_pos,
            p.getQuaternionFromEuler(table_orn),
            physicsClientId=self._sim_cid,
            useFixedBase=True,
        )
        step_sim()
        os.system("rm table.urdf")

        obj_collision_fn = (
            ".".join(
                f'{self.grab_dir}/{self.demo_data["object"]["object_mesh"]}'.split(
                    "."
                )[
                    :-1
                ]
            )
            + "_coacd.obj"
        )
        create_urdf_from_mesh(
            f'{self.grab_dir}/{self.demo_data["object"]["object_mesh"]}',
            "obj.urdf",
            collision_fn=obj_collision_fn,
            mass=0.02,
            scale=[1.0, 1.0, 1.0],
            rgba=[1, 1, 0, 1],
        )
        obj_pos = self.demo_data["object"]["params"]["transl"][step_id]
        obj_orn = self.demo_data["object"]["params"]["global_orient"][step_id]
        self.obj_id = p.loadURDF(
            "obj.urdf",
            obj_pos,
            p.getQuaternionFromEuler(obj_orn),
            useFixedBase=False,
            # globalScaling=0.8,
            physicsClientId=sim_cid,
        )
        step_sim()
        os.system("rm obj.urdf")

    def replay_hand(
        self,
        step_id: int,
        vis_pose: bool = False,
    ) -> None:

        for side in self.controllable_sides:
            trans = self.demo_data[f"{side[0]}hand"]["params"]["transl"][
                step_id
            ]
            global_orn = self.demo_data[f"{side[0]}hand"]["params"][
                "global_orient"
            ][step_id]
            full_pose = self.demo_data[f"{side[0]}hand"]["params"]["fullpose"][
                step_id
            ]
            mano_pose = np.concatenate([global_orn, full_pose], -1)
            if vis_pose:
                base_pos, base_orn = getattr(
                    self, f"{side[0]}hand"
                ).get_target_from_mano(trans, mano_pose)
                vis_frame(base_pos, base_orn, self._sim_cid, duration=1)
            getattr(self, f"{side[0]}hand").set_target_from_mano(
                trans, mano_pose
            )
            step_sim(4)

    def replay_object(
        self,
        step_id: int,
        vis_pose: bool = False,
    ) -> None:
        obj_pos = self.demo_data["object"]["params"]["transl"][step_id]
        obj_orn = self.demo_data["object"]["params"]["global_orient"][step_id]
        if vis_pose:
            vis_frame(obj_pos, obj_orn, self._sim_cid, duration=1)
        p.resetBasePositionAndOrientation(
            self.obj_id,
            obj_pos,
            p.getQuaternionFromEuler(obj_orn),
            physicsClientId=sim_cid,
        )
        step_sim()

    def replay(
        self,
        start_idx: int = None,
        end_idx: int = None,
        every_n_frame: int = 1,
        replay_object: bool = True,
    ) -> None:
        """Replay recorded demo data in simulation.

        Args:
            start_idx: start frame index from the demo sequence.
            end_idx: end frame index from the demo sequence.
            every_n_frame: read every_n_frame from the demo frame sequence.
            replay_object: set object position according to recorded data
                           instead of physics simulation.
        """
        if start_idx is None:
            start_idx = 0
        else:
            start_idx = max(0, start_idx)
        if end_idx is None:
            end_idx = self.demo_len
        else:
            end_idx = min(self.demo_len, end_idx)
        disable_collisions(self.lgripper._pid, self._sim_cid)
        disable_collisions(self.rgripper._pid, self._sim_cid)

        for step_id in tqdm(range(start_idx, end_idx, every_n_frame)):
            if step_id == start_idx:
                self.init_scene(step_id)
                if replay_object:
                    disable_collisions(self.lhand._pid, self._sim_cid)
                    disable_collisions(self.rhand._pid, self._sim_cid)
            self.replay_hand(step_id)
            if replay_object:
                self.replay_object(step_id)
            time.sleep(0.01)

    # finger tip cartesian position may help here rather than raw joint angles
    def hand2gripper(
        self, side, hand_base_pos, hand_base_orn, hand_tip_poses, hand_q, method
    ):
        assert method in ["heuristic"], "Mapping method not supported."
        if method == "heuristic":
            return heuristic_mapping(
                side, hand_base_pos, hand_base_orn, hand_tip_poses, hand_q
            )

    def transfer_gripper(
        self,
        step_id: int,
        vis_pose: bool = False,
    ) -> None:

        for side in self.controllable_sides:
            # parsing demo data
            trans = self.demo_data[f"{side[0]}hand"]["params"]["transl"][
                step_id
            ]
            global_orn = self.demo_data[f"{side[0]}hand"]["params"][
                "global_orient"
            ][step_id]
            full_pose = self.demo_data[f"{side[0]}hand"]["params"]["fullpose"][
                step_id
            ]
            mano_pose = np.concatenate([global_orn, full_pose], -1)

            # extract state from hand target
            hand = getattr(self, f"{side[0]}hand")
            h_base_pos, h_base_orn, h_q = hand.get_target_from_mano(
                trans, mano_pose
            )
            hlink_poses = hand.forward_kinematics(h_q)
            htip_poses = [hlink_poses[idx] for idx in hand._model.tip_links]
            # poses or just position?
            htip_local_poses = [
                relative_pose(h_base_pos, h_base_orn, tip_pos, tip_orn)
                for (tip_pos, tip_orn) in htip_poses
            ]
            g_base_pos, g_base_orn, g_q = self.hand2gripper(
                side, h_base_pos, h_base_orn, htip_local_poses, h_q, "heuristic"
            )
            # apply gripper action
            if vis_pose:
                vis_frame(g_base_pos, g_base_orn, self._sim_cid, duration=1)
            gripper = getattr(self, f"{side[0]}gripper")
            gripper.reset_base_pose(g_base_pos, g_base_orn)
            gripper.set_joint_positions(g_q)
            step_sim(4)

    def transfer(
        self,
        start_idx: int = None,
        end_idx: int = None,
        every_n_frame: int = 1,
        replay_object: bool = False,
        ghost_hand: bool = False,
    ) -> None:
        """Replay recorded demo data in simulation.

        Args:
            start_idx: start frame index from the demo sequence.
            end_idx: end frame index from the demo sequence.
            every_n_frame: read every_n_frame from the demo frame sequence.
            replay_object: set object position according to recorded data
                           instead of physics simulation.
        """
        if start_idx is None:
            start_idx = 0
        else:
            start_idx = max(0, start_idx)
        if end_idx is None:
            end_idx = self.demo_len
        else:
            end_idx = min(self.demo_len, end_idx)
        disable_collisions(self.lhand._pid, self._sim_cid)
        disable_collisions(self.rhand._pid, self._sim_cid)

        for step_id in tqdm(range(start_idx, end_idx, every_n_frame)):

            if step_id == start_idx:
                self.init_scene(step_id)
                if replay_object:
                    disable_collisions(self.lgripper._pid, self._sim_cid)
                    disable_collisions(self.rgripper._pid, self._sim_cid)

            self.transfer_gripper(step_id)
            if ghost_hand:
                self.replay_hand(step_id)
            if replay_object:
                self.replay_object(step_id)
            time.sleep(0.01)


if __name__ == "__main__":

    sim_cid = init_sim()

    assert os.path.isdir(args.models_dir), f"{args.models_dir} does not exist."
    assert os.path.isdir(args.grab_dir), f"{args.grab_dir} does not exist."
    assert os.path.isdir(
        f"{args.grab_dir}/tools/object_meshes"
    ), f"object meshes does not exist in {args.grab_dir}."
    assert os.path.isfile(args.demo_npz_fn), f"{args.grab_dir} does not exist."

    grab_demo = TransferDemo(
        sim_cid,
        args.models_dir,
        args.grab_dir,
        args.demo_npz_fn,
        disable_left=args.disable_left,
        disable_right=args.disable_right,
    )

    if args.replay_hand:
        grab_demo.replay(
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            every_n_frame=args.every_n_frame,
            replay_object=args.replay_object,
        )
    else:
        grab_demo.transfer(
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            every_n_frame=args.every_n_frame,
            replay_object=args.replay_object,
            ghost_hand=args.ghost_hand,
        )
