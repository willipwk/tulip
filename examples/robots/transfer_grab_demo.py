import argparse
import inspect
import os
import sys
import time
import uuid

import gym
import numpy as np
import pybullet_data
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

import pybullet as p
import tulip
from tulip.grippers.kg3_pblt import KG3
from tulip.grippers.mano_pblt import HandBody, HandModel45
from tulip.utils.mesh_utils import create_urdf_from_mesh
from tulip.utils.pblt_utils import (disable_collisions,
                                    disable_collisions_between_objects,
                                    init_sim, step_sim, vis_frame)
from tulip.utils.transform_utils import homogeneous_transform, relative_pose

parser = argparse.ArgumentParser(description="Grab demo  transfer argparsing")
parser.add_argument(
    "--mano_models_dir",
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
    default=True,
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
    side, gripper, hand_wrist_pos, hand_wrist_orn, tip_local_poses, hand_q
) -> tuple:
    if side == "left":
        coef = -1
    else:
        coef = 1
    g2h_pos = np.array([0, 0, 0])
    g2h_orn = R.from_euler(
        "xyz", [-coef * np.pi / 5, 0, coef * np.pi / 5]
    ).as_quat()
    gripper_wrist_pos, gripper_wrist_orn = homogeneous_transform(
        hand_wrist_pos, hand_wrist_orn, g2h_pos, g2h_orn
    )

    # 0: index -> 1: middle -> 2: ring --> 3: pinky -> 4: thumb
    index_pos = tip_local_poses[0][0]
    thumb_pos = tip_local_poses[-1][0]
    dist_thumb2idx = np.linalg.norm(index_pos - thumb_pos)
    gripper_q = gripper.ik(dist_thumb2idx)
    return gripper_wrist_pos, gripper_wrist_orn, gripper_q


class TransferDemoEnv(gym.Env):
    def __init__(
        self,
        sim_cid: int,
        models_dir: str,
        grab_dir: str,
        demo_npz_fn: str,
        gripper_type: str = "kg3",
        disable_left: bool = False,
        disable_right: bool = False,
        init_hands: bool = True,
        init_grippers: bool = True,
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

        self.models_dir = models_dir
        self.grab_dir = grab_dir
        self.demo_data = self.parse_demo(demo_npz_fn)
        self.demo_len = self.demo_data["object"]["params"]["transl"].shape[0]

        self.gripper_type = gripper_type
        self.controllable_sides = []
        if not disable_left:
            self.controllable_sides.append("left")
        if not disable_right:
            self.controllable_sides.append("right")

        if init_hands:
            self.init_hands(models_dir)
        if init_grippers:
            self.init_grippers(gripper=gripper_type)
        if init_hands and init_grippers:
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

    def seed(self, seed):
        np.random.seed(seed)

    def init_hands(self, models_dir: str, sim_step=500) -> None:
        """
        self.lhand_model = HandModel45(True, models_dir)
        self.lhand = HandBody(self._sim_cid, self.lhand_model)
        if sim_step > 0:
            self.lhand.set_target([0.5, 0, 1.0], [0, 0, 0, 1])
            step_sim(sim_step)
        """
        self.rhand_model = HandModel45(False, models_dir)
        self.rhand = HandBody(self._sim_cid, self.rhand_model)
        if sim_step > 0:
            self.rhand.set_target([-0.5, 0, 1.0], [0, 0, 0, 1])
            step_sim(sim_step)

    def init_grippers(self, gripper: str, sim_step=500) -> None:
        if gripper == "kg3":
            urdf_dir = inspect.getfile(tulip).replace(
                "tulip/__init__.py", "data/urdf/movo"
            )
            """
            self.lgripper = KG3(
                self._sim_cid,
                urdf_file=f"{urdf_dir}/left_kg3.urdf",
                base_pos=[0.15, 0, 1.0],
                base_orn=[0, 0, 0, 1],
            )
            if sim_step > 0:
                self.lgripper.control_base_pose(
                    self.lgripper._base_pos, self.lgripper._base_orn
                )
                self.lgripper.set_joint_positions(self.lgripper._home_pos)
                step_sim(sim_step)
            """
            self.rgripper = KG3(
                self._sim_cid,
                urdf_file=f"{urdf_dir}/right_kg3.urdf",
                base_pos=[-0.15, 0, 1.0],
                base_orn=[0, 0, 0, 1],
            )
            if sim_step > 0:
                self.rgripper.control_base_pose(
                    self.rgripper._base_pos, self.rgripper._base_orn
                )
                self.rgripper.set_joint_positions(self.rgripper._home_pos)
                step_sim(sim_step)
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

    def init_scene(
        self,
        step_id: int = None,
    ) -> None:
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
            mu=0.9,
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
            globalScaling=1,
        )
        table_pos, _ = p.getBasePositionAndOrientation(self.table_id)
        self.table_center = table_pos
        os.system("rm table.urdf")

    def init_object(
        self,
        step_id: int = None,
        obj_fn: str = None,
        scale: list = [1.0, 1.0, 1.0],
        rgba: list = [1.0, 1.0, 0.0, 1.0],
        pos_offset: np.ndarray = [0.0, 0.0, 0.0],
        rpy_offset: np.ndarray = [0.0, 0.0, 0.0],
    ) -> None:
        if obj_fn is None:
            obj_fn = (
                f'{self.grab_dir}/{self.demo_data["object"]["object_mesh"]}'
            )
        urdf_fn = f"{uuid.uuid4()}.urdf"
        obj_collision_fn = ".".join(obj_fn.split(".")[:-1]) + "_coacd.obj"
        create_urdf_from_mesh(
            obj_fn,
            urdf_fn,
            collision_fn=obj_collision_fn,
            mass=0.02,
            mu=0.5,
            scale=scale,
            rgba=rgba,
        )
        obj_pos = self.demo_data["object"]["params"]["transl"][step_id]
        obj_pos[2] = 0.075 * scale[2] + self.table_center[2]
        obj_orn = self.demo_data["object"]["params"]["global_orient"][step_id]
        self.obj_id = p.loadURDF(
            urdf_fn,
            obj_pos + pos_offset,
            p.getQuaternionFromEuler(obj_orn + rpy_offset),
            # globalScaling=0.8,
            physicsClientId=self._sim_cid,
        )
        step_sim(20)
        self.obj_aabb = p.getAABB(self.obj_id, physicsClientId=self._sim_cid)
        self.obj_aabb = [list(aabb) for aabb in self.obj_aabb]
        os.system(f"rm {urdf_fn}")

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
                wrist_pos, wrist_orn = getattr(
                    self, f"{side[0]}hand"
                ).get_target_from_mano(trans, mano_pose)
                vis_frame(wrist_pos, wrist_orn, self._sim_cid, duration=1)
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
            physicsClientId=self._sim_cid,
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
                self.init_object(step_id)
                if replay_object:
                    disable_collisions(self.lhand._pid, self._sim_cid)
                    disable_collisions(self.rhand._pid, self._sim_cid)
            self.replay_hand(step_id)
            if replay_object:
                self.replay_object(step_id)
            time.sleep(0.01)

    # finger tip cartesian position may help here rather than raw joint angles
    def hand2gripper(
        self,
        side: str,
        gripper,
        hand_wrist_pos,
        hand_wrist_orn,
        hand_tip_poses,
        hand_q,
        method,
        delta_action,
        pos_dilation=0.02,
        orn_dilation=0.2,
        ang_dilation=0.4,
    ):
        assert method in [
            "heuristic",
            "residual",
        ], "Unsupported mapping method."
        if method == "heuristic":
            return heuristic_mapping(
                side,
                gripper,
                hand_wrist_pos,
                hand_wrist_orn,
                hand_tip_poses,
                hand_q,
            )
        elif method == "residual":
            base_pos, base_quat, base_q = heuristic_mapping(
                side,
                gripper,
                hand_wrist_pos,
                hand_wrist_orn,
                hand_tip_poses,
                hand_q,
            )
            base_orn = R.from_quat(base_quat).as_euler("xyz")
            delta_pos = pos_dilation * delta_action[..., 0:3]
            delta_orn = orn_dilation * delta_action[..., 3:6]
            delta_orn[1:] /= 2
            delta_q = ang_dilation * np.array([delta_action[..., 6]] * 3)
            pos = base_pos + delta_pos
            orn = base_orn + delta_orn
            quat = R.from_euler("xyz", orn).as_quat()
            q = base_q + delta_q
            for i, q_value in enumerate(q):
                q[i] = min(
                    max(gripper.joint_limits[i][0], q_value),
                    gripper.joint_limits[i][1],
                )
            return pos, quat, q

    def transfer_gripper(
        self,
        step_id: int,
        method: str = "heuristic",
        delta_action: np.array = None,
        controllable_sides: list = None,
        vis_pose: bool = False,
    ) -> None:

        if controllable_sides is None:
            controllable_sides = self.controllable_sides
        for side in controllable_sides:
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
            h_wrist_pos, h_wrist_orn, h_q = hand.get_target_from_mano(
                trans, mano_pose
            )
            hlink_poses = hand.forward_kinematics(h_q)
            htip_poses = [hlink_poses[idx] for idx in hand._model.tip_links]
            # poses or just position?
            htip_local_poses = [
                relative_pose(h_wrist_pos, h_wrist_orn, tip_pos, tip_orn)
                for (tip_pos, tip_orn) in htip_poses
            ]

            gripper = getattr(self, f"{side[0]}gripper")
            g_wrist_pos, g_wrist_orn, g_q = self.hand2gripper(
                side,
                gripper,
                h_wrist_pos,
                h_wrist_orn,
                htip_local_poses,
                h_q,
                method,
                delta_action=delta_action,
            )
            # apply gripper action
            if vis_pose:
                vis_frame(g_wrist_pos, g_wrist_orn, self._sim_cid, duration=1)
            for _ in range(5):
                gripper.control_base_pose(g_wrist_pos, g_wrist_orn)
                gripper.set_joint_positions(g_q)
                step_sim(4)

        return h_wrist_pos

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

        poses = []
        for step_id in tqdm(range(start_idx, end_idx, every_n_frame)):

            if step_id == start_idx:
                self.init_scene(step_id)
                if replay_object:
                    disable_collisions(self.lgripper._pid, self._sim_cid)
                    disable_collisions(self.rgripper._pid, self._sim_cid)

            poses.append(self.transfer_gripper(step_id))
            if ghost_hand:
                self.replay_hand(step_id)
            if replay_object:
                self.replay_object(step_id)
            time.sleep(0.01)

    def init_rl_env(
        self, start_idx: int, end_idx: int, every_n_frame: int, ghost_hand: bool
    ) -> None:
        # setup trajectory horizon
        assert len(self.controllable_sides) == 1, "Only single hand control."
        self.action_side = self.controllable_sides[0]

        self.episode_idx = -1
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.every_n_frame = every_n_frame

        self.state_id = None
        # setup target goal
        self.obj_goal_pos = self.demo_data["object"]["params"]["transl"][
            self.end_idx
        ]
        self.obj_goal_orn = self.demo_data["object"]["params"]["global_orient"][
            self.end_idx
        ]

        # ignore necessary collisions
        self.ghost_hand = False

        # setup observation and action space
        self._observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(71,), dtype="float32"
        )  # base_pose, tip_pose * 3, keyframe_pose_diff * 4,  obj2ee_pose, q, t
        self._action_space = gym.spaces.Box(
            -1, 1, shape=(7,), dtype="float32"
        )  # d_pos, d_orn, d_q

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space

    @property
    def observation_space(self) -> gym.spaces.Box:
        return self._observation_space

    def get_obj2ee_pose(self, gripper, euler_orn=False):
        # object to ee pose
        ee_pos, ee_quat = gripper.ee_pose
        obj_pos, obj_quat = p.getBasePositionAndOrientation(
            self.obj_id, physicsClientId=self._sim_cid
        )
        obj2ee_pos = (np.array(obj_pos) - ee_pos).tolist()
        obj2ee_orn = (
            R.from_quat(obj_quat).as_euler("xyz")
            - R.from_quat(ee_quat).as_euler("xyz")
        ).tolist()
        if not euler_orn:
            obj2ee_orn = R.from_euler("xyz", obj2ee_orn).as_quat().tolist()
        return obj2ee_pos, obj2ee_orn

    def get_obj2goal_pose(self, euler_orn=False):
        # object to ee pose
        obj_pos, obj_quat = p.getBasePositionAndOrientation(
            self.obj_id, physicsClientId=self._sim_cid
        )
        obj2goal_pos = (np.array(obj_pos) - self.obj_goal_pos).tolist()
        obj2goal_orn = (
            R.from_quat(obj_quat).as_euler("xyz")
            - R.from_euler("xyz", self.obj_goal_orn).as_euler("xyz")
        ).tolist()
        if not euler_orn:
            obj2goal_orn = R.from_euler("xyz", obj2goal_orn).as_quat().tolist()
        return obj2goal_pos, obj2goal_orn

    def get_observation(self) -> np.array:
        """
        Hand tip pose sequence:
        # 0: index -> 1: middle -> 2: ring --> 3: pinky -> 4: thumb
        Grippertip pose sequence:
        # 0: thumb -> 1: middle -> 2: index
        """
        # hand related observations
        hand = getattr(self, f"{self.action_side[0]}hand")
        h_wrist_pos, h_wrist_quat = hand.base_pose
        h_wrist_pos = list(h_wrist_pos)
        h_wrist_orn = R.from_quat(h_wrist_quat).as_euler("xyz").tolist()
        h_tip_poses = [
            [list(p), list(R.from_quat(q).as_euler("xyz"))]
            for p, q in hand.tip_pose
        ]
        # h_q = hand.get_joint_positions().tolist()

        # gripper related observations
        gripper = getattr(self, f"{self.action_side[0]}gripper")
        g_wrist_pos, g_wrist_quat = gripper.base_pose
        g_wrist_pos = list(g_wrist_pos)
        g_wrist_orn = R.from_quat(g_wrist_quat).as_euler("xyz").tolist()
        g_tip_poses = [
            [list(p), list(R.from_quat(q).as_euler("xyz"))]
            for p, q in gripper.tip_pose
        ]
        g_q = gripper.get_joint_positions().mean()

        # keyframe pose differences
        poses_diff = [
            [
                [h_p - g_p for h_p, g_p in zip(h_wrist_pos, g_wrist_pos)],
                [h_o - g_o for h_o, g_o in zip(h_wrist_orn, g_wrist_orn)],
            ]
        ]
        for h_i, g_i in ((4, 0), (0, 2), (1, 1)):
            h_pos, h_orn = h_tip_poses[h_i]
            g_pos, g_orn = g_tip_poses[g_i]
            poses_diff.append(
                [
                    [h_p - g_p for h_p, g_p in zip(h_pos, g_pos)],
                    [h_o - g_o for h_o, g_o in zip(h_orn, g_orn)],
                ]
            )
        obj2ee_pos, obj2ee_orn = self.get_obj2ee_pose(gripper, euler_orn=True)
        obj2goal_pos, obj2goal_orn = self.get_obj2goal_pose(euler_orn=True)

        # aggregate observations
        # obs_dim = 6 + 6 * 3 + 6 * 4 + 3 = 6 * 8 + 3 + 1 = 52
        obs = g_wrist_pos + g_wrist_orn
        for tip_pos, tip_orn in g_tip_poses:
            obs += tip_pos + tip_orn
        for diff_pos, diff_orn in poses_diff:
            obs += diff_pos + diff_orn
        obs += self.obj_aabb[0] + self.obj_aabb[1] + self.aabb_size
        obs += obj2ee_pos + obj2ee_orn + obj2goal_pos + obj2goal_orn
        obs += [g_q]
        obs += [self.step_idx]
        return np.array(obs)

    def reset(self) -> np.ndarray:

        p.resetSimulation(physicsClientId=self._sim_cid)
        p.setTimeStep(1 / 240.0, physicsClientId=self._sim_cid)
        p.setGravity(0, 0, -9.8, physicsClientId=self._sim_cid)

        sim_step = 500
        self.init_hands(self.models_dir, sim_step=sim_step)
        self.init_grippers(self.gripper_type, sim_step=sim_step)
        self.init_scene(self.start_idx)
        self.transfer_gripper(self.start_idx)
        step_sim(sim_step)

        # init object
        x_scale = np.random.uniform(low=1.0, high=1.0)
        y_scale = x_scale + np.random.uniform(low=0.0, high=0.0)
        z_scale = np.random.uniform(low=1.0, high=1.0)
        scale = [x_scale, y_scale, z_scale]
        rgba = [1, 1, 0, 1] #np.random.uniform(low=0, high=1, size=3).tolist() + [1.0]
        # pos_offset = np.random.uniform(-0.03, 0.03, 3)
        pos_offset = np.random.uniform(0, 0, 3)
        rpy_offset = np.array([0.0, 0.0, np.random.uniform(-np.pi, np.pi)])
        obj_fn = np.random.choice(
            [
                f'{self.grab_dir}/{self.demo_data["object"]["object_mesh"]}',
                # f'{self.grab_dir}/{self.demo_data["object"]["object_mesh"].replace("waterbottle.ply", "mug.ply")}',
                # f'{self.grab_dir}/{self.demo_data["object"]["object_mesh"].replace("waterbottle.ply", "cup.ply")}',
                # f'{self.grab_dir}/{self.demo_data["object"]["object_mesh"].replace("waterbottle.ply", "cylindermedium.ply")}',
                #f'{self.grab_dir}/{self.demo_data["object"]["object_mesh"].replace("waterbottle.ply", "cubemedium.ply")}',
            ]
        )
        self.init_object(
            step_id=self.start_idx,
            obj_fn=obj_fn,
            scale=scale,
            rgba=rgba,
            pos_offset=pos_offset,
            rpy_offset=rpy_offset,
        )
        self.aabb_size = [
            (self.obj_aabb[1][0] - self.obj_aabb[0][0]) / 0.07991508751230986,
            (self.obj_aabb[1][1] - self.obj_aabb[0][1]) / 0.08382568779629374,
            (self.obj_aabb[1][2] - self.obj_aabb[0][2]) / 0.1492939400276554,
        ]
        # reset step
        self.episode_idx += 1
        self.ep_rewards = []
        self.demo_traj = []
        self.exec_traj = []
        self.step_idx = self.start_idx

        obs = self.get_observation()
        return obs

    def count_contact(self, ee, object_id) -> int:
        count = 0
        for link_idx in ee.finger_links:
            contact_points = p.getContactPoints(
                ee.id, object_id, link_idx, -1, physicsClientId=self._sim_cid
            )
            count += len(contact_points)
        return count

    def step(self, action: np.ndarray) -> tuple:
        # step action
        self.demo_traj.append(
            np.array(
                self.transfer_gripper(
                    self.step_idx,
                    method="residual",
                    delta_action=action,
                    controllable_sides=[self.action_side],
                    vis_pose=True,
                )
            )
        )

        # retrieve observation
        obs = self.get_observation()

        # reward shaping
        obj2ee_dist = np.linalg.norm(obs[-14:-11])
        """
        obj2goal_dist = np.linalg.norm(obs[-8:-5])
        self.exec_traj.append(obs[:3])
        traj_diff_dist = [
            np.linalg.norm(d_pos - e_pos)
            for (d_pos, e_pos) in zip(self.demo_traj, self.exec_traj)
        ]
        traj_diff_dist = sum(traj_diff_dist) / len(traj_diff_dist)
        """
        n_contacts = self.count_contact(
            getattr(self, f"{self.action_side[0]}gripper"), self.obj_id
        )
        reward = -(obj2ee_dist)  # + (-obj2goal_dist)  # + (-traj_diff_dist)
        #reward += n_contacts / 10.0
        self.ep_rewards.append(reward)

        # check termination
        done = self.is_terminated()
        if done:
            print(obj2ee_dist)
            #with open("waterbottle.txt", "a") as fp:
            #    fp.write(f"{obj2ee_dist}\n")

        # update step
        self.step_idx += self.every_n_frame

        return obs, reward, done, {}

    def is_terminated(self):
        if self.step_idx > self.end_idx:
            return True
        else:
            return False

    def render(self):
        pass

    def close(self):
        p.disconnect(physicsClientId=self._sim_cid)


if __name__ == "__main__":

    sim_cid = init_sim()

    assert os.path.isdir(args.models_dir), f"{args.models_dir} does not exist."
    assert os.path.isdir(args.grab_dir), f"{args.grab_dir} does not exist."
    assert os.path.isdir(
        f"{args.grab_dir}/tools/object_meshes"
    ), f"object meshes does not exist in {args.grab_dir}."
    assert os.path.isfile(args.demo_npz_fn), f"{args.grab_dir} does not exist."

    grab_demo = TransferDemoEnv(
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
