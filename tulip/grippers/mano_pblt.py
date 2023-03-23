"""This module describes the ManoModel."""

import collections
import contextlib
import functools
import os
import pickle as pkl
import tempfile
from typing import Tuple

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R
from tulip.utils.pblt_utils import restore_states, save_states, step_sim

Joint = collections.namedtuple("Joint", ["origin", "basis", "axes", "limits"])


def joint2mat(axes, angles):
    """Compose rotation matrix of a multi-dof joint.
    Arguments:
        axes {str} -- joint axes
        angles {list} -- joint angles
    Returns:
        array -- rotation matrix
    """
    rest = "".join([i for i in "xyz" if i not in axes])
    order = axes + rest
    euler = [0.0, 0.0, 0.0]
    euler[: len(axes)] = angles[: len(axes)]
    return R.from_euler(order, euler).as_matrix()


def mat2joint(mat, axes):
    """Decompose rotation matrix of a multi-dof joint.
    Arguments:
        mat {mat3} -- rotation matrix
        axes {str} -- joint axes
    Returns:
        list -- rotation angles about joint axes
    """
    rest = "".join([i for i in "xyz" if i not in axes])
    order = axes + rest
    euler = R.from_matrix(mat).as_euler(order)
    return euler[: len(axes)]


def filter_mesh(vertices, faces, vertex_mask):
    """Get a submesh from a mesh using vertex mask.
    Arguments:
        vertices {array} -- whole mesh vertices
        faces {array} -- whole mesh faces
        vertex_mask {boolean array} -- vertex filter
    """
    index_map = np.cumsum(vertex_mask) - 1
    faces_mask = np.all(vertex_mask[faces], axis=1)
    vertices_sub = vertices[vertex_mask]
    faces_sub = index_map[faces[faces_mask]]
    return vertices_sub, faces_sub


def save_mesh_obj(filename, vertices, faces):
    """Save a mesh as an obj file.
    Arguments:
        filename {str} -- output file name
        vertices {array} -- mesh vertices
        faces {array} -- mesh faces
    """
    with open(filename, "w") as obj:
        for vert in vertices:
            obj.write("v {:f} {:f} {:f}\n".format(*vert))
        for face in faces + 1:  # Faces are 1-based, not 0-based in obj files
            obj.write("f {:d} {:d} {:d}\n".format(*face))


class ManoModel:
    """The helper class to work with a MANO hand model."""

    def __init__(self, left_hand=False, models_dir=None):
        """Load the hand model from a pickled file.
        Keyword Arguments:
            left_hand {bool} -- create a left hand model (default: {False})
            models_dir {str} -- path to the pickled model files (default: {None})
        """
        if models_dir is None:
            models_dir = os.path.expandvars("$MANO_MODELS_DIR")

        fname = f'MANO_{["RIGHT", "LEFT"][left_hand]}.pkl'
        self._model = self._load(os.path.join(models_dir, fname))
        self._is_left_hand = left_hand

    @staticmethod
    @functools.lru_cache(maxsize=2)
    def _load(path):
        """Load the model from disk.
        Arguments:
            path {str} -- path to the pickled model file
        Returns:
            chumpy array -- MANO model
        """
        with open(path, "rb") as pick_file:
            return pkl.load(pick_file, encoding="latin1")

    @property
    def is_left_hand(self):
        """This is the model of a left hand.
        Returns:
            bool -- left hand flag
        """
        return self._is_left_hand

    @property
    def faces(self):
        """Hand mesh faces indices.
        Returns:
            np.ndarray -- matrix Nf x 3, where Nf - number of faces
        """
        return self._model.get("f")

    @property
    def weights(self):
        """Vertex weights.
        Returns:
            array -- matrix Nv x Nl, where Nv - number of vertices,
                     Nl - number of links
        """
        return self._model.get("weights")

    @property
    def kintree_table(self):
        """Kinematic tree.
        Returns:
            array -- matrix 2 x Nl, where Nl - number of links
        """
        return np.int32(self._model.get("kintree_table"))

    @property
    def shapedirs(self):
        """Shape mapping matrix.
        Returns:
            array -- matrix Nv x 3 x Nb, where Nv - vertices number,
                     Nb - shape coeffs number
        """
        return self._model.get("shapedirs")

    @property
    def posedirs(self):
        """Pose mapping matrix.
        Returns:
            array -- matrix Nv x 3 x ((Nl-1)*9), where Nv - vertices number,
                     Nl - links number
        """
        return self._model.get("posedirs")

    @property
    def link_names(self):
        """Human readable link names.
        Returns:
            list -- list of link names of size Nl, where Nl - number of links
        """
        fingers = ("index", "middle", "pinky", "ring", "thumb")
        return ["palm"] + [
            "{}{}".format(f, i) for f in fingers for i in range(1, 4)
        ]

    @property
    def tip_links(self):
        """Tip link indices.
        Returns:
            list -- list of tip link indices
        """
        return [3, 6, 12, 9, 15]

    def origins(self, betas=None, pose=None, trans=None):
        """Joint origins.
        Keyword Arguments:
            betas {array} -- shape coefficients, vector 1 x 10 (default: {None})
            pose {array} -- hand pose, matrix Nl x 3 (default: {None})
            trans {array} -- translation, vector 1 x 3 (default: {None})
        Returns:
            array -- matrix Nl x 3, where Nl - number of links
        """
        origins = self._model.get("J")
        if betas is not None:
            regressor = self._model.get("J_regressor")
            origins = regressor.dot(self.vertices(betas=betas))
        if pose is not None:
            raise NotImplementedError
        if trans is not None:
            origins = origins + trans
        return origins

    def vertices(self, betas=None, pose=None, trans=None):
        """Hand mesh verticies.
        Keyword Arguments:
            betas {array} -- shape coefficients, vector 1 x 10 (default: {None})
            pose {array} -- hand pose, matrix Nl x 3 (default: {None})
            trans {array} -- translation, vector 1 x 3 (default: {None})
        Returns:
            array -- matrix Nv x 3, where Nv - number of vertices
        """
        vertices = self._model.get("v_template")
        if betas is not None:
            vertices = vertices + np.dot(self.shapedirs, betas)
        if pose is not None:
            pose = np.ravel(
                [
                    R.from_rotvec(rvec).as_matrix() - np.eye(3)
                    for rvec in pose[1:]
                ]
            )
            vertices = vertices + np.dot(self.posedirs, pose)
            raise NotImplementedError
        if trans is not None:
            vertices = vertices + trans
        return vertices


class HandModel(ManoModel):
    """Base rigid hand model.
    The model provides rigid hand kinematics description and math_utils
    from the joint space to the MANO model pose.
    """

    def __init__(self, left_hand=False, models_dir=None):
        """Initialize a HandModel.
        Keyword Arguments:
            left_hand {bool} -- create a left hand model (default: False)
            models_dir {str} -- path to the pickled model files (default: None)
        """
        super().__init__(left_hand=left_hand, models_dir=models_dir)
        self._joints = self._make_joints()
        self._basis = [joint.basis for joint in self._joints]
        self._axes = [joint.axes for joint in self._joints]
        self._dofs = [
            (u - len(self._axes[i]), u)
            for i, u in enumerate(
                np.cumsum([len(joint.axes) for joint in self._joints])
            )
        ]

        assert len(self._joints) == len(self.origins()), "Wrong joints number"
        assert all(
            [len(j.axes) == len(j.limits) for j in self._joints]
        ), "Wrong limits number"
        assert not self._joints[0].axes, "Palm joint is not fixed"

    @property
    def joints(self):
        """Joint descriptions.
        Returns:
            list -- list of Joint structures
        """
        return self._joints

    @property
    def dofs_number(self):
        """Number of degrees of freedom.
        Returns:
            int -- sum of degrees of freedom of all joints
        """
        return sum([len(joint.axes) for joint in self._joints[1:]])

    @property
    def dofs_limits(self):
        """Limits corresponding to degrees of freedom.
        Returns:
            tuple -- lower limits list, upper limits list
        """
        return list(
            zip(
                *[
                    limits
                    for joint in self._joints[1:]
                    for limits in joint.limits
                ]
            )
        )

    def angles_to_mano(self, angles, palm_basis=None):
        """Convert joint angles to a MANO pose.
        Arguments:
            angles {array} -- rigid model's dofs angles
        Keyword Arguments:
            palm_basis {mat3} -- palm basis (default: {None})
        Returns:
            array -- MANO pose, array of size N*3 where N - number of links
        """
        if len(angles) != self.dofs_number:
            raise ValueError(
                f"Expected {self.dofs_number} angles (got {len(angles)})."
            )

        def joint_to_rvec(i, angles):
            return R.from_matrix(
                self._basis[i]
                @ joint2mat(self._axes[i], angles)
                @ self._basis[i].T
            ).as_rotvec()

        rvecs = [
            joint_to_rvec(i, angles[d0:d1])
            for i, (d0, d1) in enumerate(self._dofs)
        ]
        if palm_basis is not None:
            rvecs[0] = R.from_matrix(palm_basis).as_rotvec()
        return np.ravel(rvecs)

    def mano_to_angles(self, mano_pose):
        """Convert a mano pose to joint angles of the rigid model.
        It is not guaranteed that the rigid model can ideally
        recover a mano pose.
        Arguments:
            mano_pose {array} -- MANO pose, array of size N*3 where N -
                                 number of links
        Returns:
            tuple -- dofs angles, palm_basis
        """
        rvecs = np.asarray(mano_pose).reshape((-1, 3))
        if rvecs.size != 48:
            raise ValueError(
                f"Expected 48 items in the MANO pose (got {rvecs.size})."
            )

        def rvec_to_joint(i, rvec):
            return mat2joint(
                self._basis[i].T
                @ R.from_rotvec(rvec).as_matrix()
                @ self._basis[i],
                self._axes[i],
            )

        angles = [
            angle
            for i, rvec in enumerate(rvecs)
            for angle in rvec_to_joint(i, rvec)
        ]
        palm_basis = R.from_rotvec(rvecs[0]).as_matrix()
        return angles, palm_basis

    def _make_joints(self):
        """Compute joints parameters.
        Returns:
            list -- list of joints parameters
        """
        raise NotImplementedError


class HandModel20(HandModel):
    """Heuristic rigid model with 20 degrees of freedom."""

    def _make_joints(self):
        """Compute joints parameters.
        Returns:
            list -- list of joints parameters
        """
        origin = dict(zip(self.link_names, self.origins()))
        basis = {"palm": np.eye(3)}

        def make_basis(yvec, zvec):
            mat = np.vstack([np.cross(yvec, zvec), yvec, zvec])
            return mat.T / np.linalg.norm(mat.T, axis=0)

        zvec = origin["index2"] - origin["index3"]
        yvec = np.cross(zvec, [0.0, 0.0, 1.0])
        basis["index"] = make_basis(yvec, zvec)

        zvec = origin["middle2"] - origin["middle3"]
        yvec = np.cross(zvec, origin["index1"] - origin["ring1"])
        basis["middle"] = make_basis(yvec, zvec)

        zvec = origin["ring2"] - origin["ring3"]
        yvec = np.cross(zvec, origin["middle1"] - origin["ring1"])
        basis["ring"] = make_basis(yvec, zvec)

        zvec = origin["pinky2"] - origin["pinky3"]
        yvec = np.cross(zvec, origin["ring1"] - origin["pinky1"])
        basis["pinky"] = make_basis(yvec, zvec)

        yvec = origin["thumb1"] - origin["index1"]
        zvec = np.cross(yvec, origin["thumb1"] - origin["thumb2"])
        basis["thumb0"] = make_basis(yvec, zvec)

        zvec = origin["thumb2"] - origin["thumb3"]
        yvec = np.cross(zvec, [0, -np.sin(0.96), np.cos(0.96)])
        basis["thumb"] = make_basis(yvec, zvec)

        if self.is_left_hand:
            rot = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
            basis = {key: mat @ rot for key, mat in basis.items()}

        return [
            Joint(origin["palm"], basis["palm"], "", []),
            Joint(
                origin["index1"],
                basis["index"],
                "yx",
                np.deg2rad([(-20, 20), (-10, 90)]),
            ),
            Joint(
                origin["index2"], basis["index"], "x", np.deg2rad([(0, 100)])
            ),
            Joint(
                origin["index3"], basis["index"], "x", np.deg2rad([(0, 100)])
            ),
            Joint(
                origin["middle1"],
                basis["middle"],
                "yx",
                np.deg2rad([(-30, 20), (-10, 90)]),
            ),
            Joint(
                origin["middle2"], basis["middle"], "x", np.deg2rad([(0, 100)])
            ),
            Joint(
                origin["middle3"], basis["middle"], "x", np.deg2rad([(0, 100)])
            ),
            Joint(
                origin["pinky1"],
                basis["pinky"],
                "yx",
                np.deg2rad([(-40, 20), (-10, 90)]),
            ),
            Joint(
                origin["pinky2"], basis["pinky"], "x", np.deg2rad([(0, 100)])
            ),
            Joint(
                origin["pinky3"], basis["pinky"], "x", np.deg2rad([(0, 100)])
            ),
            Joint(
                origin["ring1"],
                basis["ring"],
                "yx",
                np.deg2rad([(-30, 20), (-10, 90)]),
            ),
            Joint(origin["ring2"], basis["ring"], "x", np.deg2rad([(0, 100)])),
            Joint(origin["ring3"], basis["ring"], "x", np.deg2rad([(0, 100)])),
            Joint(
                origin["thumb1"],
                basis["thumb0"],
                "yz",
                np.deg2rad([(-10, 150), (-40, 40)]),
            ),
            Joint(
                origin["thumb2"], basis["thumb"], "x", np.deg2rad([(0, 100)])
            ),
            Joint(
                origin["thumb3"], basis["thumb"], "x", np.deg2rad([(0, 100)])
            ),
        ]


class HandModel45(HandModel):
    """Rigid model with 45 degrees of freedom."""

    def _make_joints(self):
        """Compute joints parameters.
        Returns:
            list -- list of joints parameters
        """
        origin = dict(zip(self.link_names, self.origins()))
        limits = [(-np.pi, np.pi), (-np.pi / 2, np.pi / 2), (-np.pi, np.pi)]

        return [
            Joint(origin["palm"], np.eye(3), "", []),
            Joint(origin["index1"], np.eye(3), "xyz", limits),
            Joint(origin["index2"], np.eye(3), "xyz", limits),
            Joint(origin["index3"], np.eye(3), "xyz", limits),
            Joint(origin["middle1"], np.eye(3), "xyz", limits),
            Joint(origin["middle2"], np.eye(3), "xyz", limits),
            Joint(origin["middle3"], np.eye(3), "xyz", limits),
            Joint(origin["pinky1"], np.eye(3), "xyz", limits),
            Joint(origin["pinky2"], np.eye(3), "xyz", limits),
            Joint(origin["pinky3"], np.eye(3), "xyz", limits),
            Joint(origin["ring1"], np.eye(3), "xyz", limits),
            Joint(origin["ring2"], np.eye(3), "xyz", limits),
            Joint(origin["ring3"], np.eye(3), "xyz", limits),
            Joint(origin["thumb1"], np.eye(3), "xyz", limits),
            Joint(origin["thumb2"], np.eye(3), "xyz", limits),
            Joint(origin["thumb3"], np.eye(3), "xyz", limits),
        ]


class HandBody:
    """Rigid multi-link hand body."""

    FLAG_STATIC = 1  # create static (fixed) body
    FLAG_ENABLE_COLLISION_SHAPES = 2  # enable colllision shapes
    FLAG_ENABLE_VISUAL_SHAPES = 4  # enable visual shapes
    FLAG_JOINT_LIMITS = 8  # apply joint limits
    FLAG_DYNAMICS = 16  # overide default dynamics parameters
    FLAG_USE_SELF_COLLISION = 32  # enable self collision
    FLAG_DEFAULT = sum(
        [
            FLAG_ENABLE_COLLISION_SHAPES,
            FLAG_ENABLE_VISUAL_SHAPES,
            FLAG_JOINT_LIMITS,
            FLAG_DYNAMICS,
            FLAG_USE_SELF_COLLISION,
        ]
    )

    def __init__(
        self, sim_cid, hand_model, flags=FLAG_DEFAULT, shape_betas=None
    ):
        """HandBody constructor.
        Arguments:
            sim_cid {int} -- pybullet client simulation id
            hand_model {HandModel} -- rigid hand model
        Keyword Arguments:
            flags {int} -- configuration flags (default: {FLAG_DEFAULT})
            color {list} -- color RGBA (default: {None})
            shape_betas {array} -- MANO shape beta parameters  (default: {None})
        """
        self._sim_cid = sim_cid
        self._model = hand_model
        self._flags = flags
        self._vertices = hand_model.vertices(betas=shape_betas)
        self._origin = self._model.origins()[0]
        self._joint_indices = []
        self._joint_limits = []
        self._link_mapping = {}
        self.controllable_joint_indices = []
        self._pid = self._make_body()
        self._constraint_id = self._make_constraint()
        self._apply_joint_limits()
        self._apply_dynamics()
        self.state = {}

    @property
    def id(self):
        """Body unique id in the simulator.
        Returns:
            int -- body unique id in PyBullet
        """
        return self._pid

    @property
    def link_names(self):
        """Link names for the hand model.
        Returns:
            list of string as link names.
        """
        return self._model.link_names

    @property
    def num_joints(self):
        """Number of joints for the mano hand.
        Returns:
            int -- number of unique joints
        """
        return self._num_joints

    @property
    def joint_indices(self):
        """Articulated joint indices.
        Returns:
            list -- list of joint indices
        """
        return self._joint_indices

    @property
    def joint_limits(self):
        """Articulated joints angle bounds.
        Returns:
            list -- list of tuples (lower limit, upper limit)
        """
        return self._joint_limits

    @property
    def base_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Base pose of the hand.
        Returns:
            position and quaternion of the base link.
        """
        return self.get_base_pose()

    @property
    def tip_pose(self) -> tuple:
        """Cartesian poses of finger tips.
        Returns:
            Tuple of finger tip Cartesian poses as position and quaternion.
        """
        assert len(self._model.tip_links) == 5, "Wrong number of finger tips."
        tip_poses = ()
        for hand_idx in self._model.tip_links:
            tip_pos, tip_quat = self.get_link_pose(self._link_mapping[hand_idx])
            tip_poses += ((tip_pos, tip_quat),)
        return tip_poses

    def forward_kinematics(self, q, joint_indices: list = None) -> dict:
        if joint_indices is None:
            joint_indices = self.controllable_joint_indices
        assert len(q) == len(joint_indices), "Wrong number of joint positions."
        state_id = save_states(self)
        self.reset_to_states(q, joint_indices=joint_indices)
        link_poses = {}
        for hand_idx, joint_idx in self._link_mapping.items():
            link_poses[hand_idx] = self.get_link_pose(joint_idx)
        restore_states(self, state_id)
        return link_poses

    def get_state(self):
        """Get current hand state.
        Returns:
            tuple -- base position, orientation, forces, joint positions,
                     velocities, torques
        """
        base_pos, base_orn = p.getBasePositionAndOrientation(
            self._pid, physicsClientId=self._sim_cid
        )
        if self._constraint_id != -1:
            constraint_forces = p.getConstraintState(
                self._constraint_id, physicsClientId=self._sim_cid
            )
        else:
            constraint_forces = [0.0] * 6
        joint_states = p.getJointStates(
            self._pid, self._joint_indices, physicsClientId=self._sim_cid
        )
        joints_pos, joints_vel, _, joints_torque = zip(*joint_states)
        return (
            base_pos,
            base_orn,
            constraint_forces,
            joints_pos,
            joints_vel,
            joints_torque,
        )

    def reset(self, position, orientation, joint_angles=None):
        """Reset base pose and joint angles.
        Arguments:
            position {vec3} -- position
            orientation {vec4} -- quaternion x,y,z,w
        Keyword Arguments:
            joint_angles {list} -- new angles for all articulated joints
                                   (default: {None})
        """
        p.resetBasePositionAndOrientation(
            self._pid, position, orientation, physicsClientId=self._sim_cid
        )
        if joint_angles is not None:
            for i, angle in zip(self._joint_indices, joint_angles):
                p.resetJointState(
                    self._pid, i, angle, physicsClientId=self._sim_cid
                )

    def set_target(self, position, orientation, joint_angles=None):
        """Set target base pose and joint angles.
        Arguments:
            position {vec3} -- position
            orientation {vec4} -- quaternion x,y,z,w
        Keyword Arguments:
            joint_angles {list} -- new angles for all articulated joints
                                   (default: {None})
        """
        if self._constraint_id != -1:
            p.changeConstraint(
                self._constraint_id,
                jointChildPivot=position,
                jointChildFrameOrientation=orientation,
                maxForce=10.0,
                physicsClientId=self._sim_cid,
            )
        if joint_angles is not None:
            p.setJointMotorControlArray(
                bodyUniqueId=self._pid,
                jointIndices=self._joint_indices,
                controlMode=p.POSITION_CONTROL,
                targetPositions=joint_angles,
                forces=[0.5] * len(joint_angles),
                physicsClientId=self._sim_cid,
            )

    def get_base_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        pos, quat = p.getBasePositionAndOrientation(
            self._pid,
            physicsClientId=self._sim_cid,
        )
        return np.array(pos), np.array(quat)

    def get_mano_state(self):
        """Get current hand state as a MANO model state.
        Returns:
            tuple -- trans, pose
        """
        base_pos, base_orn, _, angles, _, _ = self.get_state()
        basis = R.from_quat(base_orn).as_matrix()
        trans = base_pos - self._origin + basis @ self._origin
        mano_pose = self._model.angles_to_mano(angles, basis)
        return trans, mano_pose

    def reset_from_mano(self, trans, mano_pose):
        """Reset hand state from a Mano pose.
        Arguments:
            mano_pose {array} -- pose of the Mano model
            trans {vec3} -- hand translation
        """
        angles, basis = self._model.mano_to_angles(mano_pose)
        trans = trans + self._origin - basis @ self._origin
        self.reset(trans, R.from_matrix(basis).as_quat(), angles)

    def get_target_from_mano(self, trans, mano_pose):
        """get target hand state from a Mano pose.
        Arguments:
            mano_pose {array} -- pose of the Mano model
            trans {vec3} -- hand translation
        """
        angles, basis = self._model.mano_to_angles(mano_pose)
        trans = trans + self._origin - basis @ self._origin
        tgt_pos = trans
        tgt_orn = R.from_matrix(basis).as_quat()
        return tgt_pos, tgt_orn, angles

    def set_target_from_mano(self, trans, mano_pose):
        """Set target hand state from a Mano pose.
        Arguments:
            mano_pose {array} -- pose of the Mano model
            trans {vec3} -- hand translation
        """
        angles, basis = self._model.mano_to_angles(mano_pose)
        trans = trans + self._origin - basis @ self._origin
        self.set_target(trans, R.from_matrix(basis).as_quat(), angles)

    def _make_body(self):
        joints = self._model.joints
        link_masses = [0.2]
        link_collision_indices = [
            self._make_collision_shape(0, joints[0].basis.T)
        ]
        link_visual_indices = [self._make_visual_shape(0, joints[0].basis.T)]
        link_positions = [joints[0].origin]
        link_orientations = [R.from_matrix(joints[0].basis).as_quat()]
        link_parent_indices = [0]
        link_joint_types = [p.JOINT_FIXED]
        link_joint_axis = [[0.0, 0.0, 0.0]]
        self._link_mapping[0] = 0

        for i, j in self._model.kintree_table.T[1:]:
            parent_index = self._link_mapping[i]
            origin_rel = joints[i].basis.T @ (
                joints[j].origin - joints[i].origin
            )
            basis_rel = joints[i].basis.T @ joints[j].basis

            for axis, limits in zip(joints[j].axes, joints[j].limits):
                link_masses.append(0.0)
                link_collision_indices.append(-1)
                link_visual_indices.append(-1)
                link_positions.append(origin_rel)
                link_orientations.append(R.from_matrix(basis_rel).as_quat())
                link_parent_indices.append(parent_index + 1)
                link_joint_types.append(p.JOINT_REVOLUTE)
                link_joint_axis.append(np.eye(3)[ord(axis) - ord("x")])
                origin_rel, basis_rel = [0.0, 0.0, 0.0], np.eye(3)
                parent_index = len(link_masses) - 1
                self._link_mapping[j] = parent_index
                self._joint_limits.append(limits)
                self._joint_indices.append(parent_index)
                self.controllable_joint_indices.append(parent_index)

            link_masses[-1] = 0.02
            link_visual_indices[-1] = self._make_visual_shape(
                j, joints[j].basis.T
            )
            link_collision_indices[-1] = self._make_collision_shape(
                j, joints[j].basis.T
            )
        self._num_joints = len(self._joint_indices)

        flags = p.URDF_INITIALIZE_SAT_FEATURES
        if self.FLAG_USE_SELF_COLLISION & self._flags:
            flags |= p.URDF_USE_SELF_COLLISION
            flags |= p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS

        base_mass = 0.01
        if self.FLAG_STATIC & self._flags:
            base_mass = 0.0

        return p.createMultiBody(
            baseMass=base_mass,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=-1,
            basePosition=[0.0, 0.0, 0.0],
            baseOrientation=[0.0, 0.0, 0.0, 1.0],
            linkMasses=link_masses,
            linkCollisionShapeIndices=link_collision_indices,
            linkVisualShapeIndices=link_visual_indices,
            linkPositions=link_positions,
            linkOrientations=link_orientations,
            linkInertialFramePositions=[[0.0, 0.0, 0.0]] * len(link_masses),
            linkInertialFrameOrientations=[[0.0, 0.0, 0.0, 1.0]]
            * len(link_masses),
            linkParentIndices=link_parent_indices,
            linkJointTypes=link_joint_types,
            linkJointAxis=link_joint_axis,
            # flags=flags,
            physicsClientId=self._sim_cid,
        )

    def _make_collision_shape(self, link_index, basis):
        if self.FLAG_ENABLE_COLLISION_SHAPES & self._flags:
            with self._temp_link_mesh(link_index, True) as filename:
                return p.createCollisionShape(
                    p.GEOM_MESH,
                    fileName=filename,
                    meshScale=[1.0, 1.0, 1.0],
                    collisionFramePosition=[0, 0, 0],
                    collisionFrameOrientation=R.from_matrix(basis).as_quat(),
                    physicsClientId=self._sim_cid,
                )
        return -1

    def _make_visual_shape(self, link_index, basis):
        if self.FLAG_ENABLE_VISUAL_SHAPES & self._flags:
            with self._temp_link_mesh(link_index, False) as filename:
                return p.createVisualShape(
                    p.GEOM_MESH,
                    fileName=filename,
                    meshScale=[1.0, 1.0, 1.0],
                    rgbaColor=[0.0, 1.0, 0.0, 1.0],
                    specularColor=[1.0, 1.0, 1.0],
                    visualFramePosition=[0.0, 0.0, 0.0],
                    visualFrameOrientation=R.from_matrix(basis).as_quat(),
                    physicsClientId=self._sim_cid,
                )
        return -1

    def _make_constraint(self):
        if not self.FLAG_STATIC & self._flags:
            return p.createConstraint(
                parentBodyUniqueId=self._pid,
                parentLinkIndex=-1,
                childBodyUniqueId=-1,
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=[0.0, 0.0, 0.0],
                parentFramePosition=[0.0, 0.0, 0.0],
                childFramePosition=[0.0, 0.0, 0.0],
                physicsClientId=self._sim_cid,
            )
        return -1

    def _apply_joint_limits(self):
        if self.FLAG_JOINT_LIMITS & self._flags:
            for i, limits in zip(self._joint_indices, self._joint_limits):
                p.changeDynamics(
                    bodyUniqueId=self._pid,
                    linkIndex=i,
                    jointLowerLimit=limits[0],
                    jointUpperLimit=limits[1],
                    physicsClientId=self._sim_cid,
                )

    def _apply_dynamics(self):
        if self.FLAG_DYNAMICS & self._flags:
            p.changeDynamics(
                bodyUniqueId=self._pid,
                linkIndex=-1,
                linearDamping=10.0,
                angularDamping=10.0,
                physicsClientId=self._sim_cid,
            )

            for i in [0] + self._joint_indices:
                p.changeDynamics(
                    bodyUniqueId=self._pid,
                    linkIndex=i,
                    restitution=0.5,
                    lateralFriction=5.0,
                    spinningFriction=5.0,
                    physicsClientId=self._sim_cid,
                )

    @contextlib.contextmanager
    def _temp_link_mesh(self, link_index, collision):
        with tempfile.NamedTemporaryFile("w", suffix=".obj") as temp_file:
            threshold = 0.2
            if collision and link_index in [4, 7, 10]:
                threshold = 0.7
            vertex_mask = self._model.weights[:, link_index] > threshold
            vertices, faces = filter_mesh(
                self._vertices, self._model.faces, vertex_mask
            )
            vertices -= self._model.joints[link_index].origin
            save_mesh_obj(temp_file.name, vertices, faces)
            yield temp_file.name

    def reset_to_states(self, q, dq=None, joint_indices=None):
        if joint_indices is None:
            joint_indices = self._controllable_joint_indices
        assert len(q) == len(joint_indices), "Wrong joint positions given"
        if dq is not None:
            assert len(dq) == len(joint_indices), "Wrong joint positions given"
        else:
            dq = [0 for idx in joint_indices]
        for joint_idx, joint_pos, joint_vel in zip(joint_indices, q, dq):
            p.resetJointState(
                self._pid,
                jointIndex=joint_idx,
                targetValue=joint_pos,
                targetVelocity=joint_vel,
                physicsClientId=self._sim_cid,
            )
        step_sim(1)

    def get_joint_positions(self, joint_indices: list = None) -> np.ndarray:
        """Retrieve the current joint positions.

        Args:
            joint_indices: joint_indices to query
        Returns:
            A fixed-length array of current joint positions."""
        if joint_indices is None:
            joint_indices = self.controllable_joint_indices
        joint_states = p.getJointStates(
            bodyUniqueId=self.id,
            jointIndices=joint_indices,
            physicsClientId=self._sim_cid,
        )
        joint_pos = np.array([js[0] for js in joint_states])
        return joint_pos

    def get_joint_velocities(self, joint_indices: list = None) -> np.ndarray:
        """Retrieve the current joint velocities.

        Args:
            joint_indices: joint_indices to query
        Returns:
            A fixed-length array of current joint velocities."""
        if joint_indices is None:
            joint_indices = self.controllable_joint_indices
        joint_states = p.getJointStates(
            bodyUniqueId=self.id,
            jointIndices=joint_indices,
            physicsClientId=self._sim_cid,
        )
        joint_vel = np.array([js[1] for js in joint_states])
        return joint_vel

    def get_joint_accelerations(self, joint_indices: list = None) -> np.ndarray:
        """Retrieve the current joint accelerations.

        Args:
            joint_indices: joint_indices to query
        Returns:
            A fixed-length array of current joint accelerations."""
        raise NotImplementedError(
            "PyBullet doesn't support direct acceleration query. You may \
            consider to calculate via (v_(t) - v_(t-1)) / dt."
        )

    # todo(zyuwei) parsing the dof for force/torque values
    def get_joint_torques(self, joint_indices: list = None) -> np.ndarray:
        """Retrieve the current joint torques.

        Args:
            joint_indices: joint_indices to query
        Returns:
            A fixed-length array of current joint torques."""
        if joint_indices is None:
            joint_indices = self.controllable_joint_indices
        joint_states = p.getJointStates(
            bodyUniqueId=self.id,
            jointIndices=joint_indices,
            physicsClientId=self._sim_cid,
        )
        joint_torques = np.array([js[2] for js in joint_states])
        return joint_torques

    def get_link_pose(self, link_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        link_info = p.getLinkState(
            self._pid,
            link_idx,
            # computeForwardKinematics=True,
            physicsClientId=self._sim_cid,
        )
        pos = np.array(link_info[0])
        quat = np.array(link_info[1])
        return pos, quat
