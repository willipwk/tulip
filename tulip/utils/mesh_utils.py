import os

import pybullet as p
import pymeshlab
#import trimesh


def convert_to_wavefront(in_fn: str, obj_fn=None) -> str:
    """Convert mesh file into wavefront .obj format.
    This is required for URDF, Convex decomposition support.

    Args:
        in_fn: input mesh filename.
        obj_fn: output obj filename.
    Returns:
        obj_fn: generated obj filename.
    """
    assert os.path.isfile(in_fn), f"Input {in_fn} does not exist."
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(in_fn)
    if obj_fn is None:
        obj_fn = ".".join(in_fn.split(".")[:-1]) + ".obj"
    ms.save_current_mesh(obj_fn)
    return obj_fn


def convex_decompose(
    obj_fn: str,
    out_fn: str = None,
    suffix: str = "collision",
    coacd_exec: str = None,
) -> None:
    """Convex decomposition for input mesh.

    Args:
        obj_fn: input mesh file in wavefront format.
        out_fn: convex decomposition output mesh file.
        suffix: if out_fn is not specified, it will generate according to the
                suffix.
        coacd_exec: CoACD(https://github.com/SarahWeiii/CoACD) executable path.
    """
    assert obj_fn.endswith(
        ".obj"
    ), "Convex decomposition only supports wavefront .obj format."
    if out_fn is None:
        out_fn = obj_fn.replace(".obj", f"_{suffix}.obj")
    if coacd_exec is not None:
        assert os.path.isfile(coacd_exec), "CoACD executable not found."
        os.system(f"{coacd_exec} -i {obj_fn} -o {out_fn}")
        assert os.path.isfile(out_fn), "CoACD failed to generate output."
    else:  # use vhacd
        p.vhacd(obj_fn, out_fn, "log.txt", alpha=0.04, resolution=50000)
        os.system("rm log.txt")
        assert os.path.isfile(out_fn), "VHACD failed to generate output."


"""
def get_center_of_mass(filename):
    mesh = trimesh.load(filename)
    if isinstance(mesh, trimesh.Scene):
        return mesh.centroid
    else:
        return mesh.center_mass
"""


def create_urdf_from_mesh(
    mesh_fn: str,
    urdf_fn: str,
    collision_fn: str = None,
    mass: float = 0.1,
    mu: float = 0.1,
    com: list = None,
    scale: list = [1, 1, 1],
    rgba: list = [1, 1, 1, 1],
) -> None:
    """Generate a single link urdf given input mesh file.

    Args:
        mesh_fn: input mesh filename.
        urdf_fn: output urdf filename.
        collision_fn: convex decomposed mesh file for collision.
        mass: mass in kg.
        scale: scale of x, y and z dimensions.
        rgba: color of object displayed via the URDF.
    """
    if not mesh_fn.endswith(".obj"):
        mesh_fn = convert_to_wavefront(mesh_fn)
    if collision_fn is None:
        collision_fn = mesh_fn.replace(".obj", "_collision.obj")
    gen_collision_fn = not os.path.isfile(collision_fn)
    if gen_collision_fn:
        convex_decompose(mesh_fn, collision_fn)
    if com is None:
        com = get_center_of_mass(mesh_fn)
    with open(urdf_fn, "w") as fp:
        fp.write(
            """<?xml version="1.0" ?>
<robot name="object.urdf">
  <link name="baseLink">
    <!--contact>
        <lateral_friction value="{10}" />
        <rolling_friction value="0.2" />
        <spinning_friction value="0.2" />
    </contact-->
    <inertial>
      <origin rpy="0 0 0" xyz="0 0 0"/>
       <mass value="{1}"/>
       <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
    </inertial>
    <visual>
      <origin rpy="{11} {12} {13}" xyz="0 0 0"/>
      <geometry>
        <mesh filename="{0}" scale="{2} {3} {4}"/>
      </geometry>
       <material name="white">
        <color rgba="{5} {6} {7} {8}"/>
      </material>
    </visual>
    <collision>
      <origin rpy="{11} {12} {13}" xyz="0 0 0"/>
     <geometry>
        <mesh filename="{9}" scale="{2} {3} {4}"/>
      </geometry>
    </collision>
  </link>
</robot>""".format(
                mesh_fn,
                mass,
                scale[0],
                scale[1],
                scale[2],
                rgba[0],
                rgba[1],
                rgba[2],
                rgba[3],
                collision_fn,
                mu,
                com[0],
                com[1],
                com[2],
            )
        )
    return
