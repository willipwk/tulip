import numpy as np
from tulip.utils.transform_utils import pos_quat2pose_matrix


def build_projection_matrix(
    width: float,
    height: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    near: float,
    far: float,
    flatten: bool = True,
) -> np.ndarray:
    """Get OpenGL projection matrix from camera intrinsic parameters.

    Args:
        width: image width.
        height: image height.
        fx: image focal length fx.
        fy: image focal length fy.
        cy: camera principle point cx.
        cy: camera principle point cy.
        near: distance to the nearer depth clipping plane.
        far: distance to the farther depth clipping plane.
    Returns:
        4x4 projection matrix.
    """
    perspective = np.array(
        [
            [fx, 0.0, -cx, 0.0],
            [0.0, fy, -cy, 0.0],
            [0.0, 0.0, near + far, near * far],
            [0.0, 0.0, -1.0, 0.0],
        ]
    )
    ortho = build_gl_ortho(0.0, width, height, 0.0, near, far)
    proj_matrix = np.matmul(ortho, perspective)
    if flatten:
        return proj_matrix.flatten(order="F")
    else:
        return proj_matrix


def build_gl_ortho(
    left: float,
    right: float,
    bottom: float,
    top: float,
    near: float,
    far: float,
) -> np.ndarray:
    """Build glortho as the the parallel projection transformation matrix.

    Args:
        left: cooridinate for the left vertical clipping plane.
        right: coordinate for the right vertical clipping plane.
        bottom: coordinate for the bottom horizontal clipping plane.
        top: coordinate for the top horizontal clipping plane.
        near: distance to the nearer depth clipping plane.
        far: distance to the farther depth clipping plane.
    Returns:
        4x4 gl_ortho transformation matrix.
    """
    ortho = np.diag(
        [2.0 / (right - left), 2.0 / (top - bottom), -2.0 / (far - near), 1.0]
    )
    ortho[0, 3] = -(right + left) / (right - left)
    ortho[1, 3] = -(top + bottom) / (top - bottom)
    ortho[2, 3] = -(far + near) / (far - near)
    return ortho


def build_view_matrix(
    cam_pos: np.ndarray, cam_quat: np.ndarray, flatten: bool = True
) -> np.ndarray:
    """Build the view matrix.

    Args:
        cam_pos: camera position
        cam_quat: camera orientation
    Returns:
        4x4 gl view matrix.
    """
    camera_pose = pos_quat2pose_matrix(cam_pos, cam_quat)
    gl_view_matrix = np.linalg.inv(camera_pose)  # gl_view_matrix = extrinsic
    gl_view_matrix[2, :] *= -1  # flip the Z axis
    if flatten:
        return gl_view_matrix.flatten(order="F")
    else:
        return gl_view_matrix


def zbuffer_to_depth(
    z_buffer: np.ndarray,
    near: float,
    far: float,
) -> np.ndarray:
    """Convert gl depth buffer to real distance.
    Args:
        z_buffer: z_buffer array in target shape.
        near: distance to the nearer depth clipping plane.
        far: distance to the farther depth clipping plane.
    Returns:
        true depth value in mm
    """
    depth = 1.0 * far * near / (far - (far - near) * z_buffer)
    return depth
