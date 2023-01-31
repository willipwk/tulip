import numpy as np
from PIL import Image


def vis_rgb(rgb: np.ndarray, max: int) -> None:
    """Visualize a rgb array.

    Args:
        rgb: (h, w, 3) rgb image array
        max: max value in range. 1 for [0, 1] or 255 for [0, 255] array.
    """
    vis = rgb.copy()
    vis = (vis / float(max)) * 255.0
    vis = vis.astype(np.uint8)
    vis_image = Image.fromarray(vis)
    vis_image.show()


def vis_depth(depth: np.ndarray, near: float = None, far: float = None) -> None:
    """Visualize a depth array. Value of each is the true depth value.

    Args:
        depth: (h, w) depth image array
        max: max value in range. 1 for [0, 1] or 255 for [0, 255] array.
    """
    if near is None:
        near = depth.min()
    if far is None:
        far = depth.max()
    assert far > near, "Invalid near and far input to visualize depth!"
    vis = depth.copy()
    vis = (vis - near) / (far - near)
    vis *= 255.0
    vis = vis.astype(np.uint8)
    vis_image = Image.fromarray(vis)
    vis_image.show()


def vis_seg_indices(seg: np.ndarray) -> None:
    """Visualize a segmentation indices array. Value of each is the integer
    segmentation index.

    Args:
        seg: (h, w) segmentation indices array
    """
    min_idx = seg.min()
    max_idx = seg.max()
    if max_idx == min_idx:
        vis = np.zeros(seg.shape)
    else:
        vis = seg.copy()
        vis = (vis - min_idx) / (max_idx - min_idx)
    vis *= 255.0
    vis = vis.astype(np.uint8)
    vis_image = Image.fromarray(vis)
    vis_image.show()
