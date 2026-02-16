import matplotlib.pyplot as plt
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
import numpy as np
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from nuscenes.utils.color_map import get_colormap
from PIL import Image
import copy
from nuscenes.nuscenes import NuScenes

from scipy.optimize import minimize, differential_evolution
from pyquaternion import Quaternion


def scale_box(box: Box, scale=1.5):
    """
    Returns a scaled copy of the NuScenes Box (scale is multiplicative).
    """
    new_box = copy.deepcopy(box)
    new_box.wlh[0] = box.wlh[0] * scale
    new_box.wlh[1] = box.wlh[1] * scale
    return new_box


def crop_points_to_prev_box_old(scene_pts, prev_box, scale=1.5):
    """Crop lidar points around previous box, enlarged by `scale`."""
    center, yaw, half_sizes = box_to_params_nusc(prev_box)

    # zväčšíme box (napr. 1.5x)
    half_sizes = half_sizes * scale

    # body do lokálneho systému
    local = rotate_points_inverse(scene_pts, center, yaw)

    # inside mask
    inside = np.all(np.abs(local) <= half_sizes, axis=1)
    return scene_pts[inside]


def crop_points_to_prev_box(scene_pts, box: Box, scale=1.5):
    """
    Correct crop: uses box rotation & translation directly from nuScenes Box.
    """
    # scaled box
    scaled = copy.deepcopy(box)
    scaled.wlh[:2] *= scale

    # get transform world->box
    R = scaled.orientation.rotation_matrix  # 3×3
    t = scaled.center

    # invert transform
    R_inv = R.T
    pts_local = (scene_pts - t) @ R_inv.T

    # half sizes
    w, l, h = scaled.wlh
    half = np.array([w/2, l/2, h/2])

    inside = np.all(np.abs(pts_local) <= half, axis=1)
    return scene_pts[inside]




def box_to_params_nusc(box: Box):
    """Extract center, yaw, and half sizes from nuScenes Box."""
    center = np.array(box.center)

    # yaw from quaternion
    yaw = box.orientation.yaw_pitch_roll[0]

    w, l, h = box.wlh  # width (x), length (y), height (z)
    half_sizes = np.array([w/2, l/2, h/2])
    return center, yaw, half_sizes


def rotate_points_inverse(points, center, yaw):
    """Rotate world points into local box frame."""
    c = np.cos(-yaw)
    s = np.sin(-yaw)
    R = np.array([[c, -s, 0],
                  [s,  c, 0],
                  [0,  0, 1]])
    return (points - center) @ R.T


def distance_points_to_aabb(local_points, half_sizes):
    """Distance of each point to axis-aligned box in local frame."""
    d = np.abs(local_points) - half_sizes
    d = np.maximum(d, 0)
    return np.linalg.norm(d, axis=1)


def box_fit_cost(params, scene_pts, base_center, base_yaw, half_sizes):
    dx, dy, dyaw = params

    cand_center = base_center + np.array([dx, dy, 0])
    cand_yaw = base_yaw + dyaw

    pts_local = rotate_points_inverse(scene_pts, cand_center, cand_yaw)
    dists = distance_points_to_aabb(pts_local, half_sizes)

    inside = dists < 1e-5
    frac_inside = inside.sum() / len(dists)

    mean_out = np.mean(dists[~inside]) if (~inside).any() else 0.0

    cost = mean_out + 10 * (1 - frac_inside)
    return cost


def optimize_box(scene_pts, prev_box: Box, search_xy=2.0, search_yaw=0.5):
    """Optimize the previous box to fit the current lidar point cloud."""

     # === NEW: crop points around previous box ===
    cropped = crop_points_to_prev_box(scene_pts, prev_box, scale=1.5)
    if len(cropped) < 20:
        print("Warning: too few points for optimization; skipping.")
        return prev_box

    scene_pts = cropped

    base_center, base_yaw, half_sizes = box_to_params_nusc(prev_box)

    x0 = np.array([0.0, 0.0, 0.0])  # dx, dy, dyaw

    bounds = [(-search_xy, search_xy),
              (-search_xy, search_xy),
              (-search_yaw, search_yaw)]

    # Global search
    de = differential_evolution(
        lambda p: box_fit_cost(p, scene_pts, base_center, base_yaw, half_sizes),
        bounds, maxiter=40, polish=False
    )

    # Local refine
    res = minimize(
        lambda p: box_fit_cost(p, scene_pts, base_center, base_yaw, half_sizes),
        de.x, method='Powell'
    )

    dx, dy, dyaw = res.x

    new_center = base_center + np.array([dx, dy, 0])
    new_yaw = base_yaw + dyaw

    # Construct nuScenes box
    new_box = copy.deepcopy(prev_box)
    new_box.center = new_center.tolist()
    new_box.orientation = Quaternion(axis=[0, 0, 1], angle=new_yaw)

    return new_box


def get_color(category_name: str) -> tuple[int, int, int]:
        """
        Provides the default colors based on the category names.
        This method works for the general nuScenes categories, as well as the nuScenes detection categories.
        """

        return nusc.colormap[category_name]


def render_annotation(nusc,
                      anntoken: str,
                      margin: float = 10,
                      view: np.ndarray = np.eye(4),
                      box_vis_level: BoxVisibility = BoxVisibility.ANY):

    ann_record = nusc.get('sample_annotation', anntoken)
    sample_record = nusc.get('sample', ann_record['sample_token'])

    assert 'LIDAR_TOP' in sample_record['data']

    # --- Load LIDAR data and points ---
    lidar_token = sample_record['data']['LIDAR_TOP']
    lidar_path, boxes, _ = nusc.get_sample_data(
        lidar_token,
        selected_anntokens=[anntoken]
    )

    print(boxes)

    pc = LidarPointCloud.from_file(lidar_path)
    pts = pc.points.T[:, :3]   # (x,y,z)

    # --- Create figure with 2 subplots ---
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(18, 10))
    fig.tight_layout()

    # ============================
    #       LEFT PLOT
    # ============================
    # Height map
    pc.render_height(ax_left, view=view)

    # Original box
    box = boxes[0]
    c_orig = np.array(get_color(box.name)) / 255.0
    box.render(ax_left, view=view, colors=(c_orig, c_orig, c_orig))

    # Previous / shifted box
    prev = copy.deepcopy(box)
    prev.center[0] -= 0.8
    prev.center[1] -= 0.8

    yaw = np.deg2rad(-13)
    q_rot = Quaternion(axis=[0, 0, 1], angle=yaw)
    prev.orientation = q_rot * prev.orientation

    c_prev = np.array([0.0, 0.0, 1.0])  # yellow
    prev.render(ax_left, view=view, colors=(c_prev, c_prev, c_prev))

    # Scaled box 1.5×
    scaled_prev = copy.deepcopy(prev)
    scaled_prev.wlh = prev.wlh * 1.5

    c_scaled = np.array([1.0, 0.0, 0.0])  # red
    scaled_prev.render(ax_left, view=view, colors=(c_scaled, c_scaled, c_scaled))

    # Autoscale around original box
    corners = view_points(box.corners(), view, False)[:2, :]
    ax_left.set_xlim([np.min(corners[0]) - margin, np.max(corners[0]) + margin])
    ax_left.set_ylim([np.min(corners[1]) - margin, np.max(corners[1]) + margin])
    ax_left.set_aspect('equal')
    ax_left.set_title("LIDAR + Boxes")

    # ============================
    #       RIGHT PLOT (BEV)
    # ============================

    # Použijeme tvoju funkciu na výber bodov vo vnútri boxu
    inside_pts = crop_points_to_prev_box(pts, prev)
    # scale=1.0 → pretože scaled_prev.wlh je už zväčšený o 1.5x

    ax_right.scatter(inside_pts[:, 0], inside_pts[:, 1], s=2)

    # Nakreslíme aj outline scaled boxu zhora (BEV) – pre kontrolu
    # draw BEV rectangle of scaled box
    corn = scaled_prev.corners().T  # 8×3

    # use full corner set
    print(corn)
    order = [3,7,6,2,3]

    ax_right.plot(
        corn[order, 0],
        corn[order, 1],
        linewidth=2,
        color='red'
    )


    ax_right.set_title("BEV: points inside scaled box (crop_points_to_prev_box)")
    ax_right.set_aspect('equal')

    plt.show()



nusc = NuScenes(version='v1.0-mini', dataroot='../nuscenes')

sample = nusc.sample[56]
ann_token = sample['anns'][12]

render_annotation(nusc,ann_token)
