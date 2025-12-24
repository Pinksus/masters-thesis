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

import numpy as np
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
from copy import deepcopy

def transform_box(box: Box, dx=0.0, dy=0.0, dtheta=0.0):
    """
    Posunie box o dx, dy a otočí okolo Z osi o dtheta (v radiánoch)
    """
    new_box = deepcopy(box)
    
    # Posun
    new_box.center[0] += dx
    new_box.center[1] += dy
    
    # Rotácia okolo Z
    q = Quaternion(axis=[0,0,1], radians=dtheta)
    new_box.orientation = q * new_box.orientation
    
    return new_box

def objective_fn(params, box: Box, pts):
    """
    params = [dx, dy, dtheta]
    Cieľ: maximalizovať počet bodov v boxe
    """
    dx, dy, dtheta = params
    new_box = transform_box(box, dx, dy, dtheta)
    
    # použijeme tvoju crop_points_to_prev_box funkciu
    pts_inside = crop_points_to_prev_box(pts, new_box, scale=1.0)
    
    # mínus, lebo minimize() minimalizuje
    return -len(pts_inside)

from scipy.optimize import minimize

def fit_box_to_points(box: Box, pts, max_iter=100):
    """
    Nájde dx, dy, dtheta, ktoré maximalizujú počet bodov vo vnútri boxu
    """
    x0 = [0.0, 0.0, 0.0]  # začneme bez posunu
    res = minimize(objective_fn, x0, args=(box, pts),
                   method='Powell',  # robustný pre malé dimenzie
                   options={'maxiter': max_iter, 'disp': True})
    
    dx_opt, dy_opt, dtheta_opt = res.x
    box_fitted = transform_box(box, dx_opt, dy_opt, dtheta_opt)
    
    return box_fitted, res


def crop_points_to_prev_box(scene_pts, box: Box, scale=1.5):
    """
    Correct crop: uses box rotation & translation directly from nuScenes Box.
    """
    # scaled box
    scaled = copy.deepcopy(box)
    scaled.wlh[:2] *= scale

    print(f"crop_points_to_prev_box: {scaled}")

    # get transform world->box
    R = scaled.orientation.rotation_matrix  # 3×3
    t = scaled.center

    # invert transform
    R_inv = R.T
    pts_local = (scene_pts - t) @ R_inv.T

    # half sizes
    w, l, h = scaled.wlh
    half = np.array([l/2, w/2, h/2])

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


def get_color(category_name: str) -> tuple[int, int, int]:
        """
        Provides the default colors based on the category names.
        This method works for the general nuScenes categories, as well as the nuScenes detection categories.
        """

        return nusc.colormap[category_name]

def render_annotation(nusc,
                      anntoken: str,
                      stepback: int = 1,
                      margin: float = 10,
                      view: np.ndarray = np.eye(4),
                      box_vis_level: BoxVisibility = BoxVisibility.ANY):

    # ---- Load current annotation ----
    ann_record = nusc.get('sample_annotation', anntoken)
    sample_record = nusc.get('sample', ann_record['sample_token'])
    instance_token = ann_record['instance_token']

    assert 'LIDAR_TOP' in sample_record['data']

    # --- Load LIDAR for current sample ---
    lidar_token = sample_record['data']['LIDAR_TOP']
    lidar_path, boxes, _ = nusc.get_sample_data(
        lidar_token,
        selected_anntokens=[anntoken]
    )

    # current box
    box_curr = boxes[0]
    print("Current box (local to current LIDAR):", box_curr.center)

    # ============================
    #   FIND PREVIOUS ANNOTATION
    # =============================
    frame_index = nusc.sample.index(sample_record)
    target_idx = frame_index - stepback
    print(f"Predosly sample bude: {target_idx}")

    prev_box_transformed = None

    if 0 <= target_idx < len(nusc.sample):
        prev_sample = nusc.sample[target_idx]

        # search annotation with same instance
        for a in prev_sample['anns']:
            ann_prev = nusc.get('sample_annotation', a)
            if ann_prev['instance_token'] == instance_token:

                # --- Transform global coordinates of previous annotation to current LIDAR frame ---
                p_global_prev = np.array(ann_prev['translation'])

                # Load current ego pose + sensor calibration
                sd_curr = nusc.get('sample_data', lidar_token)
                cs_curr = nusc.get('calibrated_sensor', sd_curr['calibrated_sensor_token'])
                ep_curr = nusc.get('ego_pose', sd_curr['ego_pose_token'])

                # Global → current ego
                p_ego_curr = Quaternion(ep_curr['rotation']).inverse.rotate(
                    p_global_prev - np.array(ep_curr['translation'])
                )
                # Current ego → current LIDAR
                p_lidar_curr = Quaternion(cs_curr['rotation']).inverse.rotate(
                    p_ego_curr - np.array(cs_curr['translation'])
                )

                q_global = Quaternion(ann_prev['rotation'])
                q_ego = Quaternion(ep_curr['rotation']).inverse * q_global
                q_lidar = Quaternion(cs_curr['rotation']).inverse * q_ego

                # Create Box object with same size and orientation as prev annotation
                from nuscenes.utils.data_classes import Box
                prev_box_transformed = Box(
                    center=p_lidar_curr,
                    size=ann_prev['size'],
                    orientation=q_lidar,
                    name=ann_prev['category_name'],
                    token=ann_prev['token']
                )
                break

    # ============================
    #       RENDERING
    # ============================
    pc = LidarPointCloud.from_file(lidar_path)
    pts = pc.points.T[:, :3]   # (x,y,z)


    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(18, 10))
    fig.tight_layout()

    # pointcloud
    pc.render_height(ax_left, view=view)

    # CURRENT BOX — original color
    c_curr = np.array(get_color(box_curr.name)) / 255.0
    box_curr.render(ax_left, view=view, colors=(c_curr, c_curr, c_curr))

    # PREVIOUS BOX — blue
    if prev_box_transformed is not None:
        c_prev = np.array([0.0, 0.0, 1.0])
        prev_box_transformed.render(ax_left, view=view, colors=(c_prev, c_prev, c_prev))
        print("Previous box transformed to current LIDAR frame:", prev_box_transformed.center)
    else:
        print("OBJECT DOES NOT EXIST in previous frame.")

    # Scaled box 1.5×
    scaled_prev = copy.deepcopy(prev_box_transformed)
    scaled_prev.wlh = prev_box_transformed.wlh * 1.5

    c_scaled = np.array([1.0, 0.0, 0.0])  # red
    scaled_prev.render(ax_left, view=view, colors=(c_scaled, c_scaled, c_scaled))

    # autoscale
    corners = view_points(box_curr.corners(), view, False)[:2, :]
    ax_left.set_xlim([np.min(corners[0]) - margin, np.max(corners[0]) + margin])
    ax_left.set_ylim([np.min(corners[1]) - margin, np.max(corners[1]) + margin])
    ax_left.set_aspect('equal')
    ax_left.set_title("LIDAR + Boxes")

    # ============================
    #       RIGHT PLOT (BEV)
    # ============================

    # Použijeme tvoju funkciu na výber bodov vo vnútri boxu
    inside_pts = crop_points_to_prev_box(pts, prev_box_transformed)
    # scale=1.0 → pretože scaled_prev.wlh je už zväčšený o 1.5x

    # BEV scatter
    ax_right.scatter(inside_pts[:, 0], inside_pts[:, 1], s=2)

    order = [3,7,6,2,3]

    # scaled box (red)
    corn_scaled = scaled_prev.corners().T
    ax_right.plot(
        corn_scaled[order, 0],
        corn_scaled[order, 1],
        linewidth=2,
        color='red',
        label='scaled prev box'
    )

    # original prev box (blue)
    corn_prev = prev_box_transformed.corners().T
    ax_right.plot(
        corn_prev[order, 0],
        corn_prev[order, 1],
        linewidth=2,
        linestyle='--',
        color='blue',
        label='original prev box'
    )

    # inside_pts = pôvodne cropované body
    fitted_box, result = fit_box_to_points(prev_box_transformed, inside_pts)

    # nakreslíme fitted box v BEV
    corn_fitted = fitted_box.corners().T
    order = [3,7,6,2,3]
    ax_right.plot(
        corn_fitted[order,0],
        corn_fitted[order,1],
        color='green',
        linewidth=2,
        label='fitted prev box'
    )

    ax_right.legend()
    ax_right.set_aspect('equal')
    ax_right.set_title("BEV: original vs scaled prev box")

    plt.show()

nusc = NuScenes(version='v1.0-mini', dataroot='../../nuscenes')

sample = nusc.sample[56]
ann_token = sample['anns'][12]

# render with the previous box (transformed inside)
render_annotation(nusc, ann_token, stepback=1)
