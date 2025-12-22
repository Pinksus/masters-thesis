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

def avg_distance_to_box_walls(box: Box, pts):
    """
    Vypočíta priemernú vzdialenosť bodov od stien boxu.
    - body musia byť vo vnútri boxu
    """
    # transform points to box local frame
    R = box.orientation.rotation_matrix
    t = box.center
    pts_local = (pts - t) @ R

    # nuScenes axes: x=length, y=width, z=height
    l, w, h = box.wlh[1], box.wlh[0], box.wlh[2]
    half = np.array([l/2, w/2, h/2])

    # vzdialenosť od stien = min(half - abs(coord))
    dist_x = half[0] - np.abs(pts_local[:,0])
    dist_y = half[1] - np.abs(pts_local[:,1])
    dist_z = half[2] - np.abs(pts_local[:,2])

    # priemer cez všetky body a všetky osi
    avg_dist = np.mean(np.minimum(dist_x, np.minimum(dist_y, dist_z)))
    return avg_dist


def objective_fn_centering(params, box: Box, pts_inside):
    """
    Posun (dx, dy) a rotácia (dtheta) boxu, aby sa body
    viac centrovali v boxe.
    """
    dx, dy, dtheta = params
    new_box = transform_box(box, dx, dy, dtheta)
    # všetky body vo vnútri boxu
    pts_local_inside = crop_points_to_prev_box(pts_inside, new_box, scale=1.0)
    
    # priemerna vzdialenost od stien
    avg_dist = avg_distance_to_box_walls(new_box, pts_local_inside)
    
    return avg_dist  # minimalizujeme


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


from scipy.optimize import minimize

def fit_box_two_step(box: Box, pts, max_iter=100):
    # --- krok 1: maximalizácia počtu bodov ---
    fitted_box, res1 = fit_box_to_points(box, pts, max_iter=max_iter)
    
    # --- krok 2: minimalizácia priemernej vzdialenosti od stien ---
    # vyber body vo vnútri fitted_box
    pts_inside = crop_points_to_prev_box(pts, fitted_box, scale=1.0)
    
    # optimalizácia posunu + rotácie
    x0 = [0.0, 0.0, 0.0]
    res2 = minimize(objective_fn_centering, x0, args=(fitted_box, pts_inside),
                    method='Powell', options={'maxiter': max_iter, 'disp': True})
    
    dx_opt, dy_opt, dtheta_opt = res2.x
    final_box = transform_box(fitted_box, dx_opt, dy_opt, dtheta_opt)
    
    return final_box, res1, res2


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





nusc = NuScenes(version='v1.0-mini', dataroot='../nuscenes')

sample = nusc.sample[56]
ann_token = sample['anns'][12]

# Metadata aktuálnej anotácie
ann_rec = nusc.get('sample_annotation', ann_token)
instance_token = ann_rec['instance_token']

print("Current annotation:")
print("  sample ann token:", ann_token)
print("  instance token:", instance_token)
print("  translation:", ann_rec['translation'])
print()

# krok dozadu
stepback = 3
sample_prev = nusc.sample[56 - stepback]

# Nájdeme anotáciu v staršom sample, ktorá má rovnaký instance_token
prev_ann_token = None
prev_ann_rec = None

for a in sample_prev['anns']:
    ann = nusc.get('sample_annotation', a)
    if ann['instance_token'] == instance_token:
        prev_ann_token = a
        prev_ann_rec = ann
        break

if prev_ann_token is None:
    print("Tento objekt (instance) sa v staršom sample nenachádza.")
else:
    print("Previous annotation found:")
    print("  sample ann token:", prev_ann_token)
    print("  instance token:", prev_ann_rec['instance_token'])
    print("  translation:", prev_ann_rec['translation'])
    print("  rotation:", prev_ann_rec['rotation'])
    print("  size:", prev_ann_rec['size'])
    print()

    # =============================
    #      VÝPOČET POSUNU OBJEKTU
    # =============================
    p_curr = np.array(ann_rec['translation'])
    p_prev = np.array(prev_ann_rec['translation'])
    dpos = p_curr - p_prev
    dist = np.linalg.norm(dpos)

    # časový rozdiel
    sample_curr_rec = nusc.get('sample', ann_rec['sample_token'])
    sample_prev_rec = nusc.get('sample', prev_ann_rec['sample_token'])
    dt = (sample_curr_rec['timestamp'] - sample_prev_rec['timestamp']) / 1e6

    if dt > 0:
        vel = dpos / dt
    else:
        vel = np.zeros(3)

    print("Δ position (current - previous):", dpos)
    print("Distance moved:", dist, "m")
    print("Δ time:", dt, "s")
    print("Velocity (computed):", vel, "m/s")
    print()

    # =====================================
    #      CHECK GLOBAL → LOCAL LIDAR
    # =====================================
    lidar_token_curr = sample_curr_rec['data']['LIDAR_TOP']
    sd_curr = nusc.get('sample_data', lidar_token_curr)
    cs_curr = nusc.get('calibrated_sensor', sd_curr['calibrated_sensor_token'])
    ep_curr = nusc.get('ego_pose', sd_curr['ego_pose_token'])

    # prev annotation global → current LIDAR frame
    p_global_prev = p_prev.copy()
    # global → ego
    p_ego = Quaternion(ep_curr['rotation']).inverse.rotate(p_global_prev - np.array(ep_curr['translation']))
    # ego → lidar
    p_lidar = Quaternion(cs_curr['rotation']).inverse.rotate(p_ego - np.array(cs_curr['translation']))

    print("Previous box center in current LIDAR frame:", p_lidar)

# render with the previous box (transformed inside)
render_annotation(nusc, ann_token, stepback=1)
