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


def global_to_lidar(nusc, p_global, lidar_token):
    """
    Convert a point from global coordinates to LIDAR_TOP coordinates of a sample.
    
    p_global: np.array([x, y, z])
    lidar_token: sample_data token for LIDAR_TOP
    """
    # 1. Load sensor + ego pose
    sd = nusc.get('sample_data', lidar_token)
    cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
    ep = nusc.get('ego_pose', sd['ego_pose_token'])

    # 2. Transform: global → ego
    p_ego = Quaternion(ep['rotation']).inverse.rotate(p_global - np.array(ep['translation']))

    # 3. Transform: ego → lidar
    p_lidar = Quaternion(cs['rotation']).inverse.rotate(p_ego - np.array(cs['translation']))

    return p_lidar


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

    fig, ax = plt.subplots(1, 1, figsize=(18, 10))
    fig.tight_layout()

    # pointcloud
    pc.render_height(ax, view=view)

    # CURRENT BOX — original color
    c_curr = np.array(get_color(box_curr.name)) / 255.0
    box_curr.render(ax, view=view, colors=(c_curr, c_curr, c_curr))

    # PREVIOUS BOX — blue
    if prev_box_transformed is not None:
        c_prev = np.array([0.0, 0.0, 1.0])
        prev_box_transformed.render(ax, view=view, colors=(c_prev, c_prev, c_prev))
        print("Previous box transformed to current LIDAR frame:", prev_box_transformed.center)
    else:
        print("OBJECT DOES NOT EXIST in previous frame.")

    # Scaled box 1.5×
    scaled_prev = copy.deepcopy(prev_box_transformed)
    scaled_prev.wlh = prev_box_transformed.wlh * 1.5

    c_scaled = np.array([1.0, 0.0, 0.0])  # red
    scaled_prev.render(ax, view=view, colors=(c_scaled, c_scaled, c_scaled))

    # autoscale
    corners = view_points(box_curr.corners(), view, False)[:2, :]
    ax.set_xlim([np.min(corners[0]) - margin, np.max(corners[0]) + margin])
    ax.set_ylim([np.min(corners[1]) - margin, np.max(corners[1]) + margin])
    ax.set_aspect('equal')

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
