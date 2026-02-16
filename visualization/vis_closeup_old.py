from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
import numpy as np
import matplotlib.pyplot as plt

# === Inicializácia datasetu ===
nusc = NuScenes(version='v1.0-mini', dataroot='../nuscenes', verbose=True)

TARGET = "bfc839"   # objekt ktorý trackujeme


# ------------------------------------------------------------
#                   HELPER FUNKCIE
# ------------------------------------------------------------

def get_lidar_points_flat_vehicle(nusc, lidar_token):
    """Načíta point cloud v rovnakom frame ako boxy."""
    lidar_path, _, _ = nusc.get_sample_data(lidar_token, use_flat_vehicle_coordinates=True)
    pc = LidarPointCloud.from_file(lidar_path)
    return pc.points[:3, :]


def get_pose(nusc, sample_data_token):
    """Vráti ego_pose (pozícia a rotácia auta)."""
    sd = nusc.get('sample_data', sample_data_token)
    return nusc.get('ego_pose', sd['ego_pose_token'])


def apply_transform(box, pose_from, pose_to):
    """Transformuje bounding box zo starého egopose do aktuálneho."""
    rot_from = Quaternion(pose_from['rotation'])
    rot_to = Quaternion(pose_to['rotation'])

    tf_from = transform_matrix(pose_from['translation'], rot_from, inverse=False)
    tf_to_inv = transform_matrix(pose_to['translation'], rot_to, inverse=True)

    tf = tf_to_inv @ tf_from  # prevod medzi snímkami

    box.translate(tf[:3, 3])
    box.rotate(Quaternion(matrix=tf[:3, :3]))


def get_box_by_token(nusc, sample_data_token, target_token):
    """Nájde box pre objekt podľa token prefixu."""
    _, boxes, _ = nusc.get_sample_data(sample_data_token, use_flat_vehicle_coordinates=True)
    for box in boxes:
        if box.token.startswith(target_token):
            return box.copy()
    return None


def draw_single_box(ax, box, pose_from, pose_to, color):
    """Transformuje a vykreslí jeden bounding box."""
    box = box.copy()
    apply_transform(box, pose_from, pose_to)

    corners = box.bottom_corners()[:2, :]
    ax.plot(
        np.append(corners[0, :], corners[0, 0]),
        np.append(corners[1, :], corners[1, 0]),
        color=color,
        linewidth=2
    )
    return box.center[:2]  # vráti XY center


# ------------------------------------------------------------
#                   HLAVNÁ LOGIKA
# ------------------------------------------------------------

# Vyber snímku
frame_index = 56
lidar_token = nusc.sample[frame_index]['data']['LIDAR_TOP']

# Načítaj dáta
pc = get_lidar_points_flat_vehicle(nusc, lidar_token)
ref_pose = get_pose(nusc, lidar_token)

# Priprav graf
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_aspect('equal')
ax.set_title(f"nuScenes BEV – object {TARGET}")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")

# Lidar body (rotované)
x_rot = pc[1, :]
y_rot = -pc[0, :]
ax.scatter(x_rot, y_rot, s=0.7, c='gray', alpha=0.6)


# ------------------------
#  Previous 3 frames
# ------------------------
prev3 = lidar_token
for _ in range(3):
    prev3 = nusc.get('sample_data', prev3)['prev']
    if prev3 is None:
        break

center_prev = None
if prev3:
    box_prev = get_box_by_token(nusc, prev3, TARGET)
    if box_prev:
        pose_prev = get_pose(nusc, prev3)
        center_prev = draw_single_box(ax, box_prev, pose_prev, ref_pose, color="red")


# ------------------------
#  Current frame
# ------------------------
box_curr = get_box_by_token(nusc, lidar_token, TARGET)
center_curr = None

if box_curr:
    pose_curr = get_pose(nusc, lidar_token)
    center_curr = draw_single_box(ax, box_curr, pose_curr, ref_pose, color="blue")


# ------------------------
#  AUTO-ZOOM
# ------------------------
if center_curr is not None:
    cx, cy = center_curr
else:
    cx, cy = 0, 0

R = 5  # okolí objektu
ax.set_xlim(cx - R, cx + R)
ax.set_ylim(cy - R, cy + R)

plt.show()
