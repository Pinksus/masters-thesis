from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

# === Inicializácia datasetu ===
nusc = NuScenes(version='v1.0-mini', dataroot='../nuscenes', verbose=True)

# Vyber počiatočný sample
sample = nusc.sample[50]
lidar_token = sample['data']['LIDAR_TOP']

# === Povolené kategórie (len vozidlá) ===
ALLOWED_CATEGORIES = [
    "vehicle.car", "vehicle.truck", "vehicle.bus", "vehicle.construction",
    "vehicle.motorcycle", "vehicle.bicycle",
]

def get_lidar_points_flat_vehicle(nusc, lidar_token):
    """Načíta point cloud v rovnakom 'flat vehicle' frame ako boxy."""
    lidar_path, _, _ = nusc.get_sample_data(lidar_token, use_flat_vehicle_coordinates=True)
    pc = LidarPointCloud.from_file(lidar_path)
    return pc.points[:3, :]

def get_pose(nusc, sample_data_token):
    """Načítaj ego_pose pre daný token."""
    sd = nusc.get('sample_data', sample_data_token)
    return nusc.get('ego_pose', sd['ego_pose_token'])

def apply_transform(box, pose_from, pose_to):
    """Transformuje bounding box zo starého egopose do nového."""
    # Rotácie treba zabaliť do Quaternion objektov
    rot_from = Quaternion(pose_from['rotation'])
    rot_to = Quaternion(pose_to['rotation'])

    # Transform matrix from pose_from
    tf_from = transform_matrix(pose_from['translation'], rot_from, inverse=False)

    # Transform matrix to pose_to (invertovaný)
    tf_to_inv = transform_matrix(pose_to['translation'], rot_to, inverse=True)

    # Prechod medzi snímkami
    tf = tf_to_inv @ tf_from

    # Aplikácia transformácie
    box.translate(tf[:3, 3])
    box.rotate(Quaternion(matrix=tf[:3, :3]))


def plot_boxes(ax, nusc, sample_data_token, ref_pose, color='b'):
    """Nakreslí boxy transformované do ref_pose."""
    _, boxes, _ = nusc.get_sample_data(sample_data_token, use_flat_vehicle_coordinates=True)
    curr_pose = get_pose(nusc, sample_data_token)

    for box in boxes:
        if not any(box.name.startswith(c) for c in ALLOWED_CATEGORIES):
            continue
        
        box = box.copy()  # Nekarási originálne boxy
        apply_transform(box, curr_pose, ref_pose)  # transformácia

        corners = box.bottom_corners()[:2, :]
        ax.plot(
            np.append(corners[0, :], corners[0, 0]),
            np.append(corners[1, :], corners[1, 0]),
            color=color,
            linewidth=1,
        )

# === Výstupný priečinok ===
output_dir = "bev_sequence_aligned"
os.makedirs(output_dir, exist_ok=True)

# === VIDEO ===
video_path = os.path.join(output_dir, "lidar_sequence.mp4")
fps = 20  # Framerate videa
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = None

# === Analýza rozsahu ===
print("Analyzujem rozsah všetkých snímok...")
current_token = lidar_token
x_min, x_max, y_min, y_max = np.inf, -np.inf, np.inf, -np.inf
num_frames = 300
lidar_tokens = []

for i in range(num_frames):
    if current_token is None:
        break
    lidar_tokens.append(current_token)

    pc = get_lidar_points_flat_vehicle(nusc, current_token)
    x_rot = pc[1, :]
    y_rot = -pc[0, :]
    x_min = min(x_min, np.min(x_rot))
    x_max = max(x_max, np.max(x_rot))
    y_min = min(y_min, np.min(y_rot))
    y_max = max(y_max, np.max(y_rot))

    next_token = nusc.get('sample_data', current_token)['next']
    if not next_token:
        break
    current_token = next_token

# Padding
padding = 5
x_min -= padding
x_max += padding
y_min -= padding
y_max += padding

print(f"Rozsah osí: X({x_min:.1f}, {x_max:.1f}), Y({y_min:.1f}, {y_max:.1f})")

# === Choose a single sample/frame ===
frame_index = 56  # change this to the frame you want
sample = nusc.sample[frame_index]
lidar_token = sample['data']['LIDAR_TOP']

# === Load LiDAR points ===
pc = get_lidar_points_flat_vehicle(nusc, lidar_token)

# Get reference pose for this frame
ref_pose = get_pose(nusc, lidar_token)

# Create plot
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_title(f"nuScenes BEV – frame {frame_index}")
ax.set_xlabel("x [m] (forward)")
ax.set_ylabel("y [m] (left)")
ax.set_aspect('equal')

# Rotate point cloud to flat vehicle coordinates
x_rot = pc[1, :]
y_rot = -pc[0, :]
ax.scatter(x_rot, y_rot, s=0.5, c='gray')
# Optionally, draw boxes from a previous frame
prev3 = lidar_token
for _ in range(3):
    prev3 = nusc.get('sample_data', prev3)['prev']
    if prev3 is None:
        break

if prev3:
    plot_boxes(ax, nusc, prev3, ref_pose=ref_pose, color='red')
# Draw current boxes
plot_boxes(ax, nusc, lidar_token, ref_pose=ref_pose, color='blue')



# Optionally, fix axis limits
ax.set_xlim(-50, 50)  # adjust based on your scene
ax.set_ylim(-50, 50)

# Show the plot
plt.show()