import numpy as np
import matplotlib.pyplot as plt

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Box
from pyquaternion import Quaternion


# =========================
# CONFIG
# =========================
DATAROOT = '../../nuscenes'
VERSION = 'v1.0-mini'
SAMPLE_IDX = 150


# =========================
# LOAD NUSCENES
# =========================
nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False)
sample = nusc.sample[SAMPLE_IDX]


# =========================
# LOAD + TRANSFORM LIDAR
# =========================
lidar_token = sample['data']['LIDAR_TOP']
lidar_sd = nusc.get('sample_data', lidar_token)

pc = LidarPointCloud.from_file(
    nusc.get_sample_data_path(lidar_token)
)

cs = nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])
pose = nusc.get('ego_pose', lidar_sd['ego_pose_token'])

pc.rotate(Quaternion(cs['rotation']).rotation_matrix)
pc.translate(np.array(cs['translation']))

pc.rotate(Quaternion(pose['rotation']).rotation_matrix)
pc.translate(np.array(pose['translation']))

points = pc.points[:2, :]


# =========================
# PLOT BEV
# =========================
fig, ax = plt.subplots(figsize=(10, 10))

ax.scatter(
    points[0],
    points[1],
    s=0.2,
    c='gray',
    alpha=0.5
)


# =========================
# DRAW CAR BOXES + INDEX
# =========================
order = [0, 1, 2, 3, 0]

for idx, ann_token in enumerate(sample['anns']):
    ann = nusc.get('sample_annotation', ann_token)

    if not ann['category_name'].startswith('vehicle.car'):
        continue

    box = Box(
        center=ann['translation'],
        size=ann['size'],
        orientation=Quaternion(ann['rotation'])
    )

    corners = box.corners()[:2, :].T
    bev_corners = corners[[0, 1, 2, 3]]

    ax.plot(
        bev_corners[order, 0],
        bev_corners[order, 1],
        color=(1.0, 0.65, 0.0),
        linewidth=2
    )

    ax.text(
        box.center[0],
        box.center[1],
        str(idx),
        color='red',
        fontsize=11,
        ha='center',
        va='center',
        weight='bold'
    )


# =========================
# EGO VEHICLE
# =========================
ego_xy = np.array(pose['translation'])[:2]
ax.scatter(
    ego_xy[0],
    ego_xy[1],
    c='blue',
    s=60,
    marker='x'
)


# =========================
# FINAL SETTINGS
# =========================
ax.set_aspect('equal')
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_title(f'BEV LiDAR + car annotations (sample {SAMPLE_IDX})')
ax.grid(True)

plt.show()
