from nuscenes.nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from nuscenes_utils import NuScenesUtils
from visualizer import Visualizer
import numpy as np
import time

# =====================================
#            EXAMPLE USAGE
# =====================================
nusc = NuScenes(version='v1.0-mini', dataroot='../../nuscenes')
nusc_utils = NuScenesUtils(nusc)
predict_helper = PredictHelper(nusc)

# vyber sample a anotáciu
sample = nusc.sample[150]
ann_token = sample['anns'][13]
ann = nusc.get('sample_annotation', ann_token)
instance_token = ann['instance_token']
sample_token = sample['token']

try:
    velocity_ms = predict_helper.get_velocity_for_agent(
        instance_token,
        sample_token
    )
except Exception:
    velocity_ms = 0.0  # ak nevieme rýchlosť

# prepočet na km/h
velocity_kmh = velocity_ms * 3.6

# -------------------------------
# vypocet vzdialenosti medzi LiDAR snimkami
# -------------------------------
lidar_hz = 20.0           # LiDAR sweep frequency (nuScenes)
dt = 1 / lidar_hz         # čas medzi snímkami v sekundách
distance_m = velocity_ms * dt

print(f"Speed: {velocity_kmh:.2f} km/h")
print(f"Estimated distance moved in {dt*1000:.0f} ms (1 LiDAR frame): {distance_m:.3f} m")

# vizualizacia
Visualizer.render_annotation(nusc, ann_token, distance_m, stepback=1)
