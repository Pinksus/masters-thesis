from nuscenes.nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from nuscenes_utils import NuScenesUtils
from visualizer import Visualizer
import numpy as np
import time

# =====================================
#            SETUP
# =====================================
nusc = NuScenes(version='v1.0-trainval', dataroot='../../Nuscene')
nusc_utils = NuScenesUtils(nusc)
predict_helper = PredictHelper(nusc)

MAX_DISTANCE = 50.0
MIN_LIDAR_POINTS = 5

# velocity limits
MIN_SPEED_KMH = 5.0
MAX_SPEED_KMH = 100.0
MIN_SPEED_MS = MIN_SPEED_KMH / 3.6
MAX_SPEED_MS = MAX_SPEED_KMH / 3.6

total_samples = len(nusc.sample)
print(f"Total samples: {total_samples}")

TIME_LOG_PATH = "sample_processing_time.txt"

# =====================================
#        ITERATE OVER ALL SAMPLES
# =====================================
for i, sample in enumerate(nusc.sample):

    start_time = time.time()
    print(f"Processing sample {i+1}/{total_samples}", end='\r', flush=True)

    # ego pose
    sd_token = sample['data']['LIDAR_TOP']
    ego_pose = nusc.get(
        'ego_pose',
        nusc.get('sample_data', sd_token)['ego_pose_token']
    )
    ego_translation = np.array(ego_pose['translation'])

    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)

        # -------------------------
        # 1️⃣ only vehicles
        # -------------------------
        if not ann['category_name'].startswith('vehicle'):
            continue

        # -------------------------
        # 2️⃣ enough lidar points (current frame)
        # -------------------------
        if ann['num_lidar_pts'] < MIN_LIDAR_POINTS:
            continue

        # -------------------------
        # 3️⃣ distance filter
        # -------------------------
        ann_translation = np.array(ann['translation'])
        distance = np.linalg.norm(ann_translation - ego_translation)
        if distance > MAX_DISTANCE:
            continue

        # -------------------------
        # 4️⃣ velocity filter
        # -------------------------
        instance_token = ann['instance_token']
        sample_token = sample['token']

        try:
            velocity_ms = predict_helper.get_velocity_for_agent(
                instance_token,
                sample_token
            )
        except Exception:
            continue

        if velocity_ms is None:
            continue

        if not (MIN_SPEED_MS <= velocity_ms <= MAX_SPEED_MS):
            continue

        # -------------------------
        # 5️⃣ NEW: previous frame must exist
        # -------------------------
        if ann['prev'] == '':
            continue

        prev_ann = nusc.get('sample_annotation', ann['prev'])

        # previous frame must also have enough lidar points
        if prev_ann['num_lidar_pts'] < MIN_LIDAR_POINTS:
            continue

        # -------------------------
        # passed all filters
        # -------------------------
        velocity_kmh = velocity_ms * 3.6

        lidar_hz = 20.0
        dt = 1 / lidar_hz
        distance_m = velocity_ms * dt

        Visualizer.render_annotation(
            nusc,
            ann_token,
            distance_m,
            stepback=1
        )

    elapsed_time = time.time() - start_time

    with open(TIME_LOG_PATH, "a") as f:
        f.write(f"{elapsed_time:.6f}\n")
