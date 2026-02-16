from nuscenes.nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
import numpy as np

# ===============================
# SETUP
# ===============================
nusc = NuScenes(
    version='v1.0-trainval',
    dataroot='../../Nuscene',
    verbose=True
)

predict_helper = PredictHelper(nusc)

# ===============================
# FILTER PARAMETERS
# ===============================
MAX_DISTANCE = 50.0
MIN_LIDAR_POINTS = 5

MIN_SPEED_KMH = 5.0
MAX_SPEED_KMH = 100.0
MIN_SPEED_MS = MIN_SPEED_KMH / 3.6
MAX_SPEED_MS = MAX_SPEED_KMH / 3.6

# ===============================
# FILTER FUNCTION
# ===============================
def is_valid_annotation(nusc, predict_helper, sample, ann):

    # 1️⃣ iba vozidlá
    if not ann['category_name'].startswith('vehicle'):
        return False

    # 2️⃣ aktuálny frame musí mať min počet bodov
    if ann['num_lidar_pts'] < MIN_LIDAR_POINTS:
        return False

    # 3️⃣ vzdialenosť od ego vozidla
    sd_token = sample['data']['LIDAR_TOP']
    ego_pose = nusc.get(
        'ego_pose',
        nusc.get('sample_data', sd_token)['ego_pose_token']
    )
    ego_translation = np.array(ego_pose['translation'])
    ann_translation = np.array(ann['translation'])

    distance = np.linalg.norm(ann_translation - ego_translation)
    if distance > MAX_DISTANCE:
        return False

    # 4️⃣ rýchlosť
    try:
        velocity_ms = predict_helper.get_velocity_for_agent(
            ann['instance_token'],
            sample['token']
        )
    except Exception:
        return False

    if velocity_ms is None:
        return False

    if not (MIN_SPEED_MS <= velocity_ms <= MAX_SPEED_MS):
        return False

    # 5️⃣ predchádzajúci frame musí existovať
    if ann['prev'] == '':
        return False

    prev_ann = nusc.get('sample_annotation', ann['prev'])

    # 6️⃣ aj v predchádzajúcom frame musí byť min počet bodov
    if prev_ann['num_lidar_pts'] < MIN_LIDAR_POINTS:
        return False

    return True


# ===============================
# GLOBAL STATISTICS
# ===============================
total_vehicle_annotations = 0
appeared_count = 0
disappeared_count = 0

total_compared_frames = 0
stable_frames = 0

# ===============================
# ITERATE OVER SCENES
# ===============================
for scene_idx, scene in enumerate(nusc.scene):

    sample_token = scene['first_sample_token']
    prev_vehicle_instances = None

    while sample_token != '':

        sample = nusc.get('sample', sample_token)
        current_vehicle_instances = set()

        # ===============================
        # FILTERED VEHICLES IN CURRENT FRAME
        # ===============================
        for ann_token in sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)

            if is_valid_annotation(nusc, predict_helper, sample, ann):
                total_vehicle_annotations += 1
                current_vehicle_instances.add(ann['instance_token'])

        # ===============================
        # COMPARE t-1 -> t
        # ===============================
        if prev_vehicle_instances is not None:

            total_compared_frames += 1

            appeared = current_vehicle_instances - prev_vehicle_instances
            disappeared = prev_vehicle_instances - current_vehicle_instances

            appeared_count += len(appeared)
            disappeared_count += len(disappeared)

            if len(appeared) == 0 and len(disappeared) == 0:
                stable_frames += 1

        prev_vehicle_instances = current_vehicle_instances
        sample_token = sample['next']

# ===============================
# FINAL STATISTICS
# ===============================
print("\n========== FINAL STATISTICS ==========")
print(f"Total VALID vehicle annotations: {total_vehicle_annotations}")
print(f"Appeared vehicles: {appeared_count}")
print(f"Disappeared vehicles: {disappeared_count}")

if total_vehicle_annotations > 0:
    print(f"Appeared [%]: {100.0 * appeared_count / total_vehicle_annotations:.2f}%")
    print(f"Disappeared [%]: {100.0 * disappeared_count / total_vehicle_annotations:.2f}%")

if total_compared_frames > 0:
    print("\n========== FRAME-LEVEL STATISTICS ==========")
    print(f"Total compared frames: {total_compared_frames}")
    print(f"Stable frames: {stable_frames}")
    print(f"Stable frames [%]: {100.0 * stable_frames / total_compared_frames:.2f}%")
else:
    print("No frame comparisons were performed.")
