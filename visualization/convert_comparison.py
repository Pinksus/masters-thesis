import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
#from nuscenes.utils.geometry_utils import Box

# --- Load NuScenes mini ---
nusc = NuScenes(version='v1.0-mini', dataroot='../nuscenes')

# --- Choose sample and annotation ---
sample = nusc.sample[56]
ann_token = sample['anns'][12]

# --- Current annotation metadata ---
ann_curr = nusc.get('sample_annotation', ann_token)
instance_token = ann_curr['instance_token']

print("Current annotation:")
print("  ann token:", ann_token)
print("  instance token:", instance_token)
print("  translation (global):", ann_curr['translation'])
print()

# --- Load current LIDAR and Box (local coordinates) ---
lidar_token_curr = sample['data']['LIDAR_TOP']
_, boxes_curr, _ = nusc.get_sample_data(lidar_token_curr, selected_anntokens=[ann_token])
box_curr = boxes_curr[0]  # Box in current LIDAR frame
print("Local coordinates in current LIDAR frame (NuScenes Box):", box_curr.center)
print()

# --- Step back N frames ---
stepback = 3
frame_idx = nusc.sample.index(sample)
target_idx = frame_idx - stepback

if 0 <= target_idx < len(nusc.sample):
    sample_prev = nusc.sample[target_idx]

    # Find annotation in previous sample with the same instance_token
    prev_ann_token = None
    prev_ann = None
    for a in sample_prev['anns']:
        ann = nusc.get('sample_annotation', a)
        if ann['instance_token'] == instance_token:
            prev_ann_token = a
            prev_ann = ann
            break

    if prev_ann is None:
        print("Object not found in previous sample.")
    else:
        print("Previous annotation (3 frames back):")
        print("  ann token:", prev_ann_token)
        print("  translation (global):", prev_ann['translation'])

        # --- Convert global coordinates to local LIDAR of previous frame ---
        lidar_token_prev = sample_prev['data']['LIDAR_TOP']
        sd_prev = nusc.get('sample_data', lidar_token_prev)
        cs_prev = nusc.get('calibrated_sensor', sd_prev['calibrated_sensor_token'])
        ep_prev = nusc.get('ego_pose', sd_prev['ego_pose_token'])

        p_global_prev = np.array(prev_ann['translation'])

        # Global → ego_prev
        p_ego_prev = Quaternion(ep_prev['rotation']).inverse.rotate(p_global_prev - np.array(ep_prev['translation']))
        # Ego_prev → lidar_prev
        p_lidar_prev = Quaternion(cs_prev['rotation']).inverse.rotate(p_ego_prev - np.array(cs_prev['translation']))

        print("Local coordinates in previous LIDAR frame:", p_lidar_prev)

        # --- Transform previous global coordinates into current LIDAR frame ---
        sd_curr = nusc.get('sample_data', lidar_token_curr)
        cs_curr = nusc.get('calibrated_sensor', sd_curr['calibrated_sensor_token'])
        ep_curr = nusc.get('ego_pose', sd_curr['ego_pose_token'])

        # Global → current ego
        p_ego_curr = Quaternion(ep_curr['rotation']).inverse.rotate(p_global_prev - np.array(ep_curr['translation']))
        # Current ego → current lidar
        p_lidar_curr = Quaternion(cs_curr['rotation']).inverse.rotate(p_ego_curr - np.array(cs_curr['translation']))

        print("Previous annotation transformed into current LIDAR frame:", p_lidar_curr)

else:
    print("Stepback index out of bounds.")
