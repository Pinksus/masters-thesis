from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
import numpy as np

class NuScenesUtils:
    def __init__(self, nusc: NuScenes):
        self.nusc = nusc

    def get_prev_annotation(self, ann_token: str, stepback: int = 1):
        ann_rec = self.nusc.get('sample_annotation', ann_token)
        instance_token = ann_rec['instance_token']
        sample_rec = self.nusc.get('sample', ann_rec['sample_token'])
        frame_index = self.nusc.sample.index(sample_rec)
        target_idx = frame_index - stepback

        if target_idx < 0 or target_idx >= len(self.nusc.sample):
            return None

        sample_prev = self.nusc.sample[target_idx]
        for a in sample_prev['anns']:
            ann_prev = self.nusc.get('sample_annotation', a)
            if ann_prev['instance_token'] == instance_token:
                return ann_prev
        return None

    def transform_global_to_lidar(self, p_global, lidar_token):
        sd = self.nusc.get('sample_data', lidar_token)
        cs = self.nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
        ep = self.nusc.get('ego_pose', sd['ego_pose_token'])

        p_ego = Quaternion(ep['rotation']).inverse.rotate(p_global - np.array(ep['translation']))
        p_lidar = Quaternion(cs['rotation']).inverse.rotate(p_ego - np.array(cs['translation']))
        return p_lidar
