from nuscenes.utils.data_classes import Box
import numpy as np
from pyquaternion import Quaternion


class BoxTransformer:

    def __init__(self, nusc):
        self.nusc = nusc

    def global_to_lidar(self, ann, lidar_token):
        sd = self.nusc.get('sample_data', lidar_token)
        cs = self.nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
        ep = self.nusc.get('ego_pose', sd['ego_pose_token'])

        # translation
        p_global = np.array(ann['translation'])
        p_ego = Quaternion(ep['rotation']).inverse.rotate(
            p_global - np.array(ep['translation'])
        )
        p_lidar = Quaternion(cs['rotation']).inverse.rotate(
            p_ego - np.array(cs['translation'])
        )

        # rotation
        q_global = Quaternion(ann['rotation'])
        q_ego = Quaternion(ep['rotation']).inverse * q_global
        q_lidar = Quaternion(cs['rotation']).inverse * q_ego

        return Box(
            center=p_lidar,
            size=ann['size'],
            orientation=q_lidar,
            name=ann['category_name'],
            token=ann['token']
        )
