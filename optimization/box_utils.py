from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
import numpy as np
import copy

class BoxUtils:

    @staticmethod
    def crop_points_to_prev_box(scene_pts, box: Box, scale=1.5):
        """
        Correct crop: uses box rotation & translation directly from nuScenes Box.
        """
        # scaled box
        scaled = copy.deepcopy(box)
        scaled.wlh[:2] *= scale

        #print(f"crop_points_to_prev_box: {scaled}")

        # get transform world->box
        R = scaled.orientation.rotation_matrix  # 3×3
        t = scaled.center

        # invert transform
        R_inv = R.T
        pts_local = (scene_pts - t) @ R_inv.T

        # half sizes
        w, l, h = scaled.wlh
        half = np.array([l/2, w/2, h/2])

        inside = np.all(np.abs(pts_local) <= half, axis=1)
        return scene_pts[inside]