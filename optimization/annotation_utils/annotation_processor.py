import copy
import numpy as np
from nuscenes.utils.data_classes import LidarPointCloud

from annotation_utils.annotation_loader import AnnotationLoader
from box_utils.box_transformer import BoxTransformer
from box_utils.box_optimizer import BoxOptimizer


class AnnotationProcessor:
    """
    Core processing of nuScenes annotation.
    No visualization here.
    """

    def __init__(self, nusc):
        self.nusc = nusc
        self.loader = AnnotationLoader(nusc)
        self.transformer = BoxTransformer(nusc)

    # --------------------------------------------------
    # Main processing
    # --------------------------------------------------
    def process(self, anntoken: str, distance: float, stepback: int = 1):
        """
        Process annotation and compute moved boxes.

        Returns dict with all computed data.
        """
        distance = distance * 10  # zachovávam tvoju logiku
        distance = float(distance)  # zachovávam tvoju logiku

        # ---- load current ----
        data = self.loader.load_current(anntoken)
        ann = data["annotation"]
        sample = data["sample"]
        box_curr = data["box"]

        # ---- previous annotation ----
        prev_ann = self.loader.find_prev_annotation(
            ann["instance_token"],
            self.nusc.sample.index(sample),
            stepback,
        )

        prev_box = None
        prev_box_moved = None
        scaled_box = None

        if prev_ann:
            prev_box = self.transformer.global_to_lidar(
                prev_ann, data["lidar_token"]
            )

            # ---- move previous box ----
            prev_box_moved = copy.deepcopy(prev_box)

            yaw = prev_box_moved.orientation.yaw_pitch_roll[0]
            direction = np.array([np.cos(yaw), np.sin(yaw)])
            prev_box_moved.center[:2] += direction * distance

            # ---- scaled box ----
            scaled_box = copy.deepcopy(prev_box_moved)
            scaled_box.wlh *= 1.5

        # ---- load point cloud ----
        pc = LidarPointCloud.from_file(data["lidar_path"])
        pts = pc.points.T[:, :3]

        # ✅ OPTIMIZATION STEP (FIXED)
        aligned_box, inside_pts = BoxOptimizer.compute_aligned_box(
            pts, prev_box_moved
        )

        return {
            "points": pts,
            "inside_points": inside_pts,
            "box_curr": box_curr,
            "prev_box": prev_box,
            "prev_box_moved": prev_box_moved,
            "scaled_box": scaled_box,
            "aligned_box": aligned_box,  # ⭐ IMPORTANT
            "lidar_path": data["lidar_path"],
            "distance": distance,
        }