from shapely.geometry import Polygon
from scipy.spatial import ConvexHull
import numpy as np


class BoxMetrics:
    """
    Geometry and evaluation utilities for 3D boxes.
    """

    @staticmethod
    def box_to_bev_polygon(box):
        """
        Robust BEV polygon:
        - takes 4 lowest Z corners
        - orders them CCW via ConvexHull
        - invariant to yaw + pi
        """
        corners = box.corners().T  # (8, 3)

        idx = np.argsort(corners[:, 2])[:4]
        bev = corners[idx][:, :2]

        hull = ConvexHull(bev)
        ordered = bev[hull.vertices]

        return Polygon(ordered)

    @staticmethod
    def bev_iou(box1, box2):
        if box1 is None or box2 is None:
            return 0.0

        p1 = BoxMetrics.box_to_bev_polygon(box1)
        p2 = BoxMetrics.box_to_bev_polygon(box2)

        if not p1.is_valid or not p2.is_valid:
            return 0.0

        inter = p1.intersection(p2).area
        union = p1.union(p2).area

        return 0.0 if union == 0 else inter / union