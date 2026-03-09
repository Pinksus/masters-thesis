from box_utils.box_utils import BoxUtils
import numpy as np
from scipy.optimize import minimize
import copy
from pyquaternion import Quaternion
from bev_fit_utils import BEVFitUtils
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull
import numpy as np
import time
import os


class BEVPlotter:

    @staticmethod
    def plot(
        ax,
        points,
        prev_box,
        prev_box_moved,
        box_curr,
        scaled_box,
        aligned_box=None,
        inside_pts=None,
        final_center=None,
    ):
        # ----------------------------
        # Draw cropped points
        # ----------------------------
        if inside_pts is not None and inside_pts.shape[0] > 0:
            ax.scatter(inside_pts[:, 0], inside_pts[:, 1], s=2, label="inside points")

        # ----------------------------
        # Draw centers
        # ----------------------------
        if final_center is not None and prev_box is not None:
            prev_center = prev_box.center[:2]
            ax.scatter(prev_center[0], prev_center[1], c="blue", s=80, marker="x", label="prev center")
            ax.scatter(final_center[0], final_center[1], c="cyan", s=80, marker="*", label="final center")

        # ----------------------------
        # Draw boxes
        # ----------------------------
        order = [3, 7, 6, 2, 3]

        def draw_box(box, style):
            if box is None:
                return
            c = box.corners().T
            ax.plot(c[order, 0], c[order, 1], **style)

        draw_box(aligned_box, dict(color="green", linewidth=2, label="aligned"))
        draw_box(scaled_box, dict(color="red", label="scaled"))
        draw_box(prev_box, dict(color="blue", linestyle="--", label="original"))
        draw_box(prev_box_moved, dict(color="purple", linestyle="-.", label="prev moved"))
        draw_box(box_curr, dict(color="orange", label="ground_truth"))

        ax.set_aspect("equal")
        ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1))




