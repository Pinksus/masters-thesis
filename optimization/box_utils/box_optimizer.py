"""
box_optimizer.py

Utilities for LiDAR-based bounding box refinement.
Contains pure computation (no visualization).

Author: you-from-the-future 🚀
"""

from __future__ import annotations

import copy
from typing import Optional, Tuple

import numpy as np
from pyquaternion import Quaternion
from scipy.spatial import ConvexHull

# ---- project imports (uprav podľa projektu) ----
from box_utils.box_utils import BoxUtils
from bev_fit_utils import BEVFitUtils


class BoxOptimizer:
    """
    LiDAR-based box alignment using BEV fitting.
    """

    # ================================
    # Tunable constants
    # ================================
    HEIGHT_OFFSET: float = 0.30
    CENTER_THRESHOLD: float = 0.5
    MIN_POINTS: int = 3

    # ==========================================================
    # Public API
    # ==========================================================
    @staticmethod
    def compute_aligned_box(
        points: np.ndarray,
        prev_box,
    ) -> Tuple[Optional[object], np.ndarray]:
        """
        Pure computation of aligned box from LiDAR points.

        Parameters
        ----------
        points : (N,3) ndarray
            LiDAR points in global frame.
        prev_box : Box
            Previous frame bounding box.

        Returns
        -------
        aligned_box : Box | None
            Refined box or None if insufficient points.
        inside_points : ndarray
            Points used for fitting (after filtering).
        """

        # -------------------------------
        # 1. Crop points inside prev box
        # -------------------------------
        inside_xy = BoxUtils.crop_points_to_prev_box(points, prev_box)

        # -------------------------------
        # 2. Height filtering
        # -------------------------------
        z_bottom = prev_box.center[2] - prev_box.wlh[2] / 2
        inside_pts = inside_xy[
            inside_xy[:, 2] >= z_bottom + BoxOptimizer.HEIGHT_OFFSET
        ]

        if inside_pts.shape[0] < BoxOptimizer.MIN_POINTS:
            return None, inside_pts

        # -------------------------------
        # 3. Minimum area rectangle
        # -------------------------------
        rect = BoxOptimizer._minimum_area_rectangle(inside_pts[:, :2])
        if rect is None:
            return None, inside_pts

        hull_center = rect["center"]
        hull_angle = rect["angle"]

        # -------------------------------
        # 4. Center gating
        # -------------------------------
        prev_center = prev_box.center[:2]
        center_dist = np.linalg.norm(hull_center - prev_center)

        if center_dist <= BoxOptimizer.CENTER_THRESHOLD:
            final_center = hull_center
        else:
            final_center = prev_center

        # -------------------------------
        # 5. Initialize box
        # -------------------------------
        init_box = copy.deepcopy(prev_box)
        init_box.center[:2] = final_center
        init_box.orientation = Quaternion(axis=[0, 0, 1], angle=hull_angle)

        # -------------------------------
        # 6. BEV fit
        # -------------------------------
        aligned_box, _ = BEVFitUtils.fit_box(init_box, inside_pts)

        return aligned_box, inside_pts

    # ==========================================================
    # Internal helpers
    # ==========================================================
    @staticmethod
    def _minimum_area_rectangle(points_2d: np.ndarray) -> Optional[dict]:
        """
        Compute minimum-area oriented rectangle around 2D points.
        """

        if points_2d.shape[0] < 3:
            return None

        hull = ConvexHull(points_2d)
        hull_points = points_2d[hull.vertices]

        min_area = np.inf
        best_rect = None

        for i in range(len(hull_points)):
            p1 = hull_points[i]
            p2 = hull_points[(i + 1) % len(hull_points)]

            edge_dir = p2 - p1
            edge_angle = -np.arctan2(edge_dir[1], edge_dir[0])

            R = np.array(
                [
                    [np.cos(edge_angle), -np.sin(edge_angle)],
                    [np.sin(edge_angle), np.cos(edge_angle)],
                ]
            )

            rot_points = hull_points @ R.T
            min_xy = np.min(rot_points, axis=0)
            max_xy = np.max(rot_points, axis=0)
            area = np.prod(max_xy - min_xy)

            if area < min_area:
                min_area = area
                center_rot = (min_xy + max_xy) / 2

                best_rect = {
                    "center": R.T @ center_rot,
                    "size": max_xy - min_xy,
                    "angle": -edge_angle,
                }

        return best_rect