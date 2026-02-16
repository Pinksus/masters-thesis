import numpy as np
from scipy.optimize import minimize_scalar
import copy
from pyquaternion import Quaternion

class BEVFitUtils:


    @staticmethod
    def rotate_points_around_center(points, center, yaw):
        translated_points = points - center
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        R = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ])
        rotated = np.dot(translated_points, R.T)
        return rotated + center

    @staticmethod
    def point_line_dist_vectorized(Ps, A, B):
        PA = Ps - A
        BA = B - A
        BA_norm_sq = np.dot(BA, BA) + 1e-8
        t = np.clip(np.dot(PA, BA) / BA_norm_sq, 0.0, 1.0)
        projections = A + np.outer(t, BA)
        dists = np.linalg.norm(Ps - projections, axis=1)
        return dists

    @staticmethod
    def compute_avg_distance(points, corners):
        bottom_indices = np.argsort(corners[:, 2])[:4]
        bottom_corners = corners[bottom_indices]
        centroid = np.mean(bottom_corners, axis=0)
        angles = np.arctan2(bottom_corners[:,1] - centroid[1],
                            bottom_corners[:,0] - centroid[0])
        sort_order = np.argsort(angles)
        bottom_corners = bottom_corners[sort_order]

        edges = []
        for i in range(4):
            A = bottom_corners[i][:2]
            B = bottom_corners[(i+1)%4][:2]
            edges.append((A,B))

        points_2d = points[:, :2]
        min_dists = np.full(points_2d.shape[0], np.inf)
        for A,B in edges:
            dists = BEVFitUtils.point_line_dist_vectorized(points_2d, A, B)
            min_dists = np.minimum(min_dists, dists)

        return np.mean(min_dists)

    @staticmethod
    def fit_box(prev_box, points, scale=1.0):
        MAX_YAW_DELTA = np.deg2rad(180)  # ±45°

        best_box = copy.deepcopy(prev_box)

        corners = best_box.corners().T
        bottom_indices = np.argsort(corners[:, 2])[:4]
        bottom_corners = corners[bottom_indices]
        center = np.mean(bottom_corners, axis=0)

        adjusted = corners - center

        def yaw_cost(delta_yaw):
            rotated = BEVFitUtils.rotate_points_around_center(
                adjusted,
                np.array([0, 0, 0]),
                delta_yaw
            )
            rotated += center
            return BEVFitUtils.compute_avg_distance(points, rotated)

        # ⛔ OBMEDZENIE ROTÁCIE
        res = minimize_scalar(
            yaw_cost,
            bounds=(-MAX_YAW_DELTA, MAX_YAW_DELTA),
            method='bounded'
        )

        delta_yaw_opt = res.x

        # aplikujeme LEN malú korekciu
        best_box.center = prev_box.center.copy()
        best_box.wlh = prev_box.wlh.copy()
        best_box.orientation = (
            prev_box.orientation *
            Quaternion(axis=[0, 0, 1], angle=delta_yaw_opt)
        )

        if scale != 1.0:
            best_box.wlh *= scale

        return best_box, res

